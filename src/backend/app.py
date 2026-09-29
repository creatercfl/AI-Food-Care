import uvicorn
import time
import logging
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from configuration import config
from configuration.logging_config import setup_logging
from datasync import nutrition_store

# 尽早初始化日志：在导入 chat_service（会加载嵌入模型等重依赖）之前完成，
# 以便压制第三方库导入期噪声，并确保后续所有应用日志都进入文件通道
setup_logging()

from chat_service import ChatService  # noqa: E402
from backend.schemas import Question, Answer  # noqa: E402

logger = logging.getLogger(__name__)

app = FastAPI(title="智医助手 KG-QA", version="1.0")

# 会话签名密钥从配置（环境变量）读取，杜绝硬编码泄露
app.add_middleware(
    SessionMiddleware,
    secret_key=config.SESSION_SECRET_KEY,  # 用于签名 Cookie
    max_age=3600,  # session 过期时间（秒）
    https_only=False,  # 开发时设为 False，生产建议 True
    same_site="lax"    # 安全策略
)


class RateLimitMiddleware:
    """基于客户端 IP 的滑动窗口限流（纯 ASGI 实现）。

    采用纯 ASGI 而非 BaseHTTPMiddleware，避免后者对 /chat 流式响应的缓冲干扰；
    仅约束高频写接口 POST /chat，超限返回 429，防止大模型接口被刷爆。
    进程内内存计数，适用于单实例部署。
    """

    def __init__(self, app, max_per_min: int, window: int = 60):
        self.app = app
        self.max_per_min = max(1, max_per_min)
        self.window = window
        self._hits = defaultdict(deque)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/chat" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        client_ip = client[0] if client else "unknown"
        now = time.time()
        bucket = self._hits[client_ip]
        # 单线程事件循环内“清理-判定-追加”之间无 await，天然无竞态
        while bucket and now - bucket[0] > self.window:
            bucket.popleft()
        if len(bucket) >= self.max_per_min:
            retry_after = int(self.window - (now - bucket[0])) + 1
            logger.warning("限流触发：IP=%s 已达 %d 次/分钟", client_ip, self.max_per_min)
            response = JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁，请稍后再试。"},
                headers={"Retry-After": str(max(1, retry_after))},
            )
            await response(scope, receive, send)
            return
        bucket.append(now)
        await self.app(scope, receive, send)


app.add_middleware(RateLimitMiddleware, max_per_min=config.CHAT_RATE_LIMIT_PER_MIN)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """全局异常兜底：统一返回结构化错误，避免堆栈细节直泄前端。"""
    logger.exception("未处理异常：%s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "服务器内部错误，请稍后再试。"})


app.mount("/static", StaticFiles(directory=str(config.WEB_STATIC_DIR)), name="static")

service = ChatService()

@app.get("/")
def read_root():
    return RedirectResponse("/static/index.html")

@app.get("/health")
def health():
    """健康检查：供部署探活 / 监控使用。"""
    return {"status": "ok", "service": "smart_medical_kgqa", "version": "1.0"}


@app.get("/nutrition")
def nutrition(food: str):
    """营养成分确定性查询（AI食护管家）：不经 LLM，直接返回每100g成分与营养标签。"""
    rec = nutrition_store.lookup(food)
    if not rec:
        return JSONResponse(status_code=404, content={"detail": f"营养成分表未收录：{food}"})
    return rec

@app.post("/chat")
def read_item(question: Question,request:Request):

    # 维护Session，使用Session来唯一标记维护，通过langchain agent来保存对话历史记录
    session = request.session
    # 用户首次输入时，没有session_id，需要生成session_id
    if "session_id" not in session:
        import uuid
        session["session_id"] = str(uuid.uuid4())

    current_session_id = session["session_id"]
    # print("当前 Session ID:", current_session_id)

    return StreamingResponse(service.chat(question.message,session_id=current_session_id), media_type="text/plain")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
