"""统一日志配置：控制台 + 轮转文件双通道。

动机：此前仅有控制台 basicConfig，运行期故障（如向量库损坏、工具异常）
事后无据可查。这里把日志同时落到 data/logs/app.log（按大小轮转），
便于线上问题回溯与医疗质控审查。

用法：在应用入口（backend/app.py）调用一次 setup_logging() 即可；
重复调用幂等（通过 root logger 上的标记位防重复挂载 handler）。
"""
import logging
from logging.handlers import RotatingFileHandler

from configuration import config

# 日志落盘目录与文件（data/logs 已被 .gitignore 排除）
LOG_DIR = config.ROOT_DIR / "data" / "logs"
LOG_FILE = LOG_DIR / "app.log"

_FMT = "%(asctime)s %(levelname)s [%(name)s] %(filename)s:%(lineno)d - %(message)s"

# 第三方库噪声较大，统一压低到 WARNING，突出本项目日志
_NOISY_LOGGERS = (
    "httpx", "httpcore", "openai", "urllib3",
    "sentence_transformers", "chromadb", "neo4j", "asyncio",
)

_CONFIGURED_FLAG = "_smart_medical_logging_configured"


def setup_logging(level=logging.INFO, max_bytes=5 * 1024 * 1024, backup_count=5):
    """配置根日志器：控制台 + 轮转文件。幂等，可安全多次调用。"""
    root = logging.getLogger()
    if getattr(root, _CONFIGURED_FLAG, False):
        return

    root.setLevel(level)
    formatter = logging.Formatter(_FMT)

    # 控制台通道
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # 文件通道（轮转，失败不影响主流程——降级为仅控制台）
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except Exception:  # pragma: no cover - 文件系统异常时降级
        root.warning("日志文件初始化失败，降级为仅控制台输出", exc_info=True)

    # 压低第三方库噪声
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    setattr(root, _CONFIGURED_FLAG, True)
