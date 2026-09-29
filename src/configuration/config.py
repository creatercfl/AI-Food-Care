from pathlib import Path
import os
import dotenv

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent
# 显式加载项目根目录下的 .env，避免受启动时工作目录影响（敏感信息不写入代码）
dotenv.load_dotenv(ROOT_DIR / ".env")

# 数据库配置（敏感信息从环境变量读取，参见 .env / .env.example）
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "smart_medical"),
}

NEO4J_CONFIG = {
    "uri": os.getenv("NEO4J_URI", "neo4j://localhost"),
    "auth": (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
}

# Web 安全 / 限流配置
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "dev-insecure-change-me")
CHAT_RATE_LIMIT_PER_MIN = int(os.getenv("CHAT_RATE_LIMIT_PER_MIN", "20"))

# 路径配置
WEB_STATIC_DIR = ROOT_DIR / "src" / "backend" /"templates"
EMBEDDING_MODEL_PATH = ROOT_DIR / "pretrained" / "bge-base-zh-v1.5"
VECTOR_STORE_DIR = ROOT_DIR / 'data' / 'vectorstore'
# 模型配置
LLM_MODEL = "deepseek-chat"
EMBEDDING_MODEL_NAME = "bge-base-zh-v1.5"
# 审计日志（证据溯源 / 医疗质控留痕）
AUDIT_LOG_PATH = ROOT_DIR / 'data' / 'audit' / 'audit_log.jsonl'
# 其他配置
AGENT_WITH_MEMORY = True
AGENT_STREAM_OUTPUT = True