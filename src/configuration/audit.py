"""决策审计模块：为每一次问答留存审计日志（证据溯源 / 医疗质控 / 监管审查）。

每条记录包含：会话、问题、回答、证据列表、模型版本、时延、状态等，
以 JSONL 形式追加写入 config.AUDIT_LOG_PATH。
"""
import json
import logging
from datetime import datetime

from configuration import config

logger = logging.getLogger(__name__)


def write_audit(record: dict):
    """追加一条审计记录；失败仅告警不影响主流程。"""
    try:
        path = config.AUDIT_LOG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(record)
        data.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
    except Exception:
        logger.exception("写入审计日志失败")
