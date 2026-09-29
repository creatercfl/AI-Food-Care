"""决策审计模块（configuration/audit.py）单元测试。

用临时目录替换 config.AUDIT_LOG_PATH，验证 JSONL 落盘、自动补时间戳、
非序列化对象的 default=str 兜底、以及多条追加；不污染真实审计日志。
"""
import _bootstrap  # noqa: F401  统一把 src 加入 sys.path
import json
import os
import tempfile
import unittest
from pathlib import Path

from configuration import config
from configuration.audit import write_audit


class TestWriteAudit(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = config.AUDIT_LOG_PATH
        config.AUDIT_LOG_PATH = Path(os.path.join(self._tmp.name, "audit.jsonl"))

    def tearDown(self):
        config.AUDIT_LOG_PATH = self._orig
        self._tmp.cleanup()

    def _read_lines(self):
        return config.AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines()

    def test_write_and_read(self):
        write_audit({"question": "感冒吃什么药", "answer": "多休息", "status": "ok"})
        lines = self._read_lines()
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertEqual(rec["question"], "感冒吃什么药")
        self.assertIn("ts", rec)  # 自动补时间戳

    def test_default_str_serialization(self):
        class Weird:
            def __str__(self):
                return "weird-obj"

        write_audit({"obj": Weird(), "status": "ok"})
        rec = json.loads(self._read_lines()[0])
        self.assertEqual(rec["obj"], "weird-obj")  # 不可序列化对象被 default=str 兜底

    def test_append_multiple(self):
        write_audit({"q": 1})
        write_audit({"q": 2})
        self.assertEqual(len(self._read_lines()), 2)

    def test_does_not_mutate_input(self):
        record = {"question": "x"}
        write_audit(record)
        self.assertNotIn("ts", record)  # ts 只加在副本上，不改调用方字典


if __name__ == "__main__":
    unittest.main()
