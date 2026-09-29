"""Supervisor 多专家路由（agent/supervisor.py）单元测试。

只测纯逻辑部分：关键词兜底分诊 `_keyword_route` 与专家指令拼装
`build_expert_instruction`；`route()` 依赖真实 LLM，不在此单测范围
（其端到端行为由 eval/ 跑批评测覆盖）。
"""
import _bootstrap  # noqa: F401  必须最先导入：绕过 agent/__init__，避免打开 Chroma
import unittest

from agent.supervisor import (
    _keyword_route,
    build_expert_instruction,
    EXPERTS,
    DEFAULT_EXPERT,
)


class TestKeywordRoute(unittest.TestCase):
    def test_medication(self):
        self.assertEqual(_keyword_route("感冒了吃什么药"), "medication")

    def test_exam(self):
        self.assertEqual(_keyword_route("体检发现尿酸偏高说明什么"), "exam")

    def test_special_population(self):
        self.assertEqual(_keyword_route("怀孕期间要注意什么"), "special_population")

    def test_emergency(self):
        self.assertEqual(_keyword_route("突然昏迷抽搐怎么办"), "emergency")

    def test_default_fallback(self):
        # 全不命中关键词时回落到默认专家
        self.assertEqual(_keyword_route("随便说点别的xyz"), DEFAULT_EXPERT)


class TestBuildExpertInstruction(unittest.TestCase):
    def test_basic(self):
        text = build_expert_instruction({"expert": "medication", "need_guideline": False})
        self.assertIn("临床用药专家", text)
        self.assertNotIn("guideline_search", text)

    def test_need_guideline(self):
        text = build_expert_instruction({"expert": "diet", "need_guideline": True})
        self.assertIn("guideline_search", text)


class TestExpertsIntegrity(unittest.TestCase):
    def test_six_experts(self):
        self.assertEqual(len(EXPERTS), 6)

    def test_default_in_experts(self):
        self.assertIn(DEFAULT_EXPERT, EXPERTS)

    def test_each_expert_fields(self):
        for key, exp in EXPERTS.items():
            for field in ("name", "scope", "keywords", "guidance"):
                self.assertIn(field, exp, f"{key} 缺少字段 {field}")


if __name__ == "__main__":
    unittest.main()
