"""安全护栏模块（agent/guardrail.py）单元测试。

覆盖：人群识别、红旗症状识别、用药红线比对、人群约束构建、生成后护栏整体行为。
全部为纯规则函数，确定性强、离线可跑。
"""
import _bootstrap  # noqa: F401  必须最先导入：绕过 agent/__init__，避免打开 Chroma
import unittest

from agent.guardrail import (
    detect_populations,
    detect_red_flags,
    check_drug_redlines,
    build_population_constraint,
    run_post_guardrail,
)


class TestDetectPopulations(unittest.TestCase):
    def test_pregnant(self):
        self.assertEqual(detect_populations("怀孕5个月感冒了能吃什么药"), ["孕妇"])

    def test_child(self):
        self.assertEqual(detect_populations("小孩发烧38.5度吃什么药"), ["儿童"])

    def test_lactation(self):
        self.assertEqual(detect_populations("哺乳期能吃这个药吗"), ["哺乳期妇女"])

    def test_elderly(self):
        self.assertEqual(detect_populations("老人血压高怎么办"), ["老年人"])

    def test_no_population(self):
        # 普通问题不应误判出特殊人群
        self.assertEqual(detect_populations("感冒了吃什么药"), [])


class TestDetectRedFlags(unittest.TestCase):
    def test_hit_multiple(self):
        flags = detect_red_flags("突然胸痛呼吸困难怎么办")
        self.assertIn("胸痛", flags)
        self.assertIn("呼吸困难", flags)

    def test_none(self):
        self.assertEqual(detect_red_flags("普通感冒流鼻涕"), [])


class TestDrugRedlines(unittest.TestCase):
    def test_pregnant_ibuprofen(self):
        hits = check_drug_redlines("可以服用布洛芬缓解", ["孕妇"])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["drug"], "布洛芬")
        self.assertEqual(hits[0]["population"], "孕妇")

    def test_child_aspirin(self):
        hits = check_drug_redlines("建议用阿司匹林退热", ["儿童"])
        self.assertTrue(any(h["drug"] == "阿司匹林" for h in hits))

    def test_child_ibuprofen_not_redlined(self):
        # 布洛芬不在儿童红线表内，不应误报
        self.assertEqual(check_drug_redlines("可用布洛芬", ["儿童"]), [])

    def test_no_population(self):
        self.assertEqual(check_drug_redlines("服用布洛芬", []), [])


class TestBuildPopulationConstraint(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(build_population_constraint([]), "")

    def test_pregnant(self):
        text = build_population_constraint(["孕妇"])
        self.assertIn("孕妇", text)
        self.assertIn("安全约束", text)


class TestRunPostGuardrail(unittest.TestCase):
    def test_redflag_prefixed(self):
        answer, record = run_post_guardrail("突然胸痛", "建议尽快就医")
        self.assertTrue(answer.startswith("> 🚨"))
        self.assertIn("胸痛", record["red_flags"])

    def test_redline_appended(self):
        answer, record = run_post_guardrail("怀孕感冒了", "可以吃布洛芬")
        self.assertIn("禁忌警告", answer)
        self.assertEqual(record["populations"], ["孕妇"])
        self.assertTrue(record["redlines"])

    def test_clean_passthrough(self):
        answer, record = run_post_guardrail("感冒了吃什么药", "多休息多喝水")
        self.assertEqual(answer, "多休息多喝水")  # 无红旗无红线，回答原样返回
        self.assertEqual(record["populations"], [])
        self.assertEqual(record["red_flags"], [])
        self.assertEqual(record["redlines"], [])


if __name__ == "__main__":
    unittest.main()
