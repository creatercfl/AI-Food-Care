"""营养成分库（AI食护管家）单元测试：精确/别名/模糊查询、营养标签、格式化、副本隔离。"""
import unittest

import _bootstrap  # noqa: F401  # 必须先导入：注入 src 路径

from datasync import nutrition_store


class TestLookup(unittest.TestCase):
    def test_exact_name(self):
        rec = nutrition_store.lookup("米饭(蒸)")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["matched_name"], "米饭(蒸)")
        self.assertAlmostEqual(rec["per100g"]["energy_kcal"], 116)

    def test_alias(self):
        rec = nutrition_store.lookup("鸡胸肉")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["matched_name"], "鸡肉(胸)")

    def test_fuzzy(self):
        rec = nutrition_store.lookup("西蓝花")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["matched_name"], "西兰花")

    def test_not_found(self):
        self.assertIsNone(nutrition_store.lookup("根本不存在的食物xyz"))

    def test_empty_input(self):
        self.assertIsNone(nutrition_store.lookup(""))
        self.assertIsNone(nutrition_store.lookup(None))


class TestTags(unittest.TestCase):
    def test_high_tags(self):
        rec = nutrition_store.lookup("核桃")
        self.assertIn("高能量", rec["tags"])
        self.assertIn("高脂肪", rec["tags"])

    def test_low_tag(self):
        rec = nutrition_store.lookup("冬瓜")
        self.assertIn("低能量", rec["tags"])

    def test_no_tags(self):
        rec = nutrition_store.lookup("米饭(蒸)")
        self.assertEqual(rec["tags"], [])


class TestFormatAndIsolation(unittest.TestCase):
    def test_format_contains_fields(self):
        rec = nutrition_store.lookup("牛奶")
        text = nutrition_store.format_nutrition(rec)
        self.assertIn("牛奶", text)
        self.assertIn("能量", text)
        self.assertIn("钠", text)

    def test_lookup_returns_copy(self):
        rec = nutrition_store.lookup("鸡蛋")
        rec["per100g"]["energy_kcal"] = -1
        again = nutrition_store.lookup("鸡蛋")
        self.assertEqual(again["per100g"]["energy_kcal"], 144)


if __name__ == "__main__":
    unittest.main()
