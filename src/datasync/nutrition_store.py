"""
食物营养成分库（AI食护管家 · 营养成分分析能力）。

数据来源：data/nutrition/food_nutrition.jsonl（每 100g 可食部，参考《中国食物成分表》常见值）。
提供精确名 / 别名 / 模糊三级匹配查询，并按阈值计算营养标签（高能量/高脂肪/高钠/低能量）。
本模块为纯标准库实现，不依赖向量库与嵌入模型，可离线单测。
"""
import json
import logging
from difflib import get_close_matches

from configuration import config

logger = logging.getLogger(__name__)

NUTRITION_FILE = config.ROOT_DIR / "data" / "nutrition" / "food_nutrition.jsonl"

# 营养标签阈值（每 100g 可食部）
HIGH_ENERGY_KCAL = 300
HIGH_FAT_G = 10
HIGH_SODIUM_MG = 400
LOW_ENERGY_KCAL = 30

_table = None   # name -> record
_alias = None   # lower(alias or name) -> name


def _load():
    """懒加载营养表并构建别名索引。"""
    global _table, _alias
    if _table is not None:
        return
    _table = {}
    _alias = {}
    for line in NUTRITION_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        _table[rec["name"]] = rec
        _alias[rec["name"].strip().lower()] = rec["name"]
        for a in rec.get("aliases", []):
            _alias[a.strip().lower()] = rec["name"]
    logger.info("营养成分表加载完成：%d 种食物，%d 个别名", len(_table), len(_alias))


def _tags(record):
    """按阈值计算营养标签。"""
    p = record["per100g"]
    tags = []
    if p["energy_kcal"] >= HIGH_ENERGY_KCAL:
        tags.append("高能量")
    if p["fat_g"] >= HIGH_FAT_G:
        tags.append("高脂肪")
    if p["sodium_mg"] >= HIGH_SODIUM_MG:
        tags.append("高钠")
    if p["energy_kcal"] <= LOW_ENERGY_KCAL:
        tags.append("低能量")
    return tags


def lookup(food):
    """
    查询食物营养成分。
    匹配顺序：精确名 -> 别名 -> 模糊（difflib, cutoff=0.6）。
    :return: 命中返回 dict（含 matched_name / tags / per100g 副本）；未命中返回 None。
    """
    _load()
    if not food:
        return None
    key = food.strip().lower()
    name = _alias.get(key)
    if not name and food.strip() in _table:
        name = food.strip()
    if not name:
        cands = get_close_matches(key, list(_alias.keys()), n=1, cutoff=0.6)
        if cands:
            name = _alias[cands[0]]
    if not name:
        return None
    src = _table[name]
    rec = dict(src)
    rec["per100g"] = dict(src["per100g"])
    rec["tags"] = _tags(src)
    rec["matched_name"] = name
    return rec


def format_nutrition(rec):
    """把营养记录格式化为供 LLM 组答的文本。"""
    p = rec["per100g"]
    lines = [
        f"食物：{rec['matched_name']}（每100g可食部）",
        f"能量 {p['energy_kcal']} kcal｜蛋白质 {p['protein_g']} g｜脂肪 {p['fat_g']} g"
        f"｜碳水化合物 {p['carb_g']} g｜钠 {p['sodium_mg']} mg",
    ]
    if rec.get("tags"):
        lines.append("营养标签：" + "、".join(rec["tags"]))
    if rec.get("note"):
        lines.append("提示：" + rec["note"])
    return "\n".join(lines)
