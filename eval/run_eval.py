"""智医助手 KG-QA 评测跑批脚本。

对评测集（eval/eval_set.jsonl）逐题走完整问答链路（HTTP /chat），
解析回答正文与结构化证据块，从多个维度客观打分并汇总指标，产出
JSON + Markdown 报告到 eval/reports/。

评测维度（口径见 README 与报告头）：
- success            ：HTTP 200 且非兜底文案、答案非空
- expert_ok          ：Supervisor 分诊专家命中期望（支持可接受集合）
- population_ok      ：护栏人群识别命中期望
- redflag_ok         ：红旗症状触发与期望一致（含"不该触发时不误触发"）
- source_ok          ：证据来源路由符合期望（kg=图谱有命中 / guideline=触发指南回退 / any=不校验）
- answer_ok          ：答案关键要点命中比例 >= 阈值
- evidence_ok        ：至少有一条溯源证据
- pass               ：以上关键维度全部通过

用法：
    python eval/run_eval.py --base-url http://127.0.0.1:8000
    python eval/run_eval.py --limit 2      # 只跑前 2 题（探测/冒烟）
"""
import argparse
import datetime as dt
import http.cookiejar
import json
import statistics
import time
import urllib.request
from pathlib import Path

SENT = "\n<!--EVIDENCE-->\n"
FALLBACK_PHRASES = ["系统暂时无法响应", "抱歉，暂时未能获取到相关结果"]
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SET = ROOT / "eval" / "eval_set.jsonl"
DEFAULT_OUT = ROOT / "eval" / "reports"


def load_cases(path, limit=None):
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases[:limit] if limit else cases


def ask(base_url, question, timeout=240):
    """发一题问答，独立 cookie 保证会话隔离。返回 (answer, evidence, status, latency_ms)。"""
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat",
        data=json.dumps({"message": question}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    r = op.open(req, timeout=timeout)
    text = r.read().decode("utf-8", "ignore")
    latency = int((time.time() - t0) * 1000)
    answer, evidence = text, []
    if SENT in text:
        answer, evjson = text.split(SENT, 1)
        try:
            evidence = json.loads(evjson).get("evidence", [])
        except Exception:
            evidence = []
    return answer, evidence, r.status, latency


def pick(evidence, tool):
    return next((e for e in evidence if e.get("tool") == tool), None)


def score_case(case, answer, evidence, http_status, latency):
    """对单题打分，返回结果 dict。"""
    sup = pick(evidence, "supervisor") or {}
    guard = pick(evidence, "guardrail") or {}
    queries = [e for e in evidence if e.get("tool") == "neo4j_query"]
    guide = pick(evidence, "guideline_rag")

    actual_expert = sup.get("expert")
    # 护栏记录存在时以其人群/红旗为准（生成后校验结果）
    actual_pops = guard.get("populations", []) if guard else []
    actual_flags = guard.get("red_flags", []) if guard else []
    kg_rows = sum(q.get("row_count", 0) for q in queries)
    guide_hits = guide.get("hit_count", 0) if guide else 0

    # success
    is_fallback = any(p in answer for p in FALLBACK_PHRASES)
    success = (http_status == 200) and (not is_fallback) and (len(answer.strip()) >= 20)

    # expert
    exp = case.get("expect_expert")
    if exp is None:
        expert_ok, expert_checked = True, False
    else:
        accept = exp if isinstance(exp, list) else [exp]
        expert_ok, expert_checked = (actual_expert in accept), True

    # population
    exp_pops = set(case.get("expect_populations") or [])
    population_ok = exp_pops == set(actual_pops or [])
    population_checked = bool(exp_pops)

    # red flag
    exp_flag = bool(case.get("expect_red_flags"))
    redflag_ok = exp_flag == (len(actual_flags) > 0)

    # source routing
    src = case.get("expect_source", "any")
    if src == "kg":
        source_ok = kg_rows > 0
    elif src == "guideline":
        source_ok = guide_hits > 0
    else:
        source_ok = True
    source_checked = src in ("kg", "guideline")

    # answer keywords
    kws = case.get("answer_keywords") or []
    thr = case.get("answer_keywords_threshold", 0.5)
    hit = sum(1 for k in kws if k in answer)
    hit_ratio = (hit / len(kws)) if kws else 1.0
    answer_ok = hit_ratio >= thr

    # evidence
    evidence_ok = len(evidence) >= 1

    passed = success and expert_ok and population_ok and redflag_ok and source_ok and answer_ok

    return {
        "id": case["id"],
        "category": case.get("category"),
        "question": case["question"],
        "http_status": http_status,
        "latency_ms": latency,
        "actual_expert": actual_expert,
        "expert_confidence": sup.get("confidence"),
        "need_guideline": sup.get("need_guideline"),
        "actual_populations": actual_pops,
        "actual_red_flags": actual_flags,
        "kg_rows": kg_rows,
        "guide_hits": guide_hits,
        "answer_len": len(answer.strip()),
        "keyword_hit_ratio": round(hit_ratio, 3),
        "checks": {
            "success": success,
            "expert_ok": expert_ok,
            "expert_checked": expert_checked,
            "population_ok": population_ok,
            "population_checked": population_checked,
            "redflag_ok": redflag_ok,
            "source_ok": source_ok,
            "source_checked": source_checked,
            "answer_ok": answer_ok,
            "evidence_ok": evidence_ok,
        },
        "pass": passed,
        "answer_head": answer.strip()[:120],
    }


def summarize(results):
    n = len(results)
    def cnt(pred):
        return sum(1 for r in results if pred(r))
    def rate(a, b):
        return round(a / b, 4) if b else None

    expert_denom = cnt(lambda r: r["checks"]["expert_checked"])
    pop_denom = cnt(lambda r: r["checks"]["population_checked"])
    src_denom = cnt(lambda r: r["checks"]["source_checked"])
    lat = [r["latency_ms"] for r in results if r["checks"]["success"]]

    return {
        "total": n,
        "success_rate": rate(cnt(lambda r: r["checks"]["success"]), n),
        "overall_pass_rate": rate(cnt(lambda r: r["pass"]), n),
        "expert_accuracy": rate(cnt(lambda r: r["checks"]["expert_checked"] and r["checks"]["expert_ok"]), expert_denom),
        "population_accuracy": rate(cnt(lambda r: r["checks"]["population_checked"] and r["checks"]["population_ok"]), pop_denom),
        "redflag_accuracy": rate(cnt(lambda r: r["checks"]["redflag_ok"]), n),
        "source_routing_accuracy": rate(cnt(lambda r: r["checks"]["source_checked"] and r["checks"]["source_ok"]), src_denom),
        "answer_keyword_pass_rate": rate(cnt(lambda r: r["checks"]["answer_ok"]), n),
        "evidence_coverage": rate(cnt(lambda r: r["checks"]["evidence_ok"]), n),
        "denominators": {
            "expert_checked": expert_denom,
            "population_checked": pop_denom,
            "source_checked": src_denom,
        },
        "latency_ms": {
            "avg": int(statistics.mean(lat)) if lat else None,
            "p50": int(statistics.median(lat)) if lat else None,
            "max": max(lat) if lat else None,
        },
        "by_category": _by_category(results),
    }


def _by_category(results):
    cats = {}
    for r in results:
        c = r.get("category") or "other"
        cats.setdefault(c, {"total": 0, "pass": 0})
        cats[c]["total"] += 1
        cats[c]["pass"] += 1 if r["pass"] else 0
    for c, v in cats.items():
        v["pass_rate"] = round(v["pass"] / v["total"], 4) if v["total"] else None
    return cats


def write_reports(summary, results, out_dir, base_url, set_path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"eval_report_{ts}.json"
    md_path = out_dir / f"eval_report_{ts}.md"

    payload = {
        "meta": {
            "timestamp": ts,
            "base_url": base_url,
            "eval_set": str(set_path),
            "llm_model": "deepseek-chat",
        },
        "summary": summary,
        "results": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def pct(x):
        return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "—"

    lines = []
    lines.append(f"# 智医助手 KG-QA 评测报告 · {ts}\n")
    lines.append(f"- 服务地址：`{base_url}`  评测集：`{Path(set_path).name}`（{summary['total']} 题）\n")
    lines.append("## 一、总体指标\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    lines.append(f"| 请求成功率 | {pct(summary['success_rate'])} |")
    lines.append(f"| **综合通过率** | **{pct(summary['overall_pass_rate'])}** |")
    lines.append(f"| Supervisor 分诊准确率 | {pct(summary['expert_accuracy'])} （校验 {summary['denominators']['expert_checked']} 题）|")
    lines.append(f"| 护栏人群识别准确率 | {pct(summary['population_accuracy'])} （校验 {summary['denominators']['population_checked']} 题）|")
    lines.append(f"| 红旗症状判定准确率 | {pct(summary['redflag_accuracy'])} |")
    lines.append(f"| 证据来源路由准确率 | {pct(summary['source_routing_accuracy'])} （校验 {summary['denominators']['source_checked']} 题）|")
    lines.append(f"| 答案要点达标率 | {pct(summary['answer_keyword_pass_rate'])} |")
    lines.append(f"| 证据溯源覆盖率 | {pct(summary['evidence_coverage'])} |")
    lat = summary["latency_ms"]
    lines.append(f"| 平均时延 / P50 / 最大 | {lat['avg']}ms / {lat['p50']}ms / {lat['max']}ms |")
    lines.append("\n## 二、分类通过率\n")
    lines.append("| 场景 | 通过/总数 | 通过率 |")
    lines.append("|---|---|---|")
    for c, v in summary["by_category"].items():
        lines.append(f"| {c} | {v['pass']}/{v['total']} | {pct(v['pass_rate'])} |")
    lines.append("\n## 三、逐题明细\n")
    lines.append("| ID | 问题 | 分诊 | 来源(kg/guide) | 要点命中 | 时延 | 结果 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        ok = "✅" if r["pass"] else "❌"
        lines.append(
            f"| {r['id']} | {r['question'][:16]} | {r['actual_expert']} | "
            f"{r['kg_rows']}/{r['guide_hits']} | {r['keyword_hit_ratio']:.2f} | "
            f"{r['latency_ms']}ms | {ok} |"
        )
    # 失败题诊断
    failed = [r for r in results if not r["pass"]]
    if failed:
        lines.append("\n## 四、失败题诊断\n")
        for r in failed:
            bad = [k for k, v in r["checks"].items() if v is False and not k.endswith("_checked")]
            lines.append(f"- **{r['id']}** `{r['question']}` → 未通过维度：{', '.join(bad)}")
            lines.append(f"  - 分诊={r['actual_expert']} 人群={r['actual_populations']} kg行={r['kg_rows']} 指南命中={r['guide_hits']}")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--set", default=str(DEFAULT_SET))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cases = load_cases(args.set, args.limit)
    print(f"评测集 {len(cases)} 题，目标 {args.base_url}")
    results = []
    for i, case in enumerate(cases, 1):
        try:
            answer, evidence, status, latency = ask(args.base_url, case["question"])
            r = score_case(case, answer, evidence, status, latency)
        except Exception as e:
            r = {
                "id": case["id"], "category": case.get("category"), "question": case["question"],
                "http_status": None, "latency_ms": 0, "actual_expert": None, "expert_confidence": None,
                "need_guideline": None, "actual_populations": [], "actual_red_flags": [],
                "kg_rows": 0, "guide_hits": 0, "answer_len": 0, "keyword_hit_ratio": 0.0,
                "checks": {"success": False, "expert_ok": False, "expert_checked": case.get("expect_expert") is not None,
                           "population_ok": False, "population_checked": bool(case.get("expect_populations")),
                           "redflag_ok": not case.get("expect_red_flags"), "source_ok": False,
                           "source_checked": case.get("expect_source", "any") in ("kg", "guideline"),
                           "answer_ok": False, "evidence_ok": False},
                "pass": False, "answer_head": f"ERROR {type(e).__name__}: {e}",
            }
        results.append(r)
        flag = "✅" if r["pass"] else "❌"
        print(f"[{i}/{len(cases)}] {flag} {r['id']} expert={r['actual_expert']} "
              f"kg={r['kg_rows']} guide={r['guide_hits']} {r['latency_ms']}ms")

    summary = summarize(results)
    json_path, md_path = write_reports(summary, results, args.out, args.base_url, args.set)
    print("\n===== 评测汇总 =====")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_category"}, ensure_ascii=False, indent=2))
    print(f"\n报告已保存：\n  {json_path}\n  {md_path}")


if __name__ == "__main__":
    main()
