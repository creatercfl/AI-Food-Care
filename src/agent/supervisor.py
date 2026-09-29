"""Supervisor 多专家路由：生成前对用户问题做"专家分诊"，选择合适的专家画像注入
Agent，并把路由决策写入证据与审计（可解释、可质控）。

设计取舍：
- 采用"路由 + 专家画像注入"而非并行多智能体聚合，以控制单轮时延（每问已 10s+）
  与流程风险；分诊本身是一次轻量结构化输出调用，失败自动降级为关键词分诊，
  永不阻断问答主链路。
- 分诊结果（专家/置信度/理由/是否需指南）随证据块与审计日志留痕，体现"决策可追溯"。
"""
from langchain_deepseek import ChatDeepSeek
from agent.schema import SupervisorRoute
from configuration import config
import logging

logger = logging.getLogger(__name__)

# 专家画像库：领域标识 -> {名称, 适用范围（分诊依据）, 关键词（兜底分诊用）, 专家视角指引}
EXPERTS = {
    "medication": {
        "name": "临床用药专家",
        "scope": "用药咨询：吃什么药/用什么药、用法用量、药物副作用与相互作用、能否同服",
        "keywords": ["吃什么药", "用什么药", "服药", "用药", "剂量", "副作用", "能一起吃", "开什么药", "药"],
        "guidance": "以临床用药专家视角回答：明确适应证与常用药物类别，给出用法用量要点、不良反应与相互作用提醒，强调遵医嘱、不自行增减剂量或停药。",
    },
    "diagnosis": {
        "name": "疾病诊疗专家",
        "scope": "疾病诊疗：某症状可能是什么病/怎么回事、病因与鉴别、疾病的表现与危害、病情判断",
        "keywords": ["是什么病", "什么病", "症状", "怎么回事", "原因", "诊断", "为什么会", "得了", "表现"],
        "guidance": "以疾病诊疗专家视角回答：结合症状给出可能的病因与鉴别方向，说明需关注的伴随表现，提示何时就医与需完善的检查，不下确定性诊断。",
    },
    "diet": {
        "name": "营养膳食专家",
        "scope": "营养膳食：宜吃忌口、饮食管理、食疗搭配、营养素补充",
        "keywords": ["吃什么", "忌口", "饮食", "食物", "能吃", "可不可以吃", "营养", "食谱", "调理", "热量", "卡路里", "成分"],
        "guidance": "以营养膳食专家视角回答：给出宜吃与忌口的具体食物及理由，结合疾病特点提供可操作的膳食搭配与生活方式建议。",
    },
    "exam": {
        "name": "检查检验专家",
        "scope": "检查检验：化验单/检查指标解读、异常数值（偏高/偏低）说明什么、该做哪些检查、检查意义与正常值",
        "keywords": ["检查", "化验", "指标", "偏高", "偏低", "正常值", "报告", "CT", "B超", "验血", "核酸", "体检"],
        "guidance": "以检查检验专家视角回答：解释相关检查的意义与常见指标含义，提示异常结果的可能方向与复查/进一步检查建议，不替代医生解读。",
    },
    "emergency": {
        "name": "急危重症专家",
        "scope": "急危重症：突发危重症状（胸痛/呼吸困难/昏迷/抽搐/大出血等）的紧急处置与就医",
        "keywords": ["突然", "剧烈", "昏迷", "抽搐", "呼吸困难", "胸痛", "大出血", "中毒", "急救", "晕倒"],
        "guidance": "以急危重症专家视角回答：优先给出立即就医/呼叫 120 的处置与就医前自救要点，突出时间紧迫性，避免延误。",
    },
    "special_population": {
        "name": "特殊人群安全专家",
        "scope": "特殊人群安全：孕妇/哺乳期/儿童/老年人等特定人群的用药与处置安全",
        "keywords": ["孕妇", "怀孕", "妊娠", "哺乳", "喂奶", "儿童", "小孩", "宝宝", "婴儿", "老人", "老年"],
        "guidance": "以特殊人群安全专家视角回答：突出该人群的用药与处置禁忌、安全阈值，凡涉及药物须强调咨询医生，宁可保守。",
    },
}
DEFAULT_EXPERT = "diagnosis"

_router_llm = ChatDeepSeek(model=config.LLM_MODEL).with_structured_output(SupervisorRoute)


def _keyword_route(question):
    """关键词兜底分诊：按命中关键词数打分选专家，全不中则用默认专家。"""
    best, best_hits = DEFAULT_EXPERT, 0
    for key, exp in EXPERTS.items():
        hits = sum(1 for kw in exp["keywords"] if kw in question)
        if hits > best_hits:
            best, best_hits = key, hits
    return best


def route(question):
    """Supervisor 分诊：优先 LLM 语义路由，失败降级关键词分诊。

    :return: {expert, expert_name, confidence, rationale, need_guideline, router}
    """
    try:
        expert_lines = "\n".join(f"- {k}（{v['name']}）：{v['scope']}" for k, v in EXPERTS.items())
        res = _router_llm.invoke(
            "请对以下医疗问题进行专家分诊，选择最合适的一位专家，并判断是否需要临床指南要点补充。\n"
            f"专家职责边界：\n{expert_lines}\n"
            "分诊要点：凡涉及化验单/检查指标解读、异常数值（如偏高/偏低）说明什么、该做哪些检查的问题，均归 exam；"
            "仅当问题核心是“某症状可能是什么病/病因鉴别”时才归 diagnosis。\n"
            f"问题：{question}"
        )
        expert = res.expert if res.expert in EXPERTS else _keyword_route(question)
        return {
            "expert": expert,
            "expert_name": EXPERTS[expert]["name"],
            "confidence": round(float(res.confidence), 3),
            "rationale": res.rationale,
            "need_guideline": bool(res.need_guideline),
            "router": "llm",
        }
    except Exception as e:
        logger.exception("Supervisor LLM 路由失败，降级关键词分诊")
        expert = _keyword_route(question)
        return {
            "expert": expert,
            "expert_name": EXPERTS[expert]["name"],
            "confidence": 0.3,
            "rationale": f"关键词兜底分诊（LLM 路由异常：{type(e).__name__}）",
            "need_guideline": False,
            "router": "keyword",
        }


def build_expert_instruction(route_result):
    """把专家画像与指南回退提示拼成注入指令（追加到用户问题后）。"""
    exp = EXPERTS[route_result["expert"]]
    text = f"\n\n【Supervisor 分诊·{exp['name']}】{exp['guidance']}"
    if route_result.get("need_guideline"):
        text += (
            "本问题图谱覆盖可能不足：若图谱查询无充分结果，请调用 guideline_search 检索"
            "临床指南要点作为回退补充，并在回答中注明依据来源。"
        )
    return text
