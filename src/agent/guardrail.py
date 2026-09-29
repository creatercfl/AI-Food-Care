"""独立安全护栏模块：对问答做"生成前约束注入 + 生成后规则校验"。

三层校验（规则驱动、确定性强，独立于大模型输出）：
1. 人群识别：识别问题中的特殊人群（孕妇/哺乳期/儿童/老年人），
   用于生成前注入"人群禁忌约束"，强制 Agent 查询并声明该人群注意事项；
2. 红旗症状识别：识别胸痛/呼吸困难/昏迷/抽搐等危重症状，触发急诊就医提示；
3. 用药红线比对：生成后把回答与"人群×药物"红线规则表比对，
   命中即在该人群下追加禁忌警告，拦截高风险建议。

所有校验结果随证据块与审计日志留痕，满足医疗质控审查要求。
"""

# 问题关键词 -> 标准人群组（顺序即匹配优先级）
POPULATION_KEYWORDS = {
    "孕妇": ["孕妇", "怀孕", "妊娠", "孕期", "有孕"],
    "哺乳期妇女": ["哺乳期", "哺乳", "喂奶", "产后"],
    "儿童": ["新生儿", "早产儿", "婴幼儿", "婴儿", "幼儿", "儿童", "小儿", "小孩", "未成年", "青少年", "宝宝"],
    "老年人": ["老年人", "老人", "高龄", "年老"],
}

# 危重（红旗）症状词：命中即提示急诊就医
RED_FLAG_TERMS = [
    "胸痛", "胸闷", "压榨性疼痛", "呼吸困难", "呼吸急促", "喘不上气",
    "昏迷", "意识不清", "抽搐", "惊厥",
    "大出血", "出血不止", "呕血", "便血",
    "言语不清", "口角歪斜", "肢体无力", "瘫痪",
    "自杀", "自残", "中毒", "误服",
    "持续高热", "剧烈腹痛",
]

# 急诊提示模板（前置到回答之前）
EMERGENCY_NOTICE = (
    "> 🚨 **紧急提示**：您描述的情况包含危重症状（{}），可能属于急症。"
    "请**立即拨打 120 或前往最近医院急诊科**，切勿依赖在线问答自行判断或等待观察。"
    "以下回答仅供就医前参考：\n\n"
)

# "人群 × 药物"红线规则表：该人群下回答中出现该药物即告警
DRUG_REDLINES = {
    "儿童": [
        ("阿司匹林", "儿童及青少年病毒感染期使用阿司匹林可能诱发瑞氏综合征，禁用"),
        ("尼美舒利", "尼美舒利有肝毒性风险，12 岁以下儿童禁用"),
        ("左氧氟沙星", "喹诺酮类可能影响软骨发育，18 岁以下禁用"),
        ("氧氟沙星", "喹诺酮类可能影响软骨发育，18 岁以下禁用"),
        ("莫西沙星", "喹诺酮类可能影响软骨发育，18 岁以下禁用"),
        ("四环素", "四环素类影响牙齿与骨骼发育，8 岁以下儿童禁用"),
        ("多西环素", "四环素类影响牙齿与骨骼发育，8 岁以下儿童禁用"),
    ],
    "孕妇": [
        ("布洛芬", "布洛芬等 NSAIDs 在妊娠晚期（尤其 ≥28 周）禁用"),
        ("阿司匹林", "阿司匹林孕期须经医生评估，不可自行服用"),
        ("左氧氟沙星", "喹诺酮类妊娠期禁用"),
        ("氧氟沙星", "喹诺酮类妊娠期禁用"),
        ("四环素", "四环素类妊娠期禁用"),
        ("多西环素", "四环素类妊娠期禁用"),
        ("利巴韦林", "利巴韦林有明确致畸性，妊娠期禁用"),
    ],
    "哺乳期妇女": [
        ("利巴韦林", "利巴韦林哺乳期禁用"),
        ("左氧氟沙星", "喹诺酮类哺乳期应避免使用"),
    ],
}


def detect_populations(text):
    """识别问题中涉及的特殊人群，返回标准人群组列表。"""
    return [pop for pop, kws in POPULATION_KEYWORDS.items() if any(k in text for k in kws)]


def detect_red_flags(text):
    """识别问题中的危重（红旗）症状词。"""
    return [t for t in RED_FLAG_TERMS if t in text]


def check_drug_redlines(answer, populations):
    """把回答与人群用药红线表比对，返回命中列表。"""
    hits = []
    for pop in populations:
        for drug, reason in DRUG_REDLINES.get(pop, []):
            if drug in answer:
                hits.append({"population": pop, "drug": drug, "reason": reason})
    return hits


def build_population_constraint(populations):
    """生成"人群禁忌约束"指令，拼接到用户问题后注入 Agent（安全边界模式）。"""
    if not populations:
        return ""
    names = "、".join(populations)
    return (
        f"\n\n【安全约束·人群模式】提问者属于特殊人群：{names}。回答必须满足："
        f"1) 先查询图中该人群相关信息，例如用 `MATCH (d:Disease)-[:COMMON_ON]->(p:People) WHERE p.name CONTAINS '{populations[0]}'` "
        f"定位相关疾病，再结合 (Disease)-[:COMMON_USE]->(Drug)、(Disease)-[:NO_EAT]->(Food) 给出建议；"
        f"2) 回答中必须包含针对{names}的注意事项或禁忌说明，不得将其等同于普通人群给出建议；"
        f"3) 若图中无该人群数据，必须明确说明并建议咨询医生，禁止编造。"
    )


def run_post_guardrail(question, answer):
    """生成后校验：红旗提示前置、用药红线警告追加。

    :return: (处理后的回答, 护栏记录 dict)
    """
    populations = detect_populations(question)
    red_flags = detect_red_flags(question)
    redlines = check_drug_redlines(answer, populations)

    if red_flags:
        answer = EMERGENCY_NOTICE.format("、".join(red_flags)) + answer
    if redlines:
        answer += (
            "\n\n> ⚠️ **禁忌警告**（安全护栏自动校验）：\n"
            + "\n".join(f"> - 【{r['population']}】{r['drug']}：{r['reason']}" for r in redlines)
            + "\n> 以上药物请务必先咨询医生或药师，切勿自行使用。"
        )
    record = {
        "populations": populations,
        "red_flags": red_flags,
        "redlines": redlines,
    }
    return answer, record
