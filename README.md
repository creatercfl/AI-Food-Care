# AI食护管家 · 饮食营养分析 + 知识图谱循证医疗问答智能体（KG-QA）

> 一个面向医疗咨询场景的检索增强问答系统：以 **Neo4j 医疗知识图谱** 为唯一事实源，
> 由 **DeepSeek 大模型** 作为推理编排层完成实体对齐、Text2Cypher 检索、答案生成，
> 并叠加 **证据溯源、安全护栏、指南 RAG 回退、多专家分诊、决策审计** 五层工程能力，
> 做到「有据可查、可解释、可质控」。
> 产品形态为 **AI食护管家**：在医疗问答之上提供食物营养成分分析（每100g 成分 + 营养标签）。

---

## 一、系统架构

```
用户提问
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  FastAPI 后端  (src/backend)                               │
│   · SessionMiddleware：会话隔离（签名密钥走环境变量）        │
│   · RateLimitMiddleware：/chat 按 IP 滑动窗口限流（429）     │
│   · 全局异常兜底：结构化错误，不泄露堆栈                      │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  ChatService  (src/backend/chat_service.py)                │
│   ① Supervisor 分诊：选择专家画像 + 是否需指南回退           │
│   ② 生成前护栏：特殊人群禁忌约束注入                         │
│   ③ LangGraph Agent 编排（流式）                            │
│   ④ 生成后护栏：红旗症状前置 + 用药红线警告追加              │
│   ⑤ 证据块拼接 + 决策审计落盘 + 会话自愈重试                 │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  Agent 工具层  (src/agent)                                  │
│   · entity_alignment ：口语实体 → 图谱标准实体（向量对齐）    │
│   · check_syntax_error：Cypher 合法性 LLM 自检               │
│   · neo4j_query       ：执行 Text2Cypher 检索图谱            │
│   · guideline_search  ：图谱无覆盖时回退临床指南要点         │
└──────────────────────────────────────────────────────────┘
  │            │                         │
  ▼            ▼                         ▼
Neo4j 图谱   Chroma 向量库            Chroma 指南库
(事实源)    (实体对齐 smart_medical)  (指南 smart_medical_guidelines)
```

### 反幻觉设计
系统遵循「闭域检索接地」原则：**答案必须来自图谱查询结果或权威指南要点**，
图谱与指南均无相关内容时明确声明「无数据」，而非由大模型自由编造。
所有检索/对齐/护栏/分诊动作都以结构化「证据块」随回答返回并在前端可视化。

---

## 二、核心能力（三批工程化改造）

| 能力 | 说明 |
|---|---|
| **证据溯源** | 回答末尾附 `<!--EVIDENCE-->` JSON 块，前端渲染「📚 回答依据」折叠卡片（分诊徽章 / 实体对齐 / Cypher 与返回行 / 指南命中） |
| **决策审计** | 每次问答落一条 JSONL 审计（问题/答案/证据/护栏/分诊/模型/时延/状态），满足医疗质控留痕 |
| **安全护栏** | 生成前注入特殊人群（孕妇/哺乳期/儿童/老年人）禁忌约束；生成后红旗症状（胸痛/呼吸困难等）急诊提示前置、18 条「人群×药物」红线警告追加 |
| **指南 RAG 回退** | 图谱覆盖不到时检索人工整理的临床指南要点补覆盖，回答注明「依据指南共识」，直击「库无即答不了」短板 |
| **营养成分分析**（食护管家） | 内置 41 种常见食物每100g成分表（能量/蛋白/脂肪/碳水/钠），Agent 经 `nutrition_query` 工具查询并按阈值打营养标签（高能量/高脂肪/高钠/低能量）；另提供 `/nutrition` 确定性查询端点 |
| **Supervisor 多专家** | 生成前对问题分诊到 6 类专家（用药/诊疗/营养/检查/急诊/特殊人群），注入专家画像指引，分诊决策可追溯 |
| **容错与自愈** | 工具异常降级不中断问答；会话历史污染时自动清 thread 并重试一次，杜绝「一次失败、句句失败」 |

---

## 三、技术栈

- **大模型**：DeepSeek（`deepseek-chat`），经 `langchain-deepseek` 接入
- **Agent 编排**：LangChain 1.4 / LangGraph 1.2（`create_agent` + `InMemorySaver` 会话记忆）
- **知识图谱**：Neo4j（事实源，Text2Cypher 检索）
- **向量检索**：Chroma（本地持久化）+ `bge-base-zh-v1.5` 中文嵌入模型
- **结构化存储**：MySQL（实体同义词映射表 `entity_mapping`）
- **Web 后端**：FastAPI + Uvicorn + Starlette
- **前端**：原生 HTML/CSS/JS，`marked` + `DOMPurify` 渲染 Markdown

---

## 四、目录结构

```
smart_medical/
├── data/
│   ├── annotated_data/          # CMeIE-V2 标注语料
│   ├── knowledge_graph/         # 图谱构建源数据（jsonl）
│   ├── guidelines/              # 临床指南要点（curated_guidelines.jsonl）
│   ├── nutrition/               # 食物营养成分表（food_nutrition.jsonl）
│   ├── vectorstore/             # Chroma 向量库（运行时生成，已 gitignore）
│   └── audit/                   # 审计日志（运行时生成，已 gitignore）
├── src/
│   ├── agent/                   # Agent 工具、提示词、schema、护栏、Supervisor
│   ├── backend/                 # FastAPI 应用、问答服务、前端模板
│   ├── configuration/           # 配置、依赖注入、审计、统一日志
│   └── datasync/                # 数据清洗入库、实体对齐、指南库
├── eval/                        # 评测体系（基准集 + 跑批脚本 + 报告）
├── tests/                       # 单元测试（标准库 unittest，可离线运行）
├── pretrained/                  # 本地嵌入模型（已 gitignore）
├── requirements.txt
├── requirements-dev.txt         # 开发/测试依赖（可选，pytest）
├── .env.example                 # 环境变量模板（复制为 .env 填真实值）
└── README.md
```

---

## 五、快速开始

### 1. 环境准备

```powershell
# 建议使用 conda 独立环境
conda create -n smart_medical python=3.10 -y
conda activate smart_medical
pip install -r requirements.txt
```

依赖的外部服务：
- **Neo4j**（默认 `neo4j://localhost`）——已导入医疗知识图谱
- **MySQL**（默认 `localhost:3306`）——存放实体同义词映射表
- **DeepSeek API Key**——见下一步

### 2. 配置环境变量（重要）

本项目**不在代码中硬编码任何密码/密钥**。复制模板并填入真实值：

```powershell
Copy-Item .env.example .env
# 然后编辑 .env，填入 MySQL / Neo4j 密码、DeepSeek Key、Session 密钥
```

生成强随机会话密钥：

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

> `.env` 已在 `.gitignore` 中排除，切勿提交到版本库。
> `DEEPSEEK_API_KEY` 也可配置在系统环境变量中。

### 3. 构建知识库（首次）

```powershell
# 清洗数据 → 向量化实体 → 导入 Neo4j → 实体对齐
python src/datasync/data_prepare.py
```

### 4. 启动后端

```powershell
python src/backend/app.py
# 或
uvicorn src.backend.app:app --host 0.0.0.0 --port 8000
```

浏览器访问 <http://127.0.0.1:8000> 进入问答界面；
健康检查：<http://127.0.0.1:8000/health>。

> ⚠️ **单实例铁律**：Chroma 本地持久库不支持多进程并发读写，
> 同一时间只能运行一个后端实例，否则向量库 HNSW 索引会损坏。

---

## 六、评测体系

`eval/` 为项目永久资产，提供可复现的量化评测：

- `eval/eval_set.jsonl`：30 题基准集，覆盖 7 类场景（用药/诊疗/饮食/检查/急诊/特殊人群/指南独有）
- `eval/run_eval.py`：端到端跑批（走完整 `/chat` 链路），解析证据块，7 维度客观打分，产出 JSON + Markdown 报告

运行：

```powershell
python eval/run_eval.py --base-url http://127.0.0.1:8000
```

**最近一次全量结果（30/30）：**

| 指标 | 结果 |
|---|---|
| 综合通过率 | **100%** |
| Supervisor 分诊准确率 | 100% |
| 护栏人群识别准确率 | 100% |
| 红旗症状判定准确率 | 100% |
| 证据来源路由准确率 | 100% |
| 答案要点达标率 / 证据覆盖率 | 100% / 100% |
| 平均时延 | ~10.7s |

评分维度口径详见 `run_eval.py` 文档字符串与报告头。

---

## 七、单元测试与日志

### 单元测试

`tests/` 提供纯逻辑单元测试（安全护栏规则、Supervisor 关键词兜底分诊、审计落盘），
用 Python 标准库 `unittest` 编写，**零额外依赖、完全离线**：通过 `tests/_bootstrap.py`
预置 agent 包壳，导入被测模块时不打开 Chroma 向量库、不需真实 API Key，
因此可与运行中的后端并存、随时执行。

```powershell
# 方式一：标准库（无需安装任何东西）
python -m unittest discover -s tests -t tests -v

# 方式二：pytest（可选，用例完全兼容）
pip install -r requirements-dev.txt
pytest tests
```

当前 30 条用例全部通过，覆盖：人群识别 / 红旗症状 / 用药红线 / 人群约束 /
生成后护栏 / 关键词分诊 / 专家指令拼装 / 审计 JSONL 落盘与时间戳。

> `route()` 等依赖真实大模型的端到端行为，由 `eval/` 跑批评测覆盖（见第六节）。

### 运行日志

后端启动时经 `configuration/logging_config.setup_logging()` 统一配置日志：
**控制台 + 轮转文件双通道**，文件落在 `data/logs/app.log`（单文件 5MB、保留 5 份，
已被 gitignore）。工具异常、护栏命中、限流触发、未处理异常等运行期事件均留痕，
便于线上故障事后回溯——这正是排查向量库损坏一类问题的关键依据。

---

## 八、体验建议（对应各能力）

| 试着问 | 你会看到 |
|---|---|
| 感冒了应该吃什么药 | 基础问答 + 底部「📚 回答依据」卡片 |
| 糖尿病患者日常饮食怎么管理 | 图谱无膳食数据 → **自动回退临床指南**并注明依据 |
| 怀孕感冒了能吃什么药 | 分诊到**特殊人群安全专家** + 末尾 ⚠️ **禁忌红线警告** |
| 突然胸痛呼吸困难怎么办 | 🚨 **急诊提示前置**到回答开头 |
| 体检发现尿酸偏高说明什么 | 分诊到**检查检验专家** |
| 鸡胸肉的营养成分怎么样 | 调用 `nutrition_query` → 回答给出每100g成分 + 底部「营养成分」卡片 |

---

## 九、安全说明

- 数据库密码、会话密钥、API Key 全部经环境变量注入，代码与版本库不含明文敏感信息；
- `/chat` 接口按客户端 IP 限流（默认 20 次/分钟，超限 429）；
- 未捕获异常统一兜底为结构化 500，不向前端泄露堆栈；
- 医疗场景仅为**信息参考**，不替代执业医师诊断，危急情况请立即就医。
