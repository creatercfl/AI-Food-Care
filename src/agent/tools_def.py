from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from agent.schema import CypherCheckerResponse
from agent.prompts import cypher_checker_prompt
from datasync.entity_alignment import EntityAlignment
from datasync import guideline_store
from datasync import nutrition_store
from configuration.dependency import neo4j_driver,neo4j_schema,embedding_model

import logging
import contextvars
from configuration import config

logger = logging.getLogger(__name__)

# 当前请求的证据收集器：由 backend 在每次问答开始前 set，工具执行时往里追加证据，
# 问答结束后由 backend 读取并随回答下发，用于“证据溯源”。
evidence_collector = contextvars.ContextVar("evidence_collector", default=None)


def record_evidence(item: dict):
    """向当前请求的证据收集器追加一条证据（无收集器时静默忽略）。"""
    collector = evidence_collector.get()
    if collector is not None:
        collector.append(item)


def _jsonable(obj, max_str_len=200):
    """把 neo4j 返回的 Record / Node 等对象转换为可 JSON 序列化的结构。"""
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj if len(obj) <= max_str_len else obj[:max_str_len] + "…"
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items() if "embedding" not in str(k).lower()}
    # neo4j 的 Node / Record / Relationship 等均为 Mapping
    if hasattr(obj, "items"):
        try:
            return {str(k): _jsonable(v) for k, v in obj.items() if "embedding" not in str(k).lower()}
        except Exception:
            pass
    return str(obj)

# 维护当前Neo4j当中实体所对应的index的列表，每个实体存在两个Index，以list形式存储，第一个为全文索引，第二个为向量索引
# hybrid_store_list = {
#     "CourseInfo": ["course_name_index","CourseInfo_namevector_index"],
#     "ChapterInfo": ["ChapterInfoNameIndex","ChapterInfoNameVectorIndex"],
# }
hybrid_store_list = ['category', 'subject', 'category', 'chapter']  # 旧项目残留，仅保留注释警示，不再使用
# 支持实体对齐的实体类型（对应 MySQL entity_mapping.entity_schema / Chroma metadata.type）。
# 注意：Neo4j 的 label 首字母大写（Disease/Symptom/...），工具内统一转小写后匹配。
ALIGNABLE_LABELS = {"disease", "symptom", "cause", "drug", "food", "people", "check"}

cypher_checker_llm = ChatDeepSeek(model=config.LLM_MODEL).with_structured_output(CypherCheckerResponse)
ea = EntityAlignment()

def hybrid_search_query(query_text,full_text_index_name,vector_index_name,driver=neo4j_driver,top_k=1,alpha=0.5,threshold=0.5):
    vector_list = embedding_model.embed_query(query_text)
    res = driver.execute_query(
        """
        CALL () {


    	CALL db.index.vector.queryNodes($vector_index_name, $top_k, $query_vector) 
    	YIELD node, score
    	WITH node, score LIMIT $top_k
    	return node, score*$alpha as score


    	UNION


    	CALL db.index.fulltext.queryNodes($fulltext_index_name, $query_text, {limit: $top_k})
    	YIELD node, score
    	WITH collect({node:node, score:score}) AS nodes, max(score) AS max_score
    	UNWIND nodes as n
    	RETURN n.node as node, n.score*(1 - $alpha) / max_score as score
    }
    WITH node, sum(score) AS score ORDER BY score DESC LIMIT $top_k
    RETURN node.`name` AS text, score, node {.*, `name`: Null, `name_embedding`: Null, id: Null } AS metadata
        """,
        parameters_={
            "vector_index_name": vector_index_name,
            "top_k": top_k,
            "query_vector": vector_list,
            "fulltext_index_name": full_text_index_name,
            "query_text": query_text,
            "alpha": alpha
        }
    ).records
    return [{"text":record["text"], "score":record["score"]} for record in res if record["score"] > threshold]

def entity_alignment(entitys_to_alignment:list):
    """
    当需要将用户的查询问题当中的实体对齐到图数据库当中已有的实体时，可以使用当前工具
    :param entity_to_alignment_list:
    :return:
    """
    logger.info(f"开始调用工具：entity_alignment，对齐参数为：{entitys_to_alignment}")
    for node in entitys_to_alignment:
        label = str(node.get('label', '')).strip().lower()
        if label in ALIGNABLE_LABELS:
            original = node['entity']
            try:
                res = ea(original, label)
            except Exception as e:
                # 对齐服务异常（如向量库索引损坏）不应中断整次问答，降级为不对齐
                logger.exception(f"实体对齐失败，降级为原始实体：{original}")
                record_evidence({
                    "tool": "entity_alignment",
                    "label": label,
                    "original": original,
                    "aligned": None,
                    "error": f"{type(e).__name__}: {e}",
                })
                continue
            if res:
                node['entity'] = res
                record_evidence({
                    "tool": "entity_alignment",
                    "label": label,
                    "original": original,
                    "aligned": res,
                })
    return entitys_to_alignment

def check_syntax_error(cypher:str):
    """
    当需要检查cypher语句是否存在语句错误，或者是否不符合已有schema结构时使用
    当执行Cypher语句前，必须要使用该工具来进行检测Cypher语句的合法性
    """
    logger.info("开始调用工具：check_syntax_error")
    logger.info(f"当前需要校验的Cypher语句为:{cypher}")
    prompt = PromptTemplate.from_template(cypher_checker_prompt)
    chain = prompt | cypher_checker_llm
    res = chain.invoke(
        {
            "neo4j_schema": neo4j_schema,
            "cypher": cypher
        }
    )
    logger.info(f"Cypher语句LLM校验结果为:{res}")
    return res

def neo4j_query(cypher,params=None):
    """
    当需要从neo4j数据库查询数据时使用
    :param cypher:
    :param driver:
    :param params:
    :return:
    """
    logger.info("开始调用工具：neo4j_query")
    logger.info(f"当前调用的cypher为:{cypher},当前调用的params为：{params}",)
    if not params:
        params={}

    driver = globals()['neo4j_driver']
    try:
        result = driver.execute_query(cypher,parameters_=params)
    except Exception as e:
        # 把错误返回给 Agent，由其修正 Cypher 后重试，而不是中断整次问答
        logger.exception(f"cypher 执行失败：{cypher}")
        record_evidence({
            "tool": "neo4j_query",
            "cypher": cypher,
            "params": params,
            "row_count": 0,
            "rows": [],
            "error": f"{type(e).__name__}: {e}",
        })
        return f"查询执行失败：{type(e).__name__}: {e}。请检查 Cypher 语法与 Schema 后重试。"
    logger.info(f"当前调用cypher结果为:{result.records}")
    record_evidence({
        "tool": "neo4j_query",
        "cypher": cypher,
        "params": params,
        "row_count": len(result.records),
        "rows": [_jsonable(dict(r)) for r in result.records[:10]],
    })
    return result.records

def guideline_search(query:str, top_k:int=3):
    """
    当知识图谱中查询不到相关信息、或图谱结果不足以回答用户问题时，使用该工具检索
    权威临床指南要点作为回退补充。返回带出处的指南要点，回答时须注明依据来源。
    :param query: 检索用的核心医疗主题（如"妊娠期感冒用药""儿童发热退热药"）
    :param top_k: 返回的指南要点条数，默认3条
    """
    logger.info(f"开始调用工具：guideline_search，query={query}")
    try:
        hits = guideline_store.search(query, top_k=top_k)
    except Exception as e:
        # 指南检索异常不应中断问答主链路，降级为无补充
        logger.exception("指南检索失败")
        record_evidence({
            "tool": "guideline_rag",
            "query": query,
            "hit_count": 0,
            "error": f"{type(e).__name__}: {e}",
        })
        return f"指南检索失败：{type(e).__name__}: {e}。"
    record_evidence({
        "tool": "guideline_rag",
        "query": query,
        "hit_count": len(hits),
        "hits": [{"title": h["title"], "source": h["source"], "score": h["score"]} for h in hits],
    })
    if not hits:
        return "未检索到相关临床指南要点。"
    parts = []
    for h in hits:
        parts.append(f"【{h['title']}】（来源：{h['source']}；相关度{h['score']}）\n{h['content']}")
    return "\n\n".join(parts)

def nutrition_query(food:str):
    """
    查询食物的营养成分（每100g可食部的能量/蛋白质/脂肪/碳水化合物/钠）与营养标签。
    当用户询问某食物的热量、卡路里、营养成分或"某食物营养如何"时使用。
    :param food: 食物名称（支持常见别名，如"米饭""鸡胸肉""番茄"）
    """
    logger.info(f"开始调用工具：nutrition_query，food={food}")
    try:
        rec = nutrition_store.lookup(food)
    except Exception as e:
        # 营养查询异常不应中断问答主链路，降级为无数据
        logger.exception("营养查询失败")
        record_evidence({
            "tool": "nutrition",
            "food": food,
            "found": False,
            "error": f"{type(e).__name__}: {e}",
        })
        return f"营养查询失败：{type(e).__name__}: {e}。"
    if not rec:
        record_evidence({"tool": "nutrition", "food": food, "found": False})
        return f"营养成分表未收录'{food}'，无法给出成分数值；禁止编造，可说明未收录并建议参考同类常见食物或咨询营养师。"
    record_evidence({"tool": "nutrition", "food": food, "found": True, "record": rec})
    return nutrition_store.format_nutrition(rec)

if __name__ == '__main__':
    pass