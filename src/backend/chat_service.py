from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector,Neo4jGraph
from neo4j_graphrag.types import SearchType

from agent import get_agent
from agent import guardrail
from agent import supervisor

from configuration import config
from configuration.audit import write_audit
from agent.tools_def import evidence_collector
import json
import time
import logging

logger = logging.getLogger(__name__)

# 证据块哨兵：回答文本与结构化证据之间的分隔符，前端识别后渲染为"回答依据"卡片
EVIDENCE_SENTINEL = "\n<!--EVIDENCE-->\n"




class ChatService:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(model_name = str(config.EMBEDDING_MODEL_PATH),model_kwargs = {"device": "cuda"}) # 需要下载带cuda版本的torch
        self.neo4j_store = Neo4jGraph(
            url=config.NEO4J_CONFIG["uri"],
            username=config.NEO4J_CONFIG["auth"][0],
            password=config.NEO4J_CONFIG["auth"][1],
            database="neo4j"
        )
        self.agent = get_agent(self.neo4j_store.get_schema)
        # 针对于每一个实体，维护一个Neo4jVector的实例，后面利用该实例来完成搜索

        # 仅用于演示，后面不使用neo4j所提供的索引进行检索
        # self.vector_stores = {
        #     "CourseInfo": Neo4jVector.from_existing_index(
        #         url=config.NEO4J_CONFIG["uri"],
        #         username = config.NEO4J_CONFIG['auth'][0],
        #         password = config.NEO4J_CONFIG['auth'][1],
        #         embedding=self.embeddings,
        #         index_name="CourseInfo_namevector_index",
        #         keyword_index_name="course_name_index",
        #         search_type=SearchType.HYBRID
        #     )
        # }

    def _collect_final_answer(self, user_query, agent_config):
        """
        运行 Agent，只提取最终回答（丢弃带工具调用的过程消息），
        同时通过 contextvars 收集工具执行产生的证据（溯源用）。
        :return: (final_text, evidence_list)
        """
        answers = []
        cur_id = None
        buf = []
        has_tool = False
        evidence = []
        token = evidence_collector.set(evidence)
        try:
            for msg, _meta in self.agent.stream(
                {"messages": [("user", user_query)]},
                config=agent_config,
                stream_mode="messages",
            ):
                if isinstance(msg, ToolMessage):
                    continue
                if not isinstance(msg, (AIMessage, AIMessageChunk)):
                    continue
                # 遇到新消息：结算上一条
                if msg.id != cur_id:
                    if cur_id is not None and not has_tool and buf:
                        answers.append("".join(buf))
                    cur_id, buf, has_tool = msg.id, [], False
                # 只要该消息发起过工具调用，就视为"过程消息"，整条丢弃
                if getattr(msg, "tool_call_chunks", None) or getattr(msg, "tool_calls", None):
                    has_tool = True
                content = msg.content
                if isinstance(content, str) and content:
                    buf.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            buf.append(part.get("text", ""))
        finally:
            evidence_collector.reset(token)
        # 结算最后一条
        if cur_id is not None and not has_tool and buf:
            answers.append("".join(buf))
        return (answers[-1] if answers else ""), evidence

    def _reset_thread(self, session_id):
        """清理指定会话的历史状态，避免残缺消息（带工具调用但无结果）污染后续请求。"""
        checkpointer = getattr(self.agent, "checkpointer", None)
        if checkpointer is None:
            return
        try:
            checkpointer.delete_thread(session_id)
        except Exception:
            logger.exception("清理会话历史失败")

    def _write_audit(self, session_id, question, answer, evidence, latency_ms, status, guardrail_record=None, supervisor_record=None):
        """落一条决策审计记录（失败不影响主流程）。"""
        write_audit({
            "session_id": session_id,
            "question": question,
            "answer": (answer or "")[:2000],
            "evidence_count": len(evidence),
            "evidence": evidence,
            "guardrail": guardrail_record or {},
            "supervisor": supervisor_record or {},
            "llm_model": config.LLM_MODEL,
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "latency_ms": latency_ms,
            "status": status,
        })

    def chat(self,user_query,session_id):
        """
        聊天入口，根据配置决定是否流式输出。
        流式模式下仅输出最终分析与结果，不展示中间查库过程；
        回答末尾附结构化证据块（溯源），并落审计日志。
        :param user_query:
        :param session_id:
        :return:
        """
        agent_config = {"configurable":{"thread_id":session_id}}
        start = time.time()
        status = "ok"
        evidence = []
        # Supervisor 分诊：选择专家画像并判断是否需指南回退（失败自动降级，不阻断）
        route_result = supervisor.route(user_query)
        expert_instruction = supervisor.build_expert_instruction(route_result)
        # 安全护栏：生成前识别人群并注入禁忌约束（安全边界模式）
        populations = guardrail.detect_populations(user_query)
        agent_query = user_query + expert_instruction + guardrail.build_population_constraint(populations)
        guardrail_record = {"populations": populations, "red_flags": [], "redlines": []}
        # 证据链首条固定为 Supervisor 分诊决策（可解释、可质控）
        base_evidence = [{"tool": "supervisor", **route_result}]
        # 流式输出：先收集最终回答+证据，再分块下发（隐藏检索过程）
        if config.AGENT_STREAM_OUTPUT:
            try:
                final_text, evidence = self._collect_final_answer(agent_query, agent_config)
                if not final_text:
                    final_text = "抱歉，暂时未能获取到相关结果，请稍后再试或换个问法。"
                    status = "empty"
            except Exception as e:
                # 失败可能在会话历史中留下残缺消息，导致同会话下次请求被模型拒绝；
                # 清理该会话后重试一次
                logger.exception("问答出错，清理会话后重试")
                self._reset_thread(session_id)
                try:
                    final_text, evidence = self._collect_final_answer(agent_query, agent_config)
                    if not final_text:
                        final_text = "抱歉，暂时未能获取到相关结果，请稍后再试或换个问法。"
                        status = "empty"
                    else:
                        status = f"recovered:{type(e).__name__}"
                except Exception as e2:
                    logger.exception("重试仍失败")
                    final_text = "系统暂时无法响应，请稍后再试。"
                    evidence = []
                    status = f"error:{type(e2).__name__}"
            # 安全护栏：生成后校验（红旗提示前置、用药红线警告追加）
            final_text, guardrail_record = guardrail.run_post_guardrail(user_query, final_text)
            evidence = base_evidence + evidence
            if populations or guardrail_record["red_flags"] or guardrail_record["redlines"]:
                evidence = evidence + [{"tool": "guardrail", **guardrail_record}]
            latency_ms = int((time.time() - start) * 1000)
            self._write_audit(session_id, user_query, final_text, evidence, latency_ms, status, guardrail_record, route_result)
            chunk_size = 24
            for i in range(0, len(final_text), chunk_size):
                yield final_text[i:i + chunk_size]
            # 证据溯源：回答末尾附结构化证据块，前端识别哨兵后渲染为"回答依据"卡片
            if evidence:
                yield EVIDENCE_SENTINEL + json.dumps({"evidence": evidence}, ensure_ascii=False)
        # 非流式输出
        else:
            token = evidence_collector.set(evidence)
            try:
                result = self.agent.invoke({"messages":[("user",agent_query)]},config=agent_config)
                final_text = result["messages"][-1].content
            except Exception as e:
                logger.exception("问答出错")
                final_text = "系统暂时无法响应，请稍后再试。"
                status = f"error:{type(e).__name__}"
            finally:
                evidence_collector.reset(token)
            # 安全护栏：生成后校验
            final_text, guardrail_record = guardrail.run_post_guardrail(user_query, final_text)
            evidence = base_evidence + evidence
            if populations or guardrail_record["red_flags"] or guardrail_record["redlines"]:
                evidence = evidence + [{"tool": "guardrail", **guardrail_record}]
            latency_ms = int((time.time() - start) * 1000)
            self._write_audit(session_id, user_query, final_text, evidence, latency_ms, status, guardrail_record, route_result)
            yield final_text
            if evidence:
                yield EVIDENCE_SENTINEL + json.dumps({"evidence": evidence}, ensure_ascii=False)




if __name__ == '__main__':
    pass





