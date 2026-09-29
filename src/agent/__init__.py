from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agent.schema import CheckSyntaxError, EntityAlignmentList,Neo4jQueryParams,GuidelineSearchParams,NutritionQueryParams
from agent.tools_def import check_syntax_error,neo4j_query,entity_alignment,guideline_search,nutrition_query
from agent.prompts import major_agent_system_prompt_template
from configuration import config




def get_agent(neo4j_schema):
    """
    创建agent
    :return:
    """
    # 构造LLM
    llm = ChatDeepSeek(
        model=config.LLM_MODEL
    )
    # 构造tools
    neo4j_query_tool = tool(neo4j_query, args_schema=Neo4jQueryParams)
    check_syntax_tool = tool(check_syntax_error,args_schema=CheckSyntaxError)
    entity_alignment_tool = tool(entity_alignment,args_schema=EntityAlignmentList)
    guideline_search_tool = tool(guideline_search,args_schema=GuidelineSearchParams)
    nutrition_query_tool = tool(nutrition_query,args_schema=NutritionQueryParams)

    # 构造记忆模块
    memory_saver = InMemorySaver() if config.AGENT_WITH_MEMORY else None

    # 创建Agent
    agent = create_agent(
        model=llm,
        tools=[neo4j_query_tool,check_syntax_tool,entity_alignment_tool,guideline_search_tool,nutrition_query_tool],
        checkpointer=memory_saver,
        system_prompt=major_agent_system_prompt_template.format(neo4j_schema=neo4j_schema)
    )
    return agent