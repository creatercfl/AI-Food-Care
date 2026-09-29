from typing import Dict

from pydantic import BaseModel,Field

# StructuredOutputSchema
class CypherCheckerResponse(BaseModel):
    is_legal:bool=Field(description="当前语句是否为合法语句，如果是，则返回True, 否则返回False")
    error_msg:str=Field(description="如果is_legal为False,返回具体错误信息，否则返回空字符串")
    solve_method:str=Field(description="如果is_legal为False,返回解决方法，否则返回空字符串")
    original_cypher:str=Field(description="原始输入的Cypher语句")


class SupervisorRoute(BaseModel):
    expert:str=Field(description="最匹配的专家领域标识，取值范围：medication/diagnosis/diet/exam/emergency/special_population")
    confidence:float=Field(description="路由置信度，0~1之间的小数")
    rationale:str=Field(description="一句话说明为什么选择该专家")
    need_guideline:bool=Field(description="该问题是否可能需要临床指南要点作为补充（图谱覆盖可能不足时为True）")


# ToolsSchema
class CheckSyntaxError(BaseModel):
    cypher:str =Field(description="需要检测是否合法的cypher语句")


class Neo4jQueryParams(BaseModel):
    cypher:str = Field(description="需要执行的Cypher查询语句")
    params:dict[str,str]=Field(
        default=None,
        description="cypher查询语句当中的参数，字典形式。没有参数，使用默认值{}")

class EntityAlignmentList(BaseModel):
    entitys_to_alignment:list[Dict[str,str]] = Field(description="当前所需要对齐的所有实体的列表，每个需要对齐的实体为一个字典，key包含entity和label，entity为需要对齐的实体，label为实体对应的在知识图谱当中的label名称")

class GuidelineSearchParams(BaseModel):
    query:str = Field(description="用于检索临床指南要点的查询文本，通常是用户问题中的核心医疗主题，如'妊娠期感冒用药''儿童发热退热药'")
    top_k:int = Field(default=3, description="返回的指南要点条数，默认3条")

class NutritionQueryParams(BaseModel):
    food:str = Field(description="需要查询营养成分的食物名称，如'米饭''鸡胸肉''牛奶'，支持常见别名")