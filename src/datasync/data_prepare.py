import re
import os
import html
import json
import random
from configuration import config
from tqdm import tqdm
from neo4j import GraphDatabase

from entity_alignment import entity_alignment, vector_indexing



# 先在MySQL中创建表
# create table entity_mapping
# (
#     id            varchar(255)                        null,
#     synonym       text collate utf8mb4_bin            not null comment '同义词',
#     std_name      text                                not null comment '标准词',
#     entity_schema varchar(255)                        not null comment '实体类型',
#     is_reviewed   int       default 0                 not null comment '是否已审核',
#     create_time   timestamp default CURRENT_TIMESTAMP null comment '创建时间',
#     update_time   timestamp                           null on update CURRENT_TIMESTAMP comment '更新时间'
# )
#     comment '实体映射表';


def clear_neo4j():
    """清空 Neo4j 中的约束和数据，并创建属性唯一性约束"""

    # 连接 Neo4j
    with GraphDatabase.driver(config.NEO4J_CONFIG['uri'], auth=config.NEO4J_CONFIG['auth']) as driver:
        # 删除所有约束
        records, _, _ = driver.execute_query("SHOW CONSTRAINTS")
        constraints = [record["name"] for record in records]
        for constraint in constraints:
            driver.execute_query(f"DROP CONSTRAINT {constraint} IF EXISTS")
        print( "清空约束")

        # 清空数据
        driver.execute_query("MATCH (n) DETACH DELETE n")
        print( "清空数据")

        # 创建属性唯一性约束
        for constraint in [
            "CREATE CONSTRAINT disease_disease_name IF NOT EXISTS FOR (n:Disease) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT department_department_name IF NOT EXISTS FOR (n:Department) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT symptom_symptom_name IF NOT EXISTS FOR (n:Symptom) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT cause_cause_desc IF NOT EXISTS FOR (n:Cause) REQUIRE n.desc IS UNIQUE",
            "CREATE CONSTRAINT drug_drug_name IF NOT EXISTS FOR (n:Drug) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT food_food_name IF NOT EXISTS FOR (n:Food) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT way_way_name IF NOT EXISTS FOR (n:Way) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT prevent_prevent_desc IF NOT EXISTS FOR (n:Prevent) REQUIRE n.desc IS UNIQUE",
            "CREATE CONSTRAINT check_check_name IF NOT EXISTS FOR (n:Check) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT treat_treat_name IF NOT EXISTS FOR (n:Treat) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT people_people_name IF NOT EXISTS FOR (n:People) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT duration_duration_name IF NOT EXISTS FOR (n:Duration) REQUIRE n.name IS UNIQUE",
        ]:
            driver.execute_query(constraint)
        print( "属性唯一性约束创建成功")


# --------- 读取数据 ---------


# name:       str
# desc:       str
# acompany:   list
# department: list
# symptom:    list
# cause:      str
# drug:       list
# eat:        list
# not_eat:    list
# way:        str
# prevent:    str
# check:      list
# treat:      list
# people:     str
# duration:   str
def read_json_file(path):
    """读取 json 文件中的数据"""
    with open(path, "r", encoding="utf-8") as f:
        datas = [json.loads(line) for line in f]
    return datas


# --------- 数据清洗 ---------


def _standardize_text(text: str) -> str:
    """清洗一条文本"""
    if not (text and isinstance(text, str)):
        return text

    # 移除 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)

    # 全角转半角
    res = ""
    for uchar in text:
        u_code = ord(uchar)
        # 全角空格
        if u_code == 12288:
            res += chr(32)
        # 其他全角字符 (除空格外)
        elif 65281 <= u_code <= 65374:
            res += chr(u_code - 65248)
        else:
            res += uchar

    # 去除首尾空格，并将内部多个空格合并为一个
    res = re.sub(r"\s+", " ", res).strip()

    return res


def data_cleaning(datas):
    """数据清洗"""
    for data in datas:
        for k, v in data.items():
            if isinstance(v, str):
                data[k] = _standardize_text(v)
            elif isinstance(v, list):
                for i in range(len(v)):
                    data[k][i] = _standardize_text(v[i])


# --------- 导入 Neo4j ---------


def _batched_import(session, batch_data):
    """批量导入数据"""
    query = """
        UNWIND $batch AS row
        MERGE (disease:Disease {name: row.name})
        ON CREATE SET disease.desc = row.desc
        ON MATCH SET disease.desc = CASE
            WHEN disease.desc IS NULL THEN row.desc
            ELSE disease.desc
        END

        // 并发症
        WITH row, disease
        CALL (row, disease) {
            UNWIND row.acompany AS acomp
            WITH DISTINCT acomp WHERE acomp IS NOT NULL AND trim(acomp) <> ""
            MERGE (c:Disease {name: acomp})
            MERGE (disease)-[:ACOMPANY]->(c)
        }

        // 科室
        WITH row, disease
        CALL (row, disease) {
            UNWIND row.department AS dept
            WITH DISTINCT row, disease, dept WHERE dept IS NOT NULL AND trim(dept) <> ""
            MERGE (d:Department {name: dept})
            MERGE (disease)-[:BELONG]->(d)
        }

        // 症状
        WITH row, disease
        CALL (row, disease) {
            UNWIND row.symptom AS symp
            WITH DISTINCT row, disease, symp WHERE symp IS NOT NULL AND trim(symp) <> ""
            MERGE (s:Symptom {name: symp})
            MERGE (disease)-[:HAVE]->(s)
        }

        // 诱因
        WITH row, disease
        CALL (row, disease) {
            WITH row, disease
            WHERE row.cause IS NOT NULL AND trim(row.cause) <> ""
            MERGE (cs:Cause {desc: row.cause})
            MERGE (cs)-[:CAUSE]->(disease)
        }

        // 药物
        WITH row, disease
        CALL (row, disease) {
            UNWIND row.drug AS drg
            WITH DISTINCT row, disease, drg WHERE drg IS NOT NULL AND trim(drg) <> ""
            MERGE (dr:Drug {name: drg})
            MERGE (disease)-[:COMMON_USE]->(dr)
        }

        // 宜食用
        WITH row, disease
        CALL (row, disease) {
            UNWIND row.eat AS food
            WITH DISTINCT row, disease, food WHERE food IS NOT NULL AND trim(food) <> ""
            MERGE (f:Food {name: food})
            MERGE (disease)-[:EAT]->(f)
        }

        // 忌食用
        WITH row, disease
        CALL (row, disease) {
            UNWIND row.not_eat AS bad_food
            WITH DISTINCT row, disease, bad_food WHERE bad_food IS NOT NULL AND trim(bad_food) <> ""
            MERGE (bf:Food {name: bad_food})
            MERGE (disease)-[:NO_EAT]->(bf)
        }

        // 传播方式
        WITH row, disease
        CALL (row, disease) {
            WITH row, disease
            WHERE row.way IS NOT NULL AND trim(row.way) <> ""
            MERGE (w:Way {name: row.way})
            MERGE (disease)-[:TRANSMIT]->(w)
        }

        // 预防措施
        WITH row, disease
        CALL (row, disease) {
            WITH row, disease
            WHERE row.prevent IS NOT NULL AND trim(row.prevent) <> ""
            MERGE (p:Prevent {desc: row.prevent})
            MERGE (p)-[:PREVENT]->(disease)
        }

        // 医学检查
        WITH row, disease
        CALL (row, disease) {
            UNWIND row.check AS test
            WITH DISTINCT row, disease, test WHERE test IS NOT NULL AND trim(test) <> ""
            MERGE (ch:Check {name: test})
            MERGE (ch)-[:CHECK]->(disease)
        }

        // 治疗方式
        WITH row, disease
        CALL (row, disease) {
            UNWIND row.treat AS treatment
            WITH DISTINCT row, disease, treatment WHERE treatment IS NOT NULL AND trim(treatment) <> ""
            MERGE (tr:Treat {name: treatment})
            MERGE (tr)-[:TREAT]->(disease)
        }

        // 人群类别
        WITH row, disease
        CALL (row, disease) {
            WITH row, disease
            WHERE row.people IS NOT NULL AND trim(row.people) <> ""
            MERGE (pl:People {name: row.people})
            MERGE (disease)-[:COMMON_ON]->(pl)
        }

        // 治疗周期
        WITH row, disease
        CALL (row, disease) {
            WITH row, disease
            WHERE row.duration IS NOT NULL AND trim(row.duration) <> ""
            MERGE (dur:Duration {name: row.duration})
            MERGE (disease)-[:TREAT_DURATION]->(dur)
        }
    """
    session.run(query, parameters={"batch": batch_data})


def import_data_2_neo4j(datas: list[dict]):
    """导入所有数据到 Neo4j"""
    BATCH_SIZE = 500

    # 连接数据库
    with GraphDatabase.driver(config.NEO4J_CONFIG['uri'], auth=config.NEO4J_CONFIG['auth']) as driver:
        with driver.session() as session:
            for i in tqdm(range(0, len(datas), BATCH_SIZE), desc="导入数据"):
                batch = datas[i : i + BATCH_SIZE]
                # 处理字段缺失
                batch_prepared = [
                    {
                        "name": row["name"],
                        "desc": row.get("desc", ""),
                        "acompany": row.get("acompany", []),
                        "department": row.get("department", []),
                        "symptom": row.get("symptom", []),
                        "cause": row.get("cause", ""),
                        "drug": row.get("drug", []),
                        "eat": row.get("eat", []),
                        "not_eat": row.get("not_eat", []),
                        "way": row.get("way", ""),
                        "prevent": row.get("prevent", ""),
                        "check": row.get("check", []),
                        "treat": row.get("treat", []),
                        "people": row.get("people", ""),
                        "duration": row.get("duration", ""),
                    }
                    for row in batch
                ]
                # 批量导入
                _batched_import(session, batch_prepared)


# --------- 处理标注数据 ---------


def process_annotated_data(tgt_path):
    """将标注数据转换为模型微调数据、和符合知识图谱结构的数据"""
    predicate_name_map = {
        "预防": "预防措施",
        "辅助治疗": "治疗方式",
        "化疗": "治疗方式",
        "放射治疗": "治疗方式",
        "手术治疗": "治疗方式",
        "实验室检查": "医学检查",
        "影像学检查": "医学检查",
        "辅助检查": "医学检查",
        "组织学检查": "医学检查",
        "内窥镜检查": "医学检查",
        "筛查": "医学检查",
        "多发群体": "人群类别",
        "传播途径": "传播途径",
        "并发症": "并发症",
        "相关（转化）": "诱因",
        "相关（症状）": "症状",
        "临床表现": "症状",
        "治疗后症状": "症状",
        "侵及周围组织转移的症状": "症状",
        "病因": "诱因",
        "高危因素": "诱因",
        "风险评估因素": "诱因",
        "病史": "诱因",
        "遗传因素": "诱因",
        "发病机制": "诱因",
        "病理生理": "诱因",
        "药物治疗": "药物",
        "预后状况": "治疗周期",
    }

    name_label_map = {
        "疾病": "name",
        "描述": "desc",
        "科室": "department",
        "症状": "symptom",
        "并发症": "acompany",
        "诱因": "cause",
        "药物": "drug",
        "宜食用": "eat",
        "忌食用": "not_eat",
        "传播途径": "way",
        "预防措施": "prevent",
        "医学检查": "check",
        "治疗方式": "treat",
        "人群类别": "people",
        "治疗周期": "duration",
    }

    def findall_entity_pos_in_content(content, entity):
        """返回 content 中该实体所有的位置"""
        result_list = []
        for match in re.finditer(re.escape(entity), content):
            start_idx = match.start()
            end_idx = match.end()
            result_list.append({"text": entity, "start": start_idx, "end": end_idx})
        return result_list

    with (
        open(
            config.ROOT_DIR / "data" / "annotated_data" / "CMeIE-V2.jsonl",
            "r",
            encoding="utf-8",
        ) as read_file,
        open(tgt_path, "w", encoding="utf-8") as write_kg_file,
    ):
        finetuning_data = []
        kg_data = []
        for line in read_file:
            data = json.loads(line)
            content = data["text"]
            disease_dict = {}
            for spo in data["spo_list"]:
                if spo["predicate"] not in predicate_name_map:
                    continue
                disease = disease_dict.setdefault(spo["subject"], {})
                disease.setdefault(predicate_name_map[spo["predicate"]], []).append(
                    spo["object"]["@value"]
                )

            # 处理为模型微调的数据格式
            # for disease, relations in disease_dict.items():
            #     sample = {
            #         "content": content,
            #         "prompt": "疾病",
            #         "result_list": findall_entity_pos_in_content(content, disease),
            #     }
            #     finetuning_data.append(json.dumps(sample, ensure_ascii=False))
            #     for relation, entities in relations.items():
            #         for entity in entities:
            #             sample = {
            #                 "content": content,
            #                 "prompt": relation,
            #                 "result_list": findall_entity_pos_in_content(
            #                     content, entity
            #                 ),
            #             }
            #             sample_with_relation = {
            #                 "content": content,
            #                 "prompt": f"{disease}的{relation}",
            #                 "result_list": findall_entity_pos_in_content(
            #                     content, entity
            #                 ),
            #             }
            #             finetuning_data.append(json.dumps(sample, ensure_ascii=False))
            #             finetuning_data.append(
            #                 json.dumps(sample_with_relation, ensure_ascii=False)
            #             )

            # 处理为对应知识图谱结构的数据
            for disease, relations in disease_dict.items():
                tmp_sample = {"疾病": disease}
                tmp_sample.update(relations)
                sample = {}
                for k, v in tmp_sample.items():
                    sample[name_label_map[k]] = v
                    if name_label_map[k] in [
                        "cause",
                        "way",
                        "prevent",
                        "people",
                        "duration",
                    ]:
                        sample[name_label_map[k]] = "、".join(v)
                kg_data.append(json.dumps(sample, ensure_ascii=False))

        # # 将模型微调数据写入文件
        # random.shuffle(finetuning_data)
        # total = len(finetuning_data)
        # train_end = int(total * 0.8)
        # valid_end = train_end + int(total * 0.1)
        # train_data = finetuning_data[:train_end]
        # valid_data = finetuning_data[train_end:valid_end]
        # test_data = finetuning_data[valid_end:]
        # uie_processed_dir = config.BASE_DIR / "data/uie/processed"
        # os.makedirs(uie_processed_dir, exist_ok=True)
        # with open(uie_processed_dir / "train.jsonl", "w", encoding="utf-8") as f_train:
        #     f_train.writelines(sample + "\n" for sample in train_data)
        # with open(uie_processed_dir / "valid.jsonl", "w", encoding="utf-8") as f_valid:
        #     f_valid.writelines(sample + "\n" for sample in valid_data)
        # with open(uie_processed_dir / "test.jsonl", "w", encoding="utf-8") as f_test:
        #     f_test.writelines(sample + "\n" for sample in test_data)

        # 将知识图谱数据写入文件
        write_kg_file.writelines(sample + "\n" for sample in kg_data)


if __name__ == "__main__":
    # 清空 Neo4j
    clear_neo4j()

    # 读取 json 格式的知识图谱数据
    datas = read_json_file(config.ROOT_DIR / "data" / "knowledge_graph"/ "medical_kg.jsonl")
    # 数据清洗
    data_cleaning(datas)
    # 创建向量索引
    vector_indexing(datas)
    # 导入数据到 Neo4j
    import_data_2_neo4j(datas)

    # 处理标注数据
    processed_annotated_data_path = (
        config.ROOT_DIR / "data"/ "knowledge_graph"/ "medical_kg_from_annotation.jsonl"
    )
    process_annotated_data(processed_annotated_data_path)
    # 读取标注数据处理后的数据
    datas = read_json_file(processed_annotated_data_path)
    # 数据清洗
    data_cleaning(datas)
    # 实体对齐
    entity_alignment(datas, "name")
    entity_alignment(datas, "symptom")
    entity_alignment(datas, "cause")
    entity_alignment(datas, "drug")
    entity_alignment(datas, "eat")
    entity_alignment(datas, "no_eat")
    entity_alignment(datas, "people")
    entity_alignment(datas, "check")
    # 创建向量索引
    vector_indexing(datas)
    # 导入数据到 Neo4j
    import_data_2_neo4j(datas)
