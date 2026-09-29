"""指南 RAG 回退库：当知识图谱覆盖不到用户问题时，用权威临床指南要点补覆盖。

设计要点：
- 数据源：data/guidelines/curated_guidelines.jsonl（人工整理的指南要点，含出处）；
- 独立 Chroma collection：smart_medical_guidelines，与实体对齐库（smart_medical）隔离，
  互不干扰；使用 cosine 距离空间，检索结果按相似度阈值过滤，避免不相关内容污染回答；
- 幂等构建：按指南 id 去重，可重复运行；
- 只读检索：问答链路中仅做 query，不写库，符合 Chroma 单进程铁律。
"""
from configuration import config
import chromadb
import json
import logging

logger = logging.getLogger(__name__)

GUIDELINE_COLLECTION = "smart_medical_guidelines"
GUIDELINE_FILE = config.ROOT_DIR / "data" / "guidelines" / "curated_guidelines.jsonl"

_client = None
_embedding_model = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(config.VECTOR_STORE_DIR))
    return _client


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        # 复用实体对齐的嵌入模型（同进程内单例，避免重复加载）
        from datasync.entity_alignment import get_embedding_model
        _embedding_model = get_embedding_model()
    return _embedding_model


def load_guidelines():
    """读取人工整理的指南要点（JSONL）。"""
    items = []
    for line in GUIDELINE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items


def build_index(force=False):
    """构建/更新指南向量索引（幂等）。

    :param force: True 时先删除旧 collection 再全量重建
    :return: collection 中的总条数
    """
    client = _get_client()
    model = _get_embedding_model()
    items = load_guidelines()
    if force:
        try:
            client.delete_collection(GUIDELINE_COLLECTION)
            print(f"已删除旧指南 collection: {GUIDELINE_COLLECTION}")
        except Exception as e:
            print("删除旧指南 collection 跳过:", e)
    collection = client.get_or_create_collection(
        GUIDELINE_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    existing = set(collection.get()["ids"])
    todo = [it for it in items if it["id"] not in existing]
    if not todo:
        print(f"指南索引已是最新，总计 {collection.count()} 条")
        return collection.count()

    ids = [it["id"] for it in todo]
    # 标题 + 正文一起嵌入，提升主题匹配召回
    documents = [f"{it['title']}。{it['content']}" for it in todo]
    metadatas = [
        {"title": it.get("title", ""), "source": it.get("source", ""), "content": it.get("content", "")}
        for it in todo
    ]
    embeddings = model.encode(documents, normalize_embeddings=True, show_progress_bar=True)
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    print(f"指南索引新增 {len(todo)} 条，总计 {collection.count()} 条")
    return collection.count()


def search(query, top_k=3, threshold=0.5):
    """语义检索指南要点。

    :return: [{title, source, content, score}]，score 为 cosine 相似度（越大越相似）
    """
    client = _get_client()
    try:
        collection = client.get_collection(GUIDELINE_COLLECTION)
    except Exception:
        logger.exception("指南索引不存在，请先运行 build_index()")
        return []
    total = collection.count()
    if total == 0:
        return []
    model = _get_embedding_model()
    emb = model.encode(query, normalize_embeddings=True)
    res = collection.query(query_embeddings=[emb], n_results=min(top_k, total))
    out = []
    ids = res["ids"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    for i in range(len(ids)):
        # cosine 距离 -> 相似度
        score = 1.0 - float(dists[i])
        if score < threshold:
            continue
        out.append({
            "title": metas[i].get("title", ""),
            "source": metas[i].get("source", ""),
            "content": metas[i].get("content", ""),
            "score": round(score, 4),
        })
    return out


if __name__ == "__main__":
    build_index()
