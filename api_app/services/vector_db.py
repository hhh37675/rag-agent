"""向量数据库与混合检索服务"""
import os
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from api_app.core.config import VECTOR_DB_PATH, SEARCH_K

def load_vector_db(embeddings):
    """加载向量数据库"""
    print("🤖 正在加载向量数据库...")
    vector_db = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )
    print(f"✅ 向量数据库加载成功: {VECTOR_DB_PATH}")
    return vector_db


def create_hybrid_retriever(vector_db):
    """构建混合检索器 (BM25 关键词 + Chroma 向量语义)"""
    # 1. 向量检索器
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": SEARCH_K})

    try:
        # 2. 从 Chroma 中提取所有已知文档，动态构建 BM25 倒排索引
        db_data = vector_db.get()
        if not db_data or not db_data.get('documents'):
            return vector_retriever  # 如果库是空的，降级为纯向量检索

        docs = []
        for i in range(len(db_data['documents'])):
            docs.append(Document(
                page_content=db_data['documents'][i],
                metadata=db_data['metadatas'][i]
            ))

        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = SEARCH_K

        # 3. 组装混合检索器，使用倒数秩融合 (RRF)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )
        return ensemble_retriever

    except Exception as e:
        print(f"⚠️ BM25 检索器初始化失败，降级为纯向量检索: {e}")
        return vector_retriever


def search_knowledge_base(vector_db, query: str) -> str:
    """供 Agent 调用的检索入口 (已升级为混合检索)"""
    try:
        print(f"🔍 正在执行混合检索 (BM25 + 向量)，查询: {query}")
        # 获取混合检索器
        retriever = create_hybrid_retriever(vector_db)

        # 执行双路召回与 RRF 融合
        docs = retriever.invoke(query)

        if not docs:
            return "知识库中未找到相关内容。"

        # 提取融合后的 Top-K 结果
        results = []
        for i, doc in enumerate(docs[:SEARCH_K]):
            source = doc.metadata.get('source', '未知来源')
            results.append(f"【片段 {i + 1}】（来源：{source}）\n{doc.page_content}")

        return "\n\n---\n\n".join(results)
    except Exception as e:
        return f"知识库混合检索失败: {str(e)}"