"""向量数据库与混合检索服务"""
import os
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from api_app.core.config import VECTOR_DB_PATH, RETRIEVAL_K

# 全局缓存，防止每次对话都重新提取全库构建倒排索引
_bm25_retriever_cache = None

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
    global _bm25_retriever_cache

    # 1. 向量检索器
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    try:
        # 如果缓存已存在，直接组装返回，避免卡死！
        if _bm25_retriever_cache is not None:
            return EnsembleRetriever(
                retrievers=[_bm25_retriever_cache, vector_retriever],
                weights=[0.4, 0.6]
            )

        # 2. 如果没有缓存，则初始化并缓存
        db_data = vector_db.get()
        if not db_data or not db_data.get('documents'):
            return vector_retriever  # 如果库是空的，降级为纯向量检索

        docs = []
        for i in range(len(db_data['documents'])):
            # 全获取元数据，防止 metadata 为 None 时报错
            meta = db_data['metadatas'][i] if db_data['metadatas'] and db_data['metadatas'][i] else {}
            docs.append(Document(
                page_content=db_data['documents'][i],
                metadata=meta
            ))

        print("⚙️ 首次调用，正在构建全局 BM25 倒排索引缓存...")
        _bm25_retriever_cache = BM25Retriever.from_documents(docs)
        _bm25_retriever_cache.k = RETRIEVAL_K

        # 3. 组装混合检索器
        return EnsembleRetriever(
            retrievers=[_bm25_retriever_cache, vector_retriever],
            weights=[0.4, 0.6]
        )

    except Exception as e:
        print(f"⚠️ BM25 检索器初始化失败，降级为纯向量检索: {e}")
        return vector_retriever


def search_knowledge_base(vector_db, query: str) -> str:
    """供 Agent 调用的检索入口 (已升级为混合检索)"""
    try:
        print(f"🔍 正在执行混合检索 (BM25 + 向量)，查询: {query}")
        retriever = create_hybrid_retriever(vector_db)
        docs = retriever.invoke(query)

        if not docs:
            return "知识库中未找到相关内容。"

        print("\n================ 📚 知识库检索来源 ================")
        for i, doc in enumerate(docs):
            # 获取文档来源路径，如果找不到则显示 '未知来源'
            source = doc.metadata.get('source', '未知来源')
            print(f"📄 片段 {i + 1} 来源: {source}")
        print("===================================================\n")

        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', '未知来源')
            results.append(f"【片段 {i + 1}】（来源：{source}）\n{doc.page_content}")

        return "\n\n---\n\n".join(results)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return f"知识库混合检索失败: {str(e)}"