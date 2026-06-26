"""全局配置"""
import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "local_models", "bge-large-zh-v1.5")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")

# 模型配置
EMBEDDING_MODEL = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "BAAI/bge-large-zh-v1.5"
LLM_MODEL = "qwen2.5:1.5b"
OLLAMA_BASE_URL = "http://localhost:11434"

# RAG 检索与重排配置
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# 第一阶段（粗排）：BM25 + 向量混合检索的召回数量
RETRIEVAL_K = 10

# 第二阶段（精排）：Reranker 重排后最终喂给大模型的片段数量 (取最精华的部分)
RERANK_TOP_K = 2