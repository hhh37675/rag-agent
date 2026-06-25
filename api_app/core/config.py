"""全局配置"""
import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, "local_models", "bge-large-zh-v1.5")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")

# 模型配置
EMBEDDING_MODEL = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else "BAAI/bge-large-zh-v1.5"
LLM_MODEL = "qwen2.5:1.5b"

# Ollama 配置
OLLAMA_BASE_URL = "http://localhost:11434"

# 检索配置
SEARCH_K = 3