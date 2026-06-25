"""Embedding 模型加载"""
import os
from langchain_huggingface import HuggingFaceEmbeddings
from api_app.core.config import EMBEDDING_MODEL, LOCAL_MODEL_PATH

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def load_embeddings():
    """加载 Embedding 模型"""
    print("🤖 正在加载 Embedding 模型...")

    if not os.path.exists(LOCAL_MODEL_PATH):
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print("⚠️ 本地模型不存在，使用镜像站在线加载...")
    else:
        print(f"✅ 使用本地模型: {LOCAL_MODEL_PATH}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print(f"✅ Embedding 模型加载成功: {EMBEDDING_MODEL}")
    return embeddings