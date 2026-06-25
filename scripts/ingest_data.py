import os
import sys

# 自动将项目根目录添加到系统搜索路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 然后再导入你的模块
from api_app.core.config import BASE_DIR, VECTOR_DB_PATH
import glob
from typing import List

# 导入 LangChain 相关组件
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 导入项目中已有的模块和配置
from api_app.core.config import BASE_DIR, VECTOR_DB_PATH
from api_app.services.embedding import load_embeddings
from api_app.services.vector_db import load_vector_db

# 定义默认的原始文档存放目录
DOCS_DIR = os.path.join(BASE_DIR, "data", "raw_docs")


def load_documents(source_dir: str) -> List[Document]:
    """遍历目录并加载支持的文档类型"""
    print(f"📂 正在扫描目录: {source_dir}")
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        print(f"⚠️ 目录不存在，已自动创建: {source_dir}。请放入文档后再试。")
        return []

    documents = []
    # 支持的文件格式与对应的加载器
    loaders_mapping = {
        ".txt": TextLoader,
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader
    }

    for ext, loader_class in loaders_mapping.items():
        # 查找所有匹配的文件
        glob_pattern = os.path.join(source_dir, f"**/*{ext}")
        files = glob.glob(glob_pattern, recursive=True)
        for file_path in files:
            print(f"📄 正在加载文件: {file_path}")
            try:
                # TextLoader 需要指定编码，防止中文乱码
                if ext == ".txt":
                    loader = loader_class(file_path, encoding='utf-8')
                else:
                    loader = loader_class(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"❌ 加载文件失败 {file_path}: {e}")

    print(f"✅ 共加载了 {len(documents)} 个原始文档页/段。")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """对长文本进行分块处理"""
    if not documents:
        return []

    print("✂️ 正在进行文本分块...")
    # 使用递归字符分割器，适合中文长文本
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 每个块的最大字符数
        chunk_overlap=50,  # 块与块之间的重叠字符，保留上下文连贯性
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ 文本分块完成，共生成 {len(chunks)} 个片段。")
    return chunks


def main():
    # 1. 加载并分割文档
    docs = load_documents(DOCS_DIR)
    if not docs:
        print("⏭️ 没有找到可处理的文档，程序退出。")
        return

    chunks = split_documents(docs)

    # 2. 初始化现有的 Embedding 和 VectorDB 组件
    embeddings = load_embeddings()
    vector_db = load_vector_db(embeddings)

    # 3. 向量化并入库
    print("💾 正在将文本块向量化并写入 ChromaDB...")
    vector_db.add_documents(chunks)

    # 强制持久化 (针对旧版 Chroma，新版通常会自动持久化)
    if hasattr(vector_db, "persist"):
        vector_db.persist()

    print(f"🎉 知识库更新成功！数据已保存至: {VECTOR_DB_PATH}")


if __name__ == "__main__":
    main()