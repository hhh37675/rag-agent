import os
import sys
import glob
from typing import List

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from api_app.core.config import BASE_DIR, VECTOR_DB_PATH
from api_app.services.embedding import load_embeddings
from api_app.services.vector_db import load_vector_db

DOCS_DIR = os.path.join(BASE_DIR, "data", "raw_docs")


def load_documents(source_dir: str) -> List[Document]:
    print(f"📂 正在扫描目录: {source_dir}")
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        return []

    documents = []
    # 查找常见办公文档
    files = glob.glob(os.path.join(source_dir, "**/*.*"), recursive=True)

    for file_path in files:
        ext = os.path.splitext(file_path)[1].lower()
        print(f"📄 正在解析文件: {file_path}")

        try:
            if ext == ".txt":
                # txt 仍然保留基础 loader，处理编码问题
                loader = TextLoader(file_path, encoding='utf-8')
            elif ext in [".pdf", ".docx", ".doc", ".pptx"]:
                # 使用 Unstructured 处理复杂排版
                loader = UnstructuredLoader(
                    file_path,
                    strategy="hi_res",
                    mode="elements"  # 将文档拆解为元素（标题、表格、段落等）
                )
            else:
                continue  # 跳过不支持的文件

            documents.extend(loader.load())
        except Exception as e:
            print(f"❌ 解析失败 {file_path}: {e}")

    print(f"✅ 共解析出 {len(documents)} 个原始文档元素。")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """文本分块处理"""
    if not documents:
        return []

    print("✂️ 正在进行文本分块...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ 文本分块完成，共生成 {len(chunks)} 个片段。")
    return chunks


def main():
    docs = load_documents(DOCS_DIR)
    if not docs: return
    chunks = split_documents(docs)

    embeddings = load_embeddings()
    vector_db = load_vector_db(embeddings)

    print("💾 正在将文本块向量化并写入 ChromaDB...")
    vector_db.add_documents(chunks)
    print(f"🎉 知识库更新成功！数据已保存至: {VECTOR_DB_PATH}")


if __name__ == "__main__":
    main()