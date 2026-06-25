"""知识库检索工具"""
from api_app.services.vector_db import search_knowledge_base


def create_kb_search_tool(vector_db):
    """创建一个绑定了 vector_db 的检索函数"""

    def search(query: str) -> str:
        return search_knowledge_base(vector_db, query)

    return search