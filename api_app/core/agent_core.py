import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import Tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from api_app.core.config import VECTOR_DB_PATH
from api_app.services.vector_db import search_knowledge_base


os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 准备底层组件
print("正在连接本地大模型和知识库...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)

llm = ChatOllama(model="qwen2.5:1.5b", temperature=0,num_gpu=0)

#封装外部工具
def kb_search_wrapper(query: str) -> str:
    """包装高级检索管道"""
    return search_knowledge_base(vector_db, query)

def calculate_math(query: str) -> str:
    try:
        return str(eval(query))
    except Exception:
        return "计算错误"

tools = [
    Tool(
        name="KnowledgeBaseSearch",
        func=kb_search_wrapper,
        description="用于查询企业内部规章制度、报销流程等垂直领域知识。注意：一旦获取到相关信息，必须立即停止调用该工具，并直接向用户输出最终答案！"
    ),
    Tool(
        name="Calculator",
        func=calculate_math,
        description="用于计算数学公式，输入应为合法的数学表达式（如 12*34）。"
    )
]

# 构建 Agent
print("正在初始化现代 LangGraph Agent...")
memory = MemorySaver()
agent_executor = create_react_agent(model=llm, tools=tools, checkpointer=memory)

# ================= 4. 测试运行 =================
if __name__ == "__main__":
    # 配置防暴走限制
    config = {
        "configurable": {"thread_id": "test_cmd_user"},
        "recursion_limit": 5
    }

    print("\n================ 测试 1：询问内部知识 ================")
    res1 = agent_executor.invoke(
        {"messages": [("user", "公司的请假制度是什么？")]},
        config=config
    )
    print(f"\n✅ 测试 1 最终答案: {res1['messages'][-1].content}\n")

    print("\n================ 测试 2：询问数学问题 ================")
    res2 = agent_executor.invoke(
        {"messages": [("user", "请帮我算一下 125 乘以 48 等于多少？")]},
        config=config
    )
    print(f"\n✅ 测试 2 最终答案: {res2['messages'][-1].content}\n")