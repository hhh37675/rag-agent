from langchain_classic.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

#  1. 准备底层组件
print("正在连接本地大模型和知识库...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
vector_db = Chroma(persist_directory="./vector_db", embedding_function=embeddings)
llm = Ollama(model="deepseek-r1:1.5b", temperature=0.1)

# 2. 封装外部工具
def search_knowledge_base(query: str) -> str:
    """当用户询问公司规章制度、内部文档、报销流程等问题时，必须使用此工具"""
    # 直接去数据库搜索最相关的 3 个文档片段
    docs = vector_db.similarity_search(query, k=3)
    if not docs:
        return "知识库中未找到相关内容。"

    # 把文本片段拼成一段大白话，直接丢给 Agent 自己去看
    return "\n".join([f"片段 {i + 1}: {doc.page_content}" for i, doc in enumerate(docs)])


def calculate_math(query: str) -> str:
    """进行纯数学计算时，使用此工具"""
    try:
        return str(eval(query))
    except Exception:
        return "计算错误"


tools = [
    Tool(
        name="KnowledgeBaseSearch",
        func=search_knowledge_base,
        description="用于查询企业内部规章制度、报销流程等垂直领域知识。"
    ),
    Tool(
        name="Calculator",
        func=calculate_math,
        description="用于计算数学公式，输入应为合法的数学表达式（如 12*34）。"
    )
]

# ================= 3. 构建 Agent =================
print("正在初始化经典 Agent...")
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,  # 开启日志看推理过程
    handle_parsing_errors=True
)

# ================= 4. 测试运行 =================
if __name__ == "__main__":
    print("\n================ 测试 1：询问内部知识 ================")
    # 这里的调用方法是 run
    response1 = agent.run("公司的请假制度是什么？")
    print(f"\n✅ 测试 1 最终答案: {response1}\n")

    print("\n================ 测试 2：询问数学问题 ================")
    response2 = agent.run("请帮我算一下 125 乘以 48 等于多少？")
    print(f"\n✅ 测试 2 最终答案: {response2}\n")