# Agent 框架与模型能力不匹配无法检索向量数据库
# from langchain_core.tools import Tool
# from langchain_ollama import ChatOllama
# from langgraph.prebuilt import create_react_agent
# from langgraph.checkpoint.memory import MemorySaver
#
# from api_app.core.config import OLLAMA_BASE_URL
# from api_app.tools.calculator import calculate_math
# # 引入我们在 vector_db 里写好的高级检索（混合检索 + BGE 重排）
# from api_app.services.vector_db import search_knowledge_base
#
# def create_agent(vector_db):
#     print("🚀 正在初始化现代 LangGraph 架构的智能体...")
#
#     llm = ChatOllama(
#         model="qwen2.5:1.5b",
#         temperature=0,
#         base_url=OLLAMA_BASE_URL,
#         num_gpu = 0
#     )
#
#     # 包装高级 RAG 检索工具
#     def kb_search_wrapper(query: str) -> str:
#         # 调用自带重排功能的终极检索
#         return search_knowledge_base(vector_db, query)
#
#     # 注册工具
#     tools = [
#         Tool(
#             name="KnowledgeBaseSearch",
#             func=kb_search_wrapper,
#             description="用于查询企业内部规章制度、报销流程等垂直领域知识。注意：一旦获取到相关信息，必须立即停止调用该工具，并直接向用户输出最终答案！"
#         ),
#         Tool(
#             name="Calculator",
#             func=calculate_math,
#             description="用于计算数学公式，输入应为合法的数学表达式（如 12*34）。计算完成后立即停止调用并输出结果。"
#         )
#     ]
#
#     # 构建 LangGraph Agent
#     memory = MemorySaver()
#
#     system_prompt = """你是一个企业内部的智能助手。
#     当用户询问关于企业规定、流程、制度等问题时，你【必须】首选调用 KnowledgeBaseSearch 工具来获取真实信息。
#     绝对不要凭空捏造（幻觉）答案。如果工具返回未找到，你就说不知道。拿到结果后必须立即停止调用工具并输出答案"""
#
#     agent_executor = create_react_agent(
#         model=llm,
#         tools=tools,
#         checkpointer=memory,
#         prompt = system_prompt
#     )
#
#     print("✅ LangGraph 智能体工作流构建成功！")
#     return agent_executor, llm
"""api_app/services/agent.py"""
from langchain_ollama import ChatOllama
from api_app.core.config import OLLAMA_BASE_URL
from api_app.services.vector_db import create_hybrid_retriever

def create_agent(vector_db):
    print("🚀 正在初始化RAG 架构 ")

    llm = ChatOllama(
        model="qwen2.5:1.5b",
        temperature=0.1,
        base_url=OLLAMA_BASE_URL,
        num_gpu=0
    )

    retriever = create_hybrid_retriever(vector_db)

    class SimpleRAGChain:
        def __init__(self, llm, retriever):
            self.llm = llm
            self.retriever = retriever

        def invoke(self, inputs):
            query = inputs["input"]

            # 1. 手动去知识库里查资料
            docs = self.retriever.invoke(query)

            if not docs:
                return {"answer": "抱歉，知识库中未找到相关内容，我也无法回答该问题。", "context": []}

            # 2. 把查出来的资料提取出来，拼装成一段长文本
            context_text = "\n\n".join([f"来源片段:\n{doc.page_content}" for doc in docs])

            # 3. 强行喂给 1.5B 小模型
            prompt_text = f"""请你作为企业的智能问答助手，基于以下【已知内容】专业、准确地回答用户问题。
如果【已知内容】中没有相关信息，请直接回答“抱歉，知识库中未找到相关内容”，切勿编造。

【已知内容】:
{context_text}

【用户提问】:
{query}"""

            # 4. 调用大模型生成最终回答
            response = self.llm.invoke(prompt_text)

            # 返回字典格式，完美兼容你 main.py 里的解析逻辑
            return {"answer": response.content, "context": docs}

    print("✅ 手工 RAG 问答链构建成功！")
    return SimpleRAGChain(llm, retriever), llm