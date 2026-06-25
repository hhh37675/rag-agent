"""Agent 智能体服务模块 - 终极环境兼容版"""
from langchain_classic.agents import initialize_agent, AgentType
from langchain_core.tools import Tool
from langchain_ollama import OllamaLLM

from api_app.core.config import LLM_MODEL, OLLAMA_BASE_URL
from api_app.tools.calculator import calculate_math
from api_app.tools.kb_search import create_kb_search_tool

def create_agent(vector_db):
    """手动实现对话管理以绕过环境报错"""
    print("正在初始化大语言模型 (LLM)...")

    # 使用 OllamaLLM，并使用 base_url 和 model 进行初始化
    # 将 num_gpu 放在 format_params 或者直接通过底层的 options 配置
    llm = OllamaLLM(
        model=LLM_MODEL,
        temperature=0.1,
        base_url=OLLAMA_BASE_URL,
        num_gpu=0  # 最新版 OllamaLLM 直接支持 num_gpu 参数
    )

    print(f"✅ LLM 加载完成: {LLM_MODEL}")

    #1. 挂载工具
    tools = [
        Tool(
            name="KnowledgeBaseSearch",
            func=create_kb_search_tool(vector_db),
            description="当用户提问关于企业内部文档、业务知识等内容时使用。[cite: 1]"
        ),
        Tool(
            name="Calculator",
            func=calculate_math,
            description="用于执行数学计算。[cite: 1]"
        )
    ]

    print("正在构建 Agent 工作流...[cite: 1]")

    # 2. 构建 Agent 执行器
    # 对话历史将由后端在调用时手动拼接
    agent_executor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=4,
        early_stopping_method="generate"
    )

    print("✅ Agent 工作流构建成功[cite: 1]")
    return agent_executor, llm