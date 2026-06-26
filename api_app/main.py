"""FastAPI 服务入口"""
import os

import uuid
from langchain_core.messages import HumanMessage

# 在所有导入之前执行，强制禁止所有子进程调用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import uvicorn
from contextlib import asynccontextmanager
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api_app.core.config import EMBEDDING_MODEL
from api_app.schemas.schemas import ChatRequest, ChatResponse
from api_app.services.embedding import load_embeddings
from api_app.services.vector_db import load_vector_db
from api_app.services.agent import create_agent

# 使用字典存储全局应用状态，避免全局变量污染
app_state = {}

#1. 生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 正在初始化 AI 组件...")
    embeddings = load_embeddings()
    vector_db = load_vector_db(embeddings)
    agent, llm = create_agent(vector_db)

    app_state["agent"] = agent
    app_state["llm"] = llm
    app_state["vector_db"] = vector_db
    print("✅ AI 组件初始化完成，服务已就绪")

    yield  # 这里交出控制权，应用开始运行

    print("🧹 正在清理资源...")
    app_state.clear()

# 2. 初始化 FastAPI
app = FastAPI(title="RAG Agent API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. API 路由定义
@app.get("/")
async def root():
    return {
        "service": "RAG Agent 智能问答 API",
        "version": "2.0",
        "status": "running"
    }
# Agent 框架与模型能力不匹配无法检索向量数据库
# @app.post("/api/chat", response_model=ChatResponse)
# async def chat_endpoint(request: ChatRequest):
#     try:
#         agent = app_state.get("agent")
#         # 提取或生成当前用户的唯一会话 ID
#         session_id = request.session_id if request.session_id else str(uuid.uuid4())
#
#         config = {
#             "configurable": {"thread_id": session_id},
#             "recursion_limit": 15
#         }
#
#         # 构造当前用户的新问题
#         input_message = HumanMessage(content=request.question)
#
#         result = await run_in_threadpool(
#             agent.invoke,
#             {"messages": [input_message]},
#             config
#         )
#
#         # 提取最新的一条回答 (历史记录保存在 result["messages"] 这个列表里)
#         final_answer = result["messages"][-1].content
#
#         return ChatResponse(answer=final_answer, status="success")
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"系统推理错误: {str(e)}")
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        qa_chain = app_state.get("agent")

        # 传统 RAG 不需要传 thread_id 和 messages，直接传 input 即可
        result = await run_in_threadpool(
            qa_chain.invoke,
            {"input": request.question}
        )

        # 提取回答内容
        final_answer = result["answer"]

        # 终端打印调试（证明查了数据库）
        print(f"\n🗣️ 用户提问: {request.question}")
        print("================ 📚 后台强制查到的知识库来源 ================")
        for idx, doc in enumerate(result.get("context", [])):
            print(f"来源 {idx + 1}: {doc.metadata.get('source', '未知')}")
        print("============================================================\n")
        print(f"🤖 AI 回答: {final_answer}")

        return ChatResponse(answer=final_answer, status="success")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"系统推理错误: {str(e)}")

# ================= 4. 服务启动 =================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="绑定的 Host")
    parser.add_argument("--port", type=int, default=8000, help="绑定的端口")
    parser.add_argument("--reload", action="store_true", help="是否开启热重载")
    args = parser.parse_args()

    uvicorn.run("api_app.main:app", host=args.host, port=args.port, reload=args.reload)