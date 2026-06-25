"""FastAPI 服务入口"""
import os
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
chat_history_buffer = []

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

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global chat_history_buffer

    try:
        agent = app_state.get("agent")

        # 手动拼接历史记录给模型
        history_str = "\n".join(chat_history_buffer[-5:])  # 只取最近 5 条
        full_input = f"对话历史：\n{history_str}\n\n当前问题：{request.question}"

        # 执行推理
        result = await run_in_threadpool(agent.invoke, {"input": full_input})
        answer = result.get("output", str(result)) if isinstance(result, dict) else str(result)

        # 更新历史记录
        chat_history_buffer.append(f"问：{request.question}")
        chat_history_buffer.append(f"答：{answer}")

        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"系统推理错误: {str(e)}")

@app.get("/api/health")
async def health_check():
    llm = app_state.get("llm")
    vector_db = app_state.get("vector_db")

    try:
        if llm:
            llm.invoke("ping")
        ollama_status = "connected"
    except Exception:
        ollama_status = "disconnected"

    return {
        "ollama": ollama_status,
        "vector_db": "connected" if vector_db else "disconnected",
        "embedding_model": EMBEDDING_MODEL
    }

# ================= 4. 服务启动 =================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="绑定的 Host")
    parser.add_argument("--port", type=int, default=8000, help="绑定的端口")
    parser.add_argument("--reload", action="store_true", help="是否开启热重载")
    args = parser.parse_args()

    uvicorn.run("api_app.main:app", host=args.host, port=args.port, reload=args.reload)