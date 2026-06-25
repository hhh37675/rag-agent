"""Pydantic 数据模型"""
from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    status: str = "success"