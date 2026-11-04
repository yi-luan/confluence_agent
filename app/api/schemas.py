from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["特休要提前幾天申請？"])
    conversation_id: str | None = Field(default=None, description="延續既有對話時帶入")
    space: str | None = Field(default=None, description="只搜尋指定的 Confluence space")


class Source(BaseModel):
    title: str
    url: str | None
    space: str
    score: float
    snippet: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    llm_calls: int


class AskResponse(BaseModel):
    conversation_id: str
    answer: str
    route: str
    standalone_question: str
    sources: list[Source]
    usage: Usage
    latency_ms: int


class MessageOut(BaseModel):
    role: str
    content: str
    route: str | None = None
    sources: list[Source] | None = None
    created_at: datetime


class ConversationSummary(BaseModel):
    id: str
    title: str
    updated_at: datetime
    message_count: int


class ConversationDetail(BaseModel):
    id: str
    title: str
    messages: list[MessageOut]


class IngestRequest(BaseModel):
    source: Literal["markdown", "confluence"]
    path: str | None = Field(default=None, description="source=markdown 時的資料夾路徑")
    space: str = Field(default="HANDBOOK", description="markdown 匯入時要歸到哪個 space")
    spaces: list[str] | None = Field(default=None, description="source=confluence 時要同步的 space")


class IngestResponse(BaseModel):
    added: int
    updated: int
    unchanged: int
    chunks: int
