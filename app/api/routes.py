from __future__ import annotations

import secrets

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy import func, select, text

from app.api.schemas import (
    AskRequest,
    AskResponse,
    ConversationDetail,
    ConversationSummary,
    IngestRequest,
    IngestResponse,
    Usage,
)
from app.chat_service import ConversationNotFound, usage_stats
from app.db.models import Conversation, Message
from app.rag.ingest import ingest_confluence, ingest_markdown

router = APIRouter(prefix="/api/v1")


def container(request: Request):
    return request.app.state.container


def require_admin(request: Request, x_api_key: str = Header(default="")) -> None:
    expected = request.app.state.container.settings.admin_api_key
    if not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid X-API-Key")


@router.post("/ask", response_model=AskResponse, tags=["chat"])
def ask(body: AskRequest, c=Depends(container)) -> AskResponse:
    """向知識庫 Agent 提問。帶 conversation_id 可以延續多輪對話。"""
    try:
        r = c.chat.ask(body.question, body.conversation_id, body.space)
    except ConversationNotFound:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found") from None
    return AskResponse(
        conversation_id=r.conversation_id,
        answer=r.answer,
        route=r.route,
        standalone_question=r.standalone_question,
        sources=r.sources,
        usage=Usage(
            prompt_tokens=r.prompt_tokens,
            completion_tokens=r.completion_tokens,
            llm_calls=r.llm_calls,
        ),
        latency_ms=r.latency_ms,
    )


@router.get("/conversations", response_model=list[ConversationSummary], tags=["chat"])
def list_conversations(limit: int = 20, c=Depends(container)):
    with c.session_factory() as s:
        rows = s.execute(
            select(Conversation, func.count(Message.id))
            .outerjoin(Message)
            .group_by(Conversation.id)
            .order_by(Conversation.updated_at.desc())
            .limit(min(limit, 100))
        ).all()
    return [
        ConversationSummary(
            id=conv.id, title=conv.title, updated_at=conv.updated_at, message_count=n
        )
        for conv, n in rows
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail, tags=["chat"])
def get_conversation(conversation_id: str, c=Depends(container)):
    with c.session_factory() as s:
        conv = s.get(Conversation, conversation_id)
        if conv is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
        return ConversationDetail.model_validate(
            {
                "id": conv.id,
                "title": conv.title,
                "messages": [
                    {
                        "role": m.role,
                        "content": m.content,
                        "route": m.route,
                        "sources": m.sources,
                        "created_at": m.created_at,
                    }
                    for m in conv.messages
                ],
            }
        )


@router.delete("/conversations/{conversation_id}", status_code=204, tags=["chat"])
def delete_conversation(conversation_id: str, c=Depends(container)):
    with c.session_factory() as s:
        conv = s.get(Conversation, conversation_id)
        if conv is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
        s.delete(conv)
        s.commit()


@router.post(
    "/ingest", response_model=IngestResponse, tags=["admin"], dependencies=[Depends(require_admin)]
)
def ingest(body: IngestRequest, c=Depends(container)):
    """匯入文件到知識庫（增量：內容沒變的文件會跳過）。需要 X-API-Key。"""
    try:
        if body.source == "markdown":
            if not body.path:
                raise ValueError("source=markdown 時必須提供 path")
            stats = ingest_markdown(c.store, body.path, body.space)
        else:
            stats = ingest_confluence(c.store, c.settings, body.spaces)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from None
    return IngestResponse(**vars(stats))


@router.get("/stats", tags=["admin"], dependencies=[Depends(require_admin)])
def stats(days: int = 7, c=Depends(container)):
    """最近 N 天的回答數、token 用量、延遲 p50/p95、路由分佈（直接 SQL 統計）。"""
    with c.session_factory() as s:
        return usage_stats(s, days)


health_router = APIRouter()


@health_router.get("/health", tags=["ops"])
def health(c=Depends(container)):
    try:
        with c.session_factory() as s:
            s.execute(text("SELECT 1"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"database: {e}") from None
    return {"status": "ok", "llm_provider": c.settings.llm_provider, "model": c.model_name}
