"""把 Agent 和資料庫串起來：讀歷史 → 跑 Agent → 存回答與用量。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Conversation, Message, utcnow
from app.observability.callbacks import UsageTracker
from app.rag.store import RetrievedChunk


class ConversationNotFound(Exception):
    pass


@dataclass
class ChatResult:
    conversation_id: str
    answer: str
    route: str
    standalone_question: str
    sources: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    latency_ms: int = 0


def _sources(docs: list[RetrievedChunk]) -> list[dict]:
    """同一份文件可能被切成多個片段命中，回傳時以文件為單位去重。"""
    seen: dict[str, dict] = {}
    for d in docs:
        if d.document_id not in seen:
            seen[d.document_id] = {
                "title": d.title,
                "url": d.url,
                "space": d.space,
                "score": round(d.score, 3),
                "snippet": d.content[:160],
            }
    return list(seen.values())


class ChatService:
    def __init__(
        self, graph, session_factory: sessionmaker[Session], model_name: str, history_turns: int = 6
    ):
        self.graph = graph
        self.session_factory = session_factory
        self.model_name = model_name
        self.history_turns = history_turns

    def ask(
        self, question: str, conversation_id: str | None = None, space: str | None = None
    ) -> ChatResult:
        started = time.perf_counter()
        with self.session_factory() as session:
            if conversation_id:
                conv = session.get(Conversation, conversation_id)
                if conv is None:
                    raise ConversationNotFound(conversation_id)
            else:
                conv = Conversation(title=question[:60])
                session.add(conv)
                session.flush()

            recent = session.scalars(
                select(Message)
                .where(Message.conversation_id == conv.id)
                .order_by(Message.id.desc())
                .limit(self.history_turns)
            ).all()
            history = [{"role": m.role, "content": m.content} for m in reversed(recent)]

            tracker = UsageTracker(self.model_name)
            state = self.graph.invoke(
                {"question": question, "history": history, "space": space},
                config={"callbacks": [tracker]},
            )
            latency_ms = round((time.perf_counter() - started) * 1000)
            sources = _sources(state.get("documents") or [])

            session.add(Message(conversation_id=conv.id, role="user", content=question))
            session.add(
                Message(
                    conversation_id=conv.id,
                    role="assistant",
                    content=state["answer"],
                    route=state["route"],
                    sources=sources,
                    model=self.model_name,
                    prompt_tokens=tracker.prompt_tokens,
                    completion_tokens=tracker.completion_tokens,
                    latency_ms=latency_ms,
                )
            )
            conv.updated_at = utcnow()
            session.commit()

            return ChatResult(
                conversation_id=conv.id,
                answer=state["answer"],
                route=state["route"],
                standalone_question=state["standalone_question"],
                sources=sources,
                prompt_tokens=tracker.prompt_tokens,
                completion_tokens=tracker.completion_tokens,
                llm_calls=tracker.calls,
                latency_ms=latency_ms,
            )


def usage_stats(session: Session, days: int = 7) -> dict:
    """用 SQL 直接從 messages 表算出成本與效能指標（不需要另外的監控系統也看得到）。"""
    since = func.now() - func.make_interval(0, 0, 0, days)
    base = (
        select(Message).where(Message.role == "assistant", Message.created_at >= since).subquery()
    )
    row = session.execute(
        select(
            func.count(),
            func.coalesce(func.sum(base.c.prompt_tokens), 0),
            func.coalesce(func.sum(base.c.completion_tokens), 0),
            func.percentile_cont(0.5).within_group(base.c.latency_ms),
            func.percentile_cont(0.95).within_group(base.c.latency_ms),
        )
    ).one()
    by_route = dict(
        session.execute(select(base.c.route, func.count()).group_by(base.c.route)).all()
    )
    return {
        "days": days,
        "answers": row[0],
        "prompt_tokens": int(row[1]),
        "completion_tokens": int(row[2]),
        "latency_p50_ms": round(row[3]) if row[3] is not None else None,
        "latency_p95_ms": round(row[4]) if row[4] is not None else None,
        "by_route": by_route,
    }
