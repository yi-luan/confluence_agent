"""LangGraph Agent：先判斷問題類型，再走不同的處理路線。

START → condense（多輪對話時改寫問題）→ route
  route ─ knowledge ─→ retrieve → answer → END
        ├ directory ─→ directory_lookup → END   （查 HR 通訊錄，參數化 SQL）
        └ chitchat  ─→ chitchat → END
"""

from __future__ import annotations

import logging
from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from sqlalchemy import or_, select
from sqlalchemy.orm import Session, sessionmaker

from app.agent import prompts
from app.db.models import Employee
from app.observability.metrics import AGENT_ROUTES, NO_ANSWER
from app.rag.store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

Route = Literal["knowledge", "directory", "chitchat"]


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict]  # [{"role": "user"|"assistant", "content": str}]
    space: str | None
    standalone_question: str
    route: Route
    documents: list[RetrievedChunk]
    answer: str


def _text(msg) -> str:
    content = msg.content
    if isinstance(content, list):  # 部分模型回傳 content blocks
        content = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return content.strip()


def parse_route(raw: str) -> Route:
    raw = raw.lower()
    for route in ("directory", "chitchat", "knowledge"):
        if route in raw:
            return route  # type: ignore[return-value]
    return "knowledge"  # 判斷不出來時，查知識庫是最安全的選擇


def format_context(docs: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{i}] {d.content}" for i, d in enumerate(docs, 1))


def build_graph(
    llm: BaseChatModel,
    store: VectorStore,
    session_factory: sessionmaker[Session],
    top_k: int = 5,
    min_score: float = 0.35,
    min_keyword: float = 0.3,
):
    def condense(state: AgentState, config: RunnableConfig) -> AgentState:
        history = state.get("history") or []
        if not history:
            return {"standalone_question": state["question"]}
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        msg = llm.invoke(
            [
                SystemMessage(prompts.CONDENSE_SYSTEM),
                HumanMessage(f"對話紀錄：\n{transcript}\n\n最新問題：{state['question']}"),
            ],
            config,
        )
        return {"standalone_question": _text(msg) or state["question"]}

    def route(state: AgentState, config: RunnableConfig) -> AgentState:
        msg = llm.invoke(
            [SystemMessage(prompts.ROUTER_SYSTEM), HumanMessage(state["standalone_question"])],
            config,
        )
        chosen = parse_route(_text(msg))
        AGENT_ROUTES.labels(chosen).inc()
        return {"route": chosen}

    def retrieve(state: AgentState) -> AgentState:
        hits = store.search(state["standalone_question"], k=top_k, space=state.get("space"))
        relevant = [h for h in hits if h.is_relevant(min_score, min_keyword)]
        logger.info(
            "retrieved",
            extra={
                "hits": len(hits),
                "relevant": len(relevant),
                "top_score": round(max(h.score for h in hits), 3) if hits else None,
            },
        )
        return {"documents": relevant}

    def answer(state: AgentState, config: RunnableConfig) -> AgentState:
        docs = state.get("documents") or []
        if not docs:
            # 沒有夠相關的文件就不呼叫 LLM：避免幻覺，也省 token
            NO_ANSWER.inc()
            return {"answer": prompts.NO_ANSWER}
        msg = llm.invoke(
            [
                SystemMessage(prompts.ANSWER_SYSTEM),
                HumanMessage(
                    prompts.ANSWER_USER.format(
                        context=format_context(docs), question=state["standalone_question"]
                    )
                ),
            ],
            config,
        )
        return {"answer": _text(msg)}

    def directory_lookup(state: AgentState, config: RunnableConfig) -> AgentState:
        keyword = _text(
            llm.invoke(
                [
                    SystemMessage(prompts.DIRECTORY_EXTRACT_SYSTEM),
                    HumanMessage(state["standalone_question"]),
                ],
                config,
            )
        )[:50]
        # 刻意不用 text-to-SQL：LLM 只負責抽關鍵字，查詢是固定的參數化 SQL，
        # 不會有 SQL injection，也不會被誘導去讀不該讀的資料表。
        pattern = f"%{keyword}%"
        with session_factory() as session:
            rows = session.scalars(
                select(Employee)
                .where(
                    or_(
                        Employee.name.ilike(pattern),
                        Employee.english_name.ilike(pattern),
                        Employee.department.ilike(pattern),
                        Employee.responsibilities.ilike(pattern),
                    )
                )
                .limit(5)
            ).all()
        if not rows:
            return {"answer": f"通訊錄中找不到和「{keyword}」相關的人員。"}
        table = "\n".join(
            f"- {e.name}（{e.english_name}）｜{e.department}｜{e.title}｜{e.email}"
            f"｜分機 {e.extension}｜負責：{e.responsibilities}"
            for e in rows
        )
        msg = llm.invoke(
            [
                SystemMessage(prompts.DIRECTORY_ANSWER_SYSTEM),
                HumanMessage(
                    f"【通訊錄查詢結果】\n{table}\n\n【問題】\n{state['standalone_question']}"
                ),
            ],
            config,
        )
        return {"answer": _text(msg)}

    def chitchat(state: AgentState, config: RunnableConfig) -> AgentState:
        msg = llm.invoke(
            [SystemMessage(prompts.CHITCHAT_SYSTEM), HumanMessage(state["question"])], config
        )
        return {"answer": _text(msg)}

    graph = StateGraph(AgentState)
    graph.add_node("condense", condense)
    graph.add_node("route", route)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_node("directory_lookup", directory_lookup)
    graph.add_node("chitchat", chitchat)

    graph.add_edge(START, "condense")
    graph.add_edge("condense", "route")
    graph.add_conditional_edges(
        "route",
        lambda s: s["route"],
        {"knowledge": "retrieve", "directory": "directory_lookup", "chitchat": "chitchat"},
    )
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    graph.add_edge("directory_lookup", END)
    graph.add_edge("chitchat", END)
    return graph.compile()
