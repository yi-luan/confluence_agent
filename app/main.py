"""FastAPI 進入點：組裝相依物件、掛上中介層與 /metrics。"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from app.agent.graph import build_graph
from app.api.routes import health_router, router
from app.chat_service import ChatService
from app.config import Settings, get_settings
from app.db.session import get_engine, get_session_factory, init_db
from app.llm.providers import get_chat_model, get_embeddings
from app.observability.logging import request_id_var, setup_logging
from app.observability.metrics import HTTP_LATENCY, HTTP_REQUESTS
from app.rag.store import VectorStore

logger = logging.getLogger("app")


@dataclass
class Container:
    settings: Settings
    session_factory: object
    store: VectorStore
    chat: ChatService
    model_name: str


def model_name_of(llm: BaseChatModel, settings: Settings) -> str:
    return getattr(llm, "model_name", None) or getattr(llm, "model", None) or settings.llm_model


def build_container(
    settings: Settings, llm: BaseChatModel | None = None, embeddings: Embeddings | None = None
) -> Container:
    """測試時可以注入假的 llm / embeddings，其餘照正式流程組裝。"""
    llm = llm or get_chat_model(settings)
    embeddings = embeddings or get_embeddings(settings)
    session_factory = get_session_factory()
    store = VectorStore(session_factory, embeddings, settings.chunk_size, settings.chunk_overlap)
    graph = build_graph(
        llm,
        store,
        session_factory,
        settings.retrieval_top_k,
        settings.retrieval_min_score,
        settings.retrieval_min_keyword,
    )
    name = model_name_of(llm, settings)
    chat = ChatService(graph, session_factory, name, settings.history_turns)
    return Container(settings, session_factory, store, chat, name)


def create_app(container: Container | None = None) -> FastAPI:
    settings = container.settings if container else get_settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(get_engine())
        if getattr(app.state, "container", None) is None:
            app.state.container = build_container(settings)
        logger.info(
            "startup",
            extra={"llm_provider": settings.llm_provider, "model": app.state.container.model_name},
        )
        yield

    app = FastAPI(
        title="Enterprise Knowledge Agent",
        description="Confluence / Markdown 企業知識庫問答 Agent（RAG + LangGraph + pgvector）",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.container = container
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = rid
            return response
        finally:
            elapsed = time.perf_counter() - started
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")  # 用路由樣板當 label，避免 label 爆量
            HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
            HTTP_LATENCY.labels(request.method, path).observe(elapsed)
            if path != "/metrics":
                logger.info(
                    "request",
                    extra={
                        "method": request.method,
                        "path": path,
                        "status": status,
                        "latency_ms": round(elapsed * 1000),
                    },
                )
            request_id_var.reset(token)

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    app.include_router(router)
    app.include_router(health_router)
    return app
