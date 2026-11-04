"""測試共用設定：使用真的 PostgreSQL + pgvector（CI 用 service container），
LLM 與 embedding 換成不連網的假模型。"""

import os

os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/kb_test"
)
os.environ["EMBEDDING_PROVIDER"] = "fake"
os.environ["EMBEDDING_DIM"] = "256"
os.environ["ADMIN_API_KEY"] = "test-key"
os.environ["RETRIEVAL_MIN_SCORE"] = "0.2"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.cli import SAMPLE_DIR, seed_employees  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db.models import Base  # noqa: E402
from app.db.session import get_engine, get_session_factory, init_db  # noqa: E402
from app.llm.offline import ScriptedChatModel  # noqa: E402
from app.llm.providers import HashingEmbeddings  # noqa: E402
from app.main import build_container, create_app  # noqa: E402
from app.rag.ingest import ingest_markdown  # noqa: E402
from app.rag.store import VectorStore  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def database():
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.drop_all(engine)
    init_db(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture(autouse=True)
def clean_tables(database):
    yield
    with database.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE chunks, documents, messages, conversations, employees "
                "RESTART IDENTITY CASCADE"
            )
        )


@pytest.fixture
def store() -> VectorStore:
    return VectorStore(get_session_factory(), HashingEmbeddings(256), 250, 50)


@pytest.fixture
def seeded(store):
    seed_employees()
    ingest_markdown(store, str(SAMPLE_DIR / "handbook"), "HANDBOOK")
    return store


@pytest.fixture
def llm() -> ScriptedChatModel:
    return ScriptedChatModel()


@pytest.fixture
def client(seeded, llm):
    container = build_container(get_settings(), llm=llm, embeddings=HashingEmbeddings(256))
    with TestClient(create_app(container)) as c:
        yield c
