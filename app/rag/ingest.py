"""匯入流程：API 和 CLI 共用。"""

from __future__ import annotations

from app.config import Settings
from app.rag.loaders import ConfluenceClient, load_markdown_dir
from app.rag.store import IngestStats, VectorStore


def ingest_markdown(store: VectorStore, path: str, space: str) -> IngestStats:
    return store.ingest(load_markdown_dir(path, space))


def ingest_confluence(
    store: VectorStore, settings: Settings, spaces: list[str] | None = None
) -> IngestStats:
    if not (settings.confluence_url and settings.confluence_token):
        raise ValueError("尚未設定 CONFLUENCE_URL / CONFLUENCE_TOKEN")
    spaces = spaces or settings.confluence_space_list
    if not spaces:
        raise ValueError("請指定至少一個 Confluence space")
    client = ConfluenceClient(
        settings.confluence_url, settings.confluence_token, settings.confluence_user
    )
    total = IngestStats()
    try:
        for space in spaces:
            stats = store.ingest(list(client.iter_pages(space)))
            for k in vars(total):
                setattr(total, k, getattr(total, k) + getattr(stats, k))
    finally:
        client.close()
    return total
