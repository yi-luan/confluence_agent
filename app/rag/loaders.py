"""資料來源：Confluence REST API 與本機 Markdown 目錄。"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class SourceDocument:
    id: str
    source: str
    space: str
    title: str
    content: str
    url: str | None = None
    version: int | None = None
    metadata: dict = field(default_factory=dict)


def html_to_text(html: str) -> str:
    """Confluence storage format（XHTML）轉純文字，保留段落與表格列的換行。"""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for row in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        row.replace_with(" | ".join(cells) + "\n")
    text = soup.get_text("\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def load_markdown_dir(path: str | Path, space: str) -> list[SourceDocument]:
    root = Path(path)
    docs: list[SourceDocument] = []
    for file in sorted(root.rglob("*.md")):
        content = file.read_text(encoding="utf-8")
        title = file.stem
        for line in content.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        rel = file.relative_to(root).as_posix()
        docs.append(
            SourceDocument(
                id=f"markdown:{space}/{rel}",
                source="markdown",
                space=space,
                title=title,
                content=content,
                url=rel,
            )
        )
    return docs


class ConfluenceClient:
    """Confluence REST API client。

    - Confluence Cloud：user=email、token=API token（Basic auth）
    - Server / Data Center：只給 token（Personal Access Token，Bearer auth）
    """

    page_size = 50

    def __init__(self, base_url: str, token: str, user: str | None = None, timeout: float = 30):
        headers = {"Accept": "application/json"}
        auth = None
        if user:
            auth = (user, token)
        else:
            headers["Authorization"] = f"Bearer {token}"
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url, headers=headers, auth=auth, timeout=timeout
        )

    def iter_pages(self, space: str) -> Iterator[SourceDocument]:
        start = 0
        while True:
            resp = self._client.get(
                "/rest/api/content",
                params={
                    "spaceKey": space,
                    "type": "page",
                    "status": "current",
                    "expand": "body.storage,version",
                    "start": start,
                    "limit": self.page_size,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            for page in data.get("results", []):
                webui = page.get("_links", {}).get("webui", "")
                yield SourceDocument(
                    id=f"confluence:{page['id']}",
                    source="confluence",
                    space=space,
                    title=page["title"],
                    content=html_to_text(page["body"]["storage"]["value"]),
                    url=f"{self.base_url}{webui}" if webui else None,
                    version=page.get("version", {}).get("number"),
                )
            if data.get("size", 0) < self.page_size:
                break
            start += self.page_size

    def close(self) -> None:
        self._client.close()
