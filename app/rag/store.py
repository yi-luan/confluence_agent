"""pgvector 向量儲存：寫入（增量更新）與混合檢索（向量 + 關鍵字）。

為什麼要混合檢索？
- 向量搜尋擅長「意思相近」：問「在家上班」也能找到「遠端工作」
- 關鍵字搜尋擅長「字面精確」：數字、代號、專有名詞（MFA、P1、Argo CD）
兩邊各取前幾名，用 RRF（Reciprocal Rank Fusion）合併排名，互補彼此的盲點。
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import Integer, case, delete, func, literal, select
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Chunk, Document, utcnow
from app.observability.metrics import RETRIEVAL_TOP_SCORE
from app.rag.loaders import SourceDocument

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    document_id: str
    title: str
    url: str | None
    space: str
    content: str
    score: float  # 向量 cosine similarity，1 = 完全相同
    keyword: float = 0.0  # 問題關鍵詞在片段中出現的比例（0~1）
    rrf: float = 0.0  # 合併後的排名分數，用來排序

    def is_relevant(self, min_score: float, min_keyword: float = 0.3) -> bool:
        """語意夠接近，或是問題裡大部分關鍵詞都出現在片段中，就視為相關。"""
        return self.score >= min_score or self.keyword >= min_keyword


# 問句常見、但對檢索沒有幫助的詞
_STOP_BIGRAMS = {
    "多少",
    "幾天",
    "什麼",
    "怎麼",
    "如何",
    "可以",
    "需要",
    "要多",
    "是什",
    "麼？",
    "嗎？",
    "請問",
    "之後",
    "辦？",
    "怎樣",
    "哪裡",
    "多久",
    "幾個",
    "一個",
    "我們",
    "公司",
}
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9.%,_-]*")


def keyword_terms(query: str) -> list[str]:
    """中文切成相鄰兩字（bigram）、英數字保留整個詞，去掉問句常見詞。"""
    terms: list[str] = [w.lower() for w in _WORD.findall(query) if len(w) >= 2]
    for run in _CJK_RUN.findall(query):
        terms += [run[i : i + 2] for i in range(len(run) - 1)]
    seen: dict[str, None] = {}
    for t in terms:
        if t not in _STOP_BIGRAMS:
            seen.setdefault(t)
    return list(seen)


@dataclass
class IngestStats:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    chunks: int = 0


class VectorStore:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        embeddings: Embeddings,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ):
        self.session_factory = session_factory
        self.embeddings = embeddings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )

    def split(self, doc: SourceDocument) -> list[str]:
        # 每個 chunk 前面加上文件標題，讓「片段」在檢索時仍保有上下文
        return [f"【{doc.title}】\n{part}" for part in self.splitter.split_text(doc.content)]

    def ingest(self, docs: list[SourceDocument]) -> IngestStats:
        """增量寫入：內容 hash 沒變的文件直接跳過，不重算 embedding（省錢也省時間）。"""
        stats = IngestStats()
        with self.session_factory() as session:
            for doc in docs:
                content_hash = hashlib.sha256(doc.content.encode()).hexdigest()
                existing = session.get(Document, doc.id)
                if existing and existing.content_hash == content_hash:
                    stats.unchanged += 1
                    continue

                parts = self.split(doc)
                vectors = self.embeddings.embed_documents(parts) if parts else []

                if existing:
                    session.execute(delete(Chunk).where(Chunk.document_id == doc.id))
                    record = existing
                    stats.updated += 1
                else:
                    record = Document(id=doc.id)
                    session.add(record)
                    stats.added += 1

                record.source = doc.source
                record.space = doc.space
                record.title = doc.title
                record.url = doc.url
                record.version = doc.version
                record.content_hash = content_hash
                record.updated_at = utcnow()
                session.flush()

                session.add_all(
                    Chunk(document_id=doc.id, chunk_index=i, content=text, embedding=vec)
                    for i, (text, vec) in enumerate(zip(parts, vectors, strict=True))
                )
                stats.chunks += len(parts)
            session.commit()
        logger.info("ingest finished", extra={"stats": stats.__dict__})
        return stats

    def search(
        self,
        query: str,
        k: int = 5,
        space: str | None = None,
        candidates: int = 20,
        rrf_k: int = 60,
        hybrid: bool = True,
    ) -> list[RetrievedChunk]:
        """混合檢索：向量前 N 名 + 關鍵字前 N 名，用 RRF 合併後取前 k 名。"""
        vector = self.embeddings.embed_query(query)
        terms = keyword_terms(query) if hybrid else []

        distance = Chunk.embedding.cosine_distance(vector)
        similarity = (1 - distance).label("score")
        if terms:
            # 參數化的 LIKE（不會有 SQL injection）。適合數萬個片段以內；
            # 資料量更大時可改用 pg_bigm（中日文 bigram 全文索引）。
            hits = sum(
                (
                    case((func.lower(Chunk.content).contains(t, autoescape=True), 1), else_=0)
                    for t in terms
                ),
                literal(0, Integer),
            )
            coverage = (hits * 1.0 / len(terms)).label("keyword")
        else:
            coverage = literal(0.0).label("keyword")

        base = select(
            Chunk.id,
            Chunk.content,
            Document.id.label("doc_id"),
            Document.title,
            Document.url,
            Document.space,
            similarity,
            coverage,
        ).join(Document, Chunk.document_id == Document.id)
        if space:
            base = base.where(Document.space == space)

        with self.session_factory() as session:
            by_vector = session.execute(base.order_by(distance).limit(candidates)).all()
            by_keyword = (
                session.execute(base.order_by(coverage.desc(), distance).limit(candidates)).all()
                if terms
                else []
            )

        fused: dict[int, RetrievedChunk] = {}
        for ranking in (by_vector, by_keyword):
            for rank, r in enumerate(ranking, 1):
                if ranking is by_keyword and r.keyword == 0:
                    break
                item = fused.get(r.id) or RetrievedChunk(
                    document_id=r.doc_id,
                    title=r.title,
                    url=r.url,
                    space=r.space,
                    content=r.content,
                    score=float(r.score),
                    keyword=float(r.keyword),
                )
                item.rrf += 1 / (rrf_k + rank)
                fused[r.id] = item
        results = sorted(fused.values(), key=lambda c: c.rrf, reverse=True)[:k]
        if results:
            RETRIEVAL_TOP_SCORE.observe(max(r.score for r in results))
        return results
