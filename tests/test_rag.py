from app.cli import SAMPLE_DIR
from app.rag.ingest import ingest_markdown
from app.rag.loaders import SourceDocument, html_to_text, load_markdown_dir
from app.rag.store import keyword_terms


def test_html_to_text_keeps_table_rows_and_paragraphs():
    html = (
        "<h1>標題</h1><p>第一段<br/>換行</p>"
        "<table><tr><th>等級</th><th>回應</th></tr><tr><td>P1</td><td>15 分鐘</td></tr></table>"
    )
    text = html_to_text(html)
    assert "標題" in text and "第一段\n換行" in text
    assert "P1 | 15 分鐘" in text


def test_markdown_loader_uses_h1_as_title():
    docs = load_markdown_dir(SAMPLE_DIR / "handbook", "HANDBOOK")
    titles = {d.title for d in docs}
    assert "請假規定" in titles and "正式環境部署 SOP" in titles
    assert all(d.id.startswith("markdown:HANDBOOK/") for d in docs)


def test_chunks_are_prefixed_with_document_title(store):
    doc = SourceDocument(
        id="t:1",
        source="markdown",
        space="X",
        title="測試文件",
        content="段落一。\n\n" + "內容" * 500,
    )
    parts = store.split(doc)
    assert len(parts) > 1
    assert all(p.startswith("【測試文件】") for p in parts)


def test_ingest_is_incremental(store):
    first = ingest_markdown(store, str(SAMPLE_DIR / "handbook"), "HANDBOOK")
    assert first.added == 7 and first.chunks > 7
    second = ingest_markdown(store, str(SAMPLE_DIR / "handbook"), "HANDBOOK")
    assert second.added == 0 and second.unchanged == 7 and second.chunks == 0


def test_changed_document_is_reindexed(store):
    doc = SourceDocument(id="t:2", source="markdown", space="X", title="A", content="舊內容")
    store.ingest([doc])
    doc.content = "新內容：VPN 需要 MFA"
    stats = store.ingest([doc])
    assert stats.updated == 1
    hits = store.search("VPN MFA", k=1)
    assert "新內容" in hits[0].content


def test_search_finds_relevant_document_and_filters_space(seeded):
    hits = seeded.search("特休要提前幾天申請", k=3)
    assert hits[0].title == "請假規定"
    assert hits[0].score > hits[-1].score or len(hits) == 1
    assert seeded.search("特休", k=3, space="OTHER") == []


def test_keyword_terms_split_chinese_bigrams_and_drop_question_words():
    terms = keyword_terms("VPN 換新手機要多久？")
    assert "vpn" in terms and "手機" in terms
    assert "多久" not in terms


def test_hybrid_search_ranks_exact_term_matches_first(seeded):
    question = "可以把客戶資料貼到 ChatGPT 嗎？"
    hybrid = seeded.search(question, k=3)
    assert hybrid[0].title == "資訊安全政策"
    assert hybrid[0].keyword > 0
