def ask(client, question, **kw):
    r = client.post("/api/v1/ask", json={"question": question, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_knowledge_question_returns_cited_answer_with_sources(client):
    body = ask(client, "特休要提前幾天申請？")
    assert body["route"] == "knowledge"
    assert body["answer"].endswith("[1]")
    assert body["sources"][0]["title"] == "請假規定"
    assert body["usage"]["llm_calls"] == 2  # route + answer（第一輪不需要改寫問題）
    assert body["usage"]["prompt_tokens"] > 0


def test_unrelated_question_is_refused_without_calling_llm(client, llm):
    body = ask(client, "量子糾纏的數學推導")
    assert body["route"] == "knowledge"
    assert body["sources"] == []
    assert "找不到" in body["answer"]
    assert body["usage"]["llm_calls"] == 1  # 只有路由，沒有生成


def test_multi_turn_question_is_condensed(client):
    first = ask(client, "特休要提前幾天申請？")
    second = ask(client, "那病假呢？", conversation_id=first["conversation_id"])
    assert second["conversation_id"] == first["conversation_id"]
    assert "病假" in second["standalone_question"]
    detail = client.get(f"/api/v1/conversations/{first['conversation_id']}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"] * 2


def test_directory_route_queries_hr_table(client):
    body = ask(client, "VPN 有問題要找誰？")
    assert body["route"] == "directory"
    assert "陳冠宇" in body["answer"] and "1401" in body["answer"]


def test_directory_lookup_is_safe_against_sql_injection(client):
    body = ask(client, "誰是 '; DROP TABLE employees; --")
    assert body["route"] == "directory"
    assert ask(client, "VPN 要找誰")["answer"].count("陳冠宇") >= 1  # 資料表還在


def test_chitchat(client):
    assert ask(client, "你好")["route"] == "chitchat"


def test_unknown_conversation_returns_404(client):
    r = client.post("/api/v1/ask", json={"question": "hi", "conversation_id": "nope"})
    assert r.status_code == 404


def test_ingest_requires_api_key(client):
    payload = {"source": "markdown", "path": "sample_data/handbook"}
    assert client.post("/api/v1/ingest", json=payload).status_code == 401
    r = client.post("/api/v1/ingest", json=payload, headers={"X-API-Key": "test-key"})
    assert r.status_code == 200 and r.json()["unchanged"] == 7


def test_stats_and_metrics(client):
    ask(client, "特休要提前幾天申請？")
    ask(client, "VPN 有問題要找誰？")
    stats = client.get("/api/v1/stats", headers={"X-API-Key": "test-key"}).json()
    assert stats["answers"] == 2
    assert stats["by_route"] == {"knowledge": 1, "directory": 1}
    assert stats["prompt_tokens"] > 0 and stats["latency_p95_ms"] is not None

    metrics = client.get("/metrics").text
    assert 'kb_llm_tokens_total{model="offline-scripted",type="prompt"}' in metrics
    assert 'kb_http_request_duration_seconds_count{method="POST",path="/api/v1/ask"}' in metrics
    assert "kb_agent_route_total" in metrics


def test_health_and_request_id(client):
    r = client.get("/health", headers={"x-request-id": "abc123"})
    assert r.json()["status"] == "ok"
    assert r.headers["x-request-id"] == "abc123"
