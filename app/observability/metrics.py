"""Prometheus 指標：HTTP、LLM 呼叫、Token 用量、檢索品質。"""

from prometheus_client import Counter, Histogram

HTTP_REQUESTS = Counter("kb_http_requests_total", "HTTP 請求數", ["method", "path", "status"])
HTTP_LATENCY = Histogram(
    "kb_http_request_duration_seconds",
    "HTTP 請求延遲",
    ["method", "path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 60),
)
LLM_CALLS = Counter("kb_llm_calls_total", "LLM 呼叫次數", ["model", "status"])
LLM_LATENCY = Histogram(
    "kb_llm_call_duration_seconds",
    "單次 LLM 呼叫延遲",
    ["model"],
    buckets=(0.25, 0.5, 1, 2, 4, 8, 15, 30, 60),
)
LLM_TOKENS = Counter("kb_llm_tokens_total", "LLM token 用量", ["model", "type"])
AGENT_ROUTES = Counter("kb_agent_route_total", "Agent 路由分佈", ["route"])
RETRIEVAL_TOP_SCORE = Histogram(
    "kb_retrieval_top_score",
    "檢索第一名的相似度（越低代表越可能答不出來）",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)
NO_ANSWER = Counter("kb_no_answer_total", "因為找不到相關文件而拒答的次數")
