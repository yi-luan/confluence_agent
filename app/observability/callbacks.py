"""LangChain callback：攔截每一次 LLM 呼叫，記錄延遲與 token 用量。

一個請求可能呼叫 LLM 好幾次（改寫問題 → 路由 → 回答），
UsageTracker 會把它們加總，最後寫進 messages 資料表並送到 Prometheus。
"""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.observability.metrics import LLM_CALLS, LLM_LATENCY, LLM_TOKENS

logger = logging.getLogger("llm")


class UsageTracker(BaseCallbackHandler):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self._started: dict[UUID, float] = {}

    def on_chat_model_start(self, serialized: Any, messages: Any, *, run_id: UUID, **kw: Any):
        self._started[run_id] = time.perf_counter()

    def on_llm_start(self, serialized: Any, prompts: Any, *, run_id: UUID, **kw: Any):
        self._started[run_id] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kw: Any):
        elapsed = time.perf_counter() - self._started.pop(run_id, time.perf_counter())
        prompt, completion = 0, 0
        for generations in response.generations:
            for gen in generations:
                usage = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                prompt += usage.get("input_tokens", 0)
                completion += usage.get("output_tokens", 0)
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1

        LLM_CALLS.labels(self.model_name, "ok").inc()
        LLM_LATENCY.labels(self.model_name).observe(elapsed)
        LLM_TOKENS.labels(self.model_name, "prompt").inc(prompt)
        LLM_TOKENS.labels(self.model_name, "completion").inc(completion)
        logger.info(
            "llm call",
            extra={
                "model": self.model_name,
                "latency_ms": round(elapsed * 1000),
                "prompt_tokens": prompt,
                "completion_tokens": completion,
            },
        )

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kw: Any):
        self._started.pop(run_id, None)
        LLM_CALLS.labels(self.model_name, "error").inc()
        logger.warning("llm call failed", extra={"model": self.model_name, "error": str(error)})
