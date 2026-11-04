"""RAG 評估腳本。

兩種模式：
  python -m eval.run_eval --retrieval-only   只評估檢索（不呼叫 LLM；CI 的品質門檻）
  python -m eval.run_eval                    完整評估：路由、回答關鍵字、拒答、延遲與 token

指標說明：
  hit@k        正確文件有沒有出現在前 k 名
  MRR          正確文件排第幾名的倒數平均（第 1 名=1、第 2 名=0.5…），越高越好
  route_acc    Agent 判斷問題類型的正確率
  keyword      回答裡有沒有包含標準答案的關鍵字（例如「3 個工作天」的「3」）
  refusal      知識庫沒有的問題，是否正確回答「找不到」而不是亂編
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from app.agent import prompts
from app.config import get_settings
from app.main import build_container

HERE = Path(__file__).parent


def load_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in (HERE / "dataset.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]


def eval_retrieval(
    store, cases: list[dict], k: int, min_score: float, min_keyword: float, hybrid: bool = True
) -> dict:
    hits, rr, refusals, refusal_total = 0, 0.0, 0, 0
    rows = []
    for case in cases:
        if case["route"] != "knowledge":
            continue
        found = store.search(case["question"], k=k, hybrid=hybrid)
        results = [r for r in found if r.is_relevant(min_score, min_keyword)]
        titles = [r.title for r in results]
        expected = case.get("expected_title")
        if expected is None:
            refusal_total += 1
            refusals += not results
            rows.append((case["question"], "（應拒答）", "✓" if not results else f"✗ {titles[:1]}"))
            continue
        rank = titles.index(expected) + 1 if expected in titles else 0
        hits += rank > 0
        rr += 1 / rank if rank else 0
        rows.append((case["question"], expected, f"#{rank}" if rank else "✗"))
    answerable = sum(1 for c in cases if c["route"] == "knowledge" and c.get("expected_title"))
    return {
        "hit@k": hits / answerable,
        "mrr": rr / answerable,
        "refusal": refusals / refusal_total if refusal_total else None,
        "rows": rows,
    }


def eval_end_to_end(chat, cases: list[dict]) -> dict:
    route_ok, kw_ok, kw_total, refusal_ok, refusal_total = 0, 0, 0, 0, 0
    latencies, tokens, rows = [], [], []
    for case in cases:
        t0 = time.perf_counter()
        r = chat.ask(case["question"])
        latencies.append((time.perf_counter() - t0) * 1000)
        tokens.append(r.prompt_tokens + r.completion_tokens)
        route_ok += r.route == case["route"]
        verdict = "✓"
        if case.get("must_include"):
            kw_total += 1
            ok = all(k in r.answer for k in case["must_include"])
            kw_ok += ok
            verdict = "✓" if ok else "✗ 缺關鍵字"
        if case["route"] == "knowledge" and case.get("expected_title") is None:
            refusal_total += 1
            ok = r.answer == prompts.NO_ANSWER or "沒有足夠的資訊" in r.answer
            refusal_ok += ok
            verdict = "✓ 正確拒答" if ok else "✗ 沒有拒答"
        if r.route != case["route"]:
            verdict = f"✗ 路由 {r.route}"
        rows.append((case["question"], r.route, verdict, r.answer[:60].replace("\n", " ")))
    n = len(cases)
    return {
        "route_acc": route_ok / n,
        "keyword": kw_ok / kw_total if kw_total else None,
        "refusal": refusal_ok / refusal_total if refusal_total else None,
        "latency_p50_ms": round(statistics.median(latencies)),
        "latency_p95_ms": round(sorted(latencies)[int(0.95 * (n - 1))]),
        "avg_tokens": round(statistics.mean(tokens)),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--vector-only", action="store_true", help="關掉關鍵字檢索，做對照組")
    ap.add_argument(
        "--min-hit", type=float, default=0.0, help="hit@k 低於此值時回傳失敗（CI 門檻）"
    )
    ap.add_argument("--out", type=Path, default=HERE / "results" / "latest.json")
    args = ap.parse_args()

    settings = get_settings()
    cases = load_cases()
    if args.retrieval_only:
        from app.db.session import get_session_factory
        from app.llm.providers import get_embeddings
        from app.rag.store import VectorStore

        store = VectorStore(get_session_factory(), get_embeddings(settings))
        result = eval_retrieval(
            store,
            cases,
            args.k,
            settings.retrieval_min_score,
            settings.retrieval_min_keyword,
            hybrid=not args.vector_only,
        )
        mode = "vector" if args.vector_only else "hybrid"
        print(f"\n檢索評估（{mode}, embedding={settings.embedding_provider}, k={args.k}）")
        for q, exp, v in result["rows"]:
            print(f"  {v:>6}  {q}  →  {exp}")
        refusal = "-" if result["refusal"] is None else f"{result['refusal']:.0%}"
        print(
            f"\nhit@{args.k} = {result['hit@k']:.0%}   MRR = {result['mrr']:.2f}   拒答 = {refusal}"
        )
    else:
        container = build_container(settings)
        result = eval_end_to_end(container.chat, cases)
        print(f"\n端到端評估（llm={settings.llm_provider}:{container.model_name}）")
        for q, route, v, a in result["rows"]:
            print(f"  {v:<10} [{route}] {q}\n             {a}")
        print(
            f"\n路由正確率 {result['route_acc']:.0%}   關鍵字命中 {result['keyword']:.0%}   "
            f"拒答正確 {result['refusal']:.0%}   p50 {result['latency_p50_ms']}ms   "
            f"p95 {result['latency_p95_ms']}ms   平均 {result['avg_tokens']} tokens/題"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.retrieval_only and result["hit@k"] < args.min_hit:
        print(f"✗ hit@k {result['hit@k']:.0%} 低於門檻 {args.min_hit:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
