"""Measure retrieval quality on a labeled question set.

    python -m evals.run_eval                       # offline demo embeddings, hybrid retrieval
    python -m evals.run_eval --mode all            # compare vector vs bm25 vs hybrid side by side
    python -m evals.run_eval --provider gemini     # needs GEMINI_API_KEY
    python -m evals.run_eval --answers             # also print generated answers

Metrics
- Hit@k: share of questions where a chunk from the expected file is in the top k.
- MRR (mean reciprocal rank): 1/rank of the first correct chunk, averaged. 1.0 = always ranked first.
Each question has a "type": "semantic" (paraphrased) or "exact_term" (names, codes, numbers),
and results are broken down by type, since that is where vector and keyword search differ.
Use these numbers to compare retrieval modes, chunk sizes, embedding models and MIN_SCORE values.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rag import RAGService  # noqa: E402
from app.schemas import AskRequest  # noqa: E402
from app.vectorstore import VectorStore  # noqa: E402

MODES = ("vector", "bm25", "hybrid")


def build(provider: str, chunk_size: int, overlap: int) -> RAGService:
    if provider == "gemini":
        from app.providers.gemini import GeminiEmbedder, GeminiLLM

        key = os.environ.get("GEMINI_API_KEY") or sys.exit("Set GEMINI_API_KEY first")
        embedder, llm = GeminiEmbedder(key), GeminiLLM(key)
    else:
        from app.providers.demo import ExtractiveLLM, HashingEmbedder

        embedder, llm = HashingEmbedder(), ExtractiveLLM()
    store = VectorStore(Path(tempfile.mkdtemp(prefix="documind-eval-")))
    return RAGService(embedder, llm, store, chunk_size, overlap, min_score=0.0)


def evaluate(svc: RAGService, cases: list[dict], k: int, mode: str) -> tuple[list[int | None], dict[str, dict]]:
    """Return the rank of the expected file for each case, and Hit@k/MRR per question type (plus "all")."""
    ranks: list[int | None] = []
    groups: dict[str, list[int | None]] = defaultdict(list)
    for case in cases:
        results = svc.retrieve(case["question"], top_k=k, mode=mode)
        rank = next((i for i, (c, _) in enumerate(results, 1) if c.filename == case["expected_file"]), None)
        ranks.append(rank)
        groups[case.get("type", "semantic")].append(rank)
        groups["all"].append(rank)
    metrics = {
        name: {
            "n": len(rs),
            "hit": sum(r is not None for r in rs) / len(rs),
            "mrr": sum(1 / r for r in rs if r) / len(rs),
        }
        for name, rs in groups.items()
    }
    return ranks, metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=["demo", "gemini"], default="demo")
    p.add_argument("--mode", choices=[*MODES, "all"], default="hybrid", help="retrieval mode, or 'all' to compare")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--overlap", type=int, default=150)
    p.add_argument("--answers", action="store_true", help="also generate and print answers")
    p.add_argument("--min-hit-rate", type=float, default=0.0, help="exit 1 if Hit@k is below this (for CI)")
    args = p.parse_args()

    svc = build(args.provider, args.chunk_size, args.overlap)
    for path in sorted((ROOT / "sample_docs").glob("*")):
        svc.ingest(path.name, path.read_bytes())
    cases = json.loads((ROOT / "evals" / "eval_set.json").read_text())

    modes = MODES if args.mode == "all" else (args.mode,)
    start = time.perf_counter()
    all_ranks, all_metrics = {}, {}
    for mode in modes:
        all_ranks[mode], all_metrics[mode] = evaluate(svc, cases, args.k, mode)

    # Per-question table: one rank column per mode ("-" = not in top k).
    print(" ".join(f"{m:>6}" for m in modes) + "  type        question")
    for i, case in enumerate(cases):
        cols = " ".join(f"{all_ranks[m][i] or '-':>6}" for m in modes)
        print(f"{cols}  {case.get('type', 'semantic'):<11} {case['question']}")
        if args.answers:
            answer = svc.ask(AskRequest(question=case["question"], top_k=args.k, retrieval_mode=modes[-1])).answer
            print(f"{'':9}expected: {case['expected_answer']}\n{'':9}got:      {answer}\n")

    print(f"\nprovider={args.provider} chunk_size={args.chunk_size} overlap={args.overlap} k={args.k}")
    print(f"{'mode':<8} {'subset':<11} {'n':>3}  {'Hit@' + str(args.k):>6}  {'MRR':>6}")
    for mode in modes:
        for subset, m in sorted(all_metrics[mode].items(), key=lambda kv: kv[0] != "all"):
            print(f"{mode:<8} {subset:<11} {m['n']:>3}  {m['hit']:>6.0%}  {m['mrr']:>6.3f}")
    print(f"time: {time.perf_counter() - start:.1f}s")

    failing = [m for m in modes if all_metrics[m]["all"]["hit"] < args.min_hit_rate]
    if failing:
        sys.exit(f"Hit rate below the required {args.min_hit_rate:.0%} for: {', '.join(failing)}")


if __name__ == "__main__":
    main()
