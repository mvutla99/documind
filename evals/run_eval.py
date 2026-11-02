"""Measure retrieval quality on a labeled question set.

    python -m evals.run_eval                     # offline demo embeddings
    python -m evals.run_eval --provider gemini   # needs GEMINI_API_KEY
    python -m evals.run_eval --answers           # also print generated answers

Metrics
- Hit@k: share of questions where a chunk from the expected file is in the top k.
- MRR (mean reciprocal rank): 1/rank of the first correct chunk, averaged. 1.0 = always ranked first.
Use these numbers to compare chunk sizes, embedding models and MIN_SCORE values.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rag import RAGService  # noqa: E402
from app.schemas import AskRequest  # noqa: E402
from app.vectorstore import VectorStore  # noqa: E402


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=["demo", "gemini"], default="demo")
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
    hits, rr = 0, 0.0
    start = time.perf_counter()
    print(f"{'':2} {'rank':>4}  question")
    for case in cases:
        results = svc.retrieve(case["question"], top_k=args.k)
        rank = next((i for i, (c, _) in enumerate(results, 1) if c.filename == case["expected_file"]), None)
        hits += rank is not None
        rr += 1 / rank if rank else 0
        print(f"{'✓' if rank else '✗':2} {rank or '-':>4}  {case['question']}")
        if args.answers:
            answer = svc.ask(AskRequest(question=case["question"], top_k=args.k)).answer
            print(f"{'':9}expected: {case['expected_answer']}\n{'':9}got:      {answer}\n")

    n = len(cases)
    hit_rate = hits / n
    print(f"\nprovider={args.provider} chunk_size={args.chunk_size} overlap={args.overlap} k={args.k}")
    print(f"Hit@{args.k}: {hit_rate:.0%} ({hits}/{n})   MRR: {rr / n:.3f}   time: {time.perf_counter() - start:.1f}s")
    if hit_rate < args.min_hit_rate:
        sys.exit(f"Hit rate {hit_rate:.0%} is below the required {args.min_hit_rate:.0%}")


if __name__ == "__main__":
    main()
