
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from semanticsearch.lexical.search import LexicalSearcher
from semanticsearch.semantic.config import SemanticConfig
from semanticsearch.semantic.searcher import SemanticSearcher


def ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def print_result(result) -> None:
    print(f"query: {result.query!r}")
    print(
        f"results: {result.total}  "
        f"({result.elapsed_ms:.1f} ms total; "
        f"{result.embedding_ms:.1f} ms embedding; "
        f"{result.retrieval_ms:.1f} ms retrieval)"
    )

    if not result.hits:
        print("\nNo results.")
        return

    print("\n=== SEMANTIC MATCHES ===")

    for i, hit in enumerate(result.hits, 1):
        speakers = " / ".join(hit.speakers)
        who = f", said by {hit.match_speaker}" if hit.match_speaker else ""

        print(f"\n#{i:<3} {hit.episode_title}")
        print(
            f"     score {hit.score:.4f}  "
            f"seek {ts(hit.seek_s)}  "
            f"(passage {ts(hit.start_s)}–{ts(hit.end_s)}{who})"
        )

        if speakers:
            print(f"     speakers: {speakers}")

        print("     " + hit.snippet)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Semantic search over podcast transcripts."
    )

    parser.add_argument("query")
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="number of semantic results (default: 20)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="ONNX intra-op CPU threads (default: 8)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="maximum query token length (default: 512)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="output results as JSON",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")

    if args.threads < 1:
        parser.error("--threads must be positive.")

    if args.max_seq_length < 1:
        parser.error("--max-seq-length must be positive.")

    config = SemanticConfig(
        intra_op_threads=args.threads,
        max_seq_length=args.max_seq_length,
    )

    # The lexical searcher owns the SQLite connection.
    # SemanticSearcher uses it for result hydration.
    with LexicalSearcher(args.db) as lexical:
        with SemanticSearcher(lexical, config=config) as semantic:
            result = semantic.search(args.query, limit=args.limit)

            if args.json:
                payload = {
                    "query": result.query,
                    "total": result.total,
                    "elapsed_ms": result.elapsed_ms,
                    "embedding_ms": result.embedding_ms,
                    "retrieval_ms": result.retrieval_ms,
                    "hits": [asdict(hit) for hit in result.hits],
                }
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print_result(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
