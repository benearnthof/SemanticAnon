
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from semanticsearch.lexical.search import LexicalSearcher
from semanticsearch.semantic.config import SemanticConfig
from semanticsearch.hybrid.config import HybridConfig
from semanticsearch.hybrid.searcher import HybridSearcher


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive hybrid podcast search."
    )

    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--lexical-limit", type=int, default=100)
    parser.add_argument("--semantic-limit", type=int, default=100)

    args = parser.parse_args()

    semantic_config = SemanticConfig(
        intra_op_threads=args.threads,
        max_seq_length=args.max_seq_length,
    )

    hybrid_config = HybridConfig(
        lexical_limit=args.lexical_limit,
        semantic_limit=args.semantic_limit,
        default_limit=args.limit,
    )

    print("Opening lexical index...", flush=True)

    with LexicalSearcher(args.db) as lexical:
        print("Loading semantic model...", flush=True)

        with HybridSearcher(
            lexical,
            config=hybrid_config,
            semantic_config=semantic_config,
        ) as hybrid:

            print("Model loaded. Ready for queries.")
            print("Enter a search query, or type 'exit' to quit.\n")

            while True:
                try:
                    query = input("search> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting.")
                    break

                if query.lower() in {"exit", "quit"}:
                    break

                if not query:
                    continue

                try:
                    result = hybrid.search(query, limit=args.limit)

                    print(
                        f"\n{result.total} results | "
                        f"{result.elapsed_ms:.1f} ms total | "
                        f"{result.lexical_ms:.1f} ms lexical | "
                        f"{result.semantic_ms:.1f} ms semantic"
                    )

                    print(
                        f"Candidates: "
                        f"{result.lexical_candidates} lexical / "
                        f"{result.semantic_candidates} semantic"
                    )

                    for i, item in enumerate(result.hits, 1):
                        hit = item.hit
                        print(f"\n#{i} {hit.episode_title}")
                        print(
                            f"   {hit.start_s:.1f}s–{hit.end_s:.1f}s "
                            f"| RRF {item.score:.6f}"
                        )
                        print(f"   {hit.text}")

                    print()

                except Exception as exc:
                    print(f"Search failed: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
