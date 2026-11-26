
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from semanticsearch.lexical.search import LexicalSearcher
from semanticsearch.semantic.config import SemanticConfig

from semanticsearch.hybrid.config import HybridConfig
from semanticsearch.hybrid.searcher import (
    HybridSearcher,
    HybridSearchResult,
)

from semanticsearch.hybrid.context import expand_context


def ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def print_result(result: HybridSearchResult) -> None:
    print(f"query: {result.query!r}")
    print(
        f"results: {result.total}  "
        f"({result.elapsed_ms:.1f} ms total; "
        f"{result.lexical_ms:.1f} ms lexical; "
        f"{result.semantic_ms:.1f} ms semantic)"
    )
    print(
        f"candidates: {result.lexical_candidates} lexical / "
        f"{result.semantic_candidates} semantic"
    )

    if result.total == 0:
        print("\nNo results.")

def print_passage_results(
    result: HybridSearchResult,
    lexical: LexicalSearcher,
    context: int = 0,
) -> None:
    print("\n=== HYBRID MATCHES (RRF) ===")

    for i, hybrid_hit in enumerate(result.hits, 1):
        hit = hybrid_hit.hit

        speakers = " / ".join(hit.speakers)
        who = (
            f", said by {hit.match_speaker}"
            if hit.match_speaker else ""
        )

        sources = " + ".join(hybrid_hit.sources)

        print(f"\n#{i:<3} {hit.episode_title}")

        print(
            f"     RRF {hybrid_hit.score:.6f}  "
            f"seek {ts(hit.seek_s)}  "
            f"(passage {ts(hit.start_s)}–{ts(hit.end_s)}{who})"
        )

        print(f"     sources: {sources}")

        ranks = []

        if hybrid_hit.lexical_rank is not None:
            ranks.append(f"lexical #{hybrid_hit.lexical_rank}")

        if hybrid_hit.semantic_rank is not None:
            ranks.append(f"semantic #{hybrid_hit.semantic_rank}")

        if ranks:
            print(f"     ranks: {' / '.join(ranks)}")

        if speakers:
            print(f"     speakers: {speakers}")

        # print("     " + hit.snippet)
        print("     " + hit.text)
        if context > 0:
            expanded = expand_context(
                hybrid_hit,
                lexical,
                before=context,
                after=context,
            )

            for neighbour in expanded.before:
                print(f"\n     [CONTEXT BEFORE] {ts(neighbour.start_s)}")
                print("     " + neighbour.snippet)

            for neighbour in expanded.after:
                print(f"\n     [CONTEXT AFTER] {ts(neighbour.start_s)}")
                print("     " + neighbour.snippet)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(
        description="Hybrid lexical and semantic podcast search."
    )

    parser.add_argument("query")
    parser.add_argument("--db", type=Path, default=None)

    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="number of final hybrid results (default: 20)",
    )

    parser.add_argument(
        "--lexical-limit",
        type=int,
        default=100,
        help="maximum lexical candidates (default: 100)",
    )

    parser.add_argument(
        "--semantic-limit",
        type=int,
        default=100,
        help="number of semantic candidates (default: 100)",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF rank constant (default: 60)",
    )

    parser.add_argument(
        "--lexical-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=1.0,
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
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    parser.add_argument(
    "--episodes",
    action="store_true",
    help="aggregate results and rank episodes",
    )

    parser.add_argument(
        "--support-weight",
        type=float,
        default=0.25,
        help="additional passage contribution to episode scores",
    )

    parser.add_argument(
        "--support-limit",
        type=int,
        default=5,
        help="maximum passages contributing to episode scores",
    )

    parser.add_argument(
        "--context",
        type=int,
        default=0,
        help="neighbouring passages to display on each side",
    )

    args = parser.parse_args()

    for name in (
        "limit",
        "lexical_limit",
        "semantic_limit",
        "threads",
        "max_seq_length",
        "rrf_k",
    ):
        value = getattr(args, name)

        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")

    if args.lexical_weight < 0 or args.semantic_weight < 0:
        parser.error("Search weights cannot be negative.")

    if args.support_weight < 0:
        parser.error("--support-weight cannot be negative")

    if args.support_limit < 1:
        parser.error("--support-limit must be positive")

    if args.context < 0:
        parser.error("--context cannot be negative")

    config = HybridConfig(
        lexical_limit=args.lexical_limit,
        semantic_limit=args.semantic_limit,
        default_limit=args.limit or 20,
        rrf_k=args.rrf_k,
        lexical_weight=args.lexical_weight,
        semantic_weight=args.semantic_weight,
    )

    semantic_config = SemanticConfig(
        intra_op_threads=args.threads,
        max_seq_length=args.max_seq_length,
    )

    with LexicalSearcher(args.db) as lexical:
        with HybridSearcher(
            lexical,
            config=config,
            semantic_config=semantic_config,
        ) as hybrid:
            result = hybrid.search(
                args.query,
                limit=args.limit,
                episodes=args.episodes,
                support_weight=args.support_weight,
                support_limit=args.support_limit,
            )

            if args.json:
                payload = {
                    "query": result.query,
                    "total": result.total,
                    "elapsed_ms": result.elapsed_ms,
                    "lexical_ms": result.lexical_ms,
                    "semantic_ms": result.semantic_ms,
                    "lexical_candidates": result.lexical_candidates,
                    "semantic_candidates": result.semantic_candidates,
                    "hits": (
                        [asdict(h) for h in result.episode_hits]
                        if args.episodes
                        else [asdict(h) for h in result.hits]
                    ),
                }
                print(json.dumps(payload, ensure_ascii=False, indent=2))

            print_result(result)

            if args.episodes:
                print_episode_results(result, lexical, args.context)
            else:
                print_passage_results(result, lexical, args.context)

                if args.context:
                    for i, hit in enumerate(result.hits, 1):
                        print_passage(
                            hit,
                            i,
                            context=args.context,
                            lexical=lexical,
                        )

    if args.json:
        payload = {
            "query": result.query,
            "total": result.total,
            "elapsed_ms": result.elapsed_ms,
            "lexical_ms": result.lexical_ms,
            "semantic_ms": result.semantic_ms,
            "lexical_candidates": result.lexical_candidates,
            "semantic_candidates": result.semantic_candidates,
            "hits": [asdict(hit) for hit in result.hits],
        }

        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_result(result)

    return 0


def print_passage(hit, rank: int, context: int = 0, lexical=None) -> None:
    speakers = " / ".join(hit.hit.speakers)
    print(f"\n     Passage #{rank}  RRF {hit.score:.6f}")
    print(
        f"     seek {ts(hit.hit.seek_s)}  "
        f"(passage {ts(hit.hit.start_s)}–{ts(hit.hit.end_s)})"
    )
    print(f"     sources: {' + '.join(hit.sources)}")

    if speakers:
        print(f"     speakers: {speakers}")

    if context and lexical is not None:
        expanded = expand_context(
            hit,
            lexical,
            before=context,
            after=context,
        )

        for neighbour in expanded.before:
            print(f"\n     [CONTEXT BEFORE] {ts(neighbour.start_s)}")
            print("     " + neighbour.snippet)

    # print("     " + hit.hit.snippet)
    print("     " + hit.hit.text)

    if context and lexical is not None:
        for neighbour in expanded.after:
            print(f"\n     [CONTEXT AFTER] {ts(neighbour.start_s)}")
            print("     " + neighbour.snippet)


def print_episode_results(result, lexical, context: int = 0) -> None:
    print("\n=== HYBRID EPISODE MATCHES ===")

    for i, episode in enumerate(result.episode_hits, 1):
        print(f"\n#{i:<3} {episode.episode_title}")
        print(
            f"     episode score: {episode.score:.6f}  "
            f"({episode.n_hits} relevant passages)"
        )

        for j, hit in enumerate(episode.hits, 1):
            print_passage(
                hit,
                j,
                context=context,
                lexical=lexical,
            )



if __name__ == "__main__":
    sys.exit(main())
