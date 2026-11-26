#!/usr/bin/env python
r"""
Generate Qwen3-Embedding-0.6B vectors for SemanticAnon podcast passages.

Run from the project root:
    pip install -U "sentence-transformers>=4.1" "transformers>=4.51" torch numpy tqdm
    python .\semanticsearch\build_embeddings.py --dry-run
    python .\semanticsearch\build_embeddings.py --batch-size 64

Outputs:
    semanticsearch/embeddings/qwen3-embedding-0.6b/
        embeddings.npy       float32, normalized matrix [passages, dimensions]
        passage_ids.json     passage ID corresponding to each matrix row
        metadata.json        model/configuration details

The input passages.jsonl is never modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QUERY_INSTRUCTION = (
    "Instruct: Given a search query, retrieve relevant passages from podcast "
    "transcripts that discuss the same subject, people, events, concepts, or ideas.\n"
    "Query: "
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build semantic passage embeddings with Qwen3-Embedding-0.6B."
    )
    p.add_argument("--input", type=Path, default=Path("semanticsearch/index/passages.jsonl"))
    p.add_argument(
        "--output-dir", type=Path,
        default=Path("semanticsearch/embeddings/qwen3-embedding-0.6b"),
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dimensions", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--max-seq-length", type=int, default=512,
        help="Maximum passage tokens. Increase if your passages are unusually long.",
    )
    p.add_argument(
        "--device", default=None,
        help="Optional device override, e.g. cuda, cuda:0, cpu. Defaults to CUDA if available.",
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=2000,
        help="Checkpoint after this many passages; 0 disables periodic checkpoints.",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION)
    return p.parse_args()


def read_passages(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    ids: list[str] = []
    seen: set[str] = set()

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(tqdm(f, desc="Reading passages", unit=" lines"), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

            pid = item.get("passage_id")
            text = item.get("text")
            if pid is None or not isinstance(text, str):
                raise ValueError(
                    f"Line {line_no} must contain string 'passage_id' and 'text' fields."
                )
            pid = str(pid)
            if pid in seen:
                raise ValueError(f"Duplicate passage_id {pid!r} at line {line_no}.")
            if not text.strip():
                continue

            seen.add(pid)
            ids.append(pid)
            texts.append(text.strip())

    if not texts:
        raise ValueError(f"No usable passages found in {path}")
    return texts, ids


def write_json_atomic(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def save_checkpoint(
    matrix: np.memmap, count: int, ids: list[str], output_dir: Path
) -> None:
    """Persist only completed rows; checkpoint can safely be resumed later."""
    matrix.flush()
    tmp_matrix = output_dir / "checkpoint.tmp.npy"
    np.save(tmp_matrix, np.asarray(matrix[:count], dtype=np.float32))
    os.replace(tmp_matrix, output_dir / "checkpoint.npy")
    write_json_atomic(output_dir / "checkpoint_ids.json", ids[:count])


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.dimensions < 1 or args.max_seq_length < 1:
        raise ValueError("Batch size, dimensions and max sequence length must be positive.")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every cannot be negative.")
    if not args.input.is_file():
        raise FileNotFoundError(
            f"Could not find {args.input}. Run this command from the project root."
        )

    texts, passage_ids = read_passages(args.input)
    n = len(texts)
    estimated_mib = n * args.dimensions * 4 / (1024 ** 2)

    print("\nEmbedding configuration")
    print("-----------------------")
    print(f"Passages          : {n:,}")
    print(f"Model             : {args.model}")
    print(f"Dimensions        : {args.dimensions:,}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Max sequence len  : {args.max_seq_length}")
    print(f"Output            : {args.output_dir}")
    print(f"Matrix size       : approximately {estimated_mib:,.1f} MiB (float32)")

    if args.dry_run:
        print("\nDry run: no model loaded and no output files written.")
        return 0

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            'Install dependencies with: pip install -U '
            '"sentence-transformers>=4.1" "transformers>=4.51" torch numpy tqdm'
        ) from exc

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    final_matrix = out / "embeddings.npy"
    final_ids = out / "passage_ids.json"
    final_meta = out / "metadata.json"
    checkpoint_matrix = out / "checkpoint.npy"
    checkpoint_ids = out / "checkpoint_ids.json"
    work_matrix = out / "embeddings_work.npy"

    if not args.resume and not args.overwrite:
        existing = [p for p in (final_matrix, final_ids, final_meta) if p.exists()]
        if existing:
            raise FileExistsError(
                "Final output already exists. Use --resume for a valid checkpoint "
                "or --overwrite to replace it."
            )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs: dict[str, Any] = {}
    if device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.float16
    tokenizer_kwargs = {"padding_side": "left"}

    print(f"\nLoading {args.model} on {device} ...")
    model = SentenceTransformer(
        args.model,
        device=device,
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
    )
    model.max_seq_length = args.max_seq_length

    # Validate dimensions before allocating the complete output matrix.
    probe = model.encode(
        ["dimension validation"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        truncate_dim=args.dimensions,
    )
    if probe.shape[1] != args.dimensions:
        raise RuntimeError(
            f"Requested {args.dimensions} dimensions, model returned {probe.shape[1]}."
        )

    start = 0
    if args.resume:
        if not checkpoint_matrix.exists() or not checkpoint_ids.exists():
            raise FileNotFoundError(
                "No complete checkpoint pair found. Start without --resume."
            )
        with checkpoint_ids.open("r", encoding="utf-8") as f:
            saved_ids = json.load(f)
        if len(saved_ids) > n or saved_ids != passage_ids[:len(saved_ids)]:
            raise RuntimeError(
                "Checkpoint IDs do not match the beginning of the current passage index. "
                "The input may have changed; refusing to resume."
            )
        previous = np.load(checkpoint_matrix, mmap_mode="r")
        if previous.shape != (len(saved_ids), args.dimensions):
            raise RuntimeError("Checkpoint matrix shape/dimensions are inconsistent.")
        start = len(saved_ids)
        print(f"Resuming at {start:,}/{n:,} passages.")

    matrix = np.lib.format.open_memmap(
        work_matrix,
        mode="w+",
        dtype=np.float32,
        shape=(n, args.dimensions),
    )
    if start:
        matrix[:start] = previous
        del previous

    started = time.time()
    last_checkpoint = start

    try:
        batches = range(start, n, args.batch_size)
        for batch_start in tqdm(batches, desc="Embedding", unit="batch"):
            batch_end = min(batch_start + args.batch_size, n)
            vectors = model.encode(
                texts[batch_start:batch_end],
                batch_size=args.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                truncate_dim=args.dimensions,
            )
            matrix[batch_start:batch_end] = np.asarray(vectors, dtype=np.float32)

            if (
                args.checkpoint_every
                and batch_end - last_checkpoint >= args.checkpoint_every
            ):
                save_checkpoint(matrix, batch_end, passage_ids, out)
                last_checkpoint = batch_end

        matrix.flush()
    except KeyboardInterrupt:
        matrix.flush()
        print(
            "\nInterrupted. The last completed checkpoint is safe. "
            "Re-run with --resume to continue from it."
        )
        return 130

    # Publish the completed matrix and row mapping.
    save_checkpoint(matrix, n, passage_ids, out)
    del matrix
    os.replace(work_matrix, final_matrix)
    write_json_atomic(final_ids, passage_ids)

    metadata = {
        "model": args.model,
        "dimensions": args.dimensions,
        "matrix_dtype": "float32",
        "normalized": True,
        "passage_count": n,
        "input_file": str(args.input),
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "device": device,
        "query_instruction": args.query_instruction,
        "query_encoding": (
            "Encode queries with the same model and dimensions, prepending "
            "query_instruction. Encode passage texts without an instruction."
        ),
        "embedding_file": final_matrix.name,
        "passage_ids_file": final_ids.name,
        "elapsed_seconds": round(time.time() - started, 2),
    }
    write_json_atomic(final_meta, metadata)

    print("\nBuild complete")
    print("--------------")
    print(f"Embeddings : {final_matrix}")
    print(f"Shape      : ({n:,}, {args.dimensions})")
    print(f"IDs        : {final_ids}")
    print(f"Metadata   : {final_meta}")
    print(f"Elapsed    : {(time.time() - started) / 60:.1f} minutes")
    print("The passage index was not modified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
