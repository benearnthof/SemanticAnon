#!/usr/bin/env python3
"""
Update the podcast search index from manually annotated diarization JSON.

Run from the project root:
    python semanticsearch/update_index.py
    python semanticsearch/update_index.py --dry-run
    python semanticsearch/update_index.py --device cuda --batch-size 64
    python semanticsearch/update_index.py --skip-lexical

The source of truth for passages remains data/diarized_annotated/. This script
never copies or modifies data/diarized/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/workspace/")
PASSAGES_BUILDER = PROJECT_ROOT / "SemanticAnon" / "semanticsearch" / "build_passages.py"
LEXICAL_BUILDER = PROJECT_ROOT / "SemanticAnon" / "semanticsearch" / "lexical" / "build_index.py"

DEFAULT_INDEX = PROJECT_ROOT / "SemanticAnon" / "semanticsearch" / "index"
DEFAULT_EMBEDDINGS = (
    PROJECT_ROOT / "SemanticAnon" /  "semanticsearch" / "embeddings" / "qwen3-embedding-0.6b"
)
DEFAULT_LEXICAL_DB = PROJECT_ROOT / "SemanticAnon" / "semanticsearch" / "lexical" / "lexical.db"
MANIFEST_NAME = "embedding_manifest.json"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QUERY_INSTRUCTION = (
    "Instruct: Given a search query, retrieve relevant passages from podcast "
    "transcripts that discuss the same subject, people, events, concepts, or ideas.\n"
    "Query: "
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def run(command: list[str], *, cwd: Path = PROJECT_ROOT) -> None:
    log("$ " + subprocess.list2cmdline([str(x) for x in command]))
    subprocess.run(command, cwd=cwd, check=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(value)
    return rows


def load_passages(path: Path) -> tuple[list[str], list[str], dict[str, str]]:
    ids: list[str] = []
    texts: list[str] = []
    hashes: dict[str, str] = {}
    seen: set[str] = set()

    for line_no, row in enumerate(read_jsonl(path), 1):
        pid = row.get("passage_id")
        text = row.get("text")
        if pid is None or text is None:
            raise ValueError(
                f"{path}:{line_no}: each passage requires passage_id and text"
            )
        pid = str(pid)
        text = str(text).strip()
        if not text:
            raise ValueError(f"{path}:{line_no}: passage {pid!r} has empty text")
        if pid in seen:
            raise ValueError(f"{path}:{line_no}: duplicate passage_id {pid!r}")
        seen.add(pid)
        ids.append(pid)
        texts.append(text)
        hashes[pid] = hashlib.sha256(text.encode("utf-8")).hexdigest()

    if not ids:
        raise ValueError(f"No passages found in {path}")
    return ids, texts, hashes


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp, path)


def build_embeddings(
    passages_path: Path,
    old_embedding_dir: Path,
    staged_embedding_dir: Path,
    *,
    model_name: str | None,
    dimensions: int | None,
    batch_size: int | None,
    max_seq_length: int | None,
    device: str | None,
    force_rebuild: bool,
) -> dict[str, int]:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    ids, texts, text_hashes = load_passages(passages_path)
    staged_embedding_dir.mkdir(parents=True, exist_ok=True)

    old_meta_path = old_embedding_dir / "metadata.json"
    old_ids_path = old_embedding_dir / "passage_ids.json"
    old_matrix_path = old_embedding_dir / "embeddings.npy"
    old_manifest_path = old_embedding_dir / MANIFEST_NAME

    old_meta: dict[str, Any] = {}
    if old_meta_path.is_file():
        old_meta = json.loads(old_meta_path.read_text(encoding="utf-8"))

    model_name = model_name or old_meta.get("model") or DEFAULT_MODEL
    dimensions = dimensions or int(old_meta.get("dimensions", 1024))
    batch_size = batch_size or int(old_meta.get("batch_size", 64))
    max_seq_length = max_seq_length or int(old_meta.get("max_seq_length", 512))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    query_instruction = old_meta.get("query_instruction", DEFAULT_QUERY_INSTRUCTION)

    old_ids: list[str] = []
    old_hashes: dict[str, str] = {}
    old_matrix = None
    compatible = (
        not force_rebuild
        and old_meta.get("model") == model_name
        and int(old_meta.get("dimensions", -1)) == dimensions
        and int(old_meta.get("max_seq_length", 512)) == max_seq_length
        and old_matrix_path.is_file()
        and old_ids_path.is_file()
        and old_manifest_path.is_file()
    )

    if compatible:
        old_ids = json.loads(old_ids_path.read_text(encoding="utf-8"))
        manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
        old_hashes = manifest.get("text_sha256", {})
        old_matrix = np.load(old_matrix_path, mmap_mode="r")
        if old_matrix.shape != (len(old_ids), dimensions):
            log("Existing embedding files have inconsistent shapes; rebuilding all.")
            compatible = False
            old_matrix = None
        elif len(set(old_ids)) != len(old_ids):
            log("Existing passage_ids.json contains duplicates; rebuilding all.")
            compatible = False
            old_matrix = None

    old_rows = {pid: i for i, pid in enumerate(old_ids)} if compatible else {}
    reusable: dict[str, int] = {}
    changed_positions: list[int] = []
    for position, pid in enumerate(ids):
        old_position = old_rows.get(pid)
        if (
            old_position is not None
            and old_hashes.get(pid) == text_hashes[pid]
        ):
            reusable[pid] = old_position
        else:
            changed_positions.append(position)

    log(
        f"Embeddings: {len(reusable):,} reusable, "
        f"{len(changed_positions):,} to encode, "
        f"{max(0, len(old_ids) - len(reusable)):,} removed/changed."
    )

    model_kwargs: dict[str, Any] = {}
    if device.startswith("cuda"):
        model_kwargs["torch_dtype"] = torch.float16
    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"},
    )
    model.max_seq_length = max_seq_length

    probe = model.encode(
        ["dimension validation"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        truncate_dim=dimensions,
    )
    if probe.shape[1] != dimensions:
        raise RuntimeError(
            f"Requested {dimensions} dimensions, model returned {probe.shape[1]}."
        )

    matrix_path = staged_embedding_dir / "embeddings.npy"
    matrix = np.lib.format.open_memmap(
        matrix_path, mode="w+", dtype=np.float32, shape=(len(ids), dimensions)
    )
    for new_position, old_position in (
        (i, reusable[pid]) for i, pid in enumerate(ids) if pid in reusable
    ):
        matrix[new_position] = old_matrix[old_position]

    started = time.time()
    for offset in range(0, len(changed_positions), batch_size):
        positions = changed_positions[offset:offset + batch_size]
        batch_texts = [texts[i] for i in positions]
        vectors = model.encode(
            batch_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            truncate_dim=dimensions,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(positions), dimensions):
            raise RuntimeError(
                f"Unexpected embedding batch shape {vectors.shape}; "
                f"expected {(len(positions), dimensions)}."
            )
        matrix[positions] = vectors
        log(f"Encoded {min(offset + len(positions), len(changed_positions)):,}"
            f"/{len(changed_positions):,} changed passages")

    matrix.flush()
    del matrix
    write_json(staged_embedding_dir / "passage_ids.json", ids)

    metadata = dict(old_meta)
    metadata.update({
        "model": model_name,
        "dimensions": dimensions,
        "matrix_dtype": "float32",
        "normalized": True,
        "passage_count": len(ids),
        "input_file": str(DEFAULT_INDEX / "passages.jsonl"),
        "max_seq_length": max_seq_length,
        "batch_size": batch_size,
        "device": device,
        "query_instruction": query_instruction,
        "query_encoding": (
            "Encode queries with the same model and dimensions, prepending "
            "query_instruction. Encode passage texts without an instruction."
        ),
        "embedding_file": "embeddings.npy",
        "passage_ids_file": "passage_ids.json",
        "elapsed_seconds": round(time.time() - started, 2),
    })
    write_json(staged_embedding_dir / "metadata.json", metadata)
    write_json(staged_embedding_dir / MANIFEST_NAME, {
        "format_version": 1,
        "model": model_name,
        "dimensions": dimensions,
        "max_seq_length": max_seq_length,
        "text_sha256": text_hashes,
    })

    # Verify exact row/ID alignment before publishing.
    check_ids = json.loads(
        (staged_embedding_dir / "passage_ids.json").read_text(encoding="utf-8")
    )
    check_matrix = np.load(matrix_path, mmap_mode="r")
    if check_ids != ids or check_matrix.shape != (len(ids), dimensions):
        raise RuntimeError("Staged embedding IDs and matrix failed validation.")
    del check_matrix
    del old_matrix

    return {
        "passages": len(ids),
        "reused": len(reusable),
        "encoded": len(changed_positions),
        "removed": max(0, len(old_ids) - len(reusable)) if compatible else 0,
    }


def publish_directory(staged: Path, destination: Path, filenames: list[str]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        source = staged / filename
        target = destination / filename
        os.replace(source, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--lexical-db", type=Path, default=DEFAULT_LEXICAL_DB)
    parser.add_argument("--target-seconds", type=float, default=30.0)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--min-seconds", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dimensions", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--trigram", action="store_true",
                        help="Build the lexical trigram FTS index.")
    parser.add_argument("--skip-lexical", action="store_true")
    parser.add_argument("--rebuild-embeddings", action="store_true",
                        help="Re-encode every passage, ignoring the incremental manifest.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.target_seconds <= 0 or args.max_seconds < args.target_seconds:
        parser.error("Require 0 < --target-seconds <= --max-seconds")
    if args.min_seconds < 0 or args.workers < 1:
        parser.error("--min-seconds must be nonnegative and --workers at least 1")
    if args.dimensions is not None and args.dimensions < 1:
        parser.error("--dimensions must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main() -> int:
    args = parse_args()
    for path in (PASSAGES_BUILDER, LEXICAL_BUILDER):
        if not path.is_file():
            raise FileNotFoundError(f"Required builder not found: {path}")
    annotation_dir = args.data_dir / "diarized_annotated"
    if not annotation_dir.is_dir():
        raise FileNotFoundError(
            f"Annotated diarization directory not found: {annotation_dir}. "
            "Raw data/diarized is deliberately not used."
        )

    if args.dry_run:
        command = [
            sys.executable, str(PASSAGES_BUILDER),
            "--data-dir", str(args.data_dir),
            "--output-dir", str(args.index_dir),
            "--target-seconds", str(args.target_seconds),
            "--max-seconds", str(args.max_seconds),
            "--min-seconds", str(args.min_seconds),
            "--workers", str(args.workers),
            "--dry-run",
        ]
        run(command)
        log("Dry run complete; no index, embedding, or lexical files were changed.")
        return 0

    args.index_dir.parent.mkdir(parents=True, exist_ok=True)
    args.embedding_dir.parent.mkdir(parents=True, exist_ok=True)
    args.lexical_db.parent.mkdir(parents=True, exist_ok=True)

    # Stage all outputs on the same filesystems as their destinations so final
    # os.replace operations are atomic per file.
    with tempfile.TemporaryDirectory(prefix=".index-update-", dir=args.index_dir.parent) as work:
        work_dir = Path(work)
        staged_index = work_dir / "index"
        staged_embeddings = work_dir / "embeddings"
        staged_lexical = work_dir / "lexical.db"

        run([
            sys.executable, str(PASSAGES_BUILDER),
            "--data-dir", str(args.data_dir),
            "--output-dir", str(staged_index),
            "--target-seconds", str(args.target_seconds),
            "--max-seconds", str(args.max_seconds),
            "--min-seconds", str(args.min_seconds),
            "--workers", str(args.workers),
        ])

        embedding_stats = build_embeddings(
            staged_index / "passages.jsonl",
            args.embedding_dir,
            staged_embeddings,
            model_name=args.model,
            dimensions=args.dimensions,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            device=args.device,
            force_rebuild=args.rebuild_embeddings,
        )

        if not args.skip_lexical:
            lexical_command = [
                sys.executable, str(LEXICAL_BUILDER),
                "--index-dir", str(staged_index),
                "--out", str(staged_lexical),
                "--force",
            ]
            if args.trigram:
                lexical_command.append("--trigram")
            run(lexical_command)

        # Publish the passage index and embedding bundle only after both stages
        # have completed and passed their validations.
        publish_directory(
            staged_index, args.index_dir,
            ["passages.jsonl", "episodes.jsonl", "build_stats.json"],
        )
        publish_directory(
            staged_embeddings, args.embedding_dir,
            ["embeddings.npy", "passage_ids.json", "metadata.json", MANIFEST_NAME],
        )
        if not args.skip_lexical:
            os.replace(staged_lexical, args.lexical_db)

    log("Update complete.")
    log(
        f"Passages: {embedding_stats['passages']:,}; "
        f"reused vectors: {embedding_stats['reused']:,}; "
        f"encoded: {embedding_stats['encoded']:,}; "
        f"removed/changed old rows: {embedding_stats['removed']:,}."
    )
    if args.skip_lexical:
        log("Lexical database skipped (--skip-lexical).")
    else:
        log(f"Lexical database: {args.lexical_db}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted; staged files were discarded.")
        raise SystemExit(130)
    except Exception as exc:
        log(f"ERROR: {exc}")
        raise
