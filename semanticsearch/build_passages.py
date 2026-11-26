#!/usr/bin/env python
r"""Build a deterministic, speaker-aware passage dataset from diarized annotations.

Run from the project root:
    python .\semanticsearch\build_passages.py

Optional:
    python .\semanticsearch\build_passages.py --target-seconds 30 --max-seconds 60
    python .\semanticsearch\build_passages.py --dry-run

Requires:
    pip install spacy
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import spacy
except ImportError:
    raise SystemExit(
        "Missing dependency 'spacy'. Install it with: pip install spacy"
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "semanticsearch" / "index"

# Attach punctuation to the preceding token; avoid spaces just inside brackets.
_NO_SPACE_BEFORE = set(".,!?;:%)]}»”’")
_NO_SPACE_AFTER = set("([{«“‘")


def detokenize(words: list[dict[str, Any]]) -> tuple[str, list[tuple[int, int]]]:
    """Join Whisper word tokens and return each token's character span."""
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    previous = ""

    for item in words:
        token = str(item.get("word", "")).strip()
        if not token:
            spans.append((cursor, cursor))
            continue

        needs_space = bool(pieces)
        if token[:1] in _NO_SPACE_BEFORE:
            needs_space = False
        if previous[-1:] in _NO_SPACE_AFTER:
            needs_space = False

        if needs_space:
            pieces.append(" ")
            cursor += 1

        start = cursor
        pieces.append(token)
        cursor += len(token)
        spans.append((start, cursor))
        previous = token

    return "".join(pieces), spans


def make_sentence_tokenizer():
    """A lightweight English sentence-boundary detector; no language model download."""
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer", config={"punct_chars": [".", "!", "?", "。", "！", "？"]})
    return nlp


def word_speaker(word: dict[str, Any], fallback: str | None) -> str | None:
    value = word.get("speaker") or fallback
    return str(value) if value is not None else None


def sentence_word_ranges(
    words: list[dict[str, Any]], nlp
) -> tuple[list[dict[str, Any]], str]:
    """Map sentence boundaries in reconstructed text back to word indices."""
    text, spans = detokenize(words)
    if not text.strip():
        return [], text

    doc = nlp(text)
    ranges: list[dict[str, Any]] = []
    word_index = 0

    for sent in doc.sents:
        sent_start, sent_end = sent.start_char, sent.end_char
        indices = []
        while word_index < len(spans) and spans[word_index][1] <= sent_start:
            word_index += 1

        j = word_index
        while j < len(spans) and spans[j][0] < sent_end:
            if spans[j][1] > sent_start:
                indices.append(j)
            j += 1

        if indices:
            ranges.append({
                "first": indices[0],
                "last": indices[-1] + 1,
                "text": text[sent_start:sent_end].strip(),
            })
        word_index = j

    # Any remaining tokens not captured due to unusual punctuation/tokenization.
    covered = {i for r in ranges for i in range(r["first"], r["last"])}
    for i, (start, end) in enumerate(spans):
        if end > start and i not in covered:
            ranges.append({"first": i, "last": i + 1, "text": words[i]["word"].strip()})

    ranges.sort(key=lambda r: r["first"])
    return ranges, text


def speaker_runs(words: list[dict[str, Any]], start: int, end: int,
                 fallback_speaker: str | None = None) -> tuple[list[str], list[dict[str, Any]]]:
    """Coalesce adjacent words into speaker intervals."""
    runs: list[dict[str, Any]] = []

    for word in words[start:end]:
        speaker = word_speaker(word, fallback_speaker)
        if not speaker:
            continue

        try:
            word_start = float(word["start"])
            word_end = float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue

        if runs and runs[-1]["speaker"] == speaker:
            runs[-1]["end"] = max(runs[-1]["end"], word_end)
        else:
            runs.append({"speaker": speaker, "start": word_start, "end": word_end})

    speakers = list(dict.fromkeys(run["speaker"] for run in runs))
    return speakers, runs


def split_oversized_sentence(sentence_words: list[dict[str, Any]],
                             max_seconds: float) -> list[list[dict[str, Any]]]:
    """Split a sentence longer than the hard limit at word boundaries."""
    pieces: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for word in sentence_words:
        if not current:
            current = [word]
            continue

        elapsed = float(word["end"]) - float(current[0]["start"])
        if elapsed > max_seconds:
            pieces.append(current)
            current = [word]
        else:
            current.append(word)

    if current:
        pieces.append(current)
    return pieces


def make_chunks(words: list[dict[str, Any]], sentence_ranges: list[dict[str, Any]],
                target_seconds: float, max_seconds: float,
                min_seconds: float, fallback_speaker: str | None):
    """Greedily accumulate sentences toward target duration, respecting max duration."""
    sentence_units: list[list[dict[str, Any]]] = []

    for sentence in sentence_ranges:
        sentence_words = words[sentence["first"]:sentence["last"]]
        if not sentence_words:
            continue

        duration = float(sentence_words[-1]["end"]) - float(sentence_words[0]["start"])
        if duration > max_seconds:
            sentence_units.extend(split_oversized_sentence(sentence_words, max_seconds))
        else:
            sentence_units.append(sentence_words)

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for unit in sentence_units:
        if not unit:
            continue

        if current:
            combined_duration = float(unit[-1]["end"]) - float(current[0]["start"])
            if combined_duration > max_seconds:
                chunks.append(current)
                current = []

        current.extend(unit)
        current_duration = float(current[-1]["end"]) - float(current[0]["start"])

        if current_duration >= target_seconds:
            chunks.append(current)
            current = []

    if current:
        # Avoid a tiny tail by merging it into the previous chunk when the hard
        # maximum still permits it. Otherwise retain the short final passage.
        if chunks:
            merged_duration = float(current[-1]["end"]) - float(chunks[-1][0]["start"])
            if merged_duration <= max_seconds:
                chunks[-1].extend(current)
            else:
                chunks.append(current)
        else:
            chunks.append(current)

    # Keep very short passages: discarding speech is worse than a short tail.
    # min_seconds is recorded as a tuning parameter and used for diagnostics.
    return chunks


def get_episode_id(annotation_path: Path, audio_name: str) -> str:
    source = f"{annotation_path.stem} {audio_name}"
    match = re.search(r"\[(\d+)\]", source)
    return match.group(1) if match else annotation_path.stem


def load_annotation(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Worker state is initialized once in each child process. This avoids rebuilding
# the spaCy pipeline for every annotation and avoids sending large objects with
# every task (important on Windows, which uses spawn for child processes).
_WORKER_AUDIO_NAMES: dict[str, str] = {}
_WORKER_NLP = None
_WORKER_TARGET_SECONDS = 30.0
_WORKER_MAX_SECONDS = 60.0
_WORKER_MIN_SECONDS = 10.0


def init_worker(audio_names: dict[str, str], target_seconds: float,
                max_seconds: float, min_seconds: float) -> None:
    global _WORKER_AUDIO_NAMES, _WORKER_NLP
    global _WORKER_TARGET_SECONDS, _WORKER_MAX_SECONDS, _WORKER_MIN_SECONDS
    _WORKER_AUDIO_NAMES = audio_names
    _WORKER_NLP = make_sentence_tokenizer()
    _WORKER_TARGET_SECONDS = target_seconds
    _WORKER_MAX_SECONDS = max_seconds
    _WORKER_MIN_SECONDS = min_seconds


def process_annotation(annotation_path_string: str):
    """Process one annotation file; safe to execute in a child process."""
    annotation_path = Path(annotation_path_string)
    stats = Counter()
    warnings: list[str] = []

    try:
        annotation = load_annotation(annotation_path)
    except Exception as exc:
        stats["invalid_json"] += 1
        warnings.append(f"{annotation_path.name}: could not parse JSON: {exc}")
        return [], None, dict(stats), warnings

    audio_name = str(annotation.get("audio") or "")
    audio_match = _WORKER_AUDIO_NAMES.get(Path(audio_name).name.casefold())
    if not audio_match:
        stats["missing_audio"] += 1
        warnings.append(f"{annotation_path.name}: audio file not found: {audio_name!r}")

    episode_id = get_episode_id(annotation_path, audio_name)
    title = annotation_path.stem
    if re.search(r"\s*\[\d+\]\s*$", title):
        title = re.sub(r"\s*\[\d+\]\s*$", "", title).strip()

    episode_duration = annotation.get("duration")
    language = annotation.get("language")
    segments = annotation.get("segments") or []
    global_speakers = annotation.get("speaker_global") or {}

    words: list[dict[str, Any]] = []
    missing_word_segments = 0
    for segment in segments:
        segment_speaker = segment.get("speaker")
        segment_words = segment.get("words") or []
        if not segment_words:
            missing_word_segments += 1
            continue

        for word in segment_words:
            if not isinstance(word, dict) or not word.get("word"):
                continue
            try:
                start = float(word["start"])
                end = float(word["end"])
            except (KeyError, TypeError, ValueError):
                stats["invalid_words"] += 1
                continue
            if end <= start:
                stats["invalid_words"] += 1
                continue

            local_speaker = word.get("speaker") or segment_speaker
            global_speaker = (
                global_speakers.get(local_speaker, local_speaker)
                if local_speaker is not None else None
            )
            words.append({
                "word": str(word["word"]),
                "start": start,
                "end": end,
                "speaker": global_speaker,
            })

    words.sort(key=lambda w: (w["start"], w["end"]))
    if missing_word_segments:
        stats["segments_without_words"] += missing_word_segments

    if not words:
        stats["episodes_without_words"] += 1
        warnings.append(f"{annotation_path.name}: no usable word-level timestamps")
        return [], None, dict(stats), warnings

    sentence_ranges, _ = sentence_word_ranges(words, _WORKER_NLP)
    chunks = make_chunks(
        words, sentence_ranges,
        target_seconds=_WORKER_TARGET_SECONDS,
        max_seconds=_WORKER_MAX_SECONDS,
        min_seconds=_WORKER_MIN_SECONDS,
        fallback_speaker=None,
    )

    passages: list[dict[str, Any]] = []
    for chunk_number, chunk_words in enumerate(chunks, start=1):
        start = float(chunk_words[0]["start"])
        end = float(chunk_words[-1]["end"])
        duration = max(0.0, end - start)
        text, _ = detokenize(chunk_words)
        text = text.strip()
        if not text:
            stats["empty_passages"] += 1
            continue

        speakers, speaker_segments = speaker_runs(chunk_words, 0, len(chunk_words))
        passage_id = f"{episode_id}_{chunk_number:05d}"
        passages.append({
            "passage_id": passage_id,
            "episode_id": episode_id,
            "episode_title": title,
            "audio": audio_match or audio_name,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(duration, 3),
            "text": text,
            "speakers": speakers,
            "speaker_segments": [
                {
                    "speaker": run["speaker"],
                    "start": round(run["start"], 3),
                    "end": round(run["end"], 3),
                }
                for run in speaker_segments
            ],
            "word_count": len(chunk_words),
            "language": language,
        })
        stats["short_passages"] += duration < _WORKER_MIN_SECONDS
        stats["long_passages"] += duration > _WORKER_MAX_SECONDS

    episode = {
        "episode_id": episode_id,
        "episode_title": title,
        "audio": audio_match or audio_name,
        "annotation_file": annotation_path.name,
        "duration": episode_duration,
        "language": language,
        "passage_count": len(passages),
    }
    stats["episodes_processed"] += 1
    stats["words_processed"] += len(words)
    return passages, episode, dict(stats), warnings


def build(args) -> dict[str, Any]:
    data_dir = args.data_dir.resolve()
    annotation_dir = data_dir / "diarized_annotated"
    audio_dir = data_dir / "audio"

    if not annotation_dir.is_dir():
        raise SystemExit(f"Annotation directory not found: {annotation_dir}")
    if not audio_dir.is_dir():
        raise SystemExit(f"Audio directory not found: {audio_dir}")

    annotation_paths = sorted(annotation_dir.glob("*.json"), key=lambda p: p.name.casefold())
    if not annotation_paths:
        raise SystemExit(f"No annotation JSON files found in {annotation_dir}")

    audio_names = {p.name.casefold(): p.name for p in audio_dir.iterdir() if p.is_file()}
    passages: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    warnings: list[str] = []
    stats = Counter()
    seen_episode_ids: set[str] = set()

    worker_args = (
        audio_names, args.target_seconds, args.max_seconds, args.min_seconds
    )
    if args.workers == 1:
        # Serial path is useful for debugging and for exact speed comparisons.
        init_worker(*worker_args)
        results = map(process_annotation, (str(p) for p in annotation_paths))
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=init_worker,
            initargs=worker_args,
        )
        # executor.map preserves input order, so IDs and JSONL output remain
        # deterministic regardless of which worker finishes first.
        results = executor.map(
            process_annotation,
            (str(p) for p in annotation_paths),
            chunksize=1,
        )

    try:
        for annotation_number, result in enumerate(results, start=1):
            file_passages, episode, file_stats, file_warnings = result
            passages.extend(file_passages)
            warnings.extend(file_warnings)
            stats.update(file_stats)

            if episode is not None:
                episode_id = episode["episode_id"]
                if episode_id in seen_episode_ids:
                    warnings.append(
                        f"{episode['annotation_file']}: duplicate episode_id {episode_id!r}; "
                        "IDs may need manual disambiguation."
                    )
                seen_episode_ids.add(episode_id)
                episodes.append(episode)

            if annotation_number % 25 == 0 or annotation_number == len(annotation_paths):
                print(
                    f"\rProcessed {annotation_number}/{len(annotation_paths)} "
                    f"annotations | {len(passages):,} passages "
                    f"({args.workers} workers)",
                    end="",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    print()
    stats["annotations_found"] = len(annotation_paths)
    stats["passages_created"] = len(passages)
    stats["episodes_in_output"] = len(episodes)
    stats["target_seconds"] = args.target_seconds
    stats["max_seconds"] = args.max_seconds
    stats["min_seconds"] = args.min_seconds
    stats["workers"] = args.workers
    stats["tokenizer"] = "spaCy blank(en) + sentencizer"

    output_dir = args.output_dir.resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        passages_path = output_dir / "passages.jsonl"
        episodes_path = output_dir / "episodes.jsonl"
        stats_path = output_dir / "build_stats.json"
        temp_passages = passages_path.with_suffix(".jsonl.tmp")
        temp_episodes = episodes_path.with_suffix(".jsonl.tmp")
        temp_stats = stats_path.with_suffix(".json.tmp")

        with temp_passages.open("w", encoding="utf-8", newline="\n") as handle:
            for passage in passages:
                handle.write(json.dumps(passage, ensure_ascii=False) + "\n")
        with temp_episodes.open("w", encoding="utf-8", newline="\n") as handle:
            for episode in episodes:
                handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
        with temp_stats.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(stats), handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        temp_passages.replace(passages_path)
        temp_episodes.replace(episodes_path)
        temp_stats.replace(stats_path)

    print("\nBuild summary")
    print("-------------")
    for key, value in stats.items():
        print(f"{key:28s} {value}")
    print(f"{'warnings':28s} {len(warnings)}")

    if warnings:
        print("\nWarnings (first 30)")
        for warning in warnings[:30]:
            print(f"  - {warning}")
        if len(warnings) > 30:
            print(f"  ... and {len(warnings) - 30} more")

    if args.dry_run:
        print("\nDry run: no output files written.")
    else:
        print(f"\nWrote index files to: {output_dir}")

    return dict(stats)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build sentence-aware, speaker-attributed podcast passages."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-seconds", type=float, default=30.0)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--min-seconds", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of worker processes (1 disables multiprocessing; default: 4).",
    )
    args = parser.parse_args()

    if args.target_seconds <= 0:
        parser.error("--target-seconds must be positive")
    if args.max_seconds < args.target_seconds:
        parser.error("--max-seconds must be >= --target-seconds")
    if args.min_seconds < 0:
        parser.error("--min-seconds cannot be negative")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    return args


if __name__ == "__main__":
    build(parse_args())
