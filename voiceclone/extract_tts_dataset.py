#!/usr/bin/env python3
"""Extract clean LIZ and BRACE utterances from episode-level annotations.

Expected layout:
    data/audio/*.mp3
    data/diarized_annotated/*.json

Output:
    data/tts_dataset/wavs/{liz,brace}/*.wav
    data/tts_dataset/segments.{jsonl,csv}
    data/tts_dataset/skipped_segments.csv
    data/tts_dataset/extraction_report.json

Requires FFmpeg on PATH. Audio is decoded ONCE per episode, then individual
clips are cut from the temporary 24 kHz mono PCM stream. This avoids decoding
an entire MP3 from its beginning for every individual utterance.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

TARGET_SPEAKERS = {"LIZ": "liz", "BRACE": "brace"}


def safe_name(value: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("._")
    return value or "episode"


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", value).strip() if isinstance(value, str) else ""


def intervals_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def get_word_speakers(segment: dict) -> set[str]:
    return {str(w.get("speaker")) for w in (segment.get("words") or [])
            if isinstance(w, dict) and w.get("speaker")}


def mean_word_score(segment: dict) -> float | None:
    scores = [float(w["score"]) for w in (segment.get("words") or [])
              if isinstance(w, dict) and isinstance(w.get("score"), (int, float))]
    return sum(scores) / len(scores) if scores else None


def load_diarization_intervals(data: dict) -> list[tuple[float, float, str]]:
    result = []
    for item in data.get("diarization") or []:
        try:
            start, end, speaker = float(item["start"]), float(item["end"]), str(item["speaker"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            result.append((start, end, speaker))
    return result


def has_other_speaker_overlap(start: float, end: float, target_local_speaker: str,
                              intervals: list[tuple[float, float, str]],
                              tolerance: float) -> tuple[bool, list[str], float]:
    others, maximum = set(), 0.0
    for other_start, other_end, other_speaker in intervals:
        if other_speaker == target_local_speaker:
            continue
        overlap = intervals_overlap(start, end, other_start, other_end)
        maximum = max(maximum, overlap)
        if overlap > tolerance:
            others.add(other_speaker)
    return bool(others), sorted(others), maximum


def is_valid_existing_clip(path: Path, start: float, end: float, sample_rate: int) -> bool:
    """Reject truncated/invalid WAVs left behind by an interrupted older run."""
    try:
        with wave.open(str(path), "rb") as wav:
            expected = max(1, int(round((end - start) * sample_rate)))
            return (wav.getnchannels() == 1 and wav.getsampwidth() == 2
                    and wav.getframerate() == sample_rate
                    and abs(wav.getnframes() - expected) <= max(2, int(sample_rate * 0.03)))
    except (OSError, wave.Error, EOFError):
        return False


def write_clip(raw_handle, destination: Path, start: float, end: float, sample_rate: int) -> None:
    """Copy a time range from mono signed-16-bit PCM into a WAV file."""
    start_frame = max(0, int(round(start * sample_rate)))
    end_frame = max(start_frame + 1, int(round(end * sample_rate)))
    raw_handle.seek(start_frame * 2)  # mono, signed 16-bit = 2 bytes/frame
    pcm = raw_handle.read((end_frame - start_frame) * 2)
    if len(pcm) < 2:
        raise RuntimeError("No decoded audio samples at requested timestamp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary sibling and atomically rename, so interrupted runs
    # cannot leave a truncated WAV that a later resumable run mistakes as valid.
    temp_destination = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with wave.open(str(temp_destination), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        temp_destination.replace(destination)
    finally:
        if temp_destination.exists():
            temp_destination.unlink()


def process_episode(ffmpeg: str, source: Path, jobs: list[tuple[Path, float, float, dict]],
                    sample_rate: int) -> tuple[list[dict], list[dict]]:
    """Decode one source file once, then extract all its requested clips."""
    successful, errors = [], []
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix="tts_decode_", suffix=".pcm", delete=False) as tmp:
            temp_path = Path(tmp.name)
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                   "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sample_rate),
                   "-f", "s16le", "-acodec", "pcm_s16le", str(temp_path)]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        with temp_path.open("rb") as raw:
            for destination, start, end, row in jobs:
                try:
                    write_clip(raw, destination, start, end, sample_rate)
                    successful.append(row)
                except Exception as exc:
                    errors.append({"episode": row["episode_id"],
                                   "segment_index": row["segment_id"].rsplit("__", 1)[-1],
                                   "start": start, "end": end, "speaker": row["speaker"],
                                   "reason": f"clip_write_error: {exc}"})
    except Exception as exc:
        message = str(exc)
        for destination, start, end, row in jobs:
            errors.append({"episode": row["episode_id"],
                           "segment_index": row["segment_id"].rsplit("__", 1)[-1],
                           "start": start, "end": end, "speaker": row["speaker"],
                           "reason": f"episode_decode_error: {message}"})
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return successful, errors


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("./data"))
    parser.add_argument("--annotation-dir", default="diarized_annotated")
    parser.add_argument("--audio-dir", default="audio")
    parser.add_argument("--output-dir", default="tts_dataset")
    parser.add_argument("--min-duration", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel episode decoders (4-6 recommended for a 6-core CPU).")
    parser.add_argument("--overlap-tolerance", type=float, default=0.15)
    parser.add_argument("--min-mean-word-score", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("--workers must be between 1 and 12")

    root = args.root.resolve()
    annotation_dir, audio_dir = root / args.annotation_dir, root / args.audio_dir
    output_dir = root / args.output_dir
    if not annotation_dir.is_dir(): parser.error(f"Annotation directory not found: {annotation_dir}")
    if not audio_dir.is_dir(): parser.error(f"Audio directory not found: {audio_dir}")
    ffmpeg = shutil.which("ffmpeg")
    if not args.dry_run and not ffmpeg: parser.error("ffmpeg was not found on PATH")

    annotations = sorted(annotation_dir.glob("*.json"))
    audio_by_name = {p.name: p for p in audio_dir.glob("*.mp3")}
    if not annotations: parser.error(f"No JSON annotations found in {annotation_dir}")

    report = {"annotation_files": len(annotations), "matched_audio_files": 0,
              "missing_audio_references": [], "unannotated_audio_files": [],
              "segments_seen": 0, "segments_selected": 0, "segments_extracted": 0,
              "segments_skipped": Counter(), "speaker_segments": Counter(),
              "speaker_seconds": Counter(), "speaker_hours": {},
              "sample_rate": args.sample_rate, "workers": args.workers,
              "min_duration": args.min_duration, "max_duration": args.max_duration,
              "overlap_tolerance": args.overlap_tolerance,
              "decode_strategy": "one_ffmpeg_decode_per_episode"}
    referenced_audio: set[str] = set()
    output_rows, skipped_rows = [], []
    jobs_by_source: dict[Path, list[tuple[Path, float, float, dict]]] = defaultdict(list)

    for annotation_path in annotations:
        try:
            with annotation_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            report["segments_skipped"]["invalid_json"] += 1
            skipped_rows.append({"episode": annotation_path.stem, "segment_index": "",
                                 "reason": f"invalid_json: {exc}"})
            continue
        audio_name = data.get("audio")
        if not audio_name:
            report["missing_audio_references"].append(annotation_path.name)
            continue
        referenced_audio.add(audio_name)
        audio_path = audio_by_name.get(audio_name)
        if audio_path is None:
            report["missing_audio_references"].append(audio_name)
            continue
        report["matched_audio_files"] += 1
        speaker_map = data.get("speaker_global") or {}
        diarization = load_diarization_intervals(data)
        episode_id, prefix = annotation_path.stem, safe_name(annotation_path.stem)

        for index, segment in enumerate(data.get("segments") or []):
            report["segments_seen"] += 1
            reason = None
            try:
                start, end = float(segment["start"]), float(segment["end"])
            except (KeyError, TypeError, ValueError):
                start = end = 0.0
                reason = "invalid_timestamps"
            duration = end - start
            local_speaker = segment.get("speaker")
            if reason is None and not local_speaker: reason = "missing_segment_speaker"
            global_speaker = speaker_map.get(local_speaker) if local_speaker else None
            if reason is None and global_speaker not in TARGET_SPEAKERS: reason = "not_target_speaker"
            text = clean_text(segment.get("text"))
            if reason is None and not text: reason = "empty_transcript"
            if reason is None and (duration < args.min_duration or duration > args.max_duration):
                reason = "duration_out_of_range"
            word_speakers = get_word_speakers(segment)
            if reason is None and word_speakers and (len(word_speakers) != 1 or local_speaker not in word_speakers):
                reason = "mixed_or_conflicting_word_speakers"
            other_speakers, max_overlap = [], 0.0
            if reason is None:
                overlap, other_speakers, max_overlap = has_other_speaker_overlap(
                    start, end, str(local_speaker), diarization, args.overlap_tolerance)
                if overlap: reason = "overlapping_other_speaker"
            word_score = mean_word_score(segment)
            if (reason is None and args.min_mean_word_score is not None and
                    word_score is not None and word_score < args.min_mean_word_score):
                reason = "low_mean_word_score"
            if reason:
                report["segments_skipped"][reason] += 1
                skipped_rows.append({"episode": episode_id, "segment_index": index,
                                     "start": start, "end": end,
                                     "speaker": global_speaker or "", "reason": reason})
                continue

            speaker_dir = TARGET_SPEAKERS[global_speaker]
            segment_id = f"{prefix}__{index:06d}"
            relative_audio = Path("wavs") / speaker_dir / f"{segment_id}.wav"
            destination = output_dir / relative_audio
            row = {"segment_id": segment_id, "episode_id": episode_id,
                   "speaker": global_speaker, "local_speaker": local_speaker,
                   "start": round(start, 4), "end": round(end, 4),
                   "duration": round(duration, 4), "text": text,
                   "audio_path": relative_audio.as_posix(), "source_audio": audio_name,
                   "mean_word_score": round(word_score, 5) if word_score is not None else "",
                   "overlap": False, "max_other_speaker_overlap": round(max_overlap, 4)}
            report["segments_selected"] += 1
            report["speaker_segments"][global_speaker] += 1
            report["speaker_seconds"][global_speaker] += duration
            if args.dry_run:
                output_rows.append(row)
            elif destination.exists() and not args.overwrite and is_valid_existing_clip(destination, start, end, args.sample_rate):
                output_rows.append(row)
                report["segments_extracted"] += 1
            else:
                jobs_by_source[audio_path].append((destination, start, end, row))

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        episode_jobs = list(jobs_by_source.items())
        total_clips = sum(len(jobs) for _, jobs in episode_jobs)
        print(f"Extracting {total_clips:,} clips from {len(episode_jobs):,} episodes "
              f"using {args.workers} workers (one decode per episode)...", file=sys.stderr, flush=True)
        completed_episodes = completed_clips = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_episode, ffmpeg, source, jobs, args.sample_rate):
                       (source, len(jobs)) for source, jobs in episode_jobs}
            for future in as_completed(futures):
                source, clip_count = futures[future]
                completed_episodes += 1
                try:
                    successes, errors = future.result()
                except Exception as exc:
                    successes, errors = [], [{"episode": source.stem, "segment_index": "",
                                               "reason": f"worker_error: {exc}"}]
                output_rows.extend(successes)
                skipped_rows.extend(errors)
                report["segments_extracted"] += len(successes)
                report["segments_skipped"]["extraction_error"] += len(errors)
                completed_clips += clip_count
                if completed_episodes % 10 == 0 or completed_episodes == len(episode_jobs):
                    print(f"  Episodes {completed_episodes:,}/{len(episode_jobs):,}; "
                          f"clips processed {completed_clips:,}/{total_clips:,}; "
                          f"successful {report['segments_extracted']:,}",
                          file=sys.stderr, flush=True)

    output_rows.sort(key=lambda r: (r["episode_id"], r["start"], r["segment_id"]))
    report["unannotated_audio_files"] = sorted(n for n in audio_by_name if n not in referenced_audio)
    report["segments_skipped"] = dict(report["segments_skipped"])
    report["speaker_segments"] = dict(report["speaker_segments"])
    report["speaker_seconds"] = {k: round(v, 2) for k, v in report["speaker_seconds"].items()}
    report["speaker_hours"] = {k: round(v / 3600, 2) for k, v in report["speaker_seconds"].items()}
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "segments.jsonl").open("w", encoding="utf-8") as handle:
            for row in output_rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        write_csv(output_dir / "segments.csv", output_rows,
                  ["segment_id", "episode_id", "speaker", "local_speaker", "start", "end",
                   "duration", "text", "audio_path", "source_audio", "mean_word_score",
                   "overlap", "max_other_speaker_overlap"])
        write_csv(output_dir / "skipped_segments.csv", skipped_rows,
                  ["episode", "segment_index", "start", "end", "speaker", "reason"])
        with (output_dir / "extraction_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run: print("\nDry run: no audio was extracted and no output files were written.")
    else: print(f"\nDataset written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
