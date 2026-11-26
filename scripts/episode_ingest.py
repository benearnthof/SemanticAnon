#!/usr/bin/env python3
"""
Incremental Patreon -> Faster Whisper -> WhisperX ingestion pipeline.

Expected project layout:

    project/
      scripts/
        episode_ingest.py
      cookies.txt
      secrets.env
      data/
        audio/
          downloaded.txt
        transcripts/
        diarized/

The script also recognizes existing /workspace/data and /data layouts.
Existing audio, transcript filenames, transcript formatting, and diarization
JSON structure are preserved.

Dependencies:
    pip install requests yt-dlp faster-whisper whisperx

The Hugging Face token is read from the HF_TOKEN environment variable or
from secrets.env in the project root. Never commit secrets.env to source control.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from typing import Any

import requests
from yt_dlp import YoutubeDL


# ---------------------------------------------------------------------------
# Configuration — defaults intentionally match the existing project scripts
# ---------------------------------------------------------------------------

COLLECTION_ID = "875184"
EXPECTED_EPISODES = 655

TRANSCRIPT_SUFFIX = "_large-v3-turbo-b64.txt"
WHISPER_MODEL_SIZE = "large-v3-turbo"
WHISPER_BATCH_SIZE = 64
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_CPU_THREADS = 8

LANGUAGE = "en"
MIN_SPEAKERS = 1
MAX_SPEAKERS = 8

TIMESTAMP_RE = re.compile(
    r"^\[\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\]\s*(.*)$"
)


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
# When episode_ingest.py lives in ./scripts, the project root is its parent.
# If it is run from the project root instead, keep that directory as root.
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name.lower() == "scripts" else SCRIPT_DIR


def find_data_root() -> Path:
    """Find the project's data directory, including existing remote layouts."""
    cwd = Path.cwd()

    candidates = [
        PROJECT_ROOT / "data",
        cwd / "data",
        Path("/workspace/data"),
        Path("/data"),
    ]

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)

        if (candidate / "audio").is_dir() and (
            candidate / "transcripts"
        ).is_dir():
            return candidate

    # Default for a fresh project or a project whose data folders have not
    # been created yet.
    return (PROJECT_ROOT / "data").resolve()


DATA_ROOT = find_data_root()

AUDIO_DIR = DATA_ROOT / "audio"
TRANSCRIPT_DIR = DATA_ROOT / "transcripts"
DIARIZED_DIR = DATA_ROOT / "diarized"

COOKIE_FILE = PROJECT_ROOT / "cookies.txt"
SECRETS_FILE = PROJECT_ROOT / "secrets.env"
ARCHIVE_FILE = AUDIO_DIR / "downloaded.txt"


def load_hf_token() -> str:
    """Read HF_TOKEN from the environment or a simple KEY=VALUE env file."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token.strip("\"'")

    if not SECRETS_FILE.exists():
        raise FileNotFoundError(
            f"Missing Hugging Face token. Set HF_TOKEN or create {SECRETS_FILE}"
        )

    for raw_line in SECRETS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        if key.strip() == "HF_TOKEN":
            token = value.strip().strip("\"'").strip()
            if token:
                return token

    raise RuntimeError(f"HF_TOKEN was not found in {SECRETS_FILE}")


def ensure_directories() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    DIARIZED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Stage 1: Patreon download
# ---------------------------------------------------------------------------

def create_patreon_session() -> requests.Session:
    if not COOKIE_FILE.exists():
        raise FileNotFoundError(
            f"Missing Patreon cookies: {COOKIE_FILE}\n"
            "Export your browser cookies in Netscape cookies.txt format."
        )

    cookie_jar = MozillaCookieJar(str(COOKIE_FILE))
    cookie_jar.load(ignore_discard=True, ignore_expires=True)

    session = requests.Session()
    session.cookies = cookie_jar
    session.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Content-Type": "application/vnd.api+json",
    })
    return session


def retrieve_episode_ids() -> list[str]:
    session = create_patreon_session()

    print("Retrieving Patreon collection metadata...")
    collection_url = f"https://www.patreon.com/api/collection/{COLLECTION_ID}"

    response = session.get(collection_url, timeout=30)
    response.raise_for_status()

    collection = response.json()["data"]["attributes"]
    thumbnail_url = collection["thumbnail"]["url"]

    campaign_match = re.search(r"/campaign/(\d+)", thumbnail_url)
    if not campaign_match:
        raise RuntimeError("Could not determine Patreon campaign ID")

    campaign_id = campaign_match.group(1)
    print(f"Campaign ID: {campaign_id}")

    url = (
        "https://www.patreon.com/api/posts"
        "?include=campaign"
        f"&filter[campaign_id]={campaign_id}"
        "&filter[contains_exclusive_posts]=true"
        "&filter[is_draft]=false"
        f"&filter[collection_id]={COLLECTION_ID}"
        "&filter[include_drops]=true"
        "&sort=collection_order"
        "&json-api-version=1.0"
    )

    episode_ids: list[str] = []
    print("Retrieving episode list...")

    while url:
        response = session.get(url, timeout=60)
        response.raise_for_status()
        payload = response.json()

        for post in payload.get("data", []):
            episode_ids.append(str(post["id"]))

        url = payload.get("links", {}).get("next")
        print(f"  Retrieved {len(episode_ids)} episodes...")

    print(f"Found {len(episode_ids)} episodes (expected {EXPECTED_EPISODES}).")
    if len(episode_ids) != EXPECTED_EPISODES:
        print("WARNING: Episode count differs from the historical expectation.")

    return episode_ids


def download_new_episodes() -> None:
    episode_ids = retrieve_episode_ids()

    options = {
        "cookiefile": str(COOKIE_FILE),
        "format": "bestaudio/best",
        "outtmpl": str(AUDIO_DIR / "%(title)s [%(id)s].%(ext)s"),
        "writeinfojson": True,
        "continuedl": True,
        "ignoreerrors": True,
        "download_archive": str(ARCHIVE_FILE),
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 5,
        "sleep_interval": 0.1,
        "max_sleep_interval": 0.5,
        "quiet": False,
        "no_warnings": False,
    }

    print("\nChecking Patreon archive and downloading new episodes...\n")

    with YoutubeDL(options) as ydl:
        for index, episode_id in enumerate(episode_ids, 1):
            print(f"\n[{index}/{len(episode_ids)}] Episode {episode_id}")
            try:
                ydl.download([
                    f"https://www.patreon.com/posts/{episode_id}"
                ])
            except Exception as exc:
                print(f"Download error for {episode_id}: {exc}")

    print("\nDownload stage finished.")
    print(f"Audio directory: {AUDIO_DIR}")
    print(f"Download archive: {ARCHIVE_FILE}")


# ---------------------------------------------------------------------------
# Stage 2: Faster Whisper transcription
# ---------------------------------------------------------------------------

def discover_audio_files(match: str | None = None) -> list[Path]:
    files = sorted(
        path for path in AUDIO_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() == ".mp3"
    )

    if match:
        needle = match.casefold()
        files = [path for path in files if needle in path.name.casefold()]

    return files


def transcript_path_for(audio_path: Path) -> Path:
    relative = audio_path.relative_to(AUDIO_DIR)
    return (
        TRANSCRIPT_DIR
        / relative.parent
        / f"{audio_path.stem}{TRANSCRIPT_SUFFIX}"
    )


def transcribe_episodes(
    audio_files: list[Path],
    *,
    batch_size: int,
    compute_type: str,
    cpu_threads: int,
    device: str,
    force: bool,
) -> None:
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    pending = []
    for audio_file in audio_files:
        output_file = transcript_path_for(audio_file)
        if output_file.exists() and output_file.stat().st_size > 0 and not force:
            print(f"TRANSCRIPT EXISTS: {audio_file.name}")
        else:
            pending.append(audio_file)

    if not pending:
        print("\nNo episodes require transcription.")
        return

    print("\n" + "=" * 75)
    print("FASTER WHISPER TRANSCRIPTION")
    print("=" * 75)
    print(f"Model: {WHISPER_MODEL_SIZE}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Compute type: {compute_type}")
    print(f"Episodes requiring transcription: {len(pending)}")

    print("\nLoading Faster Whisper model...")
    load_start = time.time()

    model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device=device,
        device_index=0 if device == "cuda" else 0,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
    )
    pipeline = BatchedInferencePipeline(model=model)

    print(f"Model loaded in {time.time() - load_start:.1f} seconds")

    completed = 0
    failed = 0
    total_audio_seconds = 0.0
    total_inference_seconds = 0.0

    for index, audio_file in enumerate(pending, 1):
        output_file = transcript_path_for(audio_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = output_file.with_suffix(output_file.suffix + ".tmp")

        print("\n" + "=" * 75)
        print(f"[{index}/{len(pending)}] TRANSCRIBING: {audio_file.name}")
        print("=" * 75)

        start = time.time()

        try:
            segments, info = pipeline.transcribe(
                str(audio_file),
                batch_size=batch_size,
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=True,
            )

            segment_count = 0
            with temp_file.open("w", encoding="utf-8") as handle:
                for segment in segments:
                    line = (
                        f"[{segment.start:8.2f} - {segment.end:8.2f}] "
                        f"{segment.text.strip()}"
                    )
                    handle.write(line + "\n")
                    segment_count += 1

                    if segment_count % 200 == 0:
                        print(
                            f"  Processed {segment.end / 60:.1f} / "
                            f"{info.duration / 60:.1f} audio minutes"
                        )

            elapsed = time.time() - start
            temp_file.replace(output_file)

            completed += 1
            total_audio_seconds += info.duration
            total_inference_seconds += elapsed

            speed = info.duration / elapsed if elapsed else 0.0
            print(f"COMPLETED | {info.duration / 60:.1f} min | "
                  f"{elapsed:.1f}s | {speed:.2f}x realtime")
            print(f"Language: {info.language} | Segments: {segment_count}")
            print(f"Output: {output_file}")

        except Exception as exc:
            failed += 1
            if temp_file.exists():
                temp_file.unlink()
            print(f"ERROR: {audio_file.name}: {exc!r}")

        if total_inference_seconds:
            overall_speed = total_audio_seconds / total_inference_seconds
            print(f"Successful: {completed} | Failed: {failed} | "
                  f"Average speed: {overall_speed:.2f}x realtime")

    print("\nTranscription stage finished.")
    print(f"Completed: {completed} | Failed: {failed}")


# ---------------------------------------------------------------------------
# Stage 3: WhisperX forced alignment and pyannote diarization
# ---------------------------------------------------------------------------

def load_transcript(path: Path, duration: float) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def flush() -> None:
        if current and current["text"].strip():
            start = max(0.0, current["start"])
            end = min(duration, current["end"])
            if end > start:
                segments.append({
                    "start": start,
                    "end": end,
                    "text": current["text"].strip(),
                })

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            match = TIMESTAMP_RE.match(line)
            if match:
                flush()
                start, end, text = match.groups()
                current = {
                    "start": float(start),
                    "end": float(end),
                    "text": text,
                }
            elif current is not None:
                current["text"] += " " + line

    flush()

    if not segments:
        raise ValueError(f"No valid transcript segments found in {path}")

    return segments


def make_json_serializable(value: Any) -> Any:
    """Convert tensors/NumPy values recursively without changing JSON shape."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def diarized_path_for(audio_path: Path) -> Path:
    relative = audio_path.relative_to(AUDIO_DIR)
    return DIARIZED_DIR / relative.parent / f"{audio_path.stem}.json"


def diarize_episodes(
    audio_files: list[Path],
    *,
    hf_token: str,
    device: str,
    min_speakers: int,
    max_speakers: int,
    force: bool,
) -> None:
    import torch
    import whisperx
    from whisperx.diarize import DiarizationPipeline

    pending: list[tuple[Path, Path]] = []

    for audio_file in audio_files:
        transcript_file = transcript_path_for(audio_file)
        output_file = diarized_path_for(audio_file)

        if not transcript_file.exists() or transcript_file.stat().st_size == 0:
            print(f"MISSING TRANSCRIPT: {audio_file.name}")
            continue

        if output_file.exists() and output_file.stat().st_size > 0 and not force:
            print(f"DIARIZATION EXISTS: {audio_file.name}")
            continue

        pending.append((audio_file, transcript_file))

    if not pending:
        print("\nNo episodes require diarization.")
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("\n" + "=" * 75)
    print("WHISPERX ALIGNMENT + DIARIZATION")
    print("=" * 75)
    print(f"Device: {device}")
    print(f"Episodes requiring diarization: {len(pending)}")
    print(f"Speaker range: {min_speakers}-{max_speakers}")

    print("\nLoading alignment model...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=LANGUAGE,
        device=device,
    )

    print("Loading pyannote diarization model...")
    diarize_model = DiarizationPipeline(
        token=hf_token,
        device=device,
    )

    # Retain the batch-size tuning from the existing diarization script.
    diarize_model.model.segmentation_batch_size = 32
    diarize_model.model.embedding_batch_size = 32

    successful = 0
    failed = 0

    for index, (audio_path, transcript_file) in enumerate(pending, 1):
        output_file = diarized_path_for(audio_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = output_file.with_suffix(output_file.suffix + ".tmp")

        print("\n" + "=" * 75)
        print(f"[{index}/{len(pending)}] DIARIZING: {audio_path.name}")
        print("=" * 75)

        try:
            audio = whisperx.load_audio(str(audio_path))
            duration = len(audio) / 16000
            segments = load_transcript(transcript_file, duration)

            print(f"Duration: {duration / 60:.1f} min")
            print(f"Transcript segments: {len(segments)}")

            print("Aligning words...")
            aligned = whisperx.align(
                segments,
                align_model,
                align_metadata,
                audio,
                device,
                return_char_alignments=False,
            )

            print("Running speaker diarization...")
            diarize_df, speaker_embeddings = diarize_model(
                audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_embeddings=True,
            )

            result = whisperx.assign_word_speakers(
                diarize_df,
                aligned,
                speaker_embeddings,
            )

            # Preserve the original raw diarization intervals, including
            # overlapping intervals, as the existing project expects.
            diarization = []
            for _, row in diarize_df.iterrows():
                diarization.append({
                    "start": float(row["start"]),
                    "end": float(row["end"]),
                    "speaker": str(row["speaker"]),
                })

            output = {
                "audio": audio_path.name,
                "transcript": transcript_file.name,
                "language": LANGUAGE,
                "duration": duration,
                "segments": result["segments"],
                "diarization": diarization,
                "speaker_embeddings": speaker_embeddings,
            }

            output = make_json_serializable(output)

            with temp_file.open("w", encoding="utf-8") as handle:
                json.dump(output, handle, ensure_ascii=False)

            temp_file.replace(output_file)

            speakers = sorted({
                item["speaker"] for item in diarization
            })

            successful += 1
            print(f"Speakers: {speakers}")
            print(f"Saved: {output_file}")

        except Exception as exc:
            failed += 1
            if temp_file.exists():
                temp_file.unlink()
            print(f"ERROR: {audio_path.name}: {exc!r}")

        finally:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    print("\nDiarization stage finished.")
    print(f"Successful: {successful} | Failed: {failed}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally download, transcribe, and diarize Patreon episodes."
    )

    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N matching local audio files.")
    parser.add_argument("--match", type=str, default=None,
                        help="Only process audio filenames containing this string.")

    parser.add_argument("--skip-download", action="store_true",
                        help="Do not query Patreon or download episodes.")
    parser.add_argument("--skip-transcription", action="store_true",
                        help="Do not run Faster Whisper.")
    parser.add_argument("--skip-diarization", action="store_true",
                        help="Do not run WhisperX/pyannote.")

    parser.add_argument("--force-transcription", action="store_true",
                        help="Regenerate transcripts even if they already exist.")
    parser.add_argument("--force-diarization", action="store_true",
                        help="Regenerate diarization JSON even if it already exists.")

    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=WHISPER_BATCH_SIZE)
    parser.add_argument("--compute-type", default=None,
                        help="Faster Whisper compute type (defaults to float16 on CUDA, int8 on CPU).")
    parser.add_argument("--cpu-threads", type=int, default=WHISPER_CPU_THREADS)
    parser.add_argument("--min-speakers", type=int, default=MIN_SPEAKERS)
    parser.add_argument("--max-speakers", type=int, default=MAX_SPEAKERS)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_directories()

    print("=" * 75)
    print("PODCAST EPISODE INGESTION")
    print("=" * 75)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Audio: {AUDIO_DIR}")
    print(f"Transcripts: {TRANSCRIPT_DIR}")
    print(f"Diarized: {DIARIZED_DIR}")

    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "--device cuda was requested, but torch.cuda.is_available() is false."
                )
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for CUDA processing.") from exc

    compute_type = args.compute_type or (
        WHISPER_COMPUTE_TYPE if device == "cuda" else "int8"
    )
    print(f"Device: {device}")
    print(f"Whisper compute type: {compute_type}")

    # Stage 1
    if not args.skip_download:
        download_new_episodes()
    else:
        print("\nSkipping download stage.")

    # Discover local audio after downloading so newly acquired episodes are
    # included in this same run.
    audio_files = discover_audio_files(args.match)

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        audio_files = audio_files[:args.limit]

    print(f"\nLocal MP3 files selected: {len(audio_files)}")

    # Stage 2
    if not args.skip_transcription:
        transcribe_episodes(
            audio_files,
            batch_size=args.batch_size,
            compute_type=compute_type,
            cpu_threads=args.cpu_threads,
            device=device,
            force=args.force_transcription,
        )
    else:
        print("\nSkipping transcription stage.")

    # Stage 3
    if not args.skip_diarization:
        # Only load/read the token if diarization is actually requested.
        token = load_hf_token()
        diarize_episodes(
            audio_files,
            hf_token=token,
            device=device,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            # A regenerated transcript invalidates its previous alignment.
            force=(args.force_diarization or args.force_transcription),
        )
    else:
        print("\nSkipping diarization stage.")

    print("\n" + "=" * 75)
    print("INGESTION RUN FINISHED")
    print("=" * 75)
    print(f"Audio files selected: {len(audio_files)}")
    print(f"Audio directory: {AUDIO_DIR}")
    print(f"Transcript directory: {TRANSCRIPT_DIR}")
    print(f"Diarization directory: {DIARIZED_DIR}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
