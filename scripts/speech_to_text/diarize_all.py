
import os
import re
import gc
import json
import argparse
from pathlib import Path

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

# Your shell listing shows /data, while your example
# paths mention /workspace/data. Detect either layout.
CANDIDATES = [Path("/workspace/data"), Path("/data")]

DATA_ROOT = next(
    (p for p in CANDIDATES
     if (p / "audio").is_dir() and (p / "transcripts").is_dir()),
    None
)

if DATA_ROOT is None:
    raise FileNotFoundError("Could not find audio/ and transcripts/ directories")

AUDIO_DIR = DATA_ROOT / "audio"
TRANSCRIPT_DIR = DATA_ROOT / "transcripts"
OUTPUT_DIR = DATA_ROOT / "diarized"

TRANSCRIPT_SUFFIX = "_large-v3-turbo-b64.txt"

LANGUAGE = "en"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# may need to rerun a couple of episodes in case this is incorrect
# for solo episodes etc.
MIN_SPEAKERS = 1
MAX_SPEAKERS = 8
# changed to max_speakers 8 after Episode 144

HF_TOKEN = os.environ["HF_TOKEN"]

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# --------------------------------------------------
# TRANSCRIPT PARSER
# --------------------------------------------------

TIMESTAMP_RE = re.compile(
    r"^\[\s*(\d+(?:\.\d+)?)\s*-\s*"
    r"(\d+(?:\.\d+)?)\]\s*(.*)$"
)


def load_transcript(path, duration):

    segments = []
    current = None

    def flush():
        if current and current["text"].strip():
            start = max(0.0, current["start"])
            end = min(duration, current["end"])

            if end > start:
                segments.append({
                    "start": start,
                    "end": end,
                    "text": current["text"].strip()
                })

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            match = TIMESTAMP_RE.match(line)

            if match:
                flush()

                start, end, text = match.groups()

                current = {
                    "start": float(start),
                    "end": float(end),
                    "text": text
                }

            elif current:
                # Handle possible wrapped transcript lines.
                current["text"] += " " + line

    flush()

    if not segments:
        raise ValueError(f"No valid transcript segments in {path}")

    return segments


# --------------------------------------------------
# PROCESS ONE EPISODE
# --------------------------------------------------

def process_episode(
    audio_path,
    transcript_path,
    align_model,
    align_metadata,
    diarize_model
):

    print(f"\nProcessing: {audio_path.name}")

    # Load audio
    audio = whisperx.load_audio(str(audio_path))
    duration = len(audio) / 16000

    # Load existing transcript
    segments = load_transcript(transcript_path, duration)

    print(f"  Duration: {duration / 60:.1f} minutes")
    print(f"  Transcript segments: {len(segments)}")

    # Forced alignment (NO transcription)
    print("  Aligning words...")

    aligned = whisperx.align(
        segments,
        align_model,
        align_metadata,
        audio,
        DEVICE,
        return_char_alignments=False
    )

    # Speaker diarization
    print("  Running diarization...")

    diarize_df, speaker_embeddings = diarize_model(
        audio,
        min_speakers=MIN_SPEAKERS,
        max_speakers=MAX_SPEAKERS,
        return_embeddings=True
    )

    # Assign speakers to aligned words
    result = whisperx.assign_word_speakers(
        diarize_df,
        aligned,
        speaker_embeddings
    )

    # Preserve raw diarization intervals, including overlaps
    diarization = []

    for _, row in diarize_df.iterrows():
        diarization.append({
            "start": float(row["start"]),
            "end": float(row["end"]),
            "speaker": str(row["speaker"])
        })

    # Output
    output = {
        "audio": audio_path.name,
        "transcript": transcript_path.name,
        "language": LANGUAGE,
        "duration": duration,
        "segments": result["segments"],
        "diarization": diarization,
        "speaker_embeddings": speaker_embeddings
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / f"{audio_path.stem}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    speakers = sorted({
        x["speaker"] for x in diarization
    })

    print(f"  Speakers: {speakers}")
    print(f"  Saved: {output_path}")

    return output_path


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--match", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Discover matching files
    pairs = []

    for transcript in sorted(
        TRANSCRIPT_DIR.glob(f"*{TRANSCRIPT_SUFFIX}")
    ):

        base = transcript.name[:-len(TRANSCRIPT_SUFFIX)]
        audio = AUDIO_DIR / f"{base}.mp3"

        if not audio.exists():
            print(f"Missing audio: {base}")
            continue

        if args.match and args.match.lower() not in base.lower():
            continue

        output = OUTPUT_DIR / f"{base}.json"

        if output.exists() and not args.force:
            continue

        pairs.append((audio, transcript))

    if args.limit:
        pairs = pairs[:args.limit]

    print(f"Data directory: {DATA_ROOT}")
    print(f"Found {len(pairs)} episodes to process")
    print(f"Device: {DEVICE}")

    if not pairs:
        return

    # Load models ONCE
    print("\nLoading alignment model...")

    align_model, align_metadata = whisperx.load_align_model(
        language_code=LANGUAGE,
        device=DEVICE
    )

    print("Loading diarization model...")

    diarize_model = DiarizationPipeline(
        token=HF_TOKEN,
        device=DEVICE
    )

    # Increase pyannote inference batch sizes
    diarize_model.model.segmentation_batch_size = 32
    diarize_model.model.embedding_batch_size = 32

    # Process episodes sequentially
    successful = 0
    failed = 0

    for index, (audio, transcript) in enumerate(pairs, 1):

        print(f"\n[{index}/{len(pairs)}]")

        try:
            process_episode(
                audio,
                transcript,
                align_model,
                align_metadata,
                diarize_model
            )
            successful += 1

        except Exception as e:
            failed += 1
            print(f"ERROR: {audio.name}: {e}")

        finally:
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print(f"Finished. Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
