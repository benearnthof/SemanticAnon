
import time
from pathlib import Path

from faster_whisper import WhisperModel, BatchedInferencePipeline

# --------------------------------------------------
# Configuration
# --------------------------------------------------

AUDIO_DIR = Path("/workspace/data/audio")
OUTPUT_DIR = Path("/workspace/data/transcripts")

MODEL_SIZE = "large-v3-turbo"
BATCH_SIZE = 64
COMPUTE_TYPE = "float16"
CPU_THREADS = 8

# --------------------------------------------------
# Find audio files
# --------------------------------------------------

audio_files = sorted(
    p for p in AUDIO_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() == ".mp3"
)

if not audio_files:
    raise SystemExit(f"No MP3 files found in {AUDIO_DIR}")

print("=" * 75)
print("MASS TRANSCRIPTION")
print("=" * 75)
print(f"Audio directory: {AUDIO_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Model: {MODEL_SIZE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Compute type: {COMPUTE_TYPE}")
print(f"Total episodes: {len(audio_files)}")
print("=" * 75)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load model once
# --------------------------------------------------

print("\nLoading model...")

load_start = time.time()

model = WhisperModel(
    MODEL_SIZE,
    device="cuda",
    device_index=0,
    compute_type=COMPUTE_TYPE,
    cpu_threads=CPU_THREADS,
)

pipeline = BatchedInferencePipeline(model=model)

print(f"Model loaded in {time.time() - load_start:.1f} seconds")

# --------------------------------------------------
# Transcription loop
# --------------------------------------------------

total_start = time.time()

completed = 0
skipped = 0
failed = 0

total_audio_seconds = 0
total_transcription_seconds = 0

for index, audio_file in enumerate(audio_files, start=1):

    relative_path = audio_file.relative_to(AUDIO_DIR)

    output_file = (
        OUTPUT_DIR
        / relative_path.parent
        / f"{audio_file.stem}_large-v3-turbo-b64.txt"
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    if output_file.exists() and output_file.stat().st_size > 0:
        skipped += 1
        print(f"\n[{index}/{len(audio_files)}] SKIPPED: {audio_file.name}")
        continue

    print("\n" + "=" * 75)
    print(f"[{index}/{len(audio_files)}] TRANSCRIBING")
    print(f"File: {audio_file.name}")
    print("=" * 75)

    temp_file = output_file.with_suffix(output_file.suffix + ".tmp")

    start = time.time()

    try:
        segments, info = pipeline.transcribe(
            str(audio_file),
            batch_size=BATCH_SIZE,
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=True,
        )

        segment_count = 0

        # Write progressively to a temporary file.
        # Rename only after successful completion.
        with temp_file.open("w", encoding="utf-8") as f:

            for segment in segments:

                line = (
                    f"[{segment.start:8.2f} - {segment.end:8.2f}] "
                    f"{segment.text.strip()}"
                )

                f.write(line + "\n")
                segment_count += 1

                if segment_count % 200 == 0:
                    print(
                        f"  Processed {segment.end / 60:.1f} / "
                        f"{info.duration / 60:.1f} audio minutes"
                    )

        elapsed = time.time() - start

        # Commit completed transcript
        temp_file.replace(output_file)

        completed += 1
        total_audio_seconds += info.duration
        total_transcription_seconds += elapsed

        speed = info.duration / elapsed if elapsed else 0

        print("\nCOMPLETED")
        print(f"Language: {info.language}")
        print(f"Duration: {info.duration / 60:.1f} minutes")
        print(f"Runtime: {elapsed:.2f} seconds")
        print(f"Speed: {speed:.2f}x realtime")
        print(f"Segments: {segment_count}")
        print(f"Output: {output_file}")

    except Exception as e:

        failed += 1

        if temp_file.exists():
            temp_file.unlink()

        print("\nERROR")
        print(f"Episode: {audio_file.name}")
        print(f"Reason: {repr(e)}")
        print("Continuing with the next episode...")

    # Overall progress
    processed = completed + skipped + failed

    print("\n" + "-" * 75)
    print(
        f"Progress: {processed}/{len(audio_files)} | "
        f"Completed: {completed} | "
        f"Skipped: {skipped} | "
        f"Failed: {failed}"
    )

    if total_transcription_seconds > 0:
        overall_speed = (
            total_audio_seconds / total_transcription_seconds
        )

        print(f"Average inference speed: {overall_speed:.2f}x realtime")

# --------------------------------------------------
# Final report
# --------------------------------------------------

total_elapsed = time.time() - total_start

print("\n" + "=" * 75)
print("MASS TRANSCRIPTION FINISHED")
print("=" * 75)

print(f"Total episodes found: {len(audio_files)}")
print(f"Successfully transcribed: {completed}")
print(f"Already completed/skipped: {skipped}")
print(f"Failed: {failed}")

print(f"Total runtime: {total_elapsed / 60:.1f} minutes")
print(f"Total audio transcribed: {total_audio_seconds / 3600:.2f} hours")

if total_transcription_seconds > 0:
    print(
        f"Average inference speed: "
        f"{total_audio_seconds / total_transcription_seconds:.2f}x realtime"
    )

print("=" * 75)
