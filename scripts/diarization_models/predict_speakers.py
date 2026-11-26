import csv
import json
from pathlib import Path
from collections import Counter

import joblib
import numpy as np


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIARIZED_DIR = PROJECT_ROOT / "data" / "diarized"
ANNOTATED_DIR = PROJECT_ROOT / "data" / "diarized_annotated"
MODEL_PATH = PROJECT_ROOT / "models" / "speaker_classifier.joblib"

OUTPUT_DIR = PROJECT_ROOT / "predictions"
OUTPUT_PATH = OUTPUT_DIR / "speaker_predictions.csv"

# Class-specific minimum probabilities for automatic assignment.
# OTHER is deliberately excluded: unknown guests must be identified
# manually using episode metadata/descriptions.
AUTO_ASSIGN_THRESHOLDS = {
    "BRACE": 0.875,
    "LIZ": 0.875,
    "YOUNG CHOMSKY": 0.90,
    "SOUNDBITE": 0.875,
}

# Retained as a review-prioritization threshold only.
# It does not prevent a sufficiently confident prediction from being
# automatically assigned.
MIN_SPEAKER_DURATION = 300.0  # 5 minutes

CLASSES = [
    "BRACE",
    "LIZ",
    "YOUNG CHOMSKY",
    "SOUNDBITE",
    "OTHER",
]


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_embedding(embedding):
    if embedding is None:
        return None

    embedding = np.asarray(embedding, dtype=np.float32)

    if embedding.ndim == 2:
        embedding = embedding.mean(axis=0)

    if embedding.ndim != 1:
        return None

    if not np.all(np.isfinite(embedding)):
        return None

    return embedding


def normalize_label(label):
    """Normalize known identities while preserving unknown manual labels."""

    if not label:
        return None

    label = str(label).strip()
    normalized = label.upper().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())

    aliases = {
        "BRACE": "BRACE",
        "LIZ": "LIZ",
        "YOUNG CHOMSKY": "YOUNG CHOMSKY",
        "SOUNDBITE": "SOUNDBITE",
        "OTHER": "OTHER",
    }

    return aliases.get(normalized, label)


def get_speaker_durations(data):
    """Calculate total diarized speaking time per local speaker."""

    durations = Counter()

    # Prefer the original diarization output.
    segments = data.get("diarization", [])

    # Fall back to transcript segments if necessary.
    if not segments:
        segments = data.get("segments", [])

    for segment in segments:
        speaker = segment.get("speaker")

        if not speaker:
            continue

        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue

        durations[speaker] += max(0.0, end - start)

    return dict(durations)


def load_manual_annotations():
    """Load existing human labels, keyed by episode and speaker."""

    annotations = {}

    if not ANNOTATED_DIR.exists():
        return annotations

    for path in ANNOTATED_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Could not read annotation {path.name}: {e}")
            continue

        episode = path.stem

        for speaker_id, label in data.get("speaker_global", {}).items():
            if label:
                annotations[(episode, speaker_id)] = normalize_label(label)

    return annotations


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("Loading classifier...")

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
classes = list(model.named_steps["classifier"].classes_)
expected_dimensions = bundle["embedding_dimensions"]

missing_classes = set(CLASSES) - set(classes)
if missing_classes:
    raise ValueError(
        "The loaded classifier is missing expected classes: "
        f"{sorted(missing_classes)}. "
        "Retrain the five-class model before generating predictions."
    )

print(f"Classes: {classes}")
print(f"Embedding dimensions: {expected_dimensions}")

print("\nAutomatic assignment thresholds:")
for label, threshold in AUTO_ASSIGN_THRESHOLDS.items():
    print(f"  {label:18s}: {threshold:.1%}")
print("  OTHER             : manual review only")


# ---------------------------------------------------------
# Load existing human annotations
# ---------------------------------------------------------

manual_annotations = load_manual_annotations()

print(f"\nLoaded {len(manual_annotations)} existing manual labels.")


# ---------------------------------------------------------
# Predict all episodes
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
stats = Counter()

episode_files = sorted(DIARIZED_DIR.glob("*.json"))

print(f"\nFound {len(episode_files)} diarization files.")
print("Generating predictions...\n")

for episode_index, path in enumerate(episode_files, start=1):

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Could not read {path.name}: {e}")
        stats["file_errors"] += 1
        continue

    episode = path.stem

    embeddings = data.get("speaker_embeddings", {})
    durations = get_speaker_durations(data)

    for speaker_id, raw_embedding in embeddings.items():

        embedding = load_embedding(raw_embedding)

        if embedding is None:
            stats["invalid_embeddings"] += 1
            continue

        if len(embedding) != expected_dimensions:
            print(
                f"Dimension mismatch: {episode} / {speaker_id} "
                f"({len(embedding)} instead of {expected_dimensions})"
            )
            stats["dimension_errors"] += 1
            continue

        duration = float(durations.get(speaker_id, 0.0))

        # Get model probabilities.
        probabilities = model.predict_proba(
            embedding.reshape(1, -1)
        )[0]

        probability_map = {
            label: float(probabilities[classes.index(label)])
            for label in CLASSES
        }

        predicted_label = max(
            probability_map,
            key=probability_map.get,
        )

        confidence = probability_map[predicted_label]

        # Existing human annotations always take precedence.
        manual_label = manual_annotations.get((episode, speaker_id))

        if manual_label is not None:
            status = "manually_annotated"
            final_label = manual_label
            stats["manually_annotated"] += 1

        # Automatically assign only the explicitly approved identities.
        # Duration does not gate automatic assignment.
        elif (
            predicted_label in AUTO_ASSIGN_THRESHOLDS
            and confidence >= AUTO_ASSIGN_THRESHOLDS[predicted_label]
        ):
            status = "auto_assigned"
            final_label = predicted_label
            stats[f"auto_{predicted_label}"] += 1

        # OTHER is never assigned automatically, regardless of confidence.
        elif predicted_label == "OTHER":
            status = "likely_other_review"
            final_label = ""
            stats["likely_other_review"] += 1

        # Speakers below five minutes remain in the review pool, but are
        # prioritized as short speakers rather than discarded.
        elif duration < MIN_SPEAKER_DURATION:
            status = "review_short"
            final_label = ""
            stats["review_short"] += 1

        else:
            status = "review_uncertain"
            final_label = ""
            stats["review_uncertain"] += 1

        row = {
            "episode": episode,
            "speaker_id": speaker_id,
            "duration_seconds": round(duration, 2),
            "duration_minutes": round(duration / 60, 2),
            "predicted_label": predicted_label,
            "confidence": round(confidence, 5),
            "manual_label": manual_label or "",
            "final_label": final_label,
            "status": status,
        }

        # Emit one probability column for every classifier class.
        for label in CLASSES:
            column_name = "prob_" + label.replace(" ", "_")
            row[column_name] = round(probability_map[label], 5)

        rows.append(row)

    if episode_index % 50 == 0 or episode_index == len(episode_files):
        print(f"Processed {episode_index}/{len(episode_files)} episodes...")


# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------

probability_columns = [
    "prob_" + label.replace(" ", "_")
    for label in CLASSES
]

fieldnames = [
    "episode",
    "speaker_id",
    "duration_seconds",
    "duration_minutes",
    "predicted_label",
    "confidence",
    *probability_columns,
    "manual_label",
    "final_label",
    "status",
]

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PREDICTION SUMMARY")
print("=" * 60)

print(f"\nEpisodes processed: {len(episode_files)}")
print(f"Speaker predictions: {len(rows)}")

print("\nResults:")
for key, value in sorted(stats.items()):
    print(f"  {key:25s}: {value}")

print("\nAutomatic assignment thresholds:")
for label, threshold in AUTO_ASSIGN_THRESHOLDS.items():
    print(f"  {label:18s}: {threshold:.1%}")
print("  OTHER             : manual review only")

print(f"\nCSV saved to:\n{OUTPUT_PATH}")
print("\nDone.")
