#!/usr/bin/env python3
"""
Generate speaker predictions with the residual autoencoder classifier.

Manual annotations in data/diarized_annotated always take precedence over
model predictions. Unknown/manual labels are preserved exactly as before.

Expected project layout:
    project/
      data/diarized/*.json
      data/diarized_annotated/*.json
      models/speaker_classifier_residual_autoencoder.pt
      predictions/

Dependencies:
    pip install torch numpy
"""

import csv
import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch import nn


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DIARIZED_DIR = PROJECT_ROOT / "data" / "diarized"
ANNOTATED_DIR = PROJECT_ROOT / "data" / "diarized_annotated"
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "speaker_classifier_residual_autoencoder.pt"
)

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREDICTION_BATCH_SIZE = 512


# ---------------------------------------------------------
# Model architecture
# Must match train_residual_autoencoder.py exactly.
# ---------------------------------------------------------

class ResidualBlock(nn.Module):
    """Two-layer residual MLP block with pre-activation normalization."""

    def __init__(self, width, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.fc2 = nn.Linear(width, width)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activation(x)
        x = self.dropout(self.fc1(x))
        x = self.norm2(x)
        x = self.activation(x)
        x = self.dropout(self.fc2(x))
        return residual + x


class ResidualAutoencoderClassifier(nn.Module):
    """
    Same architecture used during training.

    The decoder is restored from the checkpoint for architectural
    compatibility, although prediction only requires the classifier head.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=256,
        latent_dim=128,
        dropout=0.15,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.encoder_block1 = ResidualBlock(hidden_dim, dropout)
        self.encoder_block2 = ResidualBlock(hidden_dim, dropout)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        h = self.input_layer(x)
        h = self.encoder_block1(h)
        h = self.encoder_block2(h)
        z = self.to_latent(h)
        logits = self.classifier(z)
        reconstruction = self.decoder(z)
        return logits, reconstruction, z


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


def l2_normalize_rows(x):
    """Match the L2 normalization applied to embeddings during training."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


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

    for path in sorted(ANNOTATED_DIR.glob("*.json")):
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


def load_model():
    """Load the trained residual autoencoder checkpoint."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Residual autoencoder checkpoint not found:\n{MODEL_PATH}\n"
            "Run train_residual_autoencoder.py first."
        )

    # The checkpoint contains only tensors and basic Python metadata.
    # The fallback supports older PyTorch releases without weights_only.
    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    required = {"model_state_dict", "model_config", "classes", "embedding_dimensions"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            "The checkpoint does not have the expected residual-autoencoder "
            f"format. Missing keys: {sorted(missing)}"
        )

    classes = list(checkpoint["classes"])
    if classes != CLASSES:
        raise ValueError(
            "Checkpoint class order does not match the prediction script.\n"
            f"Checkpoint: {classes}\nExpected:   {CLASSES}"
        )

    config = checkpoint["model_config"]
    expected_dimensions = int(checkpoint["embedding_dimensions"])

    if int(config["input_dim"]) != expected_dimensions:
        raise ValueError(
            "Checkpoint input_dim and embedding_dimensions do not agree."
        )

    model = ResidualAutoencoderClassifier(
        input_dim=int(config["input_dim"]),
        num_classes=int(config["num_classes"]),
        hidden_dim=int(config["hidden_dim"]),
        latent_dim=int(config["latent_dim"]),
        dropout=float(config["dropout"]),
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()

    return model, classes, expected_dimensions, checkpoint


@torch.inference_mode()
def predict_probability_batch(model, embeddings):
    """Return softmax probabilities for a batch of normalized embeddings."""

    if len(embeddings) == 0:
        return np.empty((0, len(CLASSES)), dtype=np.float32)

    x = torch.as_tensor(embeddings, dtype=torch.float32, device=DEVICE)
    logits, _, _ = model(x)
    probabilities = torch.softmax(logits, dim=1)
    return probabilities.cpu().numpy()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    print(f"Using device: {DEVICE}")
    print("Loading residual autoencoder classifier...")

    model, classes, expected_dimensions, checkpoint = load_model()

    print(f"Classes: {classes}")
    print(f"Embedding dimensions: {expected_dimensions}")
    print(f"Training samples: {checkpoint.get('training_samples', 'unknown')}")
    print(f"Training episodes: {checkpoint.get('training_episodes', 'unknown')}")

    print("\nAutomatic assignment thresholds:")
    for label, threshold in AUTO_ASSIGN_THRESHOLDS.items():
        print(f"  {label:18s}: {threshold:.1%}")
    print("  OTHER             : manual review only")

    # -----------------------------------------------------
    # Load existing human annotations
    # -----------------------------------------------------

    manual_annotations = load_manual_annotations()
    print(f"\nLoaded {len(manual_annotations)} existing manual labels.")

    # -----------------------------------------------------
    # Gather embeddings from all episodes
    # -----------------------------------------------------

    if not DIARIZED_DIR.exists():
        raise FileNotFoundError(f"Diarization directory not found: {DIARIZED_DIR}")

    episode_files = sorted(DIARIZED_DIR.glob("*.json"))
    print(f"\nFound {len(episode_files)} diarization files.")
    print("Collecting speaker embeddings...\n")

    pending = []
    rows = []
    stats = Counter()

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

            # Human labels are still retained and take precedence below.
            pending.append({
                "episode": episode,
                "speaker_id": speaker_id,
                "duration": float(durations.get(speaker_id, 0.0)),
                "embedding": embedding,
                "manual_label": manual_annotations.get((episode, speaker_id)),
            })

        if episode_index % 50 == 0 or episode_index == len(episode_files):
            print(f"Collected {episode_index}/{len(episode_files)} episodes...")

    # -----------------------------------------------------
    # Batch inference
    # -----------------------------------------------------

    print(f"\nRunning autoencoder predictions for {len(pending)} speakers...")

    for batch_start in range(0, len(pending), PREDICTION_BATCH_SIZE):
        batch_records = pending[batch_start:batch_start + PREDICTION_BATCH_SIZE]

        batch_embeddings = np.stack(
            [record["embedding"] for record in batch_records]
        ).astype(np.float32)

        # Important: training used row-wise L2 normalization.
        batch_embeddings = l2_normalize_rows(batch_embeddings)

        batch_probabilities = predict_probability_batch(model, batch_embeddings)

        for record, probabilities in zip(batch_records, batch_probabilities):
            probability_map = {
                label: float(probabilities[classes.index(label)])
                for label in CLASSES
            }

            predicted_label = max(probability_map, key=probability_map.get)
            confidence = probability_map[predicted_label]
            duration = record["duration"]
            manual_label = record["manual_label"]

            # Existing human annotations always take precedence.
            if manual_label is not None:
                status = "manually_annotated"
                final_label = manual_label
                stats["manually_annotated"] += 1

            # Automatically assign only explicitly approved identities.
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

            # Short speakers remain in the review pool.
            elif duration < MIN_SPEAKER_DURATION:
                status = "review_short"
                final_label = ""
                stats["review_short"] += 1

            else:
                status = "review_uncertain"
                final_label = ""
                stats["review_uncertain"] += 1

            row = {
                "episode": record["episode"],
                "speaker_id": record["speaker_id"],
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

        completed = min(batch_start + len(batch_records), len(pending))
        if completed == len(pending) or completed % 500 < len(batch_records):
            print(f"Predicted {completed}/{len(pending)} speakers...")

    # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

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


if __name__ == "__main__":
    main()
