#!/usr/bin/env python3
"""
Train and evaluate a residual autoencoder speaker classifier.

This is a PyTorch replacement for the logistic-regression training script.
It preserves the original data format, episode-level GroupShuffleSplit,
five-class label mapping, confidence-threshold analysis, and JSON report.

Dependencies:
    pip install torch numpy scikit-learn joblib

Run from anywhere:
    python scripts/train_residual_autoencoder.py

Expected project layout:
    project/
      data/diarized_annotated/*.json
      models/

Each JSON file should contain:
    speaker_embeddings: {speaker_id: embedding}
    speaker_global:     {speaker_id: manual_label}

The saved .pt checkpoint contains the model state and architecture metadata.
"""

import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNOTATED_DIR = PROJECT_ROOT / "data" / "diarized_annotated"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "speaker_classifier_residual_autoencoder.pt"
REPORT_PATH = MODEL_DIR / "training_report_residual_autoencoder.json"

TEST_SIZE = 0.20
RANDOM_STATE = 42
SPLIT_ATTEMPTS = 200

CONFIDENCE_THRESHOLDS = [0.90, 0.95, 0.98, 0.99]
CLASSES = ["BRACE", "LIZ", "YOUNG CHOMSKY", "SOUNDBITE", "OTHER"]

# Architecture / training parameters.
LATENT_DIM = 128
HIDDEN_DIM = 256
DROPOUT = 0.15
BATCH_SIZE = 64
EPOCHS = 250
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RECONSTRUCTION_WEIGHT = 0.10
GRAD_CLIP_NORM = 2.0
NUM_WORKERS = 0


# ---------------------------------------------------------
# Reproducibility and device
# ---------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def normalize_label(label):
    """Map manual labels to the five classifier classes."""
    if label is None:
        return None

    label = str(label).strip().upper()
    if not label:
        return None

    canonical = {
        "BRACE": "BRACE",
        "LIZ": "LIZ",
        "YOUNG CHOMSKY": "YOUNG CHOMSKY",
        "YOUNG_CHOMSKY": "YOUNG CHOMSKY",
        "YOUNG-CHOMSKY": "YOUNG CHOMSKY",
        "SOUNDBITE": "SOUNDBITE",
    }
    return canonical.get(label, "OTHER")


def load_embedding(embedding):
    """Convert an embedding to a valid 1D float32 numpy array."""
    if embedding is None:
        return None

    embedding = np.asarray(embedding, dtype=np.float32)

    # Handle a possible collection of embeddings.
    if embedding.ndim == 2:
        embedding = embedding.mean(axis=0)

    if embedding.ndim != 1 or not np.all(np.isfinite(embedding)):
        return None

    return embedding


def l2_normalize_rows(x):
    """Equivalent in spirit to sklearn Normalizer(norm='l2')."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def choose_group_split(X, y, groups):
    """Find a group-disjoint split with all classes in training, ideally test."""
    splitter = GroupShuffleSplit(
        n_splits=SPLIT_ATTEMPTS,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    best_split = None
    best_test_class_count = -1

    for train_idx, test_idx in splitter.split(X, y, groups=groups):
        train_classes = set(y[train_idx])
        test_classes = set(y[test_idx])

        if train_classes != set(CLASSES):
            continue

        if len(test_classes) > best_test_class_count:
            best_split = (train_idx, test_idx)
            best_test_class_count = len(test_classes)

        if test_classes == set(CLASSES):
            return train_idx, test_idx, True

    if best_split is None:
        episode_counts = defaultdict(set)
        for label, episode in zip(y, groups):
            episode_counts[label].add(episode)
        details = ", ".join(
            f"{label}: {len(episode_counts[label])} episode(s)"
            for label in CLASSES
        )
        raise RuntimeError(
            "Could not create a training split containing all five classes. "
            f"Episode coverage: {details}"
        )

    return best_split[0], best_split[1], False


# ---------------------------------------------------------
# Residual autoencoder architecture
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
    Shared residual encoder, latent bottleneck, reconstruction decoder,
    and supervised classification head.

    The reconstruction objective regularizes the representation; the
    classification objective makes the latent representation discriminative.
    """

    def __init__(
        self,
        input_dim,
        num_classes,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        dropout=DROPOUT,
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


def make_model(input_dim):
    return ResidualAutoencoderClassifier(
        input_dim=input_dim,
        num_classes=len(CLASSES),
    )


def class_weights_for(y_train):
    """Inverse-frequency weights, normalized to mean approximately one."""
    counts = Counter(y_train)
    weights = np.array(
        [len(y_train) / (len(CLASSES) * counts[label]) for label in CLASSES],
        dtype=np.float32,
    )
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def train_model(X_train, y_train, input_dim, verbose=True):
    """Train a fresh residual autoencoder classifier."""
    label_to_index = {label: i for i, label in enumerate(CLASSES)}
    y_indices = np.asarray([label_to_index[label] for label in y_train], dtype=np.int64)

    x_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_indices, dtype=torch.long)

    dataset = TensorDataset(x_tensor, y_tensor)
    generator = torch.Generator()
    generator.manual_seed(RANDOM_STATE)

    loader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(dataset)),
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
        drop_last=False,
    )

    model = make_model(input_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LEARNING_RATE * 0.05,
    )

    class_weights = class_weights_for(y_train)
    classification_loss = nn.CrossEntropyLoss(weight=class_weights)
    reconstruction_loss = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_cls = 0.0
        running_recon = 0.0
        seen = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits, reconstructed, _ = model(batch_x)

            cls_loss = classification_loss(logits, batch_y)
            recon_loss = reconstruction_loss(reconstructed, batch_x)
            loss = cls_loss + RECONSTRUCTION_WEIGHT * recon_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            batch_size = len(batch_x)
            running_loss += loss.item() * batch_size
            running_cls += cls_loss.item() * batch_size
            running_recon += recon_loss.item() * batch_size
            seen += batch_size

        scheduler.step()

        if verbose and (epoch == 0 or (epoch + 1) % 25 == 0 or epoch == EPOCHS - 1):
            print(
                f"  Epoch {epoch + 1:3d}/{EPOCHS} | "
                f"loss={running_loss / seen:.4f} | "
                f"classification={running_cls / seen:.4f} | "
                f"reconstruction={running_recon / seen:.4f}"
            )

    return model


@torch.no_grad()
def predict_probabilities(model, X):
    model.eval()
    x_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(x_tensor),
        batch_size=512,
        shuffle=False,
    )

    all_probabilities = []
    for (batch_x,) in loader:
        batch_x = batch_x.to(DEVICE)
        logits, _, _ = model(batch_x)
        probabilities = torch.softmax(logits, dim=1)
        all_probabilities.append(probabilities.cpu().numpy())

    if not all_probabilities:
        return np.empty((0, len(CLASSES)), dtype=np.float32)
    return np.concatenate(all_probabilities, axis=0)


# ---------------------------------------------------------
# Load annotated data
# ---------------------------------------------------------

def main():
    print(f"Using device: {DEVICE}")
    print("Loading annotated episodes...")

    if not ANNOTATED_DIR.exists():
        raise FileNotFoundError(f"Annotated data directory not found: {ANNOTATED_DIR}")

    X = []
    y = []
    groups = []
    metadata = []

    for path in sorted(ANNOTATED_DIR.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Could not read {path.name}: {exc}")
            continue

        embeddings = data.get("speaker_embeddings", {})
        labels = data.get("speaker_global", {})

        for speaker_id, label in labels.items():
            normalized_label = normalize_label(label)
            if normalized_label is None:
                continue

            embedding = load_embedding(embeddings.get(speaker_id))
            if embedding is None:
                continue

            X.append(embedding)
            y.append(normalized_label)
            groups.append(path.stem)
            metadata.append({
                "episode": path.stem,
                "speaker_id": speaker_id,
                "true_label": normalized_label,
            })

    if not X:
        raise RuntimeError("No usable training data found.")

    dimensions = {len(row) for row in X}
    if len(dimensions) != 1:
        raise RuntimeError(f"Inconsistent embedding dimensions found: {sorted(dimensions)}")

    X = np.asarray(X, dtype=np.float32)
    X = l2_normalize_rows(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    print(f"\nLoaded {len(X)} speaker embeddings.")
    print(f"Annotated episodes: {len(set(groups))}")
    print(f"Embedding dimensions: {X.shape[1]}")

    missing_classes = set(CLASSES) - set(y)
    if missing_classes:
        raise RuntimeError(
            "The following classes have no usable labeled embeddings: "
            f"{sorted(missing_classes)}. Check speaker_global labels and embeddings."
        )

    print("\nClass distribution:")
    for label in CLASSES:
        print(f"  {label:16s}: {int(np.sum(y == label))}")

    class_episode_counts = {
        label: len(set(groups[y == label]))
        for label in CLASSES
    }
    print("\nEpisodes represented per class:")
    for label in CLASSES:
        print(f"  {label:16s}: {class_episode_counts[label]}")

    # -----------------------------------------------------
    # Episode-level train/test split
    # -----------------------------------------------------

    train_idx, test_idx, all_classes_in_test = choose_group_split(X, y, groups)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("\nEpisode-level split:")
    print(f"  Training episodes: {len(set(groups[train_idx]))}")
    print(f"  Testing episodes:  {len(set(groups[test_idx]))}")
    print(f"  Training speakers: {len(train_idx)}")
    print(f"  Testing speakers:  {len(test_idx)}")
    print("\nTraining class distribution:")
    print(Counter(y_train))
    print("\nTesting class distribution:")
    print(Counter(y_test))

    if not all_classes_in_test:
        absent = sorted(set(CLASSES) - set(y_test))
        print(
            "\nWARNING: The group-disjoint test split does not contain examples of "
            f"{absent}. Their held-out precision/recall cannot be estimated."
        )

    # -----------------------------------------------------
    # Train evaluation model
    # -----------------------------------------------------

    print("\nTraining residual autoencoder classifier...")
    evaluation_model = train_model(X_train, y_train, X.shape[1])

    probabilities = predict_probabilities(evaluation_model, X_test)
    predicted_indices = probabilities.argmax(axis=1)
    y_pred = np.asarray([CLASSES[i] for i in predicted_indices])

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=CLASSES,
        zero_division=0,
    ))

    print("Confusion matrix:")
    print("Labels:", CLASSES)
    print(confusion_matrix(y_test, y_pred, labels=CLASSES))

    # -----------------------------------------------------
    # Held-out predictions and probabilities
    # -----------------------------------------------------

    print("\nHeld-out speaker confidence:")
    test_predictions = []

    for i, original_idx in enumerate(test_idx):
        probability_map = {
            label: float(probabilities[i][class_index])
            for class_index, label in enumerate(CLASSES)
        }
        predicted_label = str(y_pred[i])
        confidence = probability_map[predicted_label]

        record = {
            **metadata[original_idx],
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": probability_map,
            "correct": bool(predicted_label == y_test[i]),
        }
        test_predictions.append(record)

        print(
            f"  {record['episode']} | "
            f"{record['speaker_id']} | "
            f"true={record['true_label']} | "
            f"pred={predicted_label} | "
            f"confidence={confidence:.3f}"
        )

    # -----------------------------------------------------
    # Threshold analysis
    # -----------------------------------------------------

    threshold_report = {}
    print("\nConfidence threshold analysis:")

    for threshold in CONFIDENCE_THRESHOLDS:
        accepted = [p for p in test_predictions if p["confidence"] >= threshold]
        correct = [p for p in accepted if p["correct"]]

        overall_precision = len(correct) / len(accepted) if accepted else None
        result = {
            "accepted": len(accepted),
            "correct": len(correct),
            "coverage": len(accepted) / len(test_predictions) if test_predictions else 0.0,
            "overall_precision": overall_precision,
            "by_predicted_class": {},
        }

        print(f"\n  Threshold: {threshold:.2f}")
        print(
            f"    Accepted: {len(accepted)}/{len(test_predictions)} "
            f"({result['coverage']:.1%})"
        )
        if overall_precision is not None:
            print(f"    Overall precision: {overall_precision:.1%}")
        else:
            print("    Overall precision: N/A")

        for label in CLASSES:
            class_accepted = [
                p for p in accepted if p["predicted_label"] == label
            ]
            class_correct = [p for p in class_accepted if p["correct"]]
            precision = (
                len(class_correct) / len(class_accepted)
                if class_accepted else None
            )
            actual_count = int(np.sum(y_test == label))
            recall = len(class_correct) / actual_count if actual_count else None

            result["by_predicted_class"][label] = {
                "accepted": len(class_accepted),
                "correct": len(class_correct),
                "precision": precision,
                "recall": recall,
                "actual_test_examples": actual_count,
            }

            if precision is not None:
                recall_text = f"{recall:.1%}" if recall is not None else "N/A"
                print(
                    f"    {label:16s}: {len(class_accepted)} accepted, "
                    f"precision={precision:.1%}, recall={recall_text}"
                )
            else:
                print(f"    {label:16s}: no accepted predictions")

        threshold_report[str(threshold)] = result

    # -----------------------------------------------------
    # Save evaluation report
    # -----------------------------------------------------

    report = {
        "model_type": "residual_autoencoder_classifier",
        "architecture": {
            "input_dim": int(X.shape[1]),
            "hidden_dim": HIDDEN_DIM,
            "latent_dim": LATENT_DIM,
            "dropout": DROPOUT,
            "reconstruction_weight": RECONSTRUCTION_WEIGHT,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
        },
        "device": str(DEVICE),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classes": CLASSES,
        "training_episodes": len(set(groups[train_idx])),
        "testing_episodes": len(set(groups[test_idx])),
        "training_speakers": len(train_idx),
        "testing_speakers": len(test_idx),
        "class_distribution": dict(Counter(y)),
        "class_episode_distribution": class_episode_counts,
        "training_class_distribution": dict(Counter(y_train)),
        "testing_class_distribution": dict(Counter(y_test)),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=CLASSES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
            labels=CLASSES,
        ).tolist(),
        "test_episodes": sorted(set(groups[test_idx])),
        "held_out_predictions": test_predictions,
        "threshold_analysis": threshold_report,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation report saved to: {REPORT_PATH}")

    # -----------------------------------------------------
    # Refit final model on all annotated data
    # -----------------------------------------------------

    print("\nRefitting final model on all annotated data...")
    final_model = train_model(X, y, X.shape[1])

    checkpoint = {
        "model_state_dict": final_model.state_dict(),
        "model_config": {
            "input_dim": int(X.shape[1]),
            "num_classes": len(CLASSES),
            "hidden_dim": HIDDEN_DIM,
            "latent_dim": LATENT_DIM,
            "dropout": DROPOUT,
        },
        "classes": CLASSES,
        "embedding_dimensions": int(X.shape[1]),
        "training_samples": int(len(X)),
        "training_episodes": int(len(set(groups))),
        "class_distribution": dict(Counter(y)),
        "class_episode_distribution": class_episode_counts,
        "random_state": RANDOM_STATE,
        "normalization": "l2",
    }

    torch.save(checkpoint, MODEL_PATH)
    print(f"Final model saved to: {MODEL_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
