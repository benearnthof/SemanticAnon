import json
from pathlib import Path
from collections import Counter, defaultdict

import joblib
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANNOTATED_DIR = PROJECT_ROOT / "data" / "diarized_annotated"
MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "speaker_classifier.joblib"
REPORT_PATH = MODEL_DIR / "training_report.json"

TEST_SIZE = 0.20
RANDOM_STATE = 42
SPLIT_ATTEMPTS = 200

CONFIDENCE_THRESHOLDS = [0.90, 0.95, 0.98, 0.99]

CLASSES = ["BRACE", "LIZ", "YOUNG CHOMSKY", "SOUNDBITE", "OTHER"]


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

    # Canonical identities. Accept a couple of common formatting variants
    # for YOUNG CHOMSKY, but always emit one canonical class name.
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
    """Convert an embedding to a valid 1D numpy array."""
    if embedding is None:
        return None

    embedding = np.asarray(embedding, dtype=np.float32)

    # Handle a possible collection of embeddings.
    if embedding.ndim == 2:
        embedding = embedding.mean(axis=0)

    if embedding.ndim != 1:
        return None

    if not np.all(np.isfinite(embedding)):
        return None

    return embedding


def make_model():
    return Pipeline([
        ("normalizer", Normalizer(norm="l2")),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                C=1.0,
            ),
        ),
    ])


def choose_group_split(X, y, groups):
    """Find a group-disjoint split with all classes in training, and ideally test."""
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

        # A class absent from training cannot be learned by the model.
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
            "Each class needs to occur in at least one training episode. "
            f"Episode coverage: {details}"
        )

    return best_split[0], best_split[1], False


# ---------------------------------------------------------
# Load annotated data
# ---------------------------------------------------------

print("Loading annotated episodes...")

X = []
y = []
groups = []
metadata = []

for path in sorted(ANNOTATED_DIR.glob("*.json")):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Could not read {path.name}: {e}")
        continue

    embeddings = data.get("speaker_embeddings", {})
    labels = data.get("speaker_global", {})

    for speaker_id, label in labels.items():
        normalized_label = normalize_label(label)

        # Skip unannotated speakers.
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

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)
groups = np.asarray(groups)

print(f"\nLoaded {len(X)} speaker embeddings.")
print(f"Annotated episodes: {len(set(groups))}")
print(f"Embedding dimensions: {X.shape[1]}")

missing_classes = set(CLASSES) - set(y)
if missing_classes:
    raise RuntimeError(
        "The following classes have no usable labeled embeddings: "
        f"{sorted(missing_classes)}. Check the spelling of speaker_global labels "
        "and confirm that those speakers have embeddings."
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


# ---------------------------------------------------------
# Episode-level train/test split
# ---------------------------------------------------------

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
        f"{absent}. Their held-out precision/recall cannot be estimated in this split. "
        "The final model will still be trained on all labeled data."
    )


# ---------------------------------------------------------
# Train evaluation model
# ---------------------------------------------------------

print("\nTraining logistic regression...")

evaluation_model = make_model()
evaluation_model.fit(X_train, y_train)

y_pred = evaluation_model.predict(X_test)
probabilities = evaluation_model.predict_proba(X_test)
model_classes = list(evaluation_model.named_steps["classifier"].classes_)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=CLASSES,
        zero_division=0,
    )
)

print("Confusion matrix:")
print("Labels:", CLASSES)
print(confusion_matrix(y_test, y_pred, labels=CLASSES))


# ---------------------------------------------------------
# Held-out predictions and probabilities
# ---------------------------------------------------------

print("\nHeld-out speaker confidence:")

test_predictions = []

for i, original_idx in enumerate(test_idx):
    probability_map = {
        label: float(probabilities[i][model_classes.index(label)])
        for label in CLASSES
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


# ---------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------

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
        true_positives = len(class_correct)
        recall = true_positives / actual_count if actual_count else None

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


# ---------------------------------------------------------
# Save evaluation report
# ---------------------------------------------------------

report = {
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


# ---------------------------------------------------------
# Refit final model on all annotated data
# ---------------------------------------------------------

print("\nRefitting final model on all annotated data...")

final_model = make_model()
final_model.fit(X, y)

joblib.dump(
    {
        "model": final_model,
        "classes": list(final_model.named_steps["classifier"].classes_),
        "embedding_dimensions": int(X.shape[1]),
        "training_samples": int(len(X)),
        "training_episodes": int(len(set(groups))),
        "class_distribution": dict(Counter(y)),
        "class_episode_distribution": class_episode_counts,
        "random_state": RANDOM_STATE,
    },
    MODEL_PATH,
)

print(f"Final model saved to: {MODEL_PATH}")
print("\nDone.")
