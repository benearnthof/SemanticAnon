
import json
from pathlib import Path

import numpy as np
import pandas as pd
import umap
import hdbscan
import plotly.express as px

from sklearn.preprocessing import normalize


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "data" / "diarized"
OUTPUT_DIR = ROOT / "data" / "speaker_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Clustering UMAP
CLUSTER_DIMENSIONS = 10
CLUSTER_NEIGHBORS = 30
CLUSTER_MIN_DIST = 0.0

# HDBSCAN
MIN_CLUSTER_SIZE = 8
MIN_SAMPLES = 3

# Visualization UMAP
VIS_NEIGHBORS = 20
VIS_MIN_DIST = 0.1


# --------------------------------------------------
# Load embeddings
# --------------------------------------------------

embeddings = []
metadata = []

files = sorted(INPUT_DIR.glob("*.json"))

print(f"Found {len(files)} diarization files")

for i, path in enumerate(files, 1):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        audio = data.get("audio", path.stem)
        episode_id = path.stem

        for speaker, vector in data.get("speaker_embeddings", {}).items():
            vector = np.asarray(vector, dtype=np.float32).flatten()

            if vector.size == 0 or not np.isfinite(vector).all():
                continue

            embeddings.append(vector)

            metadata.append({
                "episode_id": episode_id,
                "audio": audio,
                "speaker": speaker,
                "duration": data.get("duration"),
            })

    except Exception as e:
        print(f"Error reading {path.name}: {e}")

    if i % 50 == 0:
        print(f"Processed {i}/{len(files)} files")

if not embeddings:
    raise RuntimeError("No speaker embeddings found.")

# Ensure all vectors have the same dimensionality
dimensions = {len(x) for x in embeddings}

if len(dimensions) != 1:
    raise ValueError(f"Inconsistent embedding dimensions: {dimensions}")

X = np.vstack(embeddings)
df = pd.DataFrame(metadata)

print(f"\nLoaded {len(X)} speaker embeddings")
print(f"Embedding dimensions: {X.shape[1]}")
print(f"Unique episodes: {df['episode_id'].nunique()}")

# L2 normalization
X = normalize(X, norm="l2")


# --------------------------------------------------
# UMAP for clustering
# --------------------------------------------------

print("\nReducing embeddings for clustering...")

cluster_umap = umap.UMAP(
    n_components=CLUSTER_DIMENSIONS,
    n_neighbors=CLUSTER_NEIGHBORS,
    min_dist=CLUSTER_MIN_DIST,
    metric="cosine",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

X_cluster = cluster_umap.fit_transform(X)


# --------------------------------------------------
# HDBSCAN
# --------------------------------------------------

print("\nRunning HDBSCAN...")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=MIN_CLUSTER_SIZE,
    min_samples=MIN_SAMPLES,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

labels = clusterer.fit_predict(X_cluster)

df["cluster"] = labels
df["membership_probability"] = clusterer.probabilities_

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise = np.sum(labels == -1)

print(f"\nClusters found: {n_clusters}")
print(f"Noise points: {noise}/{len(labels)} ({noise / len(labels):.1%})")

print("\nCluster sizes:")
print(df["cluster"].value_counts().sort_index())

# little bit of cluster inspection
cluster_stats = (
    df[df["cluster"] != -1]
    .groupby("cluster")
    .agg(
        embeddings=("speaker", "size"),
        episodes=("episode_id", "nunique"),
        mean_probability=("membership_probability", "mean"),
        median_duration=("duration", "median"),
    )
    .sort_values("embeddings", ascending=False)
)

print(cluster_stats.to_string())

cluster_episode_overlap = (
    df[df["cluster"] != -1]
    .groupby(["cluster", "episode_id"])
    .size()
)

print("Cluster-episode combinations with multiple local speakers:")
print(cluster_episode_overlap[cluster_episode_overlap > 1])

largest = cluster_stats.head(15).copy()

largest["episode_coverage"] = (
    largest["episodes"] / df["episode_id"].nunique()
)

print(largest.round(3).to_string())

# --------------------------------------------------
# Independent 2D and 3D visualizations
# --------------------------------------------------

print("\nComputing 2D visualization...")

umap_2d = umap.UMAP(
    n_components=2,
    n_neighbors=VIS_NEIGHBORS,
    min_dist=VIS_MIN_DIST,
    metric="cosine",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

coords_2d = umap_2d.fit_transform(X)

df["x"] = coords_2d[:, 0]
df["y"] = coords_2d[:, 1]

df["cluster_label"] = df["cluster"].astype(str)
df.loc[df["cluster"] == -1, "cluster_label"] = "Noise"

fig = px.scatter(
    df,
    x="x",
    y="y",
    color="cluster_label",
    hover_data=[
        "audio",
        "speaker",
        "episode_id",
        "membership_probability",
    ],
    title="Global Speaker Clusters — 2D UMAP",
    opacity=0.8,
)

fig.update_traces(marker=dict(size=7))
fig.write_html(OUTPUT_DIR / "speakers_2d.html")

print("Saved speakers_2d.html")


print("\nComputing 3D visualization...")

umap_3d = umap.UMAP(
    n_components=3,
    n_neighbors=VIS_NEIGHBORS,
    min_dist=VIS_MIN_DIST,
    metric="cosine",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

coords_3d = umap_3d.fit_transform(X)

df["x3"] = coords_3d[:, 0]
df["y3"] = coords_3d[:, 1]
df["z3"] = coords_3d[:, 2]

fig3 = px.scatter_3d(
    df,
    x="x3",
    y="y3",
    z="z3",
    color="cluster_label",
    hover_data=[
        "audio",
        "speaker",
        "episode_id",
        "membership_probability",
    ],
    title="Global Speaker Clusters — 3D UMAP",
    opacity=0.8,
)

fig3.update_traces(marker=dict(size=3))
fig3.write_html(OUTPUT_DIR / "speakers_3d.html")


# --------------------------------------------------
# Save results
# --------------------------------------------------

df.to_csv(OUTPUT_DIR / "speaker_clusters.csv", index=False)

np.save(OUTPUT_DIR / "speaker_embeddings.npy", X)

print("\nDone.")
print(f"Results saved to: {OUTPUT_DIR}")
