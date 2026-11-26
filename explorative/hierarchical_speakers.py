
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent

ANALYSIS_DIR = ROOT / "data" / "speaker_analysis"

CLUSTERS_FILE = ANALYSIS_DIR / "speaker_clusters.csv"
EMBEDDINGS_FILE = ANALYSIS_DIR / "speaker_embeddings.npy"
LABELS_FILE = ROOT / "top25_labels.csv"

OUTPUT_DIR = ANALYSIS_DIR / "hierarchical"

# HDBSCAN noise is excluded from the hierarchy.
EXCLUDE_NOISE = True

# Candidate cuts are selected automatically from the
# distribution of linkage distances.
CUT_QUANTILES = [0.25, 0.50, 0.75]


# ============================================================
# Helpers
# ============================================================

def find_column(df, candidates):
    """Find a column case-insensitively."""
    lookup = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def load_labels(path):
    """Load manually assigned acoustic-cluster labels."""
    if not path.exists():
        print("No label file found. Continuing without annotations.")
        return {}

    labels = pd.read_csv(path)

    cluster_col = find_column(labels, ["cluster", "cluster_id"])
    label_col = find_column(labels, ["speaker", "label", "name"])

    if cluster_col is None or label_col is None:
        raise ValueError(
            "top25_labels.csv must contain columns named "
            "'Cluster' and 'Speaker'."
        )

    labels[cluster_col] = pd.to_numeric(
        labels[cluster_col], errors="coerce"
    )

    labels = labels.dropna(subset=[cluster_col])

    # Group duplicate labels rather than silently overwriting them.
    result = {}

    for cluster, group in labels.groupby(cluster_col):
        names = sorted(set(group[label_col].dropna().astype(str)))
        result[int(cluster)] = " / ".join(names)

        if len(names) > 1:
            warnings.warn(
                f"Cluster {int(cluster)} has multiple labels: {names}"
            )

    return result


def short_label(cluster_id, labels, max_length=28):
    """Compact label for plot axes."""
    name = labels.get(cluster_id, "Unlabelled")

    if len(name) > max_length:
        name = name[:max_length - 3] + "..."

    return f"{cluster_id}: {name}"


def detect_episode_order(df):
    """
    Try to find an explicit episode-order column.
    Never assume that an arbitrary numeric ID is chronological.
    """
    candidates = [
        "episode_number",
        "episode_num",
        "episode_no",
        "episode_index",
        "episode_order",
        "episode_position",
    ]

    col = find_column(df, candidates)

    if col is not None:
        values = pd.to_numeric(df[col], errors="coerce")

        if values.notna().mean() > 0.5:
            return col, values

    # Fallback: extract numbers only when explicitly written
    # as "Episode 123" or "Ep 123" in a title/filename.
    for candidate in ["episode_id", "audio", "filename", "title"]:
        col = find_column(df, [candidate])

        if col is None:
            continue

        values = df[col].astype(str).str.extract(
            r"\b(?:episode|ep)\s*[-#:]?\s*(\d+)\b",
            flags=re.IGNORECASE,
        )[0]

        values = pd.to_numeric(values, errors="coerce")

        if values.notna().mean() > 0.5:
            return col, values

    return None, None


# ============================================================
# Load data
# ============================================================

print("Loading clustering data...")

df = pd.read_csv(CLUSTERS_FILE)
embeddings = np.load(EMBEDDINGS_FILE)

cluster_col = find_column(df, ["cluster", "cluster_id"])
episode_col = find_column(df, ["episode_id", "episode", "audio"])
speaker_col = find_column(df, ["speaker"])

if cluster_col is None:
    raise ValueError("Could not find cluster column in CSV.")

if embeddings.ndim != 2:
    raise ValueError("Expected a 2D embedding matrix.")

if len(df) != len(embeddings):
    raise ValueError(
        f"CSV has {len(df)} rows but embeddings have "
        f"{len(embeddings)} rows. Their row order must match."
    )

df[cluster_col] = pd.to_numeric(df[cluster_col], errors="coerce")
df = df.dropna(subset=[cluster_col]).copy()
df[cluster_col] = df[cluster_col].astype(int)

# The original embeddings and CSV must retain the same row indices.
valid_indices = df.index.to_numpy()
X = embeddings[valid_indices].astype(np.float64)

# Normalize defensively.
norms = np.linalg.norm(X, axis=1, keepdims=True)
X = X / np.maximum(norms, 1e-12)

labels = load_labels(LABELS_FILE)

if EXCLUDE_NOISE:
    noise_count = (df[cluster_col] == -1).sum()
    df = df[df[cluster_col] != -1].copy()
    print(f"Excluded {noise_count} noise embeddings.")

print(f"Loaded {len(df)} embeddings.")
print(f"Found {df[cluster_col].nunique()} acoustic clusters.")


# ============================================================
# Organize embeddings by acoustic cluster
# ============================================================

cluster_vectors = {}
episode_sets = {}

for cluster_id, group in df.groupby(cluster_col):
    cluster_id = int(cluster_id)

    indices = group.index.to_numpy()

    # Use original CSV row indices to retrieve embeddings.
    # X is indexed by the original row positions through valid_indices.
    position_map = {
        original_idx: position
        for position, original_idx in enumerate(valid_indices)
    }

    positions = [position_map[i] for i in indices]

    cluster_vectors[cluster_id] = X[positions]

    if episode_col is not None:
        episode_sets[cluster_id] = set(
            group[episode_col].dropna().astype(str)
        )
    else:
        episode_sets[cluster_id] = set()

cluster_ids = sorted(cluster_vectors)
n_clusters = len(cluster_ids)

if n_clusters < 2:
    raise ValueError("Need at least two acoustic clusters.")

print("Prepared acoustic clusters.")


# ============================================================
# Cluster statistics
# ============================================================

order_col, episode_order = detect_episode_order(df)

if order_col is not None:
    df["_episode_order"] = episode_order.loc[df.index]

stats = []

for cluster_id, group in df.groupby(cluster_col):
    cluster_id = int(cluster_id)

    row = {
        "cluster": cluster_id,
        "label": labels.get(cluster_id, ""),
        "n_embeddings": len(group),
        "n_episodes": (
            group[episode_col].nunique()
            if episode_col is not None else np.nan
        ),
    }

    if order_col is not None:
        row.update({
            "first_episode": group["_episode_order"].min(),
            "median_episode": group["_episode_order"].median(),
            "last_episode": group["_episode_order"].max(),
        })

    if speaker_col is not None:
        row["local_speakers"] = group[speaker_col].nunique()

    stats.append(row)

stats_df = pd.DataFrame(stats).sort_values(
    "n_embeddings", ascending=False
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

stats_df.to_csv(
    OUTPUT_DIR / "acoustic_cluster_stats.csv",
    index=False,
)


# ============================================================
# Pairwise cluster similarity
# ============================================================

print("Calculating pairwise cluster similarities...")

similarity = np.eye(n_clusters)
relationship_rows = []

for i, cluster_a in enumerate(cluster_ids):
    A = cluster_vectors[cluster_a]

    for j in range(i + 1, n_clusters):
        cluster_b = cluster_ids[j]
        B = cluster_vectors[cluster_b]

        # Cross-cluster cosine similarities.
        cross = A @ B.T

        # For each sample in A, find its nearest sample in B.
        a_to_b = np.median(cross.max(axis=1))

        # Repeat in the other direction.
        b_to_a = np.median(cross.max(axis=0))

        # Symmetric nearest-neighbour similarity.
        symmetric_nn = (a_to_b + b_to_a) / 2

        # Other useful diagnostics.
        cross_median = np.median(cross)
        cross_p10 = np.percentile(cross, 10)

        centroid_a = A.mean(axis=0)
        centroid_b = B.mean(axis=0)

        centroid_a /= max(np.linalg.norm(centroid_a), 1e-12)
        centroid_b /= max(np.linalg.norm(centroid_b), 1e-12)

        centroid_similarity = float(centroid_a @ centroid_b)

        shared_episodes = len(
            episode_sets[cluster_a] & episode_sets[cluster_b]
        )

        similarity[i, j] = symmetric_nn
        similarity[j, i] = symmetric_nn

        relationship_rows.append({
            "cluster_a": cluster_a,
            "label_a": labels.get(cluster_a, ""),
            "cluster_b": cluster_b,
            "label_b": labels.get(cluster_b, ""),
            "symmetric_nn_similarity": symmetric_nn,
            "a_to_b_similarity": a_to_b,
            "b_to_a_similarity": b_to_a,
            "cross_pair_median": cross_median,
            "cross_pair_p10": cross_p10,
            "centroid_similarity": centroid_similarity,
            "shared_episode_count": shared_episodes,
        })

relationships = pd.DataFrame(relationship_rows)

relationships = relationships.sort_values(
    "symmetric_nn_similarity",
    ascending=False,
)

relationships.to_csv(
    OUTPUT_DIR / "cluster_relationships.csv",
    index=False,
)


# ============================================================
# Hierarchical clustering
# ============================================================

print("Building average-linkage hierarchy...")

# Convert similarity into cosine-style dissimilarity.
distance = 1.0 - similarity
distance = np.clip(distance, 0.0, 2.0)

np.fill_diagonal(distance, 0.0)

condensed = squareform(distance, checks=False)

Z = linkage(
    condensed,
    method="average",
    optimal_ordering=True,
)

# Save linkage information.
linkage_df = pd.DataFrame(
    Z,
    columns=["cluster_a_index", "cluster_b_index", "distance", "size"],
)

linkage_df.to_csv(
    OUTPUT_DIR / "linkage_matrix.csv",
    index=False,
)


# ============================================================
# Interactive dendrogram
# ============================================================

print("Generating dendrogram...")

ddata = dendrogram(
    Z,
    labels=[str(c) for c in cluster_ids],
    no_plot=True,
)

leaves = ddata["leaves"]

fig = go.Figure()

for xs, ys in zip(ddata["icoord"], ddata["dcoord"]):
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color="#4C78A8", width=1.5),
        hovertemplate="Distance: %{y:.4f}<extra></extra>",
        showlegend=False,
    ))

tick_labels = [
    short_label(cluster_ids[i], labels)
    for i in leaves
]

tick_values = [5 + 10 * i for i in range(n_clusters)]

fig.update_layout(
    title="Hierarchical clustering of acoustic speaker clusters",
    xaxis=dict(
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_labels,
        tickangle=-65,
        title="Acoustic clusters",
    ),
    yaxis=dict(
        title="Cosine distance (average linkage)",
    ),
    height=950,
    margin=dict(l=70, r=30, t=80, b=260),
    template="plotly_white",
)

fig.write_html(
    OUTPUT_DIR / "speaker_dendrogram.html"
)


# ============================================================
# Similarity heatmap, ordered by dendrogram
# ============================================================

ordered_similarity = similarity[np.ix_(leaves, leaves)]

ordered_labels = [
    short_label(cluster_ids[i], labels)
    for i in leaves
]

heatmap = go.Figure(data=go.Heatmap(
    z=ordered_similarity,
    x=ordered_labels,
    y=ordered_labels,
    colorscale="RdBu",
    zmin=-1,
    zmax=1,
    colorbar=dict(title="Similarity"),
    hovertemplate=(
        "X: %{x}<br>"
        "Y: %{y}<br>"
        "Similarity: %{z:.4f}"
        "<extra></extra>"
    ),
))

heatmap.update_layout(
    title="Acoustic cluster similarity (dendrogram order)",
    height=1100,
    width=1100,
    xaxis=dict(tickangle=-65),
    yaxis=dict(autorange="reversed"),
    margin=dict(l=260, r=40, t=80, b=260),
)

heatmap.write_html(
    OUTPUT_DIR / "cluster_similarity_heatmap.html"
)


# ============================================================
# Candidate hierarchy cuts
# ============================================================

print("Generating candidate identity groupings...")

merge_distances = Z[:, 2]

cut_heights = sorted(set(
    float(np.quantile(merge_distances, q))
    for q in CUT_QUANTILES
))

cut_rows = []
cut_summary = []

for cut_number, threshold in enumerate(cut_heights, start=1):

    groups = fcluster(
        Z,
        t=threshold,
        criterion="distance",
    )

    group_count = len(np.unique(groups))

    cut_summary.append({
        "cut": cut_number,
        "distance_threshold": threshold,
        "identity_group_count": group_count,
        "acoustic_cluster_count": n_clusters,
    })

    for cluster_id, group_id in zip(cluster_ids, groups):
        cut_rows.append({
            "cut": cut_number,
            "distance_threshold": threshold,
            "candidate_identity": int(group_id),
            "acoustic_cluster": cluster_id,
            "label": labels.get(cluster_id, ""),
        })

    print(
        f"Cut {cut_number}: distance={threshold:.4f}, "
        f"{group_count} candidate groups"
    )

pd.DataFrame(cut_rows).to_csv(
    OUTPUT_DIR / "candidate_identity_groups.csv",
    index=False,
)

pd.DataFrame(cut_summary).to_csv(
    OUTPUT_DIR / "candidate_cut_summary.csv",
    index=False,
)


# ============================================================
# Finish
# ============================================================

print("\nDone.")
print(f"Output directory: {OUTPUT_DIR}")

print("\nGenerated files:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {path.name}")

if order_col is None:
    print(
        "\nNote: No reliable episode-order column was detected. "
        "Episode overlap was calculated, but chronological "
        "statistics were omitted."
    )
