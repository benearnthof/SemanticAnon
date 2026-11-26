
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "speaker_analysis"
OUT = DATA / "pca"
OUT.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA / "speaker_clusters.csv"
EMBEDDINGS_PATH = DATA / "speaker_embeddings.npy"

N_PAIR_SAMPLES = 100_000
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def normalize(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def cosine_stats(values):
    return {
        "mean": np.mean(values),
        "median": np.median(values),
        "p10": np.percentile(values, 10),
        "p90": np.percentile(values, 90),
    }


def sample_within(X, groups, n_samples):
    """Sample pairs within clusters, weighting clusters equally."""
    values = []
    n_per_group = max(1, n_samples // len(groups))

    for indices in groups.values():
        if len(indices) < 2:
            continue

        n = n_per_group
        a = rng.integers(0, len(indices), size=n)
        b = rng.integers(0, len(indices), size=n)

        # Ensure distinct observations.
        same = a == b
        while np.any(same):
            b[same] = rng.integers(0, len(indices), size=np.sum(same))
            same = a == b

        values.append(np.sum(
            X[indices[a]] * X[indices[b]], axis=1
        ))

    return np.concatenate(values)


def sample_cross(X, groups, n_samples):
    """Sample pairs between clusters, weighting cluster pairs equally."""
    cluster_ids = list(groups.keys())
    cluster_pairs = [
        (a, b)
        for i, a in enumerate(cluster_ids)
        for b in cluster_ids[i + 1:]
    ]

    pair_choices = rng.integers(
        0, len(cluster_pairs), size=n_samples
    )

    values = np.empty(n_samples)

    for pair_index in np.unique(pair_choices):
        a, b = cluster_pairs[pair_index]
        positions = np.flatnonzero(pair_choices == pair_index)

        ia = rng.choice(groups[a], size=len(positions))
        ib = rng.choice(groups[b], size=len(positions))

        values[positions] = np.sum(
            X[ia] * X[ib], axis=1
        )

    return values


# --------------------------------------------------
# Load and validate data
# --------------------------------------------------

df = pd.read_csv(CSV_PATH)
X = np.load(EMBEDDINGS_PATH).astype(np.float64)

cluster_col = next(
    (c for c in df.columns if c.lower() in ("cluster", "cluster_id")),
    None
)

if cluster_col is None:
    raise ValueError("Could not find a cluster ID column in CSV.")

if len(df) != len(X):
    raise ValueError(
        f"Row mismatch: CSV has {len(df)} rows, "
        f"embeddings have {len(X)}."
    )

labels = pd.to_numeric(df[cluster_col], errors="coerce").to_numpy()

valid = np.isfinite(labels) & (labels != -1)
X = normalize(X[valid])
labels = labels[valid].astype(int)

groups = {
    cluster: np.flatnonzero(labels == cluster)
    for cluster in np.unique(labels)
}

print(f"Embeddings: {len(X)}")
print(f"Clusters:   {len(groups)}")
print(f"Dimensions: {X.shape[1]}")


# --------------------------------------------------
# Global mean direction
# --------------------------------------------------

mean_vector = X.mean(axis=0)
mean_direction = mean_vector / np.linalg.norm(mean_vector)

mean_cosines = X @ mean_direction

print("\nCosine similarity to global mean direction:")
print(pd.Series(mean_cosines).describe(
    percentiles=[0.1, 0.5, 0.9]
))


# --------------------------------------------------
# PCA on mean-centered embeddings
# --------------------------------------------------

X_centered = X - X.mean(axis=0)

# SVD gives principal directions and explained variance.
_, singular_values, Vt = np.linalg.svd(
    X_centered, full_matrices=False
)

explained_variance = singular_values ** 2
explained_ratio = explained_variance / explained_variance.sum()
cumulative_ratio = np.cumsum(explained_ratio)

print("\nPrincipal component explained variance:")
for i in range(min(10, len(explained_ratio))):
    print(
        f"PC{i + 1:2d}: "
        f"{explained_ratio[i] * 100:6.2f}% "
        f"(cumulative {cumulative_ratio[i] * 100:6.2f}%)"
    )

# Remove only the first principal component.
first_pc = Vt[0]
projection = X_centered @ first_pc

X_pc_removed = X_centered - np.outer(projection, first_pc)

# Normalize all representations before cosine comparisons.
representations = {
    "original": normalize(X),
    "centered": normalize(X_centered),
    "pc1_removed": normalize(X_pc_removed),
}


# --------------------------------------------------
# Within- and between-cluster similarity
# --------------------------------------------------

results = []
distributions = {}

for name, representation in representations.items():
    print(f"\nSampling similarities: {name}")

    within = sample_within(
        representation, groups, N_PAIR_SAMPLES
    )
    cross = sample_cross(
        representation, groups, N_PAIR_SAMPLES
    )

    distributions[(name, "within")] = within
    distributions[(name, "cross")] = cross

    for kind, values in [
        ("within", within),
        ("cross", cross),
    ]:
        results.append({
            "representation": name,
            "distribution": kind,
            **cosine_stats(values),
        })

summary = pd.DataFrame(results)
summary.to_csv(OUT / "similarity_summary.csv", index=False)

print("\nSimilarity summary:")
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# --------------------------------------------------
# Save PCA statistics
# --------------------------------------------------

pca_df = pd.DataFrame({
    "component": np.arange(1, len(explained_ratio) + 1),
    "explained_variance_ratio": explained_ratio,
    "cumulative_explained_variance": cumulative_ratio,
})

pca_df.to_csv(OUT / "pca_explained_variance.csv", index=False)


# --------------------------------------------------
# Scree plot
# --------------------------------------------------

fig = make_subplots(specs=[[{"secondary_y": True}]])

components_to_show = min(30, len(explained_ratio))
component_numbers = np.arange(1, components_to_show + 1)

fig.add_trace(
    go.Bar(
        x=component_numbers,
        y=explained_ratio[:components_to_show] * 100,
        name="Individual variance (%)",
    ),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        x=component_numbers,
        y=cumulative_ratio[:components_to_show] * 100,
        name="Cumulative variance (%)",
        mode="lines+markers",
    ),
    secondary_y=True,
)

fig.update_layout(
    title="PCA explained variance",
    xaxis_title="Principal component",
    template="plotly_white",
)

fig.update_yaxes(
    title_text="Individual variance (%)",
    secondary_y=False,
)
fig.update_yaxes(
    title_text="Cumulative variance (%)",
    secondary_y=True,
)

fig.write_html(OUT / "pca_scree_plot.html")


# --------------------------------------------------
# Similarity distribution plot
# --------------------------------------------------

fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[
        "Original",
        "Mean-centered",
        "First PC removed",
    ],
)

for col, name in enumerate(representations, start=1):
    for kind in ("cross", "within"):
        fig.add_trace(
            go.Histogram(
                x=distributions[(name, kind)],
                name=f"{kind} ({name})",
                opacity=0.65,
                nbinsx=80,
                histnorm="probability density",
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )

fig.update_layout(
    title="Within- and between-cluster cosine similarity",
    barmode="overlay",
    template="plotly_white",
)

fig.update_xaxes(title_text="Cosine similarity", range=[-1, 1])
fig.write_html(OUT / "similarity_distributions.html")


print(f"\nFinished. Results saved to: {OUT}")
