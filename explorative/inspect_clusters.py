
import json
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DIARIZED_DIR = ROOT / "data" / "diarized"
AUDIO_DIR = ROOT / "data" / "audio"
CLUSTER_FILE = ROOT / "data" / "speaker_analysis" / "speaker_clusters.csv"

SNIPPET_PADDING = 1.5
MAX_SNIPPET_SECONDS = 20
RANDOM_SEED = 42


# --------------------------------------------------
# Load data
# --------------------------------------------------

@st.cache_data
def load_clusters():
    df = pd.read_csv(CLUSTER_FILE)

    # Preserve integer cluster labels
    df["cluster"] = df["cluster"].astype(int)

    return df


@st.cache_data
def load_episode(episode_id):
    path = DIARIZED_DIR / f"{episode_id}.json"

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_audio(audio_name):
    path = AUDIO_DIR / audio_name

    if not path.exists():
        return None

    return path


# --------------------------------------------------
# Audio snippet extraction
# --------------------------------------------------

def get_speaker_intervals(data, speaker):
    intervals = [
        (float(segment["start"]), float(segment["end"]))
        for segment in data["diarization"]
        if segment["speaker"] == speaker
    ]

    return sorted(intervals)


def select_snippet(intervals):
    """
    Select a reasonably substantial contiguous interval
    for the given local speaker.
    """

    if not intervals:
        return None

    candidates = []

    for start, end in intervals:
        duration = end - start

        if duration >= 3:
            candidates.append((start, end))

    if not candidates:
        return None

    # Prefer longer uninterrupted speech segments,
    # but don't let a very long segment dominate.
    candidates.sort(
        key=lambda x: min(x[1] - x[0], 20),
        reverse=True,
    )

    start, end = candidates[0]

    start = max(0, start - SNIPPET_PADDING)
    end = min(
        end + SNIPPET_PADDING,
        start + MAX_SNIPPET_SECONDS,
    )

    return start, end


@st.cache_data
def extract_audio(audio_path, start, end):
    """
    Extract an MP3 snippet using ffmpeg.
    Return the bytes for Streamlit playback.
    """

    with tempfile.NamedTemporaryFile(
        suffix=".mp3",
        delete=False,
    ) as tmp:
        output_path = Path(tmp.name)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss", str(start),
                "-i", str(audio_path),
                "-t", str(end - start),
                "-vn",
                "-ac", "1",
                "-b:a", "96k",
                str(output_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        return output_path.read_bytes()

    finally:
        output_path.unlink(missing_ok=True)


# --------------------------------------------------
# Streamlit interface
# --------------------------------------------------

st.set_page_config(
    page_title="Speaker Cluster Inspector",
    layout="wide",
)

st.title("Global Speaker Cluster Inspector")

df = load_clusters()

n_episodes = df["episode_id"].nunique()
n_embeddings = len(df)

st.caption(
    f"{n_embeddings:,} embeddings · "
    f"{n_episodes:,} episodes · "
    f"{df['cluster'].nunique()} cluster labels"
)


# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------

st.sidebar.header("Selection")

cluster_stats = (
    df.groupby("cluster")
    .agg(
        embeddings=("speaker", "size"),
        episodes=("episode_id", "nunique"),
        mean_probability=("membership_probability", "mean"),
    )
    .sort_values("embeddings", ascending=False)
)

cluster_options = cluster_stats.index.tolist()

selected_clusters = st.sidebar.multiselect(
    "Clusters",
    options=cluster_options,
    default=cluster_options[:10],
    format_func=lambda c: (
        f"Noise (-1)" if c == -1 else f"Cluster {c}"
    ) + (
        f" · {cluster_stats.loc[c, 'embeddings']} embeddings"
    ),
)

representative_count = st.sidebar.slider(
    "Representative examples per cluster",
    min_value=1,
    max_value=10,
    value=3,
)

random_count = st.sidebar.slider(
    "Additional random examples per cluster",
    min_value=0,
    max_value=10,
    value=2,
)

snippet_length = st.sidebar.slider(
    "Maximum snippet length (seconds)",
    min_value=5,
    max_value=30,
    value=20,
)

include_noise = st.sidebar.checkbox(
    "Include noise points",
    value=False,
)

if include_noise and -1 not in selected_clusters:
    selected_clusters = selected_clusters + [-1]


# --------------------------------------------------
# Cluster inspection
# --------------------------------------------------

for cluster in selected_clusters:

    cluster_df = df[df["cluster"] == cluster].copy()

    if cluster_df.empty:
        continue

    cluster_df = cluster_df.sort_values(
        "membership_probability",
        ascending=False,
    )

    stats = cluster_stats.loc[cluster]

    label = "Noise" if cluster == -1 else f"Cluster {cluster}"

    st.divider()
    st.header(label)

    c1, c2, c3 = st.columns(3)

    c1.metric("Embeddings", int(stats["embeddings"]))
    c2.metric("Episodes", int(stats["episodes"]))
    c3.metric(
        "Mean probability",
        f"{stats['mean_probability']:.3f}",
    )

    # Representative examples
    representatives = cluster_df.head(representative_count)

    # Random examples, avoiding already selected episodes
    excluded = set(representatives["episode_id"])

    remaining = cluster_df[
        ~cluster_df["episode_id"].isin(excluded)
    ]

    if len(remaining) > 0 and random_count > 0:
        random_examples = remaining.sample(
            n=min(random_count, len(remaining)),
            random_state=RANDOM_SEED + int(cluster),
        )
    else:
        random_examples = remaining.head(0)

    examples = []

    for _, row in representatives.iterrows():
        examples.append(("Representative", row))

    for _, row in random_examples.iterrows():
        examples.append(("Random", row))

    for example_type, row in examples:

        episode_id = row["episode_id"]
        speaker = row["speaker"]

        data = load_episode(episode_id)

        audio_path = find_audio(data["audio"])

        if audio_path is None:
            st.warning(f"Audio missing: {data['audio']}")
            continue

        intervals = get_speaker_intervals(data, speaker)

        if not intervals:
            st.warning(
                f"No diarization intervals for {episode_id} / {speaker}"
            )
            continue

        # Select the longest contiguous speaker interval
        snippet = select_snippet(intervals)

        if snippet is None:
            st.warning(
                f"No sufficiently long speech interval: {episode_id}"
            )
            continue

        start, end = snippet

        # Respect user-selected maximum snippet length
        end = min(end, start + snippet_length)

        st.subheader(
            f"{example_type} · {data['audio']}"
        )

        st.caption(
            f"Speaker: {speaker} · "
            f"Probability: {row['membership_probability']:.3f} · "
            f"Timestamp: {start:.1f}s–{end:.1f}s"
        )

        try:
            audio_bytes = extract_audio(
                str(audio_path),
                start,
                end,
            )

            st.audio(audio_bytes, format="audio/mp3")

        except Exception as e:
            st.error(f"Audio extraction failed: {e}")

        with st.expander("Episode details"):
            st.write({
                "Episode": data["audio"],
                "Episode ID": episode_id,
                "Speaker": speaker,
                "Cluster": int(cluster),
                "Membership probability": float(
                    row["membership_probability"]
                ),
                "All speaker intervals": intervals,
            })
