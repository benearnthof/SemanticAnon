from __future__ import annotations

import csv
import json
import math
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import streamlit as st


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
AUDIO_DIR = ROOT / "data" / "audio"
DIARIZED_DIR = ROOT / "data" / "diarized"
OUTPUT_DIR = ROOT / "data" / "diarized_annotated"
PREDICTIONS_PATH = ROOT / "predictions" / "speaker_predictions.csv"

# Number of examples shown for each local speaker.
SAMPLES_PER_SPEAKER = 6

# Keep snippets short enough to review quickly.
MIN_TURN_SECONDS = 1.0
SNIPPET_PADDING_SECONDS = 0.45
MAX_SNIPPET_SECONDS = 12.0

GLOBAL_LABEL_EXAMPLES = (
    "Liz, Brace, Young Chomsky, Soundbite, guest name, clip/archival audio, unknown"
)


# ---------------------------------------------------------------------------
# Data discovery and loading
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False, max_entries=20)
def load_json_cached(path_string: str, file_mtime_ns: int) -> dict[str, Any]:
    """Cache large JSON reads; the mtime invalidates the entry after a file changes."""
    del file_mtime_ns
    return load_json(Path(path_string))


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write safely so an interrupted save does not leave a truncated JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    temporary.replace(path)


def build_audio_index() -> dict[str, Path]:
    """Index MP3s by both filename and stem for robust matching."""
    index: dict[str, Path] = {}
    if AUDIO_DIR.exists():
        for path in AUDIO_DIR.glob("*.mp3"):
            index[path.name.lower()] = path
            index[path.stem.lower()] = path
    return index


def find_audio_file(diarized: dict[str, Any], json_path: Path,
                    audio_index: dict[str, Path]) -> Path | None:
    audio_value = diarized.get("audio")
    if audio_value:
        audio_name = Path(str(audio_value)).name
        match = audio_index.get(audio_name.lower())
        if match:
            return match
        match = audio_index.get(Path(audio_name).stem.lower())
        if match:
            return match

    # Fallback: try the diarization JSON filename/stem.
    for candidate in (json_path.stem.lower(),
                      json_path.stem.replace("_diarized", "").lower()):
        if candidate in audio_index:
            return audio_index[candidate]
    return None


def normalize_title(value: str) -> str:
    """Normalize titles for matching descriptions.txt entries to episode metadata."""
    return " ".join(str(value).casefold().strip().split())


def load_description_entries() -> list[tuple[str, str]]:
    """
    Read descriptions.txt as alternating title/description lines:
        episode title
        episode description
        next episode title
        next episode description
    Blank lines are ignored.
    """
    path = ROOT / "descriptions.txt"
    if not path.exists():
        return []

    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
             if line.strip()]
    return [(lines[i], lines[i + 1]) for i in range(0, len(lines) - 1, 2)]


@st.cache_data(ttl=3600, show_spinner=False)
def discover_episodes() -> list[dict[str, Any]]:
    if not DIARIZED_DIR.exists():
        return []

    audio_index = build_audio_index()
    description_entries = load_description_entries()
    descriptions_by_title = {
        normalize_title(title): description
        for title, description in description_entries
    }
    episodes = []

    for episode_index, json_path in enumerate(sorted(DIARIZED_DIR.glob("*.json"))):
        try:
            diarized = load_json(json_path)
        except Exception:
            continue

        audio_path = find_audio_file(diarized, json_path, audio_index)
        if audio_path is None:
            continue

        title = json_path.stem
        description = ""
        metadata_path = audio_path.with_suffix(".info.json")
        if metadata_path.exists():
            try:
                metadata = load_json(metadata_path)
                title = metadata.get("title") or title
                description = metadata.get("description") or ""
            except Exception:
                pass

        # Prefer the audio metadata description. Otherwise match descriptions.txt
        # by title, with episode-order fallback for title variants.
        if not description:
            description = descriptions_by_title.get(normalize_title(title), "")
        if not description and len(episodes) < len(description_entries):
            description = description_entries[len(episodes)][1]

        output_path = OUTPUT_DIR / json_path.name
        episodes.append({
            "key": json_path.name,
            "title": title,
            "description": description,
            "json_path": json_path,
            "audio_path": audio_path,
            "output_path": output_path,
        })

    return episodes


# ---------------------------------------------------------------------------
# Classifier predictions and active-learning priority
# ---------------------------------------------------------------------------
# Keep these thresholds in sync with scripts/predict_speakers.py.
# OTHER is intentionally absent: it always requires manual identification.
AUTO_ASSIGN_THRESHOLDS = {
    "BRACE": 0.875,
    "LIZ": 0.875,
    "YOUNG CHOMSKY": 0.90,
    "SOUNDBITE": 0.875,
}


@st.cache_data(show_spinner=False, max_entries=4)
def load_predictions_cached(csv_path: str, file_mtime_ns: int) -> dict[tuple[str, str], dict[str, Any]]:
    """Load the classifier CSV once; file mtime invalidates the cache."""
    del file_mtime_ns
    result: dict[tuple[str, str], dict[str, Any]] = {}
    path = Path(csv_path)
    if not path.exists():
        return result

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            episode = str(row.get("episode", "")).strip()
            speaker_id = str(row.get("speaker_id", "")).strip()
            if not episode or not speaker_id:
                continue
            try:
                # Normalize labels so both the current five-class CSV and
                # older CSVs remain readable (e.g. YOUNG_CHOMSKY -> YOUNG CHOMSKY).
                predicted_label = " ".join(
                    str(row.get("predicted_label", "")).strip().upper()
                    .replace("_", " ").replace("-", " ").split()
                )
                result[(episode, speaker_id)] = {
                    "predicted_label": predicted_label,
                    "confidence": float(row.get("confidence") or 0.0),
                    "prob_BRACE": float(row.get("prob_BRACE") or 0.0),
                    "prob_LIZ": float(row.get("prob_LIZ") or 0.0),
                    "prob_YOUNG_CHOMSKY": float(
                        row.get("prob_YOUNG_CHOMSKY") or row.get("prob_YOUNG CHOMSKY") or 0.0
                    ),
                    "prob_SOUNDBITE": float(row.get("prob_SOUNDBITE") or 0.0),
                    "prob_OTHER": float(row.get("prob_OTHER") or 0.0),
                    "duration_seconds": float(row.get("duration_seconds") or 0.0),
                    "status": str(row.get("status", "")),
                }
            except (TypeError, ValueError):
                continue
    return result


@st.cache_data(show_spinner=False, max_entries=4)
def load_all_manual_labels_cached(
    directory_string: str, directory_fingerprint: tuple
) -> dict[tuple[str, str], str]:
    """Read saved annotations once; fingerprint invalidates after any annotation save."""
    del directory_fingerprint
    directory = Path(directory_string)
    labels: dict[tuple[str, str], str] = {}
    if not directory.exists():
        return labels

    for path in directory.glob("*.json"):
        try:
            data = load_json(path)
        except Exception:
            continue
        for speaker_id, label in (data.get("speaker_global") or {}).items():
            cleaned = str(label).strip()
            if cleaned:
                labels[(path.stem, str(speaker_id))] = cleaned
    return labels


def get_manual_labels_index() -> dict[tuple[str, str], str]:
    """Cheap directory stat fingerprint keeps manual labels fresh after saving."""
    fingerprint = []
    if OUTPUT_DIR.exists():
        for path in OUTPUT_DIR.glob("*.json"):
            try:
                stat = path.stat()
                fingerprint.append((path.name, stat.st_mtime_ns, stat.st_size))
            except OSError:
                continue
    return load_all_manual_labels_cached(str(OUTPUT_DIR), tuple(fingerprint))


def get_prediction_index() -> dict[tuple[str, str], dict[str, Any]]:
    if not PREDICTIONS_PATH.exists():
        return {}
    try:
        mtime_ns = PREDICTIONS_PATH.stat().st_mtime_ns
    except OSError:
        return {}
    return load_predictions_cached(str(PREDICTIONS_PATH), mtime_ns)


def is_auto_assign_prediction(prediction: dict[str, Any] | None) -> bool:
    """Whether this prediction meets its class-specific auto-prefill threshold."""
    if not prediction:
        return False

    label = prediction.get("predicted_label", "")
    threshold = AUTO_ASSIGN_THRESHOLDS.get(label)

    # OTHER and any unrecognized class are always review-only.
    return threshold is not None and prediction.get("confidence", 0.0) >= threshold


def build_episode_priority(
    episodes: list[dict[str, Any]],
    predictions: dict[tuple[str, str], dict[str, Any]],
    manual_labels: dict[tuple[str, str], str],
) -> dict[str, float]:
    """
    Duration-weighted uncertainty score per episode:
        sum(duration_seconds * (1 - max_class_probability))
    Human-labeled and high-confidence automatic assignments are excluded.
    """
    scores = {ep["key"]: 0.0 for ep in episodes}
    known_stems = {ep["json_path"].stem: ep["key"] for ep in episodes}

    for (episode_stem, speaker_id), prediction in predictions.items():
        episode_key = known_stems.get(episode_stem)
        if episode_key is None:
            continue
        if (episode_stem, speaker_id) in manual_labels:
            continue
        if is_auto_assign_prediction(prediction):
            continue

        duration = max(0.0, prediction.get("duration_seconds", 0.0))
        confidence = min(1.0, max(0.0, prediction.get("confidence", 0.0)))
        scores[episode_key] += duration * (1.0 - confidence)

    return scores


# ---------------------------------------------------------------------------
# Speaker and turn handling
# ---------------------------------------------------------------------------
def collect_speakers(data: dict[str, Any]) -> list[str]:
    found = set()

    for item in data.get("diarization", []):
        speaker = item.get("speaker")
        if speaker:
            found.add(str(speaker))

    for segment in data.get("segments", []):
        speaker = segment.get("speaker")
        if speaker:
            found.add(str(speaker))
        for word in segment.get("words", []) or []:
            speaker = word.get("speaker")
            if speaker:
                found.add(str(speaker))

    # Retain speakers that have embeddings even if no turn was found.
    for speaker in (data.get("speaker_embeddings") or {}).keys():
        found.add(str(speaker))

    return sorted(found)


def speaker_turns(data: dict[str, Any], speaker_id: str) -> list[dict[str, float]]:
    """Get turn intervals, preferring diarization intervals over transcript segments."""
    turns = []

    for item in data.get("diarization", []):
        if str(item.get("speaker")) != speaker_id:
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            turns.append({"start": start, "end": end})

    if not turns:
        for item in data.get("segments", []):
            if str(item.get("speaker")) != speaker_id:
                continue
            try:
                start = float(item["start"])
                end = float(item["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if end > start:
                turns.append({"start": start, "end": end})

    return sorted(turns, key=lambda x: x["start"])


def choose_representative_turns(
    turns: list[dict[str, float]],
    duration: float,
    count: int = SAMPLES_PER_SPEAKER,
) -> list[dict[str, float]]:
    """
    Choose relatively long turns spread over the episode.
    First select the best candidate in each temporal bin, then fill any gaps
    with the longest remaining turns.
    """
    candidates = [
        t for t in turns
        if t["end"] - t["start"] >= MIN_TURN_SECONDS
    ]
    if not candidates:
        candidates = turns[:]

    if not candidates:
        return []

    selected: list[dict[str, float]] = []
    bins = max(1, count)

    if duration > 0:
        for i in range(bins):
            left = duration * i / bins
            right = duration * (i + 1) / bins
            in_bin = [
                t for t in candidates
                if t["start"] < right and t["end"] > left
            ]
            if in_bin:
                best = max(in_bin, key=lambda t: t["end"] - t["start"])
                if best not in selected:
                    selected.append(best)

    remaining = sorted(
        (t for t in candidates if t not in selected),
        key=lambda t: t["end"] - t["start"],
        reverse=True,
    )
    for turn in remaining:
        if len(selected) >= count:
            break
        selected.append(turn)

    selected.sort(key=lambda x: x["start"])
    return selected[:count]


def make_snippet_bounds(
    turn: dict[str, float], duration: float
) -> tuple[float, float]:
    start = max(0.0, turn["start"] - SNIPPET_PADDING_SECONDS)
    end = min(duration, turn["end"] + SNIPPET_PADDING_SECONDS)

    if end - start > MAX_SNIPPET_SECONDS:
        midpoint = (turn["start"] + turn["end"]) / 2
        start = max(0.0, midpoint - MAX_SNIPPET_SECONDS / 2)
        end = min(duration, start + MAX_SNIPPET_SECONDS)
        start = max(0.0, end - MAX_SNIPPET_SECONDS)

    return start, end


@st.cache_data(show_spinner=False, max_entries=300)
def extract_audio_snippet(
    audio_path_string: str,
    file_mtime: float,
    start: float,
    end: float,
) -> bytes:
    """Extract a small WAV snippet with FFmpeg; cache it across reruns."""
    del file_mtime  # Included in the cache key to invalidate after audio changes.

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", audio_path_string,
        "-t", f"{max(0.1, end - start):.3f}",
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-f", "wav",
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        error = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(error[-1200:] or "FFmpeg could not extract this snippet.")
    return result.stdout


# ---------------------------------------------------------------------------
# Background episode prefetch
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_prefetch_executor() -> ThreadPoolExecutor:
    """One small background pool shared across Streamlit reruns."""
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="episode-prefetch")


@st.cache_resource(show_spinner=False)
def get_prefetch_registry() -> dict[tuple[str, int, str, int], Future]:
    """Keep in-flight/completed payloads available as the user moves through the queue."""
    return {}


def read_episode_payload(
    original_path_string: str,
    original_mtime_ns: int,
    output_path_string: str,
    output_mtime_ns: int,
) -> dict[str, Any]:
    """Background-safe file loader; deliberately makes no Streamlit UI calls."""
    del original_mtime_ns, output_mtime_ns  # Included in the Future key by the caller.
    original = load_json(Path(original_path_string))
    output_path = Path(output_path_string)
    if output_path.exists():
        try:
            enriched = load_json(output_path)
            original["speaker_global"] = enriched.get("speaker_global", {})
        except Exception:
            pass
    original.setdefault("speaker_global", {})
    return original


def episode_payload_key(episode: dict[str, Any]) -> tuple[str, int, str, int]:
    original_path = episode["json_path"]
    output_path = episode["output_path"]
    original_mtime = original_path.stat().st_mtime_ns
    try:
        output_mtime = output_path.stat().st_mtime_ns
    except OSError:
        output_mtime = -1
    return (str(original_path), original_mtime, str(output_path), output_mtime)


def schedule_episode_prefetch(episode: dict[str, Any]) -> None:
    """Load an upcoming episode on a worker while the current one is being labeled."""
    try:
        key = episode_payload_key(episode)
    except OSError:
        return

    registry = get_prefetch_registry()
    if key not in registry:
        registry[key] = get_prefetch_executor().submit(read_episode_payload, *key)

    # Bound retained Futures so navigating/searching through the corpus does not
    # keep every prefetched episode's JSON resident indefinitely.
    if len(registry) > 8:
        for old_key, future in list(registry.items()):
            if old_key != key and future.done():
                registry.pop(old_key, None)
            if len(registry) <= 6:
                break


# ---------------------------------------------------------------------------
# Annotation persistence
# ---------------------------------------------------------------------------
def load_annotated_copy(episode: dict[str, Any]) -> dict[str, Any]:
    """
    Always start from the original diarization JSON. If an enriched copy exists,
    reuse only its speaker_global mapping. This avoids accidentally accumulating
    changes to legacy diarization fields.
    """
    try:
        key = episode_payload_key(episode)
        future = get_prefetch_registry().get(key)
        if future is not None:
            try:
                return future.result()
            except Exception:
                # Fall back to the normal cached loader if background I/O failed.
                pass
    except OSError:
        pass

    original_path = episode["json_path"]
    original = load_json_cached(
        str(original_path), original_path.stat().st_mtime_ns
    )
    output_path = episode["output_path"]

    if output_path.exists():
        try:
            enriched = load_json_cached(
                str(output_path), output_path.stat().st_mtime_ns
            )
            original["speaker_global"] = enriched.get("speaker_global", {})
        except Exception:
            pass

    original.setdefault("speaker_global", {})
    return original


def save_episode(episode: dict[str, Any], labels: dict[str, str]) -> None:
    enriched = load_json(episode["json_path"])
    enriched["speaker_global"] = {
        str(local_id): str(global_id).strip()
        for local_id, global_id in labels.items()
        if str(global_id).strip()
    }
    enriched["speaker_annotation"] = {
        "schema_version": 1,
        "method": "manual_streamlit_annotation",
    }
    write_json_atomic(episode["output_path"], enriched)


def persist_speaker_draft(draft_key: str, speaker_id: str, widget_key: str) -> None:
    """Copy a text-input value into non-widget session state before reruns."""
    drafts = dict(st.session_state.get(draft_key, {}))
    drafts[speaker_id] = str(st.session_state.get(widget_key, "")).strip()
    st.session_state[draft_key] = drafts


def persist_active_speaker_draft(
    draft_key: str, speaker_id: str | None, widget_key: str | None
) -> None:
    """Preserve the currently visible input before switching speakers or saving."""
    if speaker_id and widget_key:
        persist_speaker_draft(draft_key, speaker_id, widget_key)


@st.cache_data(show_spinner=False, max_entries=5000)
def _is_fully_labeled(
    json_path_str: str,
    json_mtime_ns: int,
    output_path_str: str,
    output_mtime_ns: int,
) -> bool:
    """Cached per episode; the mtimes in the key invalidate it when a file changes."""
    if output_mtime_ns < 0:  # no annotated copy yet
        return False
    try:
        speakers = collect_speakers(load_json(Path(json_path_str)))
        mapping = load_json(Path(output_path_str)).get("speaker_global", {})
    except Exception:
        return False
    return bool(speakers) and all(str(mapping.get(s, "")).strip() for s in speakers)


def completion_map(episodes: list[dict[str, Any]]) -> dict[str, bool]:
    """episode key -> fully labeled? Costs only a stat() per file on cache hits."""
    result: dict[str, bool] = {}
    for ep in episodes:
        try:
            json_mtime = ep["json_path"].stat().st_mtime_ns
        except OSError:
            result[ep["key"]] = False
            continue
        try:
            out_mtime = ep["output_path"].stat().st_mtime_ns
        except OSError:
            out_mtime = -1
        result[ep["key"]] = _is_fully_labeled(
            str(ep["json_path"]), json_mtime, str(ep["output_path"]), out_mtime
        )
    return result


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TrueAnon Speaker Annotator",
    page_icon="🎙️",
    layout="wide",
)

if not DIARIZED_DIR.exists() or not AUDIO_DIR.exists():
    st.error(
        "Could not find the expected data directories. Expected:\n\n"
        f"- `{AUDIO_DIR}`\n- `{DIARIZED_DIR}`"
    )
    st.stop()

episodes = discover_episodes()
if not episodes:
    st.error(
        "No matching episode/audio pairs were found. Check that each diarization "
        "JSON's `audio` field matches an MP3 filename in `data/audio/`."
    )
    st.stop()

completion = completion_map(episodes)
completed, total = sum(completion.values()), len(episodes)
progress = completed / total if total else 0.0

top_left, top_right = st.columns([3, 1])
with top_left:
    st.progress(progress, text=f"{completed} of {total} episodes fully labeled")
with top_right:
    st.metric("Episodes found", total)

# Classifier outputs and current human annotations are cached independently.
predictions = get_prediction_index()
manual_labels_index = get_manual_labels_index()
priority_scores = build_episode_priority(episodes, predictions, manual_labels_index)

if not PREDICTIONS_PATH.exists():
    st.warning(
        "Classifier predictions were not found. Run `python scripts/predict_speakers.py` "
        "to enable automatic identity prefill and active-learning prioritization. "
        "Manual annotation remains fully functional."
    )

# Search, ordering, and episode navigation.
search_col, sort_col, hide_col = st.columns([3, 2, 1], vertical_alignment="bottom")
with search_col:
    search = st.text_input(
        "Find an episode", placeholder="Search episode title or filename"
    )
with sort_col:
    sort_mode = st.selectbox(
        "Episode order",
        ["Active-learning priority", "Alphabetical"],
        index=0,
    )
with hide_col:
    hide_labeled = st.checkbox(
        "Skip fully labeled",
        key="hide_labeled",
        help="Hide episodes where every local speaker already has a global label.",
    )

filtered = [
    ep for ep in episodes
    if search.lower() in (ep["title"] + " " + ep["key"]).lower()
    and not (hide_labeled and completion[ep["key"]])
]

if sort_mode == "Active-learning priority":
    filtered.sort(
        key=lambda ep: (priority_scores.get(ep["key"], 0.0), ep["title"].lower()),
        reverse=True,
    )
else:
    filtered.sort(key=lambda ep: ep["title"].lower())

if not filtered:
    if hide_labeled and not search:
        st.success("All episodes are fully labeled. Untick “Skip fully labeled” to review them.")
    else:
        st.warning("No episodes match that search.")
    st.stop()

episode_keys = [ep["key"] for ep in filtered]
if "selected_episode_key" not in st.session_state:
    st.session_state.selected_episode_key = episode_keys[0]
if st.session_state.selected_episode_key not in episode_keys:
    st.session_state.selected_episode_key = episode_keys[0]

selected_index = episode_keys.index(st.session_state.selected_episode_key)
nav_prev, nav_select, nav_next = st.columns([1, 6, 1])

with nav_prev:
    if st.button("← Previous", disabled=(selected_index == 0), use_container_width=True):
        st.session_state.selected_episode_key = episode_keys[selected_index - 1]
        st.session_state.pop("active_speaker_id", None)
        st.rerun()

with nav_select:
    chosen_key = st.selectbox(
        "Episode",
        options=episode_keys,
        index=selected_index,
        format_func=lambda key: (
            f"{next(ep['title'] for ep in filtered if ep['key'] == key)}"
            f"  ·  priority {priority_scores.get(key, 0.0) / 60:.1f} uncertain min"
        ) if sort_mode == "Active-learning priority" else
        next(ep["title"] for ep in filtered if ep["key"] == key),
        label_visibility="collapsed",
    )
    if chosen_key != st.session_state.selected_episode_key:
        st.session_state.selected_episode_key = chosen_key
        st.session_state.pop("active_speaker_id", None)
        st.rerun()

with nav_next:
    if st.button("Next →", disabled=(selected_index >= len(episode_keys) - 1),
                 use_container_width=True):
        st.session_state.selected_episode_key = episode_keys[selected_index + 1]
        st.session_state.pop("active_speaker_id", None)
        st.rerun()

episode = next(ep for ep in filtered if ep["key"] == st.session_state.selected_episode_key)
# Prefetch the next two episodes in the already-ranked queue while this episode
# is on screen. Their JSON and saved annotation mappings will be ready on arrival.
for upcoming_episode in filtered[selected_index + 1:selected_index + 3]:
    schedule_episode_prefetch(upcoming_episode)

data = load_annotated_copy(episode)
speakers = collect_speakers(data)
audio_path = episode["audio_path"]

st.divider()
st.subheader(episode["title"])
if episode.get("description"):
    st.markdown(episode["description"])
st.caption(f"Audio: `{audio_path.name}`")

if not speakers:
    st.warning("No speaker IDs were found in this diarization file.")
    st.stop()

duration = float(data.get("duration") or 0.0)
if duration <= 0:
    # Infer duration from the latest diarization/segment endpoint.
    endpoints = []
    for key in ("diarization", "segments"):
        for item in data.get(key, []):
            try:
                endpoints.append(float(item["end"]))
            except (KeyError, TypeError, ValueError):
                pass
    duration = max(endpoints, default=0.0)

existing_labels = data.get("speaker_global", {}) or {}
# Streamlit removes widget state for text inputs that are no longer rendered.
# Keep drafts in a separate session-state dictionary so switching speakers
# cannot discard labels entered for speakers that are currently hidden.
draft_key = f"speaker_drafts::{episode['key']}"
if draft_key not in st.session_state:
    st.session_state[draft_key] = {}
speaker_drafts = dict(st.session_state.get(draft_key, {}))
st.write(
    f"**{len(speakers)} local speakers** · "
    f"{sum(bool(str(existing_labels.get(s, '')).strip()) for s in speakers)} labeled"
)

# Build speaker review rows. Only the selected speaker's snippets are extracted/rendered.
# This avoids doing work for every collapsed speaker on each Streamlit rerun.
edited_labels: dict[str, str] = {}
speaker_info: dict[str, dict[str, Any]] = {}

for speaker_id in speakers:
    turns = speaker_turns(data, speaker_id)
    total_speech = sum(max(0.0, t["end"] - t["start"]) for t in turns)
    prediction = predictions.get((episode["json_path"].stem, speaker_id))
    manual_label = str(existing_labels.get(speaker_id, "")).strip()

    auto_assignment = is_auto_assign_prediction(prediction)
    suggested_label = prediction["predicted_label"] if auto_assignment else ""
    initial_label = manual_label or suggested_label

    widget_key = f"global_label::{episode['key']}::{speaker_id}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = speaker_drafts.get(speaker_id, initial_label)
    elif speaker_id in speaker_drafts and not str(st.session_state.get(widget_key, "")).strip():
        st.session_state[widget_key] = speaker_drafts[speaker_id]
    elif manual_label and not str(st.session_state.get(widget_key, "")).strip():
        # Recover legacy/current saved labels if an old widget state is still present.
        st.session_state[widget_key] = manual_label
    elif not str(st.session_state.get(widget_key, "")).strip() and suggested_label:
        # Prefill high-confidence classifier suggestions when migrating from an older app session.
        st.session_state[widget_key] = suggested_label

    # Keep an in-progress user edit in session state; saved human labels are the fallback.
    current_value = str(st.session_state.get(widget_key, initial_label)).strip()
    edited_labels[speaker_id] = current_value

    confidence = prediction.get("confidence") if prediction else None
    predicted_label = prediction.get("predicted_label", "") if prediction else ""

    if manual_label:
        status_text = f"human: {manual_label}"
        review_priority = 0.0
    elif auto_assignment:
        status_text = f"auto {suggested_label} · {confidence:.1%}"
        review_priority = 0.0
    elif prediction:
        status_text = f"review · {predicted_label} {confidence:.1%}"
        review_priority = total_speech * (1.0 - (confidence or 0.0))
    else:
        status_text = "no prediction"
        review_priority = total_speech

    speaker_info[speaker_id] = {
        "turns": turns,
        "total_speech": total_speech,
        "prediction": prediction,
        "manual_label": manual_label,
        "auto_assignment": auto_assignment,
        "status_text": status_text,
        "review_priority": review_priority,
        "widget_key": widget_key,
    }

# Prioritize the most consequential uncertain speakers within the selected episode.
ordered_speakers = sorted(
    speakers,
    key=lambda sid: (
        speaker_info[sid]["review_priority"],
        speaker_info[sid]["total_speech"],
    ),
    reverse=True,
)

# Only speakers without a saved label or a high-confidence auto-assignment
# belong in the work queue.
unresolved = [
    sid for sid in ordered_speakers
    if not speaker_info[sid]["manual_label"] and not speaker_info[sid]["auto_assignment"]
]

active_id = st.session_state.get("active_speaker_id")
if active_id not in unresolved:
    active_id = unresolved[0] if unresolved else None
    st.session_state.active_speaker_id = active_id

auto_count = sum(speaker_info[s]["auto_assignment"] and not speaker_info[s]["manual_label"] for s in speakers)
manual_count = sum(bool(speaker_info[s]["manual_label"]) for s in speakers)
review_count = len(speakers) - auto_count - manual_count

metric_cols = st.columns(4)
metric_cols[0].metric("Local speakers", len(speakers))
metric_cols[1].metric("Human-labeled", manual_count)
metric_cols[2].metric("Auto-assigned", auto_count)
metric_cols[3].metric("Need review", review_count)

episode_priority = priority_scores.get(episode["key"], 0.0)
st.caption(
    f"Episode review priority: **{episode_priority / 60:.1f} uncertain speaker-minutes**. "
    "High-confidence BRACE, LIZ, YOUNG CHOMSKY, and SOUNDBITE suggestions are prefilled but are not written until you save."
)

st.markdown("#### Speaker queue")
st.caption(f"{len(unresolved)} speakers still need a label.")

for speaker_id in unresolved:
    info = speaker_info[speaker_id]
    prediction = info["prediction"]
    duration_min = info["total_speech"] / 60.0
    confidence = prediction.get("confidence") if prediction else None
    confidence_text = f" · {confidence:.0%}" if confidence is not None else ""
    button_label = f"○ {speaker_id} · {duration_min:.1f} min{confidence_text}"

    if st.button(
        button_label,
        key=f"open_speaker::{episode['key']}::{speaker_id}",
        use_container_width=True,
        type="primary" if speaker_id == active_id else "secondary",
    ):
        previous_id = st.session_state.get("active_speaker_id")
        previous_widget_key = (
            speaker_info[previous_id]["widget_key"] if previous_id in speaker_info else None
        )
        persist_active_speaker_draft(draft_key, previous_id, previous_widget_key)
        st.session_state.active_speaker_id = speaker_id
        st.rerun()

if not unresolved:
    st.success("All speakers in this episode already have labels.")

if active_id:
    info = speaker_info[active_id]
    turns = info["turns"]
    samples = choose_representative_turns(turns, duration)
    prediction = info["prediction"]

    st.divider()
    st.subheader(f"Reviewing {active_id}")

    if info["manual_label"]:
        st.info(f"Existing human annotation: **{info['manual_label']}**")
    elif info["auto_assignment"]:
        st.success(
            f"Classifier suggestion: **{prediction['predicted_label']}** "
            f"({prediction['confidence']:.1%} confidence). "
            "This is prefilled; change it if the snippets disagree."
        )
    elif prediction:
        if prediction["predicted_label"] == "OTHER":
            st.warning(
                f"Classifier suggestion: **OTHER** ({prediction['confidence']:.1%} confidence). "
                "This is deliberately not auto-assigned. Identify the guest or clip manually "
                "using the episode description, metadata, or other context."
            )
        else:
            threshold = AUTO_ASSIGN_THRESHOLDS.get(prediction["predicted_label"])
            threshold_note = (
                f" The automatic-assignment threshold is {threshold:.1%}."
                if threshold is not None else ""
            )
            st.warning(
                f"Classifier suggestion: **{prediction['predicted_label']}** "
                f"({prediction['confidence']:.1%} confidence). "
                f"Below its automatic-assignment threshold.{threshold_note} "
                "Treat this as a suggestion, not a label."
            )
    else:
        st.info("No classifier prediction is available for this speaker.")

    col_label, col_stats = st.columns([3, 1])
    with col_label:
        label = st.text_input(
            f"Global identity for {active_id}",
            key=info["widget_key"],
            placeholder="e.g. Liz, Brace, Producer, guest name, clip/archival audio",
            help="Use the same spelling for a person across all episodes.",
            on_change=persist_speaker_draft,
            args=(draft_key, active_id, info["widget_key"]),
        )
        edited_labels[active_id] = label.strip()

    with col_stats:
        st.metric("Detected turns", len(turns))
        st.caption(f"Approx. {info['total_speech'] / 60:.1f} min of speech")

    if prediction:
        probability_cols = st.columns(5)
        probability_cols[0].metric("BRACE", f"{prediction['prob_BRACE']:.1%}")
        probability_cols[1].metric("LIZ", f"{prediction['prob_LIZ']:.1%}")
        probability_cols[2].metric(
            "YOUNG CHOMSKY", f"{prediction['prob_YOUNG_CHOMSKY']:.1%}"
        )
        probability_cols[3].metric("SOUNDBITE", f"{prediction['prob_SOUNDBITE']:.1%}")
        probability_cols[4].metric("OTHER", f"{prediction['prob_OTHER']:.1%}")

    if not samples:
        st.info("No usable turns found for this speaker.")
    else:
        for sample_start in range(0, len(samples), 2):
            sample_cols = st.columns(2)
            for offset, col in enumerate(sample_cols):
                sample_index = sample_start + offset
                if sample_index >= len(samples):
                    continue

                turn = samples[sample_index]
                clip_start, clip_end = make_snippet_bounds(turn, duration)
                with col:
                    st.caption(
                        f"Example {sample_index + 1} · "
                        f"{math.floor(turn['start'] // 60):02d}:"
                        f"{int(turn['start'] % 60):02d} – "
                        f"{math.floor(turn['end'] // 60):02d}:"
                        f"{int(turn['end'] % 60):02d}"
                    )
                    try:
                        snippet = extract_audio_snippet(
                            str(audio_path),
                            audio_path.stat().st_mtime,
                            clip_start,
                            clip_end,
                        )
                        st.audio(snippet, format="audio/wav")
                    except Exception as exc:
                        st.error(f"Could not extract snippet: {exc}")

                    matching_text = []
                    for segment in data.get("segments", []):
                        try:
                            overlaps = (
                                float(segment["start"]) < turn["end"]
                                and float(segment["end"]) > turn["start"]
                            )
                        except (KeyError, TypeError, ValueError):
                            overlaps = False
                        if overlaps and str(segment.get("speaker")) == active_id:
                            text = str(segment.get("text", "")).strip()
                            if text:
                                matching_text.append(text)
                    if matching_text:
                        st.caption(" ".join(matching_text)[:500])

# Collect all labels, including drafts for speakers whose text inputs are not
# currently rendered, untouched classifier prefills, and legacy annotations.
persist_active_speaker_draft(
    draft_key,
    active_id,
    speaker_info[active_id]["widget_key"] if active_id in speaker_info else None,
)
speaker_drafts = dict(st.session_state.get(draft_key, {}))
for speaker_id in speakers:
    widget_key = speaker_info[speaker_id]["widget_key"]
    widget_value = st.session_state.get(widget_key)
    if widget_value is not None:
        edited_labels[speaker_id] = str(widget_value).strip()
        # Mirror visible widget state into durable per-episode draft state.
        speaker_drafts[speaker_id] = str(widget_value).strip()
    elif speaker_id in speaker_drafts:
        edited_labels[speaker_id] = str(speaker_drafts[speaker_id]).strip()
    else:
        edited_labels[speaker_id] = str(edited_labels.get(speaker_id, "")).strip()
st.session_state[draft_key] = speaker_drafts

# Save controls.
st.divider()
labeled_count = sum(bool(v) for v in edited_labels.values())
save_col, info_col = st.columns([1, 3])

with save_col:
    if st.button("💾 Save episode labels", type="primary", use_container_width=True):
        save_episode(episode, edited_labels)
        # The enriched JSON is now the source of truth for this episode.
        st.session_state.pop(draft_key, None)
        for sid in speakers:
            st.session_state.pop(speaker_info[sid]["widget_key"], None)
        # Do not clear the global JSON cache here. Cache keys include file mtimes,
        # so the just-saved annotation naturally gets a fresh entry while all
        # other episodes (including the prefetched next episode) remain warm.
        # The manual-label and completion caches also include directory/file
        # fingerprints and therefore invalidate themselves on the next rerun.

        if hide_labeled:
            # If this episode is now complete it will vanish from the list,
            # so jump to the next unlabeled one (wrapping around to the start).
            updated = completion_map(episodes)
            if updated[episode["key"]]:
                pending = [
                    k for k in episode_keys[selected_index + 1:] + episode_keys[:selected_index]
                    if not updated[k]
                ]
                if pending:
                    st.session_state.selected_episode_key = pending[0]
        st.rerun()

with info_col:
    st.caption(
        f"{labeled_count}/{len(speakers)} speakers currently have labels. "
        "Saving writes a separate enriched JSON, including any accepted classifier prefill; the original remains untouched."
    )

with st.expander("Output format and paths"):
    st.code(
        "data/\n"
        "├── audio/                 # Original MP3 files\n"
        "├── diarized/              # Original diarization JSON (read-only)\n"
        "└── diarized_annotated/    # Enriched copies written by this app\n"
        "\n"
        'Each enriched JSON contains e.g.:\n'
        '"speaker_global": {\n'
        '  "SPEAKER_01": "Liz",\n'
        '  "SPEAKER_02": "Brace",\n'
        '  "SPEAKER_03": "Producer"\n'
        '}',
        language="text",
    )
    st.caption(
        "The original local speaker IDs remain unchanged in `segments`, "
        "`diarization`, and word-level records. The new mapping provides the "
        "episode-specific local-to-global identity lookup."
    )