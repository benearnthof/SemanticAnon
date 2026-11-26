from __future__ import annotations

import json
import math
import subprocess
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

# Number of examples shown for each local speaker.
SAMPLES_PER_SPEAKER = 6

# Keep snippets short enough to review quickly.
MIN_TURN_SECONDS = 1.0
SNIPPET_PADDING_SECONDS = 0.45
MAX_SNIPPET_SECONDS = 12.0

GLOBAL_LABEL_EXAMPLES = (
    "Liz, Brace, Producer, guest name, clip/archival audio, unknown"
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


@st.cache_data(ttl=3600, show_spinner=False)
def discover_episodes() -> list[dict[str, Any]]:
    if not DIARIZED_DIR.exists():
        return []

    audio_index = build_audio_index()
    episodes = []

    for json_path in sorted(DIARIZED_DIR.glob("*.json")):
        try:
            diarized = load_json(json_path)
        except Exception:
            continue

        audio_path = find_audio_file(diarized, json_path, audio_index)
        if audio_path is None:
            continue

        title = json_path.stem
        metadata_path = audio_path.with_suffix(".info.json")
        if metadata_path.exists():
            try:
                metadata = load_json(metadata_path)
                title = metadata.get("title") or title
            except Exception:
                pass

        output_path = OUTPUT_DIR / json_path.name
        episodes.append({
            "key": json_path.name,
            "title": title,
            "json_path": json_path,
            "audio_path": audio_path,
            "output_path": output_path,
        })

    return episodes


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
# Annotation persistence
# ---------------------------------------------------------------------------
def load_annotated_copy(episode: dict[str, Any]) -> dict[str, Any]:
    """
    Always start from the original diarization JSON. If an enriched copy exists,
    reuse only its speaker_global mapping. This avoids accidentally accumulating
    changes to legacy diarization fields.
    """
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

st.title("🎙️ TrueAnon — Global Speaker Annotation")
st.caption(
    "Assign persistent identities to episode-local diarization speakers. "
    "Original diarization files are never modified."
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

# Search and episode navigation.
search_col, hide_col = st.columns([4, 1], vertical_alignment="bottom")
with search_col:
    search = st.text_input(
        "Find an episode", placeholder="Search episode title or filename"
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
        st.rerun()

with nav_select:
    chosen_key = st.selectbox(
        "Episode",
        options=episode_keys,
        index=selected_index,
        format_func=lambda key: next(ep["title"] for ep in filtered if ep["key"] == key),
        label_visibility="collapsed",
    )
    if chosen_key != st.session_state.selected_episode_key:
        st.session_state.selected_episode_key = chosen_key
        st.rerun()

with nav_next:
    if st.button("Next →", disabled=(selected_index >= len(episode_keys) - 1),
                 use_container_width=True):
        st.session_state.selected_episode_key = episode_keys[selected_index + 1]
        st.rerun()

episode = next(ep for ep in filtered if ep["key"] == st.session_state.selected_episode_key)
data = load_annotated_copy(episode)
speakers = collect_speakers(data)
audio_path = episode["audio_path"]

st.divider()
st.subheader(episode["title"])
st.caption(f"Audio: `{audio_path.name}`  ·  Diarization: `{episode['json_path'].name}`")

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
st.write(
    f"**{len(speakers)} local speakers** · "
    f"{sum(bool(str(existing_labels.get(s, '')).strip()) for s in speakers)} labeled"
)

# Render one review panel per local speaker.
edited_labels: dict[str, str] = {}

for speaker_id in speakers:
    turns = speaker_turns(data, speaker_id)
    samples = choose_representative_turns(turns, duration)

    current_label = str(existing_labels.get(speaker_id, ""))
    heading_label = current_label if current_label else "Unlabeled"

    with st.expander(
        f"{speaker_id}  —  {heading_label}  "
        f"({len(turns)} turns; {len(samples)} examples)",
        expanded=(not bool(current_label)),
    ):
        col_label, col_stats = st.columns([3, 1])

        widget_key = f"global_label::{episode['key']}::{speaker_id}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = current_label

        with col_label:
            label = st.text_input(
                f"Global identity for {speaker_id}",
                key=widget_key,
                placeholder="e.g. Liz, Brace, Producer, guest name, clip/archival audio",
                help="Use the same spelling for a person across all episodes.",
            )
            edited_labels[speaker_id] = label.strip()

        with col_stats:
            st.metric("Detected turns", len(turns))
            if turns:
                total_speech = sum(t["end"] - t["start"] for t in turns)
                st.caption(f"Approx. {total_speech / 60:.1f} min of speech")

        if not samples:
            st.info("No usable turns found for this speaker.")
            continue

        # Two-column snippet layout keeps the review compact.
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

                    # Show transcript text overlapping this example, when available.
                    matching_text = []
                    for segment in data.get("segments", []):
                        try:
                            overlaps = (
                                float(segment["start"]) < turn["end"]
                                and float(segment["end"]) > turn["start"]
                            )
                        except (KeyError, TypeError, ValueError):
                            overlaps = False
                        if overlaps and str(segment.get("speaker")) == speaker_id:
                            text = str(segment.get("text", "")).strip()
                            if text:
                                matching_text.append(text)
                    if matching_text:
                        st.caption(" ".join(matching_text)[:500])

# Save controls.
st.divider()
labeled_count = sum(bool(v) for v in edited_labels.values())
save_col, info_col = st.columns([1, 3])

with save_col:
    if st.button("💾 Save episode labels", type="primary", use_container_width=True):
        save_episode(episode, edited_labels)
        load_json_cached.clear()

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
        "Saving writes a separate enriched JSON and leaves the original untouched."
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