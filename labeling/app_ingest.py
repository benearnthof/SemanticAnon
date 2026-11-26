from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import streamlit as st


# ---------------------------------------------------------------------------
# Configuration — retain the existing project layout and output format.
# ---------------------------------------------------------------------------
# TODO: unify root paths such that every file is consistent. 
# Data should be located in /workspace/data 
# expected /workspace/SemanticAnon/labeling/data/audio
# expected /workspace/SemanticAnon/labeling/data/diarized
ROOT = Path("/workspace/")
AUDIO_DIR = ROOT / "data" / "audio"
DIARIZED_DIR = ROOT / "data" / "diarized"
OUTPUT_DIR = ROOT / "data" / "diarized_annotated"

SAMPLES_PER_SPEAKER = 6
MIN_TURN_SECONDS = 1.0
SNIPPET_PADDING_SECONDS = 0.45
MAX_SNIPPET_SECONDS = 12.0


# ---------------------------------------------------------------------------
# Small, on-demand file helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def collect_speakers(data: dict[str, Any]) -> list[str]:
    """Collect speaker IDs from diarization, segments, words, and embeddings."""
    found: set[str] = set()

    for item in data.get("diarization", []) or []:
        if item.get("speaker"):
            found.add(str(item["speaker"]))

    for segment in data.get("segments", []) or []:
        if segment.get("speaker"):
            found.add(str(segment["speaker"]))
        for word in segment.get("words", []) or []:
            if word.get("speaker"):
                found.add(str(word["speaker"]))

    found.update(str(speaker) for speaker in (data.get("speaker_embeddings") or {}))
    return sorted(found)


def find_audio_file(diarized: dict[str, Any], json_path: Path) -> Path | None:
    """Resolve the audio by its recorded filename, then by the matching stem."""
    audio_value = diarized.get("audio")
    if audio_value:
        audio_name = Path(str(audio_value)).name
        candidates = (AUDIO_DIR / audio_name, AUDIO_DIR / f"{Path(audio_name).stem}.mp3")
        for candidate in candidates:
            if candidate.is_file():
                return candidate

    # episode_ingest.py writes diarization JSON using the audio file's stem.
    direct = AUDIO_DIR / f"{json_path.stem}.mp3"
    if direct.is_file():
        return direct

    # Retain support for projects with nested audio/diarized directories.
    matches = list(AUDIO_DIR.rglob(f"{json_path.stem}.mp3"))
    return matches[0] if matches else None


def discover_episodes() -> list[dict[str, Any]]:
    """
    Discover only episodes that have not been annotated yet.

    This deliberately does not open every diarization JSON at startup. The
    expensive JSON (including word alignment and speaker embeddings) is read
    only after the user selects an episode.
    """
    episodes: list[dict[str, Any]] = []
    if not DIARIZED_DIR.exists() or not AUDIO_DIR.exists():
        return episodes

    for json_path in sorted(DIARIZED_DIR.rglob("*.json")):
        relative = json_path.relative_to(DIARIZED_DIR)
        output_path = OUTPUT_DIR / relative

        # Existing enriched files are already in the annotation archive.
        # New work is restricted to episodes with no saved annotation file.
        if output_path.exists():
            continue

        # Match ingestion's stem-based naming without loading the diarization.
        audio_path = AUDIO_DIR / relative.with_suffix(".mp3")
        if not audio_path.is_file():
            # Fallback for legacy files whose audio field does not share the stem.
            # Read only this unmatched candidate, not the whole corpus.
            try:
                audio_path = find_audio_file(load_json(json_path), json_path)
            except (OSError, json.JSONDecodeError):
                audio_path = None
        if audio_path is None:
            continue

        title = json_path.stem
        description = ""
        metadata_path = audio_path.with_suffix(".info.json")
        if metadata_path.is_file():
            try:
                metadata = load_json(metadata_path)
                title = str(metadata.get("title") or title)
                description = str(metadata.get("description") or "")
            except (OSError, json.JSONDecodeError):
                pass

        episodes.append({
            "key": relative.as_posix(),
            "title": title,
            "description": description,
            "json_path": json_path,
            "audio_path": audio_path,
            "output_path": output_path,
        })

    return episodes


@st.cache_data(show_spinner=False, max_entries=4)
def load_selected_episode(path_string: str, mtime_ns: int) -> dict[str, Any]:
    """Cache only the currently selected episode; mtime invalidates changed data."""
    del mtime_ns
    return load_json(Path(path_string))


def speaker_turns(data: dict[str, Any], speaker_id: str) -> list[dict[str, float]]:
    turns: list[dict[str, float]] = []
    for item in data.get("diarization", []) or []:
        if str(item.get("speaker")) != speaker_id:
            continue
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            turns.append({"start": start, "end": end})

    if not turns:
        for item in data.get("segments", []) or []:
            if str(item.get("speaker")) != speaker_id:
                continue
            try:
                start, end = float(item["start"]), float(item["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if end > start:
                turns.append({"start": start, "end": end})

    return sorted(turns, key=lambda turn: turn["start"])


def choose_representative_turns(
    turns: list[dict[str, float]], duration: float,
    count: int = SAMPLES_PER_SPEAKER,
) -> list[dict[str, float]]:
    candidates = [
        turn for turn in turns
        if turn["end"] - turn["start"] >= MIN_TURN_SECONDS
    ] or turns[:]
    if not candidates:
        return []

    selected: list[dict[str, float]] = []
    if duration > 0:
        for index in range(count):
            left, right = duration * index / count, duration * (index + 1) / count
            in_bin = [
                turn for turn in candidates
                if turn["start"] < right and turn["end"] > left
            ]
            if in_bin:
                best = max(in_bin, key=lambda turn: turn["end"] - turn["start"])
                if best not in selected:
                    selected.append(best)

    remaining = sorted(
        (turn for turn in candidates if turn not in selected),
        key=lambda turn: turn["end"] - turn["start"],
        reverse=True,
    )
    for turn in remaining:
        if len(selected) >= count:
            break
        selected.append(turn)

    return sorted(selected[:count], key=lambda turn: turn["start"])


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


@st.cache_data(show_spinner=False, max_entries=120)
def extract_audio_snippet(
    audio_path_string: str, file_mtime_ns: int, start: float, end: float
) -> bytes:
    del file_mtime_ns
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-i", audio_path_string,
        "-t", f"{max(0.1, end - start):.3f}",
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", "-f", "wav", "pipe:1",
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if result.returncode != 0 or not result.stdout:
        message = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(message[-1200:] or "FFmpeg could not extract this snippet.")
    return result.stdout


def save_episode(episode: dict[str, Any], labels: dict[str, str]) -> None:
    """
    Preserve the original diarization structure and add the established
    speaker_global / speaker_annotation fields used by downstream scripts.
    """
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


# ---------------------------------------------------------------------------
# Streamlit application
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TrueAnon Speaker Annotator",
    page_icon="🎙️",
    layout="wide",
)

if not DIARIZED_DIR.is_dir() or not AUDIO_DIR.is_dir():
    st.error(
        "Could not find the expected data directories:\n\n"
        f"- `{AUDIO_DIR}`\n- `{DIARIZED_DIR}`"
    )
    st.stop()

episodes = discover_episodes()
total_pending = len(episodes)

st.title("Podcast speaker annotation")
st.caption(
    "This queue contains only episodes without an existing file in "
    "`data/diarized_annotated/`. Previously annotated episodes are excluded."
)

if not episodes:
    st.success("No new, unlabeled episodes found.")
    st.stop()

# Keep the selected episode stable as the queue shrinks after each save.
episode_by_key = {episode["key"]: episode for episode in episodes}
episode_keys = [episode["key"] for episode in episodes]
if st.session_state.get("selected_episode_key") not in episode_by_key:
    st.session_state.selected_episode_key = episode_keys[0]

selected_key = st.session_state.selected_episode_key
selected_index = episode_keys.index(selected_key)

progress_col, count_col = st.columns([3, 1])
with progress_col:
    st.progress(
        0.0,
        text=f"{total_pending} episode(s) awaiting annotation in this run",
    )
with count_col:
    st.metric("Unlabeled episodes", total_pending)

search = st.text_input("Find an episode", placeholder="Search title or filename")
visible_keys = [
    key for key in episode_keys
    if search.casefold() in (
        episode_by_key[key]["title"] + " " + key
    ).casefold()
]
if not visible_keys:
    st.info("No episodes match that search.")
    st.stop()

if selected_key not in visible_keys:
    selected_key = visible_keys[0]
    st.session_state.selected_episode_key = selected_key
selected_index = episode_keys.index(selected_key)

nav_prev, nav_select, nav_next = st.columns([1, 6, 1])
with nav_prev:
    if st.button(
        "← Previous", disabled=selected_index == 0, use_container_width=True
    ):
        st.session_state.selected_episode_key = episode_keys[selected_index - 1]
        st.rerun()
with nav_select:
    chosen = st.selectbox(
        "Episode",
        options=visible_keys,
        index=visible_keys.index(selected_key),
        format_func=lambda key: episode_by_key[key]["title"],
        label_visibility="collapsed",
    )
    if chosen != selected_key:
        st.session_state.selected_episode_key = chosen
        st.rerun()
with nav_next:
    if st.button(
        "Next →",
        disabled=selected_index >= len(episode_keys) - 1,
        use_container_width=True,
    ):
        st.session_state.selected_episode_key = episode_keys[selected_index + 1]
        st.rerun()

episode = episode_by_key[st.session_state.selected_episode_key]
try:
    data = load_selected_episode(
        str(episode["json_path"]), episode["json_path"].stat().st_mtime_ns
    )
except Exception as exc:
    st.error(f"Could not load diarization JSON: {exc}")
    st.stop()

speakers = collect_speakers(data)
if not speakers:
    st.warning("No speaker IDs were found in this diarization file.")
    st.stop()

audio_path = episode["audio_path"]
duration = float(data.get("duration") or 0.0)
if duration <= 0:
    endpoints: list[float] = []
    for field in ("diarization", "segments"):
        for item in data.get(field, []) or []:
            try:
                endpoints.append(float(item["end"]))
            except (KeyError, TypeError, ValueError):
                pass
    duration = max(endpoints, default=0.0)

st.divider()
st.subheader(episode["title"])
if episode.get("description"):
    st.markdown(episode["description"])
st.caption(f"Audio: `{audio_path.name}`")

st.write(f"**{len(speakers)} local speaker(s)** to identify.")
draft_key = f"speaker_labels::{episode['key']}"
if draft_key not in st.session_state:
    st.session_state[draft_key] = {}
labels = dict(st.session_state[draft_key])

# Only render/extract audio for the currently selected speaker. This keeps
# reruns and speaker switches independent of the number of speakers in an episode.
speaker_col, label_col = st.columns([1, 2])
with speaker_col:
    active_speaker = st.selectbox("Speaker to identify", speakers)
widget_key = f"speaker_label::{episode['key']}::{active_speaker}"
if widget_key not in st.session_state:
    st.session_state[widget_key] = labels.get(active_speaker, "")

with label_col:
    active_label = st.text_input(
        f"Global identity for {active_speaker}",
        key=widget_key,
        placeholder="e.g. Liz, Brace, Producer, guest name, clip/archival audio",
        help="Use the same spelling for a person across all episodes.",
    )
labels[active_speaker] = active_label.strip()

turns = speaker_turns(data, active_speaker)
total_speech = sum(turn["end"] - turn["start"] for turn in turns)
st.caption(f"{len(turns)} detected turns · approximately {total_speech / 60:.1f} min of speech")

samples = choose_representative_turns(turns, duration)
if not samples:
    st.info("No usable turns found for this speaker.")
else:
    for sample_index, turn in enumerate(samples):
        start, end = make_snippet_bounds(turn, duration)
        st.caption(
            f"Example {sample_index + 1} · "
            f"{math.floor(turn['start'] // 60):02d}:"
            f"{int(turn['start'] % 60):02d} – "
            f"{math.floor(turn['end'] // 60):02d}:"
            f"{int(turn['end'] % 60):02d}"
        )
        try:
            audio_bytes = extract_audio_snippet(
                str(audio_path), audio_path.stat().st_mtime_ns, start, end
            )
            st.audio(audio_bytes, format="audio/wav")
        except Exception as exc:
            st.error(f"Could not extract snippet: {exc}")

        transcript_parts = []
        for segment in data.get("segments", []) or []:
            try:
                overlaps = (
                    float(segment["start"]) < turn["end"]
                    and float(segment["end"]) > turn["start"]
                )
            except (KeyError, TypeError, ValueError):
                overlaps = False
            if overlaps and str(segment.get("speaker")) == active_speaker:
                text = str(segment.get("text", "")).strip()
                if text:
                    transcript_parts.append(text)
        if transcript_parts:
            st.caption(" ".join(transcript_parts)[:500])

# Preserve all per-speaker drafts, including speakers not currently selected.
for speaker_id in speakers:
    key = f"speaker_label::{episode['key']}::{speaker_id}"
    if key in st.session_state:
        labels[speaker_id] = str(st.session_state[key]).strip()

st.session_state[draft_key] = labels
missing = [speaker for speaker in speakers if not labels.get(speaker, "").strip()]

st.divider()
save_col, status_col = st.columns([1, 3])
with save_col:
    if st.button(
        "Save episode labels",
        type="primary",
        use_container_width=True,
        disabled=bool(missing),
    ):
        save_episode(episode, labels)
        st.session_state.pop(draft_key, None)
        for speaker_id in speakers:
            st.session_state.pop(
                f"speaker_label::{episode['key']}::{speaker_id}", None
            )

        # Move directly to the next unannotated episode. The saved file makes
        # this episode disappear from the next discovery pass.
        remaining = [key for key in episode_keys if key != episode["key"]]
        if remaining:
            st.session_state.selected_episode_key = remaining[
                min(selected_index, len(remaining) - 1)
            ]
        st.rerun()

with status_col:
    if missing:
        st.warning(
            f"{len(missing)} speaker label(s) still required before saving: "
            + ", ".join(missing)
        )
    else:
        st.caption(
            f"All {len(speakers)} speakers have labels. Saving writes an enriched "
            "JSON copy; the original diarization file remains unchanged."
        )

with st.expander("Output format and paths"):
    st.code(
        "data/\n"
        "├── audio/                 # Original MP3 files\n"
        "├── diarized/              # Original diarization JSON (read-only)\n"
        "└── diarized_annotated/    # Enriched copies written by this app\n\n"
        'Each enriched JSON retains the original fields and adds:\n'
        '"speaker_global": {\n'
        '  "SPEAKER_01": "Liz",\n'
        '  "SPEAKER_02": "Brace"\n'
        '},\n'
        '"speaker_annotation": {\n'
        '  "schema_version": 1,\n'
        '  "method": "manual_streamlit_annotation"\n'
        '}',
        language="text",
    )
    st.caption(
        "Local speaker IDs and all original segment, diarization, and word-level "
        "records are preserved. The speaker_global mapping remains compatible "
        "with downstream processing."
    )
