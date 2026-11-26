#!/usr/bin/env python3
"""
build_index.py - build lexical.db (SQLite + FTS5) from index/passages.jsonl and
index/episodes.jsonl (as written by build_passages.py).

Steps
-----
1. create schema in a temp file next to the target
2. load episodes, then stream-load passages (single transaction, batched)
3. derive episodes.n_passages and episodes.speakers
4. rebuild the FTS5 indexes from the content table and optimize them
5. validate (counts vs build_stats.json / episodes.jsonl, referential integrity,
   FTS integrity-check, phrase round-trip on a random sample, SQLite check)
6. atomically move the temp file to the final path

Usage (from semanticsearch/lexical/):
    python build_index.py
    python build_index.py --trigram
    python build_index.py --index-dir ../index --out lexical.db --force
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import schema

# --------------------------------------------------------------------------- #
# Source field aliases (first match wins). Matches build_passages.py output.
# --------------------------------------------------------------------------- #
PASSAGE_FIELDS = {
    "uid": ("passage_id", "id", "pid", "uid"),
    "episode_id": ("episode_id", "episode", "audio_id", "audio", "source"),
    "start": ("start", "start_s", "start_time"),
    "end": ("end", "end_s", "end_time"),
    "text": ("text", "passage", "content"),
    "speakers": ("speakers", "speaker_names", "speaker"),
    "segments": ("speaker_segments", "segments"),
    "word_count": ("word_count", "n_words"),
}
# Present on every passage but already stored once per episode (or derivable):
# dropped from passages.meta_json to avoid duplicating them 96k times.
PASSAGE_REDUNDANT = {"episode_title", "audio", "duration", "language"}

EPISODE_FIELDS = {
    "episode_id": ("episode_id", "id", "episode", "audio_id"),
    "title": ("episode_title", "title", "name"),
    "audio": ("audio", "audio_file", "file", "filename"),
    "annotation_file": ("annotation_file",),
    "duration": ("duration", "duration_s", "length"),
    "language": ("language", "lang"),
    "speakers": ("speakers", "speaker_names", "speaker_global"),
    "passage_count": ("passage_count", "n_passages"),
}

MEDIA_ID_RE = re.compile(r"\[(\d+)\]")
BATCH = 5000


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class BuildError(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick(d: dict, names: tuple[str, ...]) -> tuple[Any, str | None]:
    """Return (value, key_used) for the first alias present with a non-None value."""
    for n in names:
        if n in d and d[n] is not None:
            return d[n], n
    return None, None


def stem_of(name: str) -> str:
    """'Foo [123].mp3' / 'Foo [123].json' -> 'Foo [123]'."""
    return re.sub(r"\.(mp3|json|wav|m4a|txt)$", "", str(name), flags=re.I)


def speaker_list(v: Any) -> list[str]:
    """Normalize list / dict (speaker_global) / str into a de-duplicated list of names."""
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    elif isinstance(v, dict):
        v = list(v.values())
    out: list[str] = []
    for x in v:
        x = str(x).strip()
        if x and x not in out:
            out.append(x)
    return out


def read_jsonl(path: Path, skip_bad: bool) -> Iterator[tuple[int, dict]]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("line is not a JSON object")
            except (json.JSONDecodeError, ValueError) as e:
                if skip_bad:
                    log(f"WARNING: {path.name}:{lineno} skipped ({e})")
                    continue
                raise BuildError(f"{path.name}:{lineno}: {e}") from e
            yield lineno, obj


def dumps(obj: Any) -> str | None:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) if obj else None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_episodes(
    conn: sqlite3.Connection, path: Path, skip_bad: bool
) -> tuple[int, dict[str, int]]:
    """Insert episodes; return (count, {episode_id: passage_count claimed by the source})."""
    rows = []
    claimed: dict[str, int] = {}
    for lineno, ep in read_jsonl(path, skip_bad):
        eid, k_id = pick(ep, EPISODE_FIELDS["episode_id"])
        audio, k_audio = pick(ep, EPISODE_FIELDS["audio"])
        title, k_title = pick(ep, EPISODE_FIELDS["title"])
        afile, k_afile = pick(ep, EPISODE_FIELDS["annotation_file"])
        dur, k_dur = pick(ep, EPISODE_FIELDS["duration"])
        lang, k_lang = pick(ep, EPISODE_FIELDS["language"])
        spk, k_spk = pick(ep, EPISODE_FIELDS["speakers"])
        cnt, k_cnt = pick(ep, EPISODE_FIELDS["passage_count"])

        if eid is None:
            if audio is None and title is None:
                raise BuildError(f"{path.name}:{lineno}: no episode id / audio / title field")
            eid = stem_of(audio if audio is not None else title)
        eid = str(eid)
        if eid in claimed:
            raise BuildError(
                f"{path.name}:{lineno}: duplicate episode_id {eid!r} (title {title!r}). "
                "Two annotation files share the same [id]; build_passages.py would also "
                "generate colliding passage ids. Remove/rename one of the annotation files."
            )
        claimed[eid] = int(cnt) if cnt is not None else -1

        title = str(title) if title is not None else stem_of(audio if audio else eid)
        m = MEDIA_ID_RE.search(str(audio or eid))
        used = {k for k in (k_id, k_audio, k_title, k_afile, k_dur, k_lang, k_spk, k_cnt) if k}
        extra = {k: v for k, v in ep.items() if k not in used}
        rows.append(
            (
                eid, title,
                str(audio) if audio is not None else None,
                str(afile) if afile is not None else None,
                int(m.group(1)) if m else None,
                float(dur) if dur is not None else None,
                str(lang) if lang is not None else None,
                schema.join_speakers(speaker_list(spk)),
                dumps(extra),
            )
        )
    conn.executemany(
        "INSERT INTO episodes(episode_id, title, audio, annotation_file, media_id, "
        "duration_s, language, speakers, meta_json) VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )
    return len(rows), claimed


def load_passages(
    conn: sqlite3.Connection, path: Path, skip_bad: bool
) -> tuple[int, dict[str, list[str]]]:
    """Stream-insert passages; return (count, {episode_id: distinct speakers in order})."""
    sql = (
        "INSERT INTO passages(passage_id, ordinal, passage_uid, episode_id, start_s, end_s, "
        "text, speakers, speaker_segments, n_words, meta_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)"
    )
    batch: list[tuple] = []
    ep_speakers: dict[str, list[str]] = {}
    n = 0
    t0 = time.time()
    for lineno, p in read_jsonl(path, skip_bad):
        text, k_text = pick(p, PASSAGE_FIELDS["text"])
        eid, k_eid = pick(p, PASSAGE_FIELDS["episode_id"])
        start, k_start = pick(p, PASSAGE_FIELDS["start"])
        end, k_end = pick(p, PASSAGE_FIELDS["end"])
        uid, k_uid = pick(p, PASSAGE_FIELDS["uid"])
        spk, k_spk = pick(p, PASSAGE_FIELDS["speakers"])
        segs, k_segs = pick(p, PASSAGE_FIELDS["segments"])
        wc, k_wc = pick(p, PASSAGE_FIELDS["word_count"])

        missing = [name for name, v in
                   (("text", text), ("episode_id", eid), ("start", start), ("end", end))
                   if v is None]
        if missing:
            raise BuildError(
                f"{path.name}:{lineno}: missing field(s) {missing}; keys present: {sorted(p)}. "
                "Add aliases to PASSAGE_FIELDS in build_index.py."
            )
        text = str(text).strip()
        eid = stem_of(eid) if k_eid in ("audio", "source") else str(eid)
        start, end = float(start), float(end)
        n += 1
        uid = str(uid) if uid is not None else f"{eid}@{start:.2f}"

        names = speaker_list(spk)
        seen = ep_speakers.setdefault(eid, [])
        for s in names:
            if s not in seen:
                seen.append(s)

        used = {k for k in (k_text, k_eid, k_start, k_end, k_uid, k_spk, k_segs, k_wc) if k}
        extra = {k: v for k, v in p.items() if k not in used and k not in PASSAGE_REDUNDANT}
        batch.append(
            (n, n - 1, uid, eid, start, end, text, schema.join_speakers(names),
             dumps(segs), int(wc) if wc is not None else len(text.split()), dumps(extra))
        )
        if len(batch) >= BATCH:
            conn.executemany(sql, batch)
            batch.clear()
            if n % (BATCH * 4) == 0:
                log(f"  loaded {n:,} passages ({n / (time.time() - t0):,.0f}/s)")
    if batch:
        conn.executemany(sql, batch)
    return n, ep_speakers


def finalize_episodes(conn: sqlite3.Connection, ep_speakers: dict[str, list[str]]) -> None:
    conn.execute(
        "UPDATE episodes SET n_passages = "
        "(SELECT COUNT(*) FROM passages p WHERE p.episode_id = episodes.episode_id)"
    )
    # Source episodes.jsonl has no speakers: derive from passages unless already given.
    conn.executemany(
        "UPDATE episodes SET speakers = ? WHERE episode_id = ? AND (speakers IS NULL OR speakers = '')",
        [(schema.join_speakers(v), k) for k, v in ep_speakers.items()],
    )


# --------------------------------------------------------------------------- #
# FTS
# --------------------------------------------------------------------------- #
def build_fts(conn: sqlite3.Connection, with_trigram: bool) -> None:
    for name in schema.fts_tables(with_trigram):
        t = time.time()
        log(f"building {name} ...")
        conn.execute(f"INSERT INTO {name}({name}) VALUES('rebuild')")
        conn.commit()
        log(f"  optimizing {name} ...")
        conn.execute(f"INSERT INTO {name}({name}) VALUES('optimize')")
        conn.commit()
        log(f"  {name} done in {time.time() - t:.1f}s")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate(
    conn: sqlite3.Connection,
    *,
    n_loaded: int,
    expected_passages: int | None,
    expected_episodes: int | None,
    claimed_counts: dict[str, int],
    with_trigram: bool,
    sample: int,
    seed: int,
) -> list[str]:
    """Return a list of warnings; raise BuildError on hard failures."""
    warnings: list[str] = []
    q1 = lambda sql, *a: conn.execute(sql, a).fetchone()[0]  # noqa: E731

    # --- counts -----------------------------------------------------------
    n_pass = q1("SELECT COUNT(*) FROM passages")
    n_eps = q1("SELECT COUNT(*) FROM episodes")
    log(f"validate: {n_pass:,} passages, {n_eps:,} episodes")
    if n_pass != n_loaded:
        raise BuildError(f"passages table has {n_pass} rows but {n_loaded} were loaded")
    if expected_passages is not None and n_pass != expected_passages:
        raise BuildError(f"expected {expected_passages:,} passages, found {n_pass:,}")
    if expected_episodes is not None and n_eps != expected_episodes:
        raise BuildError(f"expected {expected_episodes:,} episodes, found {n_eps:,}")

    # per-episode passage_count claimed by episodes.jsonl vs. what was loaded
    bad = [
        (eid, claimed_counts.get(eid), got)
        for eid, got in conn.execute("SELECT episode_id, n_passages FROM episodes")
        if claimed_counts.get(eid, -1) not in (-1, got)
    ]
    if bad:
        ex = "; ".join(f"{e!r}: source says {c}, loaded {g}" for e, c, g in bad[:5])
        raise BuildError(f"{len(bad)} episode(s) with passage_count mismatch: {ex}")

    # --- basic row sanity ---------------------------------------------------
    if q1("SELECT COUNT(*) FROM passages WHERE LENGTH(TRIM(text)) = 0"):
        raise BuildError("there are passages with empty text")
    if n_pass and (q1("SELECT MIN(ordinal) FROM passages") != 0
                   or q1("SELECT MAX(ordinal) FROM passages") != n_pass - 1):
        raise BuildError("ordinal is not a contiguous 0..N-1 range")
    zero_dur = q1("SELECT COUNT(*) FROM passages WHERE end_s = start_s")
    if zero_dur:
        warnings.append(f"{zero_dur} passages have zero duration")

    # --- referential integrity ---------------------------------------------
    orphans = conn.execute(
        "SELECT DISTINCT p.episode_id FROM passages p "
        "LEFT JOIN episodes e USING(episode_id) WHERE e.episode_id IS NULL LIMIT 10"
    ).fetchall()
    if orphans:
        raise BuildError(
            "passages reference unknown episodes: " + ", ".join(repr(r[0]) for r in orphans)
        )
    if conn.execute("PRAGMA foreign_key_check").fetchall():
        raise BuildError("PRAGMA foreign_key_check reported violations")

    empty_eps = conn.execute(
        "SELECT episode_id FROM episodes WHERE n_passages = 0 LIMIT 20"
    ).fetchall()
    if empty_eps:
        warnings.append(f"episodes without passages, e.g. {empty_eps[0][0]!r}")

    over = q1(
        "SELECT COUNT(*) FROM passages p JOIN episodes e USING(episode_id) "
        "WHERE e.duration_s IS NOT NULL AND p.end_s > e.duration_s + 5"
    )
    if over:
        warnings.append(f"{over} passages end more than 5 s after the episode duration")

    # --- FTS consistency ----------------------------------------------------
    for name in schema.fts_tables(with_trigram):
        if name != schema.FTS_TRIGRAM:  # trigram table here has no docsize requirement
            n_idx = q1(f"SELECT COUNT(*) FROM {name}_docsize")
            if n_idx != n_pass:
                raise BuildError(f"{name}: indexed {n_idx} docs but passages has {n_pass}")
        try:
            # rank=1 additionally compares the index against the content table
            conn.execute(f"INSERT INTO {name}({name}, rank) VALUES('integrity-check', 1)")
        except sqlite3.DatabaseError as e:
            raise BuildError(f"{name}: FTS5 integrity-check failed: {e}") from e
        log(f"validate: {name} integrity-check ok")

    # --- phrase round-trip on a random sample --------------------------------
    rng = random.Random(seed)
    ids = [r[0] for r in conn.execute("SELECT passage_id FROM passages")]
    picked = rng.sample(ids, min(sample, len(ids)))
    failures = checked = 0
    for pid in picked:
        text = conn.execute("SELECT text FROM passages WHERE passage_id = ?", (pid,)).fetchone()[0]
        toks = schema.words(text)
        if len(toks) < 3:
            continue
        i = rng.randrange(0, len(toks) - 2)
        phrase = " ".join(toks[i:i + 3])
        checked += 1
        hit = conn.execute(
            f"SELECT 1 FROM {schema.FTS_EXACT} WHERE {schema.FTS_EXACT} MATCH ? AND rowid = ?",
            (f"text : {schema.quote_phrase(phrase)}", pid),
        ).fetchone()
        if not hit:
            failures += 1
            if failures <= 5:
                log(f"  phrase miss: passage {pid} phrase {phrase!r}")
    log(f"validate: phrase round-trip {checked - failures}/{checked} ok")
    if checked and failures / checked > 0.01:
        raise BuildError(f"phrase round-trip failed for {failures}/{checked} samples")
    if failures:
        warnings.append(f"{failures}/{checked} sampled phrases not found (tokenizer edge cases)")

    # --- SQLite's own check ---------------------------------------------------
    res = [r[0] for r in conn.execute("PRAGMA integrity_check")]
    if res != ["ok"]:
        raise BuildError(f"PRAGMA integrity_check: {res[:5]}")
    log("validate: PRAGMA integrity_check ok")

    return warnings


def read_build_stats(stats_path: Path) -> dict:
    if not stats_path.exists():
        return {}
    try:
        return json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    # Titles contain characters such as U+FF1A that a Windows console may not encode.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", type=Path, default=here.parent / "index")
    ap.add_argument("--out", type=Path, default=here / "lexical.db")
    ap.add_argument("--trigram", action="store_true",
                    help="also build a trigram FTS table (substring / misspelling search; larger DB)")
    ap.add_argument("--expected-passages", type=int, default=None,
                    help="fail if the passage count differs "
                         "(default: 'passages_created' from build_stats.json if present)")
    ap.add_argument("--skip-bad", action="store_true", help="skip malformed jsonl lines instead of failing")
    ap.add_argument("--force", action="store_true", help="overwrite an existing database")
    ap.add_argument("--no-vacuum", action="store_true", help="skip the final VACUUM")
    ap.add_argument("--sample", type=int, default=2000, help="number of phrase round-trip checks")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    passages_path = args.index_dir / "passages.jsonl"
    episodes_path = args.index_dir / "episodes.jsonl"
    for p in (passages_path, episodes_path):
        if not p.exists():
            log(f"ERROR: {p} not found")
            return 2
    if args.out.exists() and not args.force:
        log(f"ERROR: {args.out} exists (use --force to overwrite)")
        return 2

    stats = read_build_stats(args.index_dir / "build_stats.json")
    expected_passages = args.expected_passages
    if expected_passages is None:
        expected_passages = stats.get("passages_created")
    expected_episodes = stats.get("episodes_in_output")
    if expected_passages is not None:
        log(f"expected from build_stats.json: {expected_passages:,} passages, "
            f"{expected_episodes if expected_episodes is not None else '?'} episodes")

    tmp = args.out.with_suffix(args.out.suffix + ".tmp")

    def cleanup_tmp() -> None:
        for suffix in ("", "-journal", "-wal", "-shm"):
            Path(str(tmp) + suffix).unlink(missing_ok=True)

    cleanup_tmp()
    t_start = time.time()
    conn = schema.connect(tmp, bulk_load=True)
    try:
        log(f"SQLite {sqlite3.sqlite_version}; building {tmp}")
        schema.create_schema(conn, with_trigram=args.trigram)

        conn.execute("BEGIN")
        n_eps, claimed = load_episodes(conn, episodes_path, args.skip_bad)
        log(f"loaded {n_eps:,} episodes")
        n_pass, ep_speakers = load_passages(conn, passages_path, args.skip_bad)
        log(f"loaded {n_pass:,} passages")
        finalize_episodes(conn, ep_speakers)
        conn.commit()

        build_fts(conn, args.trigram)

        warnings = validate(
            conn, n_loaded=n_pass,
            expected_passages=expected_passages, expected_episodes=expected_episodes,
            claimed_counts=claimed, with_trigram=args.trigram,
            sample=args.sample, seed=args.seed,
        )

        for k, v in (
            ("schema_version", schema.SCHEMA_VERSION),
            ("built_at", time.strftime("%Y-%m-%dT%H:%M:%S%z")),
            ("sqlite_version", sqlite3.sqlite_version),
            ("n_passages", n_pass),
            ("n_episodes", n_eps),
            ("has_trigram", int(args.trigram)),
            ("speaker_sep", schema.SPEAKER_SEP),
            ("tokenizers", json.dumps(schema.TOKENIZERS)),
            ("source_passages", passages_path),
            ("source_build_stats", json.dumps(stats)),
        ):
            schema.set_meta(conn, k, v)
        conn.commit()

        conn.execute("ANALYZE")
        conn.commit()
        conn.execute("PRAGMA journal_mode = DELETE")  # final file: no sidecars
        if not args.no_vacuum:
            log("vacuum ...")
            conn.execute("VACUUM")
        conn.close()
    except BaseException:
        conn.close()
        cleanup_tmp()
        raise

    os.replace(tmp, args.out)
    size_mb = args.out.stat().st_size / 1e6
    for w in warnings:
        log(f"WARNING: {w}")
    log(f"OK: {args.out} ({size_mb:,.0f} MB) in {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BuildError as e:
        log(f"BUILD FAILED: {e}")
        sys.exit(1)