#!/usr/bin/env python3
"""
test_lexical.py - run example queries and sanity checks against lexical.db.

Usage (from semanticsearch/lexical/):
    python test_lexical.py                          # full battery + checks
    python test_lexical.py -q "Barack Hussein Obama"            # one query, mode=phrase
    python test_lexical.py -q "late capitalism" -m and          # all words, any order
    python test_lexical.py -q "Obam*" -m raw                    # raw FTS5 syntax
    python test_lexical.py -q "Hussein" -m trigram              # needs --trigram build
    python test_lexical.py -q "Barack Hussein Obama" -m boundary

Modes
    phrase    exact consecutive words (unicode61, no stemming), ranked by bm25
    stem      phrase on the porter-stemmed index ("running" ~ "runs")
    and       every word must occur in the passage, any order
    or        any word may occur, bm25 ranked
    near      words within 10 tokens of each other
    raw       FTS5 query string as-is (prefix*, NEAR(), column filters, ...)
    trigram   substring search (only if the DB was built with --trigram)
    boundary  phrases that straddle two consecutive passages
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import schema

MODES = ("phrase", "stem", "and", "or", "near", "raw", "trigram", "boundary")


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def short(s: str, n: int = 70) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# --------------------------------------------------------------------------- #
# Search primitives
# --------------------------------------------------------------------------- #
def fts_search(conn: sqlite3.Connection, table: str, match: str, limit: int, col: int = 0):
    """Return (rows, total_hits, seconds) for an FTS5 MATCH, ranked by bm25."""
    t0 = time.perf_counter()
    rows = conn.execute(
        f"""
        SELECT p.passage_id, p.ordinal, p.episode_id, e.title, p.start_s, p.end_s,
               p.speakers, p.text,
               snippet({table}, {col}, '[', ']', ' … ', 28) AS snip
        FROM {table}
        JOIN passages p ON p.passage_id = {table}.rowid
        JOIN episodes e ON e.episode_id = p.episode_id
        WHERE {table} MATCH ?
        ORDER BY {table}.rank
        LIMIT ?
        """,
        (match, limit),
    ).fetchall()
    total = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {table} MATCH ?", (match,)
    ).fetchone()[0]
    return rows, total, time.perf_counter() - t0


def show(rows, total: int, secs: float, match: str, table: str, limit: int) -> None:
    print(f"table={table}  match={match!r}")
    print(f"{total:,} matching passages in {secs * 1000:.1f} ms; showing top {min(limit, len(rows))}")
    for i, r in enumerate(rows, 1):
        spk = " / ".join(schema.split_speakers(r["speakers"]))
        print(f"\n#{i:<2} {short(r['title'])}  [{ts(r['start_s'])}–{ts(r['end_s'])}]  "
              f"passage {r['passage_id']}  ({spk})")
        print("    " + r["snip"].replace("\n", " "))
    if rows:
        by_ep: dict[str, int] = defaultdict(int)
        titles: dict[str, str] = {}
        # Count per episode over ALL hits, not just the displayed page.
        for eid, title, n in conn_episode_counts(match, table):
            by_ep[eid] = n
            titles[eid] = title
        top = sorted(by_ep.items(), key=lambda kv: -kv[1])[:5]
        print("\n    Episodes with most hits: " +
              "; ".join(f"{short(titles[e], 40)} ({n})" for e, n in top))


_CONN: sqlite3.Connection | None = None


def conn_episode_counts(match: str, table: str):
    assert _CONN is not None
    return _CONN.execute(
        f"""
        SELECT p.episode_id, e.title, COUNT(*) AS n
        FROM {table}
        JOIN passages p ON p.passage_id = {table}.rowid
        JOIN episodes e ON e.episode_id = p.episode_id
        WHERE {table} MATCH ?
        GROUP BY p.episode_id
        """,
        (match,),
    ).fetchall()


def build_match(mode: str, q: str) -> tuple[str, str]:
    """Return (table, FTS5 match string) for a query in the given mode."""
    if mode == "phrase":
        return schema.FTS_EXACT, f"text : {schema.quote_phrase(q)}"
    if mode == "stem":
        return schema.FTS_STEM, f"text : {schema.quote_phrase(q)}"
    if mode == "and":
        return schema.FTS_EXACT, f"text : ({schema.quote_terms_and(q)})"
    if mode == "or":
        return schema.FTS_EXACT, f"text : ({schema.quote_terms_or(q)})"
    if mode == "near":
        terms = " ".join(schema.quote_phrase(t) for t in schema.words(q))
        return schema.FTS_EXACT, f"text : NEAR({terms}, 10)"
    if mode == "raw":
        return schema.FTS_EXACT, q
    if mode == "trigram":
        return schema.FTS_TRIGRAM, schema.quote_phrase(q)
    raise ValueError(mode)


def run_query(conn, mode: str, q: str, limit: int):
    if mode == "boundary":
        return run_boundary(conn, q, limit)
    table, match = build_match(mode, q)
    rows, total, secs = fts_search(conn, table, match, limit, col=0)
    show(rows, total, secs, match, table, limit)
    return rows, total, secs


# --------------------------------------------------------------------------- #
# Phrases straddling a passage boundary
# --------------------------------------------------------------------------- #
def boundary_matches(conn: sqlite3.Connection, phrase: str, max_candidates: int = 3000):
    """Find occurrences of `phrase` split across passage N and passage N+1 of one episode.

    For each split point k, passages ending with phrase[:k] are found with FTS,
    then the next passage (same episode, ordinal+1) is checked to start with phrase[k:].
    """
    toks = [t.casefold() for t in schema.words(phrase)]
    found = []
    for k in range(1, len(toks)):
        head, tail = toks[:k], toks[k:]
        cands = conn.execute(
            f"""SELECT p.passage_id, p.ordinal, p.episode_id, p.start_s, p.end_s, p.text
                FROM {schema.FTS_EXACT}
                JOIN passages p ON p.passage_id = {schema.FTS_EXACT}.rowid
                WHERE {schema.FTS_EXACT} MATCH ? LIMIT ?""",
            (f"text : {schema.quote_phrase(' '.join(head))}", max_candidates),
        ).fetchall()
        for c in cands:
            ctoks = [t.casefold() for t in schema.words(c["text"])]
            if ctoks[-k:] != head:
                continue
            nxt = conn.execute(
                "SELECT passage_id, start_s, end_s, text FROM passages "
                "WHERE ordinal = ? AND episode_id = ?",
                (c["ordinal"] + 1, c["episode_id"]),
            ).fetchone()
            if not nxt:
                continue
            ntoks = [t.casefold() for t in schema.words(nxt["text"])]
            if ntoks[: len(tail)] == tail:
                found.append((c, nxt, k))
    return found


def run_boundary(conn, q: str, limit: int):
    t0 = time.perf_counter()
    found = boundary_matches(conn, q)
    secs = time.perf_counter() - t0
    print(f"phrase={q!r}: {len(found)} occurrence(s) split across two passages ({secs * 1000:.0f} ms)")
    for c, nxt, k in found[:limit]:
        title = conn.execute("SELECT title FROM episodes WHERE episode_id=?", (c["episode_id"],)).fetchone()[0]
        print(f"\n  {short(title)}  passages {c['passage_id']}→{nxt['passage_id']}  "
              f"[{ts(c['start_s'])}–{ts(nxt['end_s'])}]")
        print("    …" + " ".join(c["text"].split()[-8:]) + "  ||  " + " ".join(nxt["text"].split()[:8]) + "…")
    return found, len(found), secs


# --------------------------------------------------------------------------- #
# Battery of examples + checks
# --------------------------------------------------------------------------- #
class Checks:
    def __init__(self) -> None:
        self.results: list[tuple[bool, str]] = []

    def check(self, ok: bool, msg: str) -> None:
        self.results.append((bool(ok), msg))
        print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")

    def summary(self) -> int:
        failed = [m for ok, m in self.results if not ok]
        header(f"CHECKS: {len(self.results) - len(failed)}/{len(self.results)} passed")
        for m in failed:
            print(f"  FAILED: {m}")
        return 1 if failed else 0


def contains_phrase(text: str, phrase: str) -> bool:
    t = [x.casefold() for x in schema.words(text)]
    p = [x.casefold() for x in schema.words(phrase)]
    return any(t[i:i + len(p)] == p for i in range(len(t) - len(p) + 1))


def battery(conn: sqlite3.Connection, limit: int) -> int:
    checks = Checks()

    header("0. Database overview")
    meta = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM meta")}
    for k in ("schema_version", "built_at", "sqlite_version", "n_passages", "n_episodes", "has_trigram"):
        print(f"  {k:16s} {meta.get(k)}")
    n_p = conn.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
    n_e = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    checks.check(str(n_p) == meta.get("n_passages"), f"passage count {n_p:,} matches meta")
    checks.check(str(n_e) == meta.get("n_episodes"), f"episode count {n_e:,} matches meta")
    checks.check(meta.get("speaker_sep") == schema.SPEAKER_SEP, "speaker separator stored in meta")

    sp = conn.execute(
        "SELECT speakers, COUNT(*) n FROM passages GROUP BY speakers ORDER BY n DESC LIMIT 5"
    ).fetchall()
    print("\n  Most common speaker combinations:")
    for r in sp:
        print(f"    {r['n']:>7,}  {r['speakers'] or '(none)'}")

    # ---- 1. exact phrase -------------------------------------------------
    header('1. Exact phrase: "Barack Hussein Obama"  (exact matches first)')
    q = "Barack Hussein Obama"
    rows, n_phrase, secs = run_query(conn, "phrase", q, limit)
    checks.check(all(contains_phrase(r["text"], q) for r in rows), "every phrase hit really contains the phrase")
    checks.check(secs < 1.0, f"phrase query ran in {secs * 1000:.0f} ms (< 1 s)")

    # ---- 2. relaxed matches --------------------------------------------------
    header('2. Relaxed lexical variants of the same query')
    print("-- all words, any order (mode=and)")
    _, n_and, _ = run_query(conn, "and", q, 3)
    print("\n-- words within 10 tokens (mode=near)")
    _, n_near, _ = run_query(conn, "near", q, 3)
    print("\n-- stemmed phrase (mode=stem)")
    _, n_stem, _ = run_query(conn, "stem", q, 3)
    checks.check(n_phrase <= n_near <= n_and, f"hit counts nest: phrase {n_phrase} <= near {n_near} <= and {n_and}")
    checks.check(n_phrase <= n_stem, f"stemmed index finds at least as many phrase hits ({n_stem} >= {n_phrase})")

    # ---- 3. stemming -------------------------------------------------------------
    header('3. Stemming: "he runs" on the exact vs. the porter-stemmed index')
    _, n_ex, _ = run_query(conn, "phrase", "he runs", 2)
    _, n_st, _ = run_query(conn, "stem", "he runs", 2)
    checks.check(n_st >= n_ex, f'stem ("he runs") {n_st} >= exact {n_ex}')

    # ---- 4. prefix and raw FTS5 syntax ----------------------------------------
    header('4. Prefix search with raw FTS5 syntax: Obam*')
    _, n_pre, _ = run_query(conn, "raw", "text : Obam*", 3)
    checks.check(n_pre >= n_phrase, f"prefix 'Obam*' ({n_pre}) >= phrase hits ({n_phrase})")

    # ---- 5. speaker filter -----------------------------------------------------------
    header("5. Speaker column filter: passages where a given speaker says a word")
    top_speakers = conn.execute(
        "SELECT speakers FROM passages WHERE speakers != '' AND speakers NOT LIKE ? "
        "GROUP BY speakers ORDER BY COUNT(*) DESC LIMIT 1", (f"%{schema.SPEAKER_SEP}%",)
    ).fetchone()
    if top_speakers:
        spk = top_speakers[0]
        match = f'speakers : {schema.quote_phrase(spk)} AND text : "the"'
        rows, total, secs = fts_search(conn, schema.FTS_EXACT, match, 2)
        show(rows, total, secs, match, schema.FTS_EXACT, 2)
        checks.check(all(spk in schema.split_speakers(r["speakers"]) for r in rows),
                     f"speaker filter returns only passages featuring {spk!r}")

    # ---- 6. boundary --------------------------------------------------------------------
    header(f"6. Phrases split across consecutive passages: {q!r}")
    found, n_b, _ = run_boundary(conn, q, 3)
    print(f"\n  (these {n_b} occurrence(s) are invisible to a plain phrase query)")

    # ---- 7. trigram -----------------------------------------------------------------
    header("7. Substring search via trigram index")
    if meta.get("has_trigram") == "1":
        rows, n_tri, _ = run_query(conn, "trigram", "Hussein", 3)
        checks.check(n_tri > 0, "trigram substring search returns results")
    else:
        print("  skipped: database built without --trigram")

    # ---- 8. query safety ------------------------------------------------------------------
    header("8. Hostile / odd query strings are handled safely")
    for nasty in ['he said "stop"', "C++ AND OR NOT", "'; DROP TABLE passages;--", "***", "   "]:
        try:
            if not schema.words(nasty):
                print(f"  {nasty!r}: no word tokens, skipped (caller should return no results)")
                continue
            table, match = build_match("phrase", nasty)
            fts_search(conn, table, match, 1)
            checks.check(True, f"phrase-quoted query {nasty!r} executes")
        except sqlite3.Error as e:
            checks.check(False, f"query {nasty!r} raised {e}")

    return checks.summary()


# --------------------------------------------------------------------------- #
def main() -> int:
    global _CONN
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=here / "lexical.db")
    ap.add_argument("-q", "--query", help="run a single query instead of the battery")
    ap.add_argument("-m", "--mode", choices=MODES, default="phrase")
    ap.add_argument("-n", "--limit", type=int, default=5)
    args = ap.parse_args()

    if not args.db.exists():
        print(f"database not found: {args.db}")
        return 2
    conn = schema.connect(args.db, readonly=True)
    _CONN = conn

    if args.query:
        if args.mode == "trigram" and schema.get_meta(conn, "has_trigram") != "1":
            print("This database was built without --trigram.")
            return 2
        try:
            run_query(conn, args.mode, args.query, args.limit)
        except sqlite3.OperationalError as e:
            print(f"FTS5 query error: {e}")
            return 1
        return 0
    return battery(conn, args.limit)


if __name__ == "__main__":
    sys.exit(main())