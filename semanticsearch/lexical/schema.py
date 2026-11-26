"""
schema.py - SQLite schema, FTS5 configuration and connection helpers for the
lexical half of the podcast search index.

Layout
------
episodes            one row per episode
passages            one row per 30-60 s snippet (the unit that is embedded)
passages_fts_exact  FTS5, unicode61 (no stemming): exact words / phrases
passages_fts_stem   FTS5, porter+unicode61: inflection-tolerant matching
passages_fts_tri    FTS5, trigram (optional): substring / misspelling matching
meta                key/value build information

The FTS tables are *external-content* tables (content='passages'), so the
passage text is stored exactly once. The database is built once and treated as
read-only afterwards, therefore no sync triggers are defined; rebuild with
build_index.py if the data changes.

`passages.passage_id` is 1-based and follows the line order of passages.jsonl.
The 0-based row `ordinal` (= passage_id - 1) is stored explicitly so it can be
used to align with the embedding matrix row order.

Speaker lists are stored as strings joined with SPEAKER_SEP (speaker names may
contain spaces, e.g. "YOUNG CHOMSKY"); split with `split_speakers()`.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1
SPEAKER_SEP = " | "

# --------------------------------------------------------------------------- #
# Tokenizer configuration
# --------------------------------------------------------------------------- #
# unicode61: splits on anything that is not a unicode letter/digit, case-folds,
#            remove_diacritics 2 folds accents correctly ("Beyonce" == "Beyoncé").
#            Apostrophes split tokens ("don't" -> "don" "t"); this is applied
#            identically at index and query time so phrase queries stay exact.
# porter:    English stemmer wrapped around unicode61 ("running" == "runs").
# trigram:   3-char grams, enables LIKE/GLOB/substring MATCH (needs SQLite >= 3.34).
TOKENIZERS: dict[str, str] = {
    "exact": "unicode61 remove_diacritics 2",
    "stem": "porter unicode61 remove_diacritics 2",
    "trigram": "trigram case_sensitive 0",
}

FTS_EXACT = "passages_fts_exact"
FTS_STEM = "passages_fts_stem"
FTS_TRIGRAM = "passages_fts_tri"

# --------------------------------------------------------------------------- #
# DDL
# --------------------------------------------------------------------------- #
TABLES_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS episodes (
    episode_id       TEXT PRIMARY KEY,      -- bracketed numeric id of the file name, else file stem
    title            TEXT NOT NULL,         -- episode_title (file stem without " [id]")
    audio            TEXT,                  -- audio file name
    annotation_file  TEXT,                  -- diarized_annotated/*.json it came from
    media_id         INTEGER,               -- number in "[...]" of the audio name, if any
    duration_s       REAL,
    language         TEXT,
    speakers         TEXT,                  -- distinct speakers, joined with SPEAKER_SEP
    n_passages       INTEGER NOT NULL DEFAULT 0,
    meta_json        TEXT                   -- remaining source fields (lossless)
);

CREATE TABLE IF NOT EXISTS passages (
    passage_id        INTEGER PRIMARY KEY,  -- 1-based, jsonl line order (FTS rowid)
    ordinal           INTEGER NOT NULL UNIQUE,  -- 0-based, == embedding row index
    passage_uid       TEXT    NOT NULL UNIQUE,  -- "<episode_id>_<nnnnn>" from passages.jsonl
    episode_id        TEXT    NOT NULL REFERENCES episodes(episode_id),
    start_s           REAL    NOT NULL,
    end_s             REAL    NOT NULL,
    text              TEXT    NOT NULL,
    speakers          TEXT    NOT NULL DEFAULT '',  -- distinct speakers, joined with SPEAKER_SEP
    speaker_segments  TEXT,                 -- JSON [{"speaker","start","end"}, ...]
    n_words           INTEGER NOT NULL,
    meta_json         TEXT,
    CHECK (end_s >= start_s)
);

CREATE INDEX IF NOT EXISTS idx_passages_episode_time
    ON passages(episode_id, start_s);
"""


def _fts_ddl(name: str, tokenizer: str, columns: str) -> str:
    return (
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} USING fts5("
        f"{columns}, content='passages', content_rowid='passage_id', "
        f"tokenize='{tokenizer}');"
    )


def fts_ddl(with_trigram: bool = False) -> str:
    parts = [
        _fts_ddl(FTS_EXACT, TOKENIZERS["exact"], "text, speakers"),
        _fts_ddl(FTS_STEM, TOKENIZERS["stem"], "text, speakers"),
    ]
    if with_trigram:
        parts.append(_fts_ddl(FTS_TRIGRAM, TOKENIZERS["trigram"], "text"))
    return "\n".join(parts)


def fts_tables(with_trigram: bool = False) -> list[str]:
    names = [FTS_EXACT, FTS_STEM]
    if with_trigram:
        names.append(FTS_TRIGRAM)
    return names


# --------------------------------------------------------------------------- #
# Connection
# --------------------------------------------------------------------------- #
def connect(
    path: str | Path,
    *,
    readonly: bool = False,
    bulk_load: bool = False,
) -> sqlite3.Connection:
    """Open the database with sane pragmas.

    readonly:  open via URI mode=ro (for search.py)
    bulk_load: disable journaling/fsync for the one-off build (the build writes
               to a temp file and renames it, so a crash just discards it)
    """
    path = Path(path)
    if readonly:
        conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -262144")  # 256 MiB
    if readonly:
        conn.execute("PRAGMA mmap_size = 1073741824")
        conn.execute("PRAGMA query_only = ON")
    elif bulk_load:
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.execute("PRAGMA locking_mode = EXCLUSIVE")
    return conn


def require_fts5(conn: sqlite3.Connection, need_trigram: bool = False) -> None:
    """Raise a clear error if this SQLite build lacks FTS5 (or trigram)."""
    opts = {r[0] for r in conn.execute("PRAGMA compile_options")}
    if "ENABLE_FTS5" not in opts:
        raise RuntimeError("This SQLite build does not include FTS5.")
    if need_trigram and sqlite3.sqlite_version_info < (3, 34, 0):
        raise RuntimeError(
            f"trigram tokenizer needs SQLite >= 3.34 (found {sqlite3.sqlite_version})."
        )


def create_schema(conn: sqlite3.Connection, with_trigram: bool = False) -> None:
    require_fts5(conn, need_trigram=with_trigram)
    conn.executescript(TABLES_DDL)
    conn.executescript(fts_ddl(with_trigram))


def set_meta(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None):
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


# --------------------------------------------------------------------------- #
# Helpers (used by build_index.py and search.py)
# --------------------------------------------------------------------------- #
def join_speakers(names) -> str:
    return SPEAKER_SEP.join(names)


def split_speakers(s: str | None) -> list[str]:
    return [x for x in (s or "").split(SPEAKER_SEP) if x]


_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)  # roughly what unicode61 keeps


def words(s: str) -> list[str]:
    """Split a string into word tokens approximating the FTS5 unicode61 tokenizer."""
    return _WORD_RE.findall(s)


def quote_phrase(s: str) -> str:
    """Turn arbitrary user text into a safe FTS5 phrase query: "barack hussein obama"."""
    return '"' + s.replace('"', '""') + '"'


def quote_terms_and(s: str) -> str:
    """All words must occur in the passage, any order: "a" AND "b" AND "c"."""
    return " AND ".join(quote_phrase(t) for t in words(s))


def quote_terms_or(s: str) -> str:
    """Any word may occur (ranked by bm25): "a" OR "b" OR "c"."""
    return " OR ".join(quote_phrase(t) for t in words(s))