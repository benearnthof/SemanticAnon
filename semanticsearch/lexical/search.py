#!/usr/bin/env python3
"""
search.py - tiered lexical search over lexical.db.

Result tiers, in order (a result never appears in two tiers):
  1. phrase           the whole query as an exact consecutive-word phrase (bm25 ordered)
  2. phrase_spanning  the phrase split across two consecutive passages of one episode
  3. lexical          passages containing most of the query's content words (bm25 ordered)

Policy for the phrase tier
  * Every query gets a total hit count (exact up to COUNT_CAP, then a lower bound).
  * "Short" queries (fewer than 3 words, only stopwords, or < 10 letters) carry little
    filtering information, so at most `short_phrase_cap` (default 50) hits are returned
    per page.  Longer queries return up to `max_page` (default 200) per page.
    Either way `offset` pages through the rest.
  * No Boolean / FTS5 syntax is ever taken from the user: the query is reduced to
    word tokens and quoted.

Timestamps
  Passages carry only start/end, so the position of a match inside a passage is
  estimated by interpolating its character offset over the passage duration.
  `seek_s` backs off a few seconds from that estimate; `match_speaker` is the
  speaker active at the estimated time.  Both are approximations (a few seconds).

The hybrid layer can use:
  LexicalSearcher.phrase_hits / spanning_hits / bm25_hits   individual tiers
  LexicalSearcher.hydrate(ordinals)    turn embedding row indices into Hits
  LexicalSearcher.context(hit, ...)    neighbouring passages
  group_by_episode / merge_adjacent    result consolidation

CLI:
    python search.py "Barack Hussein Obama"
    python search.py "late stage capitalism" -n 20 --speaker LIZ --episodes
    python search.py "the" --offset 50        # next page of a short query
    python search.py "Noam Chomsky" --json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import threading
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from semanticsearch.lexical import schema

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
TIER_PHRASE = "phrase"
TIER_SPANNING = "phrase_spanning"
TIER_LEXICAL = "lexical"
TIER_DENSE = "dense"
TIER_CONTEXT = "context"
TIER_RANK = {TIER_PHRASE: 0, TIER_SPANNING: 1, TIER_LEXICAL: 2, TIER_DENSE: 3, TIER_CONTEXT: 4}

COUNT_CAP = 10_000            # total-hit counting stops here (reported as lower bound)
SEEK_BACKOFF_S = 3.0          # start playback this long before the estimated match
SPANNING_CANDIDATES = 5_000   # max FTS candidates examined per split point
MIN_INFO_TOKENS = 3           # fewer tokens than this => "short" query
MIN_INFO_LETTERS = 10         # fewer letters than this => "short" query
MAX_LEXICAL_TERMS = 20

STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its
itself just like me more most my myself no nor not now of off on once only or other our ours
ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were
what when where which while who whom why will with would yeah you your yours yourself
yourselves um uh oh
""".split())

# yt-dlp style filename sanitising used fullwidth look-alikes for reserved characters.
_TITLE_MAP = str.maketrans({
    "：": ":", "？": "?", "⧸": "/", "＊": "*", "＜": "<", "＞": ">", "｜": "|",
    "＂": '"', "＼": "\\", "／": "/",
})


class SearchTimeout(RuntimeError):
    """A query exceeded the time budget and was aborted."""


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass
class Hit:
    passage_id: int                 # first passage (for spanning hits)
    ordinal: int                    # == embedding row of passage_id
    episode_id: str
    episode_title: str              # display title (fullwidth punctuation normalised)
    audio: str | None
    start_s: float
    end_s: float
    seek_s: float                   # suggested playback position (estimate)
    speakers: list[str]
    speaker_segments: list[dict]
    text: str
    snippet: str
    tier: str
    score: float = 0.0              # higher is better (-bm25); 0 for spanning hits
    match_time_s: float | None = None
    match_speaker: str | None = None
    passage_ids: tuple[int, ...] = ()
    ordinals: tuple[int, ...] = ()


@dataclass
class PhraseResult:
    hits: list[Hit]
    total: int                      # matching passages (capped at COUNT_CAP)
    total_is_lower_bound: bool
    offset: int
    page_cap: int                   # maximum page size that applied to this query
    is_short_query: bool

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.hits) < self.total


@dataclass
class SearchResult:
    query: str
    tokens: list[str]
    phrase: PhraseResult
    spanning: list[Hit]
    lexical: list[Hit]
    notes: list[str]
    elapsed_ms: float

    def all_hits(self) -> list[Hit]:
        """Hits in tier order: exact phrase first, then spanning, then lexical."""
        return [*self.phrase.hits, *self.spanning, *self.lexical]

    def episodes(self) -> list["EpisodeGroup"]:
        return group_by_episode(self.all_hits())


@dataclass
class EpisodeGroup:
    episode_id: str
    episode_title: str
    n_hits: int
    best_tier: str
    best_score: float
    hits: list[Hit]                 # chronological
    seek_points: list[float]


@dataclass
class Region:
    """Adjacent passages of one episode merged into one span."""
    episode_id: str
    episode_title: str
    start_s: float
    end_s: float
    seek_s: float
    ordinals: list[int]
    best: Hit
    hits: list[Hit]


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def clean_title(title: str) -> str:
    return (title or "").translate(_TITLE_MAP)


def fold(s: str) -> str:
    """Case- and diacritic-fold, mirroring unicode61 remove_diacritics."""
    s = unicodedata.normalize("NFKD", s.casefold())
    return "".join(c for c in s if not unicodedata.combining(c))


def query_tokens(query: str) -> list[str]:
    return schema.words(unicodedata.normalize("NFKC", query or ""))


def is_short_query(tokens: list[str]) -> bool:
    if len(tokens) < MIN_INFO_TOKENS:
        return True
    if all(fold(t) in STOPWORDS for t in tokens):
        return True
    return sum(len(t) for t in tokens) < MIN_INFO_LETTERS


def content_terms(tokens: list[str]) -> list[str]:
    """Distinct, folded, non-stopword terms (longest first, capped)."""
    seen: dict[str, None] = {}
    for t in tokens:
        f = fold(t)
        if f not in STOPWORDS and f not in seen:
            seen[f] = None
    terms = list(seen)
    if len(terms) > MAX_LEXICAL_TERMS:
        terms = sorted(terms, key=len, reverse=True)[:MAX_LEXICAL_TERMS]
    return terms


_BOUND_L = r"(?<![^\W_])"
_BOUND_R = r"(?![^\W_])"


def locate(text: str, tokens: list[str], last: bool = False) -> int | None:
    """Character offset of the token sequence in text (first or last occurrence)."""
    if not tokens:
        return None
    pat = _BOUND_L + r"[\W_]+".join(re.escape(t) for t in tokens) + _BOUND_R
    ms = list(re.finditer(pat, text, re.IGNORECASE | re.UNICODE))
    if not ms:
        return None
    return (ms[-1] if last else ms[0]).start()


def locate_any(text: str, terms: list[str]) -> int | None:
    """Earliest offset of any of the terms."""
    if not terms:
        return None
    pat = _BOUND_L + "(?:" + "|".join(re.escape(t) for t in terms) + ")" + _BOUND_R
    m = re.search(pat, text, re.IGNORECASE | re.UNICODE)
    return m.start() if m else None


def speaker_at(segments: list[dict], t: float, fallback: list[str]) -> str | None:
    best, best_d = None, math.inf
    for seg in segments:
        s, e = seg.get("start", 0.0), seg.get("end", 0.0)
        if s <= t <= e:
            return seg.get("speaker")
        d = min(abs(t - s), abs(t - e))
        if d < best_d:
            best, best_d = seg.get("speaker"), d
    if best is not None:
        return best
    return fallback[0] if len(fallback) == 1 else None


def _loads(s: str | None) -> list:
    if not s:
        return []
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return []


# --------------------------------------------------------------------------- #
# Consolidation helpers (module level so the hybrid layer can reuse them)
# --------------------------------------------------------------------------- #
def group_by_episode(hits: list[Hit]) -> list[EpisodeGroup]:
    """Group hits per episode; episodes with exact-phrase hits come first."""
    groups: dict[str, list[Hit]] = {}
    for h in hits:
        groups.setdefault(h.episode_id, []).append(h)
    out = []
    for eid, hs in groups.items():
        hs_sorted = sorted(hs, key=lambda h: h.start_s)
        best_tier = min((h.tier for h in hs), key=lambda t: TIER_RANK.get(t, 99))
        out.append(EpisodeGroup(
            episode_id=eid,
            episode_title=hs[0].episode_title,
            n_hits=len(hs),
            best_tier=best_tier,
            best_score=max(h.score for h in hs),
            hits=hs_sorted,
            seek_points=[h.seek_s for h in hs_sorted],
        ))
    out.sort(key=lambda g: (TIER_RANK.get(g.best_tier, 99), -g.n_hits, -g.best_score))
    return out


def merge_adjacent(hits: list[Hit]) -> list[Region]:
    """Merge hits on consecutive passages of one episode into regions.

    Regions are returned in the order of their best-ranked member in `hits`.
    """
    rank = {id(h): i for i, h in enumerate(hits)}
    by_ep: dict[str, list[Hit]] = {}
    for h in hits:
        by_ep.setdefault(h.episode_id, []).append(h)

    regions: list[Region] = []
    for eid, hs in by_ep.items():
        hs = sorted(hs, key=lambda h: h.ordinals[0])
        cur: list[Hit] = []
        for h in hs:
            if cur and h.ordinals[0] > cur[-1].ordinals[-1] + 1:
                regions.append(_make_region(eid, cur))
                cur = []
            cur.append(h)
        if cur:
            regions.append(_make_region(eid, cur))

    regions.sort(key=lambda r: min(rank[id(h)] for h in r.hits))
    return regions


def _make_region(eid: str, hs: list[Hit]) -> Region:
    best = min(hs, key=lambda h: (TIER_RANK.get(h.tier, 99), -h.score))
    ords = sorted({o for h in hs for o in h.ordinals})
    return Region(
        episode_id=eid, episode_title=hs[0].episode_title,
        start_s=min(h.start_s for h in hs), end_s=max(h.end_s for h in hs),
        seek_s=best.seek_s, ordinals=ords, best=best, hits=hs,
    )


# --------------------------------------------------------------------------- #
# Searcher
# --------------------------------------------------------------------------- #
class LexicalSearcher:
    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        short_phrase_cap: int = 50,
        max_page: int = 200,
        spanning_cap: int = 50,
        timeout_s: float = 5.0,
    ) -> None:
        path = Path(db_path) if db_path else Path(__file__).resolve().parent / "lexical.db"
        if not path.exists():
            raise FileNotFoundError(f"database not found: {path}")
        self.db_path = path
        self.short_phrase_cap = short_phrase_cap
        self.max_page = max_page
        self.spanning_cap = spanning_cap
        self.timeout_s = timeout_s

        # check_same_thread=False + RLock: safe to share (e.g. st.cache_resource).
        self._lock = threading.RLock()
        self._depth = 0
        self._deadline: float | None = None
        self.conn = sqlite3.connect(
            f"file:{path.resolve().as_posix()}?mode=ro", uri=True, check_same_thread=False
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA query_only = ON")
        self.conn.execute("PRAGMA cache_size = -131072")
        self.conn.execute("PRAGMA mmap_size = 1073741824")
        self.conn.set_progress_handler(self._progress, 20_000)

        version = schema.get_meta(self.conn, "schema_version")
        if version != str(schema.SCHEMA_VERSION):
            raise RuntimeError(
                f"{path.name} has schema_version {version!r}, code expects {schema.SCHEMA_VERSION}. "
                "Rebuild with build_index.py."
            )
        self.n_passages = int(schema.get_meta(self.conn, "n_passages", "0"))

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "LexicalSearcher":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- timeout guard (re-entrant) ------------------------------------------
    def _progress(self) -> int:
        return 1 if self._deadline is not None and time.monotonic() > self._deadline else 0

    @contextmanager
    def _guard(self):
        with self._lock:
            if self._depth == 0:
                self._deadline = time.monotonic() + self.timeout_s
            self._depth += 1
            try:
                yield
            except sqlite3.OperationalError as e:
                if "interrupt" in str(e).lower():
                    raise SearchTimeout(f"query exceeded {self.timeout_s:.1f}s") from e
                raise
            finally:
                self._depth -= 1
                if self._depth == 0:
                    self._deadline = None

    # -- SQL fragments -------------------------------------------------------
    @staticmethod
    def _speaker_clause(speaker: str | None) -> tuple[str, list]:
        """Exact label equality inside the stored 'A | B | C' list."""
        if not speaker:
            return "", []
        sep = schema.SPEAKER_SEP
        return " AND INSTR(? || p.speakers || ?, ?) > 0", [sep, sep, sep + speaker + sep]

    def _row_to_hit(
        self, row: sqlite3.Row, tier: str, *, snippet: str | None = None, score: float = 0.0,
        tokens: list[str] | None = None, terms: list[str] | None = None,
    ) -> Hit:
        text = row["text"]
        start, end = row["start_s"], row["end_s"]
        segs = _loads(row["speaker_segments"])
        speakers = schema.split_speakers(row["speakers"])

        span = locate(text, tokens) if tokens else (locate_any(text, terms) if terms else None)
        match_t = seek = match_spk = None
        if span is not None:
            match_t = start + span / max(len(text), 1) * (end - start)
            match_spk = speaker_at(segs, match_t, speakers)
        seek = max(start, match_t - SEEK_BACKOFF_S) if match_t is not None else start

        return Hit(
            passage_id=row["passage_id"], ordinal=row["ordinal"], episode_id=row["episode_id"],
            episode_title=clean_title(row["title"]), audio=row["audio"],
            start_s=start, end_s=end, seek_s=seek, speakers=speakers, speaker_segments=segs,
            text=text, snippet=snippet if snippet is not None else text[:240],
            tier=tier, score=score, match_time_s=match_t, match_speaker=match_spk,
            passage_ids=(row["passage_id"],), ordinals=(row["ordinal"],),
        )

    # ------------------------------------------------------------------ #
    # Tier 1: exact phrase
    # ------------------------------------------------------------------ #
    def phrase_hits(
        self, query: str, *, limit: int | None = None, offset: int = 0, speaker: str | None = None
    ) -> PhraseResult:
        tokens = query_tokens(query)
        short = is_short_query(tokens)
        cap = self.short_phrase_cap if short else self.max_page
        if not tokens:
            return PhraseResult([], 0, False, offset, cap, True)

        page = cap if limit is None else max(0, min(limit, cap))
        T = schema.FTS_EXACT
        match = f"text : {schema.quote_phrase(' '.join(tokens))}"
        spk_sql, spk_args = self._speaker_clause(speaker)

        with self._guard():
            rows = self.conn.execute(
                f"""
                SELECT p.passage_id, p.ordinal, p.episode_id, e.title, e.audio, p.start_s, p.end_s,
                       p.speakers, p.speaker_segments, p.text,
                       snippet({T}, 0, '[', ']', ' … ', 28) AS snip,
                       -bm25({T}, 1.0, 0.0) AS score
                FROM {T}
                JOIN passages p ON p.passage_id = {T}.rowid
                JOIN episodes e ON e.episode_id = p.episode_id
                WHERE {T} MATCH ? {spk_sql}
                ORDER BY bm25({T}, 1.0, 0.0)
                LIMIT ? OFFSET ?
                """,
                [match, *spk_args, page, max(0, offset)],
            ).fetchall()
            join = "JOIN passages p ON p.passage_id = {T}.rowid".format(T=T) if speaker else ""
            n = self.conn.execute(
                f"SELECT COUNT(*) FROM (SELECT 1 FROM {T} {join} WHERE {T} MATCH ? {spk_sql} "
                f"LIMIT {COUNT_CAP + 1})",
                [match, *spk_args],
            ).fetchone()[0]

        hits = [
            self._row_to_hit(r, TIER_PHRASE, snippet=r["snip"].replace("\n", " "),
                             score=r["score"], tokens=tokens)
            for r in rows
        ]
        lower = n > COUNT_CAP
        return PhraseResult(hits, min(n, COUNT_CAP), lower, offset, cap, short)

    # ------------------------------------------------------------------ #
    # Tier 2: phrase split across two consecutive passages
    # ------------------------------------------------------------------ #
    def spanning_hits(
        self, query: str, *, limit: int | None = None, speaker: str | None = None
    ) -> tuple[list[Hit], bool]:
        """Return (hits, truncated). Needs >= 2 query words."""
        tokens = query_tokens(query)
        if len(tokens) < 2:
            return [], False
        limit = self.spanning_cap if limit is None else min(limit, self.spanning_cap)
        ftoks = [fold(t) for t in tokens]
        T = schema.FTS_EXACT
        found: dict[int, Hit] = {}
        truncated = False

        with self._guard():
            for k in range(1, len(tokens)):
                head, tail = tokens[:k], tokens[k:]
                fhead, ftail = ftoks[:k], ftoks[k:]
                # Search with the longer (more selective) side; join the neighbour.
                if len(tail) > len(head):
                    side, delta = tail, -1
                else:
                    side, delta = head, +1
                rows = self.conn.execute(
                    f"""
                    SELECT p.passage_id AS m_id, p.ordinal AS m_ord, p.episode_id AS episode_id,
                           e.title AS title, e.audio AS audio,
                           p.start_s AS m_start, p.end_s AS m_end, p.speakers AS m_spk,
                           p.speaker_segments AS m_segs, p.text AS m_text,
                           n.passage_id AS o_id, n.ordinal AS o_ord,
                           n.start_s AS o_start, n.end_s AS o_end, n.speakers AS o_spk,
                           n.speaker_segments AS o_segs, n.text AS o_text
                    FROM {T}
                    JOIN passages p ON p.passage_id = {T}.rowid
                    JOIN passages n ON n.episode_id = p.episode_id AND n.ordinal = p.ordinal + ?
                    JOIN episodes e ON e.episode_id = p.episode_id
                    WHERE {T} MATCH ?
                    LIMIT ?
                    """,
                    (delta, f"text : {schema.quote_phrase(' '.join(side))}", SPANNING_CANDIDATES + 1),
                ).fetchall()
                if len(rows) > SPANNING_CANDIDATES:
                    truncated = True
                    rows = rows[:SPANNING_CANDIDATES]

                for r in rows:
                    if delta == +1:
                        a, b = "m", "o"
                    else:
                        a, b = "o", "m"
                    a_toks = [fold(t) for t in schema.words(r[f"{a}_text"])]
                    b_toks = [fold(t) for t in schema.words(r[f"{b}_text"])]
                    if a_toks[-k:] != fhead or b_toks[: len(ftail)] != ftail:
                        continue
                    key = r[f"{a}_id"]
                    if key in found:
                        continue
                    hit = self._make_spanning_hit(r, a, b, head)
                    if speaker and speaker not in hit.speakers:
                        continue
                    found[key] = hit

        hits = sorted(found.values(), key=lambda h: (h.episode_id, h.start_s))
        if len(hits) > limit:
            truncated = True
            hits = hits[:limit]
        return hits, truncated

    def _make_spanning_hit(self, r: sqlite3.Row, a: str, b: str, head: list[str]) -> Hit:
        a_text, b_text = r[f"{a}_text"], r[f"{b}_text"]
        a_start, a_end = r[f"{a}_start"], r[f"{a}_end"]
        segs = _loads(r[f"{a}_segs"]) + _loads(r[f"{b}_segs"])
        speakers = list(dict.fromkeys(
            schema.split_speakers(r[f"{a}_spk"]) + schema.split_speakers(r[f"{b}_spk"])
        ))
        span = locate(a_text, head, last=True)
        match_t = match_spk = None
        if span is not None:
            match_t = a_start + span / max(len(a_text), 1) * (a_end - a_start)
            match_spk = speaker_at(segs, match_t, speakers)
        seek = max(a_start, match_t - SEEK_BACKOFF_S) if match_t is not None else a_start
        snippet = (
            "… " + " ".join(a_text.split()[-14:]) + "  ‖  " + " ".join(b_text.split()[:14]) + " …"
        )
        return Hit(
            passage_id=r[f"{a}_id"], ordinal=r[f"{a}_ord"], episode_id=r["episode_id"],
            episode_title=clean_title(r["title"]), audio=r["audio"],
            start_s=a_start, end_s=r[f"{b}_end"], seek_s=seek, speakers=speakers,
            speaker_segments=segs, text=a_text + " " + b_text, snippet=snippet,
            tier=TIER_SPANNING, score=0.0, match_time_s=match_t, match_speaker=match_spk,
            passage_ids=(r[f"{a}_id"], r[f"{b}_id"]), ordinals=(r[f"{a}_ord"], r[f"{b}_ord"]),
        )

    # ------------------------------------------------------------------ #
    # Tier 3: ordinary lexical relevance (bm25 over content words)
    # ------------------------------------------------------------------ #
    def bm25_hits(
        self, query: str, *, limit: int = 50, offset: int = 0, speaker: str | None = None,
        exclude_phrase: bool = False, min_term_fraction: float = 0.6,
    ) -> list[Hit]:
        """Passages containing at least `min_term_fraction` of the query's distinct
        content words, ranked by bm25.  `offset` pages over the filtered candidates.
        With exclude_phrase=True, passages that contain the exact phrase are omitted.
        """
        tokens = query_tokens(query)
        terms = content_terms(tokens)
        if len(terms) < 2:      # a single content word is fully covered by the phrase tier
            return []
        need = max(1, math.ceil(min_term_fraction * len(terms)))
        T = schema.FTS_EXACT
        or_q = " OR ".join(schema.quote_phrase(t) for t in terms)
        match = f"text : ({or_q})"
        if exclude_phrase and len(tokens) > 1:
            match = f"({match}) NOT (text : {schema.quote_phrase(' '.join(tokens))})"
        spk_sql, spk_args = self._speaker_clause(speaker)
        candidates = min(1000, max(200, (offset + limit) * 10))

        with self._guard():
            rows = self.conn.execute(
                f"""
                SELECT p.passage_id, p.ordinal, p.episode_id, e.title, e.audio, p.start_s, p.end_s,
                       p.speakers, p.speaker_segments, p.text,
                       snippet({T}, 0, '[', ']', ' … ', 28) AS snip,
                       -bm25({T}, 1.0, 0.0) AS score
                FROM {T}
                JOIN passages p ON p.passage_id = {T}.rowid
                JOIN episodes e ON e.episode_id = p.episode_id
                WHERE {T} MATCH ? {spk_sql}
                ORDER BY bm25({T}, 1.0, 0.0)
                LIMIT ?
                """,
                [match, *spk_args, candidates],
            ).fetchall()

        term_set = set(terms)
        hits: list[Hit] = []
        for r in rows:
            present = {fold(t) for t in schema.words(r["text"])} & term_set
            if len(present) < need:
                continue
            hits.append(self._row_to_hit(
                r, TIER_LEXICAL, snippet=r["snip"].replace("\n", " "), score=r["score"],
                terms=sorted(present),
            ))
        return hits[offset: offset + limit]

    # ------------------------------------------------------------------ #
    # Combined search
    # ------------------------------------------------------------------ #
    def search(
        self, query: str, *, limit: int | None = None, offset: int = 0,
        speaker: str | None = None, include_spanning: bool = True,
        include_lexical: bool = True, lexical_limit: int = 50,
    ) -> SearchResult:
        """Run all tiers. `offset` pages the phrase tier; the other tiers are only
        computed for the first page (offset == 0)."""
        t0 = time.perf_counter()
        tokens = query_tokens(query)
        notes: list[str] = []

        phrase = self.phrase_hits(query, limit=limit, offset=offset, speaker=speaker)
        if not tokens:
            notes.append("The query contains no searchable words.")
            return SearchResult(query, tokens, phrase, [], [], notes, (time.perf_counter() - t0) * 1e3)

        count = f"{phrase.total:,}{'+' if phrase.total_is_lower_bound else ''}"
        if phrase.is_short_query and phrase.has_more:
            notes.append(
                f"Short query: {count} passages match; showing {len(phrase.hits)} per page "
                f"(use offset to continue). Longer phrases filter more precisely."
            )
        elif phrase.has_more:
            notes.append(f"{count} passages match; showing {len(phrase.hits)} (use offset to continue).")

        spanning: list[Hit] = []
        lexical: list[Hit] = []
        if offset == 0:
            only_stop = all(fold(t) in STOPWORDS for t in tokens)
            if include_spanning and len(tokens) >= 2 and not only_stop:
                spanning, truncated = self.spanning_hits(query, speaker=speaker)
                if truncated:
                    notes.append("Phrase-spanning search was truncated (very common words).")
            if include_lexical:
                lexical = self.bm25_hits(
                    query, limit=lexical_limit, speaker=speaker, exclude_phrase=True,
                )
                taken = {pid for h in spanning for pid in h.passage_ids}
                lexical = [h for h in lexical if h.passage_id not in taken]
                if len(tokens) > 1 and len(content_terms(tokens)) < 2:
                    notes.append("Lexical tier skipped: fewer than two content words.")

        return SearchResult(
            query, tokens, phrase, spanning, lexical, notes, (time.perf_counter() - t0) * 1e3
        )

    # ------------------------------------------------------------------ #
    # Helpers for the hybrid layer
    # ------------------------------------------------------------------ #
    def hydrate(
        self, ordinals: list[int], *, scores: list[float] | None = None, tier: str = TIER_DENSE
    ) -> list[Hit]:
        """Turn embedding row indices (== passages.ordinal) into Hits, preserving order."""
        if not ordinals:
            return []
        by_ord: dict[int, sqlite3.Row] = {}
        with self._guard():
            for i in range(0, len(ordinals), 500):
                chunk = [int(o) for o in ordinals[i:i + 500]]
                ph = ",".join("?" * len(chunk))
                for r in self.conn.execute(
                    f"""SELECT p.passage_uid AS passage_id, p.ordinal, p.episode_id, e.title, e.audio, p.start_s,
                               p.end_s, p.speakers, p.speaker_segments, p.text
                        FROM passages p JOIN episodes e ON e.episode_id = p.episode_id
                        WHERE p.ordinal IN ({ph})""",
                    chunk,
                ):
                    by_ord[r["ordinal"]] = r
        out = []
        for i, o in enumerate(ordinals):
            r = by_ord.get(int(o))
            if r is not None:
                out.append(self._row_to_hit(r, tier, score=float(scores[i]) if scores else 0.0))
        return out

    def context(self, hit: Hit, before: int = 1, after: int = 1) -> list[Hit]:
        """Neighbouring passages (same episode), in chronological order, excluding `hit`."""
        lo, hi = hit.ordinals[0] - before, hit.ordinals[-1] + after
        with self._guard():
            rows = self.conn.execute(
                """SELECT p.passage_id, p.ordinal, p.episode_id, e.title, e.audio, p.start_s,
                          p.end_s, p.speakers, p.speaker_segments, p.text
                   FROM passages p JOIN episodes e ON e.episode_id = p.episode_id
                   WHERE p.episode_id = ? AND p.ordinal BETWEEN ? AND ? ORDER BY p.ordinal""",
                (hit.episode_id, lo, hi),
            ).fetchall()
        skip = set(hit.ordinals)
        return [self._row_to_hit(r, TIER_CONTEXT) for r in rows if r["ordinal"] not in skip]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _print_hit(i: int, h: Hit) -> None:
    who = f", said by {h.match_speaker}" if h.match_speaker else ""
    spk = " / ".join(h.speakers)
    print(f"\n#{i:<3} {h.episode_title}")
    print(f"     seek {ts(h.seek_s)}  (passage {ts(h.start_s)}–{ts(h.end_s)}{who})  [{spk}]")
    print("     " + h.snippet)


def print_result(res: SearchResult, as_episodes: bool) -> None:
    p = res.phrase
    count = f"{p.total:,}{'+' if p.total_is_lower_bound else ''}"
    print(f"query: {res.query!r}  ->  tokens {res.tokens}   ({res.elapsed_ms:.0f} ms)")
    for n in res.notes:
        print(f"note: {n}")

    if as_episodes:
        for g in res.episodes():
            print(f"\n{g.episode_title}  [{g.best_tier}, {g.n_hits} hit(s)]")
            print("     seek at " + ", ".join(ts(s) for s in g.seek_points[:8]))
        return

    print(f"\n=== EXACT PHRASE: {count} passage(s); showing {len(p.hits)} from offset {p.offset} ===")
    for i, h in enumerate(p.hits, p.offset + 1):
        _print_hit(i, h)
    if res.spanning:
        print(f"\n=== PHRASE SPLIT ACROSS TWO PASSAGES: {len(res.spanning)} ===")
        for i, h in enumerate(res.spanning, 1):
            _print_hit(i, h)
    if res.lexical:
        print(f"\n=== RELATED LEXICAL MATCHES (bm25): {len(res.lexical)} ===")
        for i, h in enumerate(res.lexical, 1):
            _print_hit(i, h)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query")
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("-n", "--limit", type=int, default=None, help="page size (clamped to the cap)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--speaker", default=None, help="exact speaker label, e.g. LIZ")
    ap.add_argument("--no-spanning", action="store_true")
    ap.add_argument("--no-lexical", action="store_true")
    ap.add_argument("--episodes", action="store_true", help="group output by episode")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    with LexicalSearcher(args.db) as s:
        res = s.search(
            args.query, limit=args.limit, offset=args.offset, speaker=args.speaker,
            include_spanning=not args.no_spanning, include_lexical=not args.no_lexical,
        )
    if args.json:
        print(json.dumps(asdict(res), ensure_ascii=False, indent=2))
    else:
        print_result(res, args.episodes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
