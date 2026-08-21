"""Lexical retrieval, and the one variable this repository exists to measure.

Two fields over identical text, scored by the same BM25, differing only in
whether snowball's Turkish stemmer ran. Everything else — the index, the
scoring, the queries — is held constant, so whatever difference appears is
morphology and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass

from tlr.db import FIELDS

CONFIGS = tuple(FIELDS)

# Real BM25, from pg_search. Postgres' own ts_rank has no IDF weighting, so a
# 300-character legal query OR-ed across its terms matched 26,205 of 31,304
# passages and put a document 1,360th when searched with its own text —
# "kanun", "madde" and "fıkra" carried the same weight as anything rare.
#
# The query text goes in whole. pg_search analyses it with the same tokenizer as
# the field, which is the point: a stemmed field must be queried by a stemmed
# query or the comparison measures nothing.
_SEARCH = """
SELECT id
FROM passages
WHERE {field} @@@ paradedb.match('{field}', %(text)s)
ORDER BY paradedb.score(id) DESC, id
LIMIT %(k)s
"""


@dataclass(frozen=True)
class Ranking:
    query_id: int
    gold_id: str
    retrieved: list[str]


def search(conn, config: str, text: str, k: int) -> list[str]:
    if config not in CONFIGS:
        raise ValueError(f"unknown configuration {config!r}")

    sql = _SEARCH.format(field=FIELDS[config])
    with conn.cursor() as cur:
        cur.execute(sql, {"text": text, "k": k})
        return [row[0] for row in cur.fetchall()]


def rank_gold_set(conn, config: str, k: int, limit: int | None = None) -> list[Ranking]:
    """Runs every gold query through one configuration."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, query, gold_id FROM gold_pairs ORDER BY id"
            + (f" LIMIT {int(limit)}" if limit else "")
        )
        pairs = cur.fetchall()

    return [
        Ranking(query_id=qid, gold_id=gold, retrieved=search(conn, config, text, k))
        for qid, text, gold in pairs
    ]
