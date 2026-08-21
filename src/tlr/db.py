"""The one store. Passages, their lexical index, and later their vectors.

Sparse and dense retrieval in the same database is the point rather than a
convenience: hybrid ranking becomes a query instead of a system, with no second
store to keep in step and no two ideas of what a document is.
"""

from __future__ import annotations

import os

import psycopg

DSN = os.environ.get(
    "TLR_DATABASE_URL", "postgresql://tlr:tlr@localhost:5433/tlr"
)

# Two BM25 indexes over the same column, differing only in whether snowball's
# Turkish stemmer runs.
#
# The stemmer is expected to hurt, which is the opposite of what stemming is
# for. Snowball maps `kanun` — the single most common noun in this corpus — to
# `kan`, which is a different word meaning blood, while mapping its own plural
# `kanunlar` to `kanun`. One lemma, two stems, and neither matches the other.
# `maddesi` becomes `maddes` where `maddeler` becomes `madde`. The postposition
# `göre` becomes the verb root `gör`.
#
# Whether that costs anything measurable is what the two indexes are for.
SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;

CREATE TABLE IF NOT EXISTS passages (
    id            text PRIMARY KEY,
    law_number    integer NOT NULL,
    kind          text    NOT NULL,
    article_number integer NOT NULL,
    law_name      text    NOT NULL,
    heading       text    NOT NULL,
    body          text    NOT NULL,

    -- The same text a second time, so one index can hold two analyses of it.
    -- pg_search permits one BM25 index per table, and an index covers fields
    -- rather than analysers — so two tokenizers means two fields.
    body_stemmed  text GENERATED ALWAYS AS (body) STORED
);

CREATE TABLE IF NOT EXISTS gold_pairs (
    id        bigserial PRIMARY KEY,
    query     text NOT NULL,
    gold_id   text NOT NULL REFERENCES passages (id),
    source_id text NOT NULL,
    citation  text NOT NULL
);
"""

# One index, two fields, two analysers over identical text. Built after the
# rows are in, because pg_search indexes on write and loading 31,000 articles
# through both analysers is slower than loading once and building afterwards.
#
# `body` is the corpus as written. `body_stemmed` is the same characters read
# through snowball's Turkish stemmer. Which field a query targets is the whole
# experiment.
BM25_INDEX = """
CREATE INDEX passages_bm25 ON passages
USING bm25 (id, body, body_stemmed)
WITH (key_field='id',
      text_fields='{
        "body":         {"tokenizer": {"type": "default"}},
        "body_stemmed": {"tokenizer": {"type": "default", "stemmer": "Turkish"}}
      }')
"""

FIELDS = {"plain": "body", "stemmed": "body_stemmed"}


def connect() -> psycopg.Connection:
    return psycopg.connect(DSN)


def create_schema() -> None:
    with connect() as conn:
        conn.execute(SCHEMA)
        conn.commit()


def build_indexes() -> None:
    with connect() as conn:
        conn.execute("DROP INDEX IF EXISTS passages_bm25")
        conn.execute(BM25_INDEX)
        conn.commit()
