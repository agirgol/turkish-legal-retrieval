"""Puts the corpus and the gold set into the database."""

from __future__ import annotations

from tlr.corpus import Passage, load_passages
from tlr.db import build_indexes, connect, create_schema
from tlr.goldset import GoldPair, build


def load_all() -> tuple[int, int]:
    create_schema()
    passages = load_passages()
    pairs = build(passages)

    with connect() as conn:
        conn.execute("TRUNCATE gold_pairs, passages RESTART IDENTITY")
        _insert_passages(conn, passages)
        _insert_pairs(conn, pairs)
        conn.commit()

    build_indexes()
    return len(passages), len(pairs)


def _insert_passages(conn, passages: list[Passage]) -> None:
    with conn.cursor() as cur, cur.copy(
        "COPY passages (id, law_number, kind, article_number, law_name, heading, body)"
        " FROM STDIN"
    ) as copy:
        for p in passages:
            copy.write_row(
                (
                    str(p.id),
                    p.id.law_number,
                    str(p.id.kind),
                    p.id.article_number,
                    p.law_name,
                    p.heading,
                    p.text,
                )
            )


def _insert_pairs(conn, pairs: list[GoldPair]) -> None:
    with conn.cursor() as cur, cur.copy(
        "COPY gold_pairs (query, gold_id, source_id, citation) FROM STDIN"
    ) as copy:
        for pair in pairs:
            copy.write_row(
                (pair.query, str(pair.gold), str(pair.source), pair.citation)
            )


if __name__ == "__main__":
    passages, pairs = load_all()
    print(f"{passages:,} passages, {pairs:,} gold pairs")
