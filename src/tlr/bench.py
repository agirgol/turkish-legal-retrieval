"""One command that runs a benchmark and prints what it measured.

    uv run tlr load     # corpus and gold set into Postgres
    uv run tlr check    # does the harness retrieve a passage by its own text
    uv run tlr bench    # every gold query through every configuration

`check` exists because a lexical harness that cannot find a document when
searched with that document's own text is broken, and the first version of this
one was — twice, for different reasons. It is cheap, it runs in a second, and it
is the difference between a number and a number worth reporting.
"""

from __future__ import annotations

import argparse
import time

from tlr.db import connect
from tlr.lexical import CONFIGS, rank_gold_set, search
from tlr.load import load_all
from tlr.metrics import score

# Not every passage can retrieve itself, and that is a property of legislation
# rather than a defect. Laws end with the same sentence — "Bu Kanunu Bakanlar
# Kurulu yürütür" — followed by the same amendment boilerplate, so dozens of
# final articles are identical apart from a law number. A harness cannot rank
# one of those first and should not be asked to.
_SELF_RETRIEVAL_FLOOR = 0.85


def check() -> int:
    """Every configuration must retrieve most passages searched by their own text.

    A lexical harness that cannot find a document when given that document's own
    words is broken, and this one was — twice. First an unquoted `4369/82` in a
    tsquery turned a term into an operator; then a scorer without IDF ranked the
    document 1,360th behind everything sharing the word "kanun". Both looked like
    a hard task rather than a broken tool.
    """
    failed = False

    with connect() as conn:
        rows = conn.execute(
            "SELECT id, body FROM passages WHERE length(body) BETWEEN 400 AND 2000"
            " ORDER BY id LIMIT 40"
        ).fetchall()

        for config in CONFIGS:
            missed = [
                passage_id
                for passage_id, body in rows
                if passage_id not in search(conn, config, body[:300], k=1)
            ]
            rate = 1 - len(missed) / len(rows)
            verdict = "ok" if rate >= _SELF_RETRIEVAL_FLOOR else "TOO LOW"
            print(
                f"  {config:<9} {len(rows) - len(missed)}/{len(rows)} "
                f"retrieved themselves at rank 1  ({rate:.0%}, {verdict})"
            )

            if missed:
                print(f"            not retrieved: {', '.join(missed[:4])}")
            if rate < _SELF_RETRIEVAL_FLOOR:
                failed = True

        failed |= _check_dense(conn, rows)

    return 1 if failed else 0


def _check_dense(conn, rows) -> bool:
    """The same question of any dense model that has been embedded.

    A vector pipeline fails differently from a lexical one — a prefix omitted, a
    normalisation skipped, a dimension mismatch — and all of those produce
    plausible-looking neighbours rather than an error.
    """
    from tlr.dense import MODELS, load_encoder
    from tlr.dense import search as dense_search

    failed = False
    for key, model in MODELS.items():
        embedded = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = %s",
            (model.table,),
        ).fetchone()[0]
        if not embedded:
            continue

        encoder = load_encoder(model)
        missed = [
            passage_id
            for passage_id, body in rows
            if passage_id not in dense_search(conn, model, encoder, body[:300], k=1)
        ]
        rate = 1 - len(missed) / len(rows)
        verdict = "ok" if rate >= _SELF_RETRIEVAL_FLOOR else "TOO LOW"
        print(
            f"  {key:<9} {len(rows) - len(missed)}/{len(rows)} "
            f"retrieved themselves at rank 1  ({rate:.0%}, {verdict})"
        )
        if rate < _SELF_RETRIEVAL_FLOOR:
            failed = True

    return failed


def bench(k: int, limit: int | None, models: list[str]) -> int:
    with connect() as conn:
        for config in CONFIGS:
            _report(config, lambda c=config: rank_gold_set(conn, c, k=k, limit=limit))

        for key in models:
            from tlr.dense import MODELS
            from tlr.dense import rank_gold_set as rank_dense

            model = MODELS[key]
            _report(key, lambda m=model: rank_dense(conn, m, k=k, limit=limit))

    return 0


def _report(label: str, run) -> None:
    started = time.monotonic()
    rankings = run()
    elapsed = time.monotonic() - started
    print(
        f"  {score(rankings).as_row(label)}"
        f"  ({elapsed:.0f}s, {len(rankings) / max(elapsed, 1e-9):.0f} q/s)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(prog="tlr")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("load", help="load the corpus and gold set")
    commands.add_parser("check", help="verify the harness before trusting it")

    run = commands.add_parser("bench", help="score every configuration")
    run.add_argument("-k", type=int, default=10, help="depth to retrieve to")
    run.add_argument("--limit", type=int, default=None, help="use only N queries")
    run.add_argument(
        "--dense",
        action="append",
        default=[],
        metavar="MODEL",
        help="also score a dense model; repeatable",
    )

    args = parser.parse_args()

    if args.command == "load":
        passages, pairs = load_all()
        print(f"  {passages:,} passages, {pairs:,} gold pairs")
        return 0

    if args.command == "check":
        return check()

    return bench(args.k, args.limit, args.dense)


if __name__ == "__main__":
    raise SystemExit(main())
