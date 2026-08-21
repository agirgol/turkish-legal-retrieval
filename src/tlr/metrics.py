"""Recall@k and MRR, over rankings that have exactly one right answer.

One gold passage per query is a property of how the set was built — a citation
names one article — and it makes the metrics simple. It also makes them strict:
there is no partial credit for retrieving something related, which is the right
severity for a benchmark whose labels are this clean.
"""

from __future__ import annotations

from dataclasses import dataclass

from tlr.lexical import Ranking


@dataclass(frozen=True)
class Scores:
    queries: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    found_nothing: int

    def as_row(self, label: str) -> str:
        return (
            f"{label:<10} R@1 {self.recall_at_1:.3f}  R@5 {self.recall_at_5:.3f}  "
            f"R@10 {self.recall_at_10:.3f}  MRR {self.mrr:.3f}  "
            f"boş {self.found_nothing}"
        )


def score(rankings: list[Ranking]) -> Scores:
    hits = {1: 0, 5: 0, 10: 0}
    reciprocal = 0.0
    empty = 0

    for ranking in rankings:
        if not ranking.retrieved:
            empty += 1
            continue

        if ranking.gold_id in ranking.retrieved:
            position = ranking.retrieved.index(ranking.gold_id) + 1
            reciprocal += 1 / position
            for k in hits:
                if position <= k:
                    hits[k] += 1

    n = len(rankings) or 1
    return Scores(
        queries=len(rankings),
        recall_at_1=hits[1] / n,
        recall_at_5=hits[5] / n,
        recall_at_10=hits[10] / n,
        mrr=reciprocal / n,
        found_nothing=empty,
    )
