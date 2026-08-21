"""The corpus: Turkish legislation, one passage per article.

Source is `muhammetakkurt/mevzuat-gov-dataset` on the Hugging Face Hub, which is
mevzuat.gov.tr scraped and structured — 907 laws, article by article, MIT
licensed.

The texts themselves need no licence. FSEK article 31 puts officially published
laws, decrees, regulations, communiqués, circulars and judicial decisions
outside copyright entirely: reproducing, distributing and adapting them is free.
That is unusually clean ground for a benchmark, and it is why this corpus rather
than something scraped from a publisher.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

DATASET = "muhammetakkurt/mevzuat-gov-dataset"


class ArticleKind(StrEnum):
    """Which numbering an article belongs to.

    A law numbers its articles from one, and then numbers its *provisional*
    articles from one again. "Geçici Madde 5" and "Madde 5" are different
    passages in the same law, and treating the number alone as an identifier
    silently pairs citations with the wrong text — 8,000 times over this corpus,
    which is enough to move any measurement taken on top of it.
    """

    ORDINARY = "ordinary"
    PROVISIONAL = "provisional"


@dataclass(frozen=True)
class PassageId:
    law_number: int
    kind: ArticleKind
    article_number: int

    def __str__(self) -> str:
        prefix = "gecici-" if self.kind is ArticleKind.PROVISIONAL else ""
        return f"{self.law_number}/{prefix}{self.article_number}"


@dataclass(frozen=True)
class Passage:
    id: PassageId
    law_name: str
    heading: str
    text: str


_NUMBER = re.compile(r"(\d+)")
# "Geçici Madde 5", "GEÇİCİ MADDE 5", "Geçici Madde5" — the spacing and the
# casing both vary, and Turkish uppercase turns i into İ.
_PROVISIONAL = re.compile(r"^\s*(?:GEÇİCİ|Geçici|geçici|GEÇICI)")


def _classify(heading: str) -> ArticleKind:
    return (
        ArticleKind.PROVISIONAL
        if _PROVISIONAL.match(heading)
        else ArticleKind.ORDINARY
    )


def load_passages() -> list[Passage]:
    """Every article in the corpus, as a retrieval passage."""
    from datasets import load_dataset

    rows = load_dataset(DATASET, split="train")
    passages: list[Passage] = []

    for row in rows:
        law_raw = str(row["kanun_numarasi"]).strip()
        if not law_raw.isdigit():
            # A handful carry no usable number. Nothing can cite them by one
            # either, so they cannot be a gold answer.
            continue

        law_number = int(law_raw)
        law_name = (row["Kanun Adı"] or "").strip()

        for article in row["maddeler"]:
            heading = (article.get("madde_numarasi") or "").strip()
            text = (article.get("text") or "").strip()
            number = _NUMBER.search(heading)
            if not (number and text):
                continue

            passages.append(
                Passage(
                    id=PassageId(law_number, _classify(heading), int(number.group(1))),
                    law_name=law_name,
                    heading=heading,
                    text=text,
                )
            )

    return passages
