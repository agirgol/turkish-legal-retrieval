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


# An article whose content was moved into the law it amends leaves a pointer
# behind: "İlgili Kanunlara işlenmiştir", "yürürlükten kaldırılmıştır", a bare
# "(Mülga: 22/7/1998-4369/82 md.)". Measured over the corpus, these are 6.4%
# of articles before the repealed ones are counted.
#
# They stay in the index, because a retriever does have to sift past them. They
# cannot be gold answers, because there is nothing in them to retrieve: a pair
# pointing at one is unanswerable by construction, and including such pairs
# lowers every system's score by the same amount while telling you nothing.
_POINTER = re.compile(
    r"(işlenmiştir|yürürlükten\s+kaldırıl|^\s*[(–\-\s]*Mülga)", re.IGNORECASE
)
_POINTER_MAX_CHARS = 200


@dataclass(frozen=True)
class Passage:
    id: PassageId
    law_name: str
    heading: str
    text: str

    @property
    def is_substantive(self) -> bool:
        """Whether there is anything here to retrieve."""
        return not (
            len(self.text) < _POINTER_MAX_CHARS and _POINTER.search(self.text)
        )


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

    return _one_passage_per_article(passages)


def _one_passage_per_article(passages: list[Passage]) -> list[Passage]:
    """Collapses articles that appear more than once, keeping the longest text.

    The source is scraped, and some laws are tables rather than prose. Law 3520
    renumbers the articles of other laws, so it is a two-column list — and the
    scraper turned every row of that list into an article with the same
    heading. "Geçici Madde 1" of that law appears 162 times, each time holding a
    fragment like "2.1.1961 203".

    Measured over the corpus: 40,496 rows carry 31,304 distinct article numbers.
    Left alone, a citation resolving to one of them would have up to 135
    candidate answers, none of them a provision, and every metric computed over
    it would be measuring the scraper.

    Longest wins. Where an article genuinely appears twice — fourteen laws are
    on more than one row of the source — the fuller text is the one a citation
    is about.
    """
    best: dict[PassageId, Passage] = {}
    for passage in passages:
        current = best.get(passage.id)
        if current is None or len(passage.text) > len(current.text):
            best[passage.id] = passage

    return list(best.values())
