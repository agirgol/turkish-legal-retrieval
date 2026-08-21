"""Turning the corpus's own cross-references into a retrieval benchmark.

Each citation gives a pair: the sentence around it is the query, and the article
it names is the passage that should be retrieved. The label was written by a
drafter deciding one provision was relevant to another, years before anyone
thought to measure a retriever with it.

What this is not: a set of natural-language questions. A citing sentence reads
like legislation, because it is. That is a real limitation and it is why the
plan carries a second, generated set — the interesting result being whether the
two rank pipelines the same way. If they do, this one is free and unbiased and
the other one is not needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tlr.citations import Citation, find_citations
from tlr.corpus import ArticleKind, Passage, PassageId

# How much text around the citation becomes the query. Enough to carry what the
# citing article was talking about; short enough that the answer is not simply
# sitting in it.
_CONTEXT_CHARS = 300


@dataclass(frozen=True)
class GoldPair:
    query: str
    gold: PassageId
    source: PassageId
    citation: str


def build(passages: list[Passage]) -> list[GoldPair]:
    """Every citation that resolves to a passage in this corpus."""
    by_id = {p.id: p for p in passages}
    pairs: list[GoldPair] = []

    for passage in passages:
        for citation in find_citations(passage.text):
            gold = _resolve(citation, within=passage.id.law_number)
            if gold is None or gold not in by_id:
                # The corpus is 907 laws, not every law. Roughly half of all
                # citations point outside it, and an unresolvable citation is
                # not a defect — it is a passage this benchmark cannot ask
                # about.
                continue

            target = by_id[gold]
            if not target.is_substantive:
                # The citation is real; the article it names has had its content
                # moved elsewhere. There is nothing at that address to find.
                continue

            if gold == passage.id:
                # An article citing itself is a formatting artefact, not a
                # retrieval task with an answer somewhere else.
                continue

            pairs.append(
                GoldPair(
                    query=_context_around(passage.text, citation),
                    gold=gold,
                    source=passage.id,
                    citation=citation.raw,
                )
            )

    return pairs


def _resolve(citation: Citation, *, within: int) -> PassageId | None:
    law = citation.law_number if citation.law_number is not None else within
    if law is None:
        return None

    # Only ordinary articles. A citation that does not say "geçici" means the
    # ordinary numbering, and matching it against a provisional article of the
    # same number would be a wrong label that looks right.
    return PassageId(law, ArticleKind.ORDINARY, citation.article_number)


def _context_around(text: str, citation: Citation) -> str:
    """The citing sentence and its neighbourhood, with the citation removed.

    The citation is cut out on purpose. Left in, the query contains the law and
    article number of its own answer, and any retriever that can match digits
    scores perfectly while understanding nothing.
    """
    start = _word_boundary_at_or_after(text, max(0, citation.start - _CONTEXT_CHARS // 2))
    end = _word_boundary_at_or_before(text, min(len(text), citation.end + _CONTEXT_CHARS // 2))

    window = text[start : citation.start] + " " + text[citation.end : end]
    return re.sub(r"\s+", " ", window).strip()


def _word_boundary_at_or_after(text: str, index: int) -> int:
    """Moves forward to the start of a word.

    Cutting on a character count alone opens queries mid-word — "arihten" for
    "tarihten". A tokenizer then sees a fragment that exists in no vocabulary,
    which is noise in a benchmark whose subject is how Turkish words are split.
    """
    if index == 0 or text[index - 1].isspace():
        return index
    space = text.find(" ", index)
    return index if space == -1 or space >= len(text) else space + 1


def _word_boundary_at_or_before(text: str, index: int) -> int:
    if index >= len(text) or text[index].isspace():
        return index
    space = text.rfind(" ", 0, index)
    return index if space == -1 else space
