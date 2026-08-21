"""Finding, in the text of Turkish legislation, the places it points at itself.

Turkish law is written in cross-reference: an article amends, excepts or defers
to another article, and says so in a formula that barely varies across sixty
years of drafting. That formula is a gold label nobody had to write. A citation
names a passage the drafter considered relevant to the passage doing the citing,
which is what a retrieval benchmark needs and what generated questions only
approximate.

The alternative — showing a model a chunk and asking it for a question — makes a
benchmark that is easy in exactly the way the generator was: the answer is in
the chunk it was written from, so retrieval looks solved and whichever chunking
strategy produced those chunks wins. Citations have no such bias. They were
written by people with no idea a retriever would read them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The ordinal that turns a number into "the Nth". Turkish vowel harmony gives
# eight of them and drafters use every one — measured over the corpus: inci
# 9079, üncü 6487, nci 6133, uncu 3265, ncı 2516, ıncı 538, and a tail of ncü,
# ncu. They also appear uppercased in headings, where Turkish casing turns i
# into İ and I into ı.
#
# Matched as "a short word between the number and the word madde" rather than by
# listing sixteen spellings. Anything two to four letters long in that position
# is an ordinal; nothing else fits there.
_ORDINAL = r"\w{2,4}"

# Turkish case folding is not ASCII case folding: uppercase of "i" is "İ", and
# lowercase of "I" is "ı". `re.IGNORECASE` gets both wrong, matching "I" to "i"
# and missing "İ" entirely. Every letter that differs is spelled out instead.
_M = r"[Mm][Aa][Dd][Dd][Ee]"
_K = r"[Kk][Aa][Nn][Uu][Nn]"
_SAYILI = r"[Ss][Aa][Yy][Iiı][Ll][Iiı]"
_AYNI = r"[Aa][Yy][Nn][Iiı]"

_APOSTROPHE = r"[’'ʼ]?"


@dataclass(frozen=True)
class Citation:
    """One reference, resolved as far as the text allows.

    `law_number` is None when the text says "the same Law" — the reference is
    to the document doing the citing, which the caller knows and this parser
    does not.
    """

    law_number: int | None
    article_number: int
    raw: str
    start: int
    end: int

    @property
    def is_internal(self) -> bool:
        return self.law_number is None


# "5846 sayılı ... Kanunun 31 inci maddesi" — the law number, then anything up
# to the word Kanun (the law's name, which we do not need and which varies in
# how it is abbreviated), then the article.
#
# The gap is bounded and forbids sentence punctuation. Without a bound it will
# happily cross from one law's number to an unrelated article three sentences
# later, and produce a confident wrong label.
_EXTERNAL = re.compile(
    rf"(?P<law>\d{{3,5}})\s*{_SAYILI}"
    rf"(?P<between>[^.;:\n]{{0,80}}?)"
    # Kanun, its case suffix, and possibly an apostrophe before it: Kanunun,
    # Kanun'un, Kanununun.
    rf"{_K}[\w’'ʼ]*"
    # Then a qualifier the drafter may have put in front of the number: "mülga"
    # for a repealed article, "ek" or "geçici" for an appended one. Bounded and
    # digit-free so it cannot swallow the number it is supposed to precede.
    rf"[^\d.;:\n]{{0,25}}"
    rf"(?P<article>\d+)\s*{_APOSTROPHE}\s*(?:{_ORDINAL}\s+)?{_M}\w*"
)

# "aynı Kanunun 5 inci maddesi" — extremely common in amending laws, where an
# article changes one law over and over.
_INTERNAL = re.compile(
    rf"{_AYNI}\s+{_K}[\w’'ʼ]*[^\d.;:\n]{{0,25}}"
    rf"(?P<article>\d+)\s*{_APOSTROPHE}\s*(?:{_ORDINAL}\s+)?{_M}\w*"
)

# "5 ilâ 9 uncu maddeleri" — a range, which is several citations written once.
_RANGE = re.compile(
    rf"(?P<first>\d+)\s*{_APOSTROPHE}\s*(?:{_ORDINAL}\s+)?(?:ilâ|ila)\s*"
    rf"(?P<last>\d+)\s*{_APOSTROPHE}\s*(?:{_ORDINAL}\s+)?{_M}\w*"
)

# Used to give a range the law it belongs to. A range rarely names one itself:
# "3065 sayılı Kanunun 5 ilâ 9 uncu maddeleri" puts the number at the front of
# a phrase the range regex does not reach.
_LAW_NUMBER = re.compile(rf"(?P<law>\d{{3,5}})\s*{_SAYILI}")

# A range that runs the wrong way, or across half the code, is a parse failure
# rather than a citation. Bounded so one bad match cannot emit hundreds of
# labels.
_MAX_RANGE = 20


def find_citations(text: str) -> list[Citation]:
    """Every citation in one article's text, in the order they appear."""
    found: list[Citation] = []

    for match in _EXTERNAL.finditer(text):
        found.append(
            Citation(
                law_number=int(match.group("law")),
                article_number=int(match.group("article")),
                raw=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )

    for match in _INTERNAL.finditer(text):
        found.append(
            Citation(
                law_number=None,
                article_number=int(match.group("article")),
                raw=match.group(0),
                start=match.start(),
                end=match.end(),
            )
        )

    found.extend(_expand_ranges(text, found))
    return sorted(found, key=lambda c: (c.start, c.article_number))


def _expand_ranges(text: str, already: list[Citation]) -> list[Citation]:
    """Turns "5 ilâ 9 uncu maddeleri" into five citations."""
    covered = {(c.start, c.end) for c in already}
    expanded: list[Citation] = []

    for match in _RANGE.finditer(text):
        if (match.start(), match.end()) in covered:
            continue

        first, last = int(match.group("first")), int(match.group("last"))
        if last < first or last - first >= _MAX_RANGE:
            continue

        # A range inherits whichever law the nearest preceding citation named.
        # "3065 sayılı Kanunun 5 ilâ 9 uncu maddeleri" is one law's five
        # articles, and reading the range alone loses that.
        law = _law_in_effect_before(match.start(), already, text)

        expanded.extend(
            Citation(
                law_number=law,
                article_number=n,
                raw=match.group(0),
                start=match.start(),
                end=match.end(),
            )
            for n in range(first, last + 1)
        )

    return expanded


def _law_in_effect_before(
    position: int, citations: list[Citation], text: str
) -> int | None:
    """The last law named before this point, whether or not it parsed as a citation.

    Reading only the citations misses the common case: in "3065 sayılı Kanunun
    5 ilâ 9 uncu maddeleri" there is no complete citation before the range —
    the law number belongs to the range itself.
    """
    preceding = [c for c in citations if c.end <= position and c.law_number is not None]
    if preceding:
        return preceding[-1].law_number

    mentions = list(_LAW_NUMBER.finditer(text[:position]))
    return int(mentions[-1].group("law")) if mentions else None
