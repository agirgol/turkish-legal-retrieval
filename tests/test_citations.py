"""The parser, against sentences taken from the corpus rather than invented.

Every string here appears in Turkish legislation. Inventing them would test the
regex against the shape it was written for, which is not a test.
"""

from tlr.citations import find_citations


def only(text: str):
    found = find_citations(text)
    assert len(found) == 1, f"expected one citation, got {[c.raw for c in found]}"
    return found[0]


class TestExternalReferences:
    def test_law_number_and_article(self):
        c = only("3065 sayılı Kanunun 16 ncı maddesinin birinci fıkrasında")
        assert (c.law_number, c.article_number) == (3065, 16)

    def test_the_law_may_be_named_in_between(self):
        c = only("3065 sayılı Katma Değer Vergisi Kanununun 13 üncü maddesinin (a) bendi")
        assert (c.law_number, c.article_number) == (3065, 13)

    def test_a_repealed_article_is_still_a_citation(self):
        # "mülga" — repealed. It points at a passage that existed, and the
        # citing article is still about it.
        c = only("193 sayılı Kanunun mülga 69 uncu maddesi")
        assert (c.law_number, c.article_number) == (193, 69)

    def test_every_ordinal_suffix(self):
        # Vowel harmony gives eight, and drafters use all of them.
        for suffix, article in [
            ("inci", 31), ("ıncı", 5), ("uncu", 69), ("üncü", 94),
            ("nci", 16), ("ncı", 16), ("ncu", 40), ("ncü", 4),
        ]:
            c = only(f"5846 sayılı Kanunun {article} {suffix} maddesi")
            assert c.article_number == article

    def test_apostrophe_instead_of_space(self):
        c = only("5846 sayılı Kanun'un 31'inci maddesi")
        assert (c.law_number, c.article_number) == (5846, 31)

    def test_uppercase_is_not_ascii_uppercase(self):
        # MADDE uppercased in Turkish keeps the dotted İ nowhere here, but
        # "SAYILI" turns i into İ and I into ı. re.IGNORECASE gets this wrong.
        c = only("5846 SAYILI KANUNUN 31 İNCİ MADDESİ")
        assert (c.law_number, c.article_number) == (5846, 31)


class TestInternalReferences:
    def test_the_same_law(self):
        c = only("aynı Kanunun 5 inci maddesine aşağıdaki fıkra eklenmiştir")
        assert c.law_number is None
        assert c.article_number == 5
        assert c.is_internal

    def test_dotless_i_spelling(self):
        c = only("Aynı Kanunun 12 nci maddesi")
        assert (c.law_number, c.article_number) == (None, 12)


class TestRanges:
    def test_a_range_is_several_citations(self):
        found = find_citations("3065 sayılı Kanunun 5 ilâ 9 uncu maddeleri")
        articles = sorted({c.article_number for c in found})
        assert articles == [5, 6, 7, 8, 9]

    def test_a_range_inherits_the_law_it_follows(self):
        found = find_citations("3065 sayılı Kanunun 5 ilâ 7 nci maddeleri")
        assert {c.law_number for c in found} == {3065}

    def test_an_implausible_range_is_not_expanded(self):
        # A parse failure, not a citation of two hundred articles.
        found = find_citations("1 ilâ 400 üncü maddeleri")
        assert found == []


class TestRefusals:
    def test_a_law_number_does_not_reach_across_a_sentence(self):
        # The gap between the number and the article is bounded and stops at
        # punctuation. Unbounded, this pairs 5846 with article 7 and produces a
        # confident wrong label — the failure mode that quietly ruins a gold set.
        text = "5846 sayılı Kanun yürürlüktedir. Bu Kanunun 7 nci maddesi ayrıdır."
        assert [(c.law_number, c.article_number) for c in find_citations(text)] == []

    def test_a_bare_article_reference_is_not_a_citation(self):
        # "üçüncü maddesi" with no number and no law names nothing resolvable.
        assert find_citations("üçüncü maddesinde belirtilen") == []

    def test_prose_about_laws_is_not_a_citation(self):
        assert find_citations("Bu kanunun amacı, kamu yararını gözetmektir.") == []


class TestQueryWindow:
    """The context cut around a citation, which becomes the query."""

    def test_the_citation_is_removed_from_its_own_query(self):
        from tlr.corpus import ArticleKind, Passage, PassageId
        from tlr.goldset import build

        # A query containing "2577 sayılı Kanunun 46 ncı maddesi" would let any
        # retriever that matches digits score perfectly while reading nothing.
        source = Passage(
            id=PassageId(9999, ArticleKind.ORDINARY, 1),
            law_name="Test",
            heading="Madde 1",
            text="Bu hüküm bakımından 2577 sayılı Kanunun 46 ncı maddesi uygulanır.",
        )
        target = Passage(
            id=PassageId(2577, ArticleKind.ORDINARY, 46),
            law_name="Test",
            heading="Madde 46",
            text="Temyiz süresi otuz gündür.",
        )

        pairs = build([source, target])
        assert len(pairs) == 1
        assert "2577" not in pairs[0].query
        assert "46 ncı" not in pairs[0].query

    def test_the_query_does_not_start_mid_word(self):
        from tlr.corpus import ArticleKind, Passage, PassageId
        from tlr.goldset import build

        prose = "tarihten itibaren geçerli olmak üzere uygulanacak hükümler " * 6
        source = Passage(
            id=PassageId(9999, ArticleKind.ORDINARY, 1),
            law_name="Test",
            heading="Madde 1",
            text=f"{prose}2577 sayılı Kanunun 46 ncı maddesi saklıdır.",
        )
        target = Passage(
            id=PassageId(2577, ArticleKind.ORDINARY, 46),
            law_name="Test",
            heading="Madde 46",
            text="Temyiz süresi otuz gündür.",
        )

        query = build([source, target])[0].query
        assert query.split()[0] in prose.split(), f"query opens mid-word: {query[:40]!r}"
