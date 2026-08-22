# turkish-legal-retrieval

Retrieval measured on Turkish legislation, with a gold set nobody had to write.

Turkish law is drafted in cross-reference: an article amends, excepts or defers
to another article and says so in a formula that has barely changed in sixty
years. Every one of those references is a label — somebody decided one provision
was relevant to another, years before anyone thought to measure a retriever with
it. Parsing them gives **5,010 query and passage pairs over 907 laws**, 79% of
them pointing across laws rather than within one.

The usual alternative is to show a language model a chunk and ask it for a
question. That makes a benchmark easy in exactly the way the generator was: the
answer sits in the chunk it was written from, retrieval looks close to solved,
and whichever chunking strategy produced those chunks wins its own comparison.
Citations have no such bias.

## What stemming does to Turkish

The headline, and it went the opposite way to the prediction.

Snowball's Turkish stemmer — the one Postgres and Tantivy both ship — is wrong
about the most important word in this corpus:

| written | stemmed |
|---|---|
| **kanun** (law) | **kan** (blood) |
| kanunlar (laws) | kanun |
| kanunların, kanuna, kanunu, kanunda | kan |
| maddesi | maddes |
| maddeler, maddesinin | madde |
| göre (according to) | gör (see) |

One lemma splits into two stems that do not match each other, the singular
collides with an unrelated word that appears throughout health legislation, and
a postposition collapses into a verb root.

Predicting from that table, stemming should hurt. Measured over all 5,010
queries, with BM25 through `pg_search`, two fields over identical text and
everything else held constant:

| | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|
| no stemming | 0.009 | 0.095 | 0.139 | 0.046 |
| Turkish stemmer | 0.012 | 0.101 | **0.156** | **0.052** |

It helps — about 12% relative at R@10. Turkish inflects so heavily that
inconsistent normalisation still beats matching surface forms. The table is a
good reason to expect a better stemmer to help more; it was not a good reason to
expect this one to hurt.

Absolute numbers are low because the task is hard for lexical matching: the
query is what an article says *about* another article, not what that article is
about. Retrieving 10 of 31,304 passages at random would score 0.0003.

## Dense against lexical

Same gold set, same metric, exact search on both sides.

| | R@1 | R@5 | R@10 | MRR | q/s |
|---|---|---|---|---|---|
| BM25, no stemming | 0.009 | 0.095 | 0.139 | 0.046 | 38 |
| BM25 + Turkish stemmer | 0.012 | 0.101 | **0.156** | 0.052 | 36 |
| `ytu-ce-cosmos/turkish-e5-large` | **0.022** | 0.099 | 0.154 | **0.056** | 15 |
| `intfloat/multilingual-e5-small` | 0.017 | 0.073 | 0.114 | 0.042 | 116 |

Neither side wins. Dense is roughly twice as precise at rank 1 — a 118M-parameter
model beats both lexical baselines there — and lexical has the better recall at
depth 10. That split is the textbook argument for hybrid ranking, and here it is
on one corpus with one metric rather than as an assertion.

A 200-query sample had `turkish-e5` ahead at R@10 as well, 0.150 against 0.125.
Over all 5,010 it is 0.154 against 0.156. The sample was not wrong so much as
too small, which is worth stating because a 200-query run takes twenty seconds
and a full one takes ten minutes.

### The index was charging the model for its own errors

Dense results were initially far worse — until the harness check was pointed at
them. Searching 40 passages by their own text:

| | retrieved itself |
|---|---|
| `query:` prefix, HNSW | 29/40 |
| `passage:` prefix, HNSW | 26/40 |
| `passage:` prefix, exact scan | **34/40** |

The gap is not the encoder's asymmetric prefixes, it is the approximate index.
Leaving it in would have charged the model for the index's error and then
compared that total against a lexical baseline with no equivalent handicap. The
benchmark searches exactly; the HNSW index stays, because a deployment needs one,
and what it costs is worth measuring on its own.

## The harness has to prove itself first

```
$ uv run tlr check
  plain      39/40 retrieved themselves at rank 1  (98%, ok)
             not retrieved: 1072/6
  stemmed    39/40 retrieved themselves at rank 1  (98%, ok)
  e5-small   34/40 retrieved themselves at rank 1  (85%, ok)
  turkish-e5 37/40 retrieved themselves at rank 1  (92%, ok)
```

A passage searched with its own text must come back first. This one did not,
twice, for different reasons — and both times the symptom read as a hard task
rather than a broken tool.

First, unquoted lexemes: legislation is full of tokens like `4369/82` and
`22/7/1998`, which `to_tsquery` reads as operators rather than terms. Then, no
IDF: Postgres' own `ts_rank` weights `kanun` the same as anything rare, so a
300-character query OR-ed across its terms matched **26,205 of 31,304 passages**
and ranked the document 1,360th. That is why the lexical baseline here is real
BM25 from `pg_search` rather than Postgres full-text ranking.

`1072/6` is a genuine miss, not a bug: laws end with the same sentence — *"Bu
Kanunu Bakanlar Kurulu yürütür"* — and the same amendment boilerplate, so dozens
of final articles are identical apart from a number.

## Running it

```sh
docker compose up -d
uv run tlr load     # corpus and gold set into Postgres
uv run tlr check    # verify the harness before trusting it
uv run tlr bench    # every gold query through every configuration
```

Both halves of retrieval live in one database — `pg_search` for BM25, `pgvector`
for embeddings — so hybrid ranking will be a query rather than a system.

## What the corpus needed before it was usable

The source is `muhammetakkurt/mevzuat-gov-dataset`, mevzuat.gov.tr scraped and
structured. Three things had to be measured and handled:

**40,496 rows carry 31,304 distinct articles.** Law 3520 renumbers the articles
of other laws, so it is a table rather than prose, and the scraper made every
row of that table an article with the same heading — "Geçici Madde 1" of that
law appears 162 times holding fragments like `2.1.1961 203`. Deduplicated by
keeping the longest text.

**6.4% of articles are pointers, not provisions**: *"İlgili Kanunlara
işlenmiştir"*, *"(Mülga: 22/7/1998-4369/82 md.)"*. They stay in the index,
because a retriever does have to sift past them, and they cannot be gold answers
because there is nothing at that address to find. That took the gold set from
6,283 pairs to 5,010.

**Provisional articles are a second numbering.** "Geçici Madde 5" and "Madde 5"
are different passages of the same law, and there are more than 8,000 of them.
Treating the number alone as an identifier would mislabel thousands of pairs in
a way that looks right.

## Licence

The texts need none. FSEK article 31 puts officially published laws, decrees,
regulations, communiqués, circulars and judicial decisions outside copyright
entirely — reproducing, distributing and adapting them is free. That is
unusually clean ground for a benchmark.

Code is MIT.

## Status

| | |
|---|---|
| Gold set from citations, 5,010 pairs | ✅ |
| BM25 baseline, with and without stemming | ✅ |
| Dense retrieval, exact, against the lexical baselines | ✅ |
| The rest of TR-MTEB's top models — `bge-m3`, `e5-large-instruct` | ⬜ |
| What the HNSW index costs, measured on its own | ⬜ |
| Hybrid ranking | ⬜ |
| Chunking strategies | ⬜ |
| A second, generated question set — and whether it ranks pipelines the same way | ⬜ |
| Reranker fine-tuned on this corpus, scored on this metric | ⬜ |
| Quantized and measured against the same metric | ⬜ |
