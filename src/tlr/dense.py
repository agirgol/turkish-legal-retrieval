"""Dense retrieval: passages as vectors, in the same database as the BM25 index.

The models are the ones TR-MTEB puts at the top for Turkish retrieval. That
leaderboard already answers "which embedding model is best at Turkish", so this
is not trying to discover it — the question here is what those models do on one
domain, against a lexical baseline, under the pipeline choices a practitioner
actually makes.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg


@dataclass(frozen=True)
class Model:
    """A model and the exact strings it expects around its inputs.

    E5 was trained with `query:` and `passage:` prefixes and expects them at
    inference. Omitting them does not fail — it quietly costs several points of
    recall, which is the worst kind of mistake to make in a benchmark, because
    the number still looks like a number.
    """

    key: str
    hub_id: str
    dimensions: int
    query_prefix: str = ""
    passage_prefix: str = ""

    @property
    def table(self) -> str:
        return f"emb_{self.key.replace('-', '_')}"


MODELS = {
    m.key: m
    for m in [
        # Small and fast. Here to validate the pipeline end to end before an
        # afternoon is spent embedding with something twenty times its size.
        Model(
            key="e5-small",
            hub_id="intfloat/multilingual-e5-small",
            dimensions=384,
            query_prefix="query: ",
            passage_prefix="passage: ",
        ),
        # TR-MTEB rank 3 for Turkish retrieval.
        Model(
            key="e5-large",
            hub_id="intfloat/multilingual-e5-large-instruct",
            dimensions=1024,
            query_prefix="query: ",
            passage_prefix="passage: ",
        ),
        # TR-MTEB rank 2, and Turkish-specific.
        Model(
            key="turkish-e5",
            hub_id="ytu-ce-cosmos/turkish-e5-large",
            dimensions=1024,
            query_prefix="query: ",
            passage_prefix="passage: ",
        ),
        # No prefixes by design — bge-m3 was trained without them, and adding
        # them because e5 needs them is the same mistake in the other direction.
        Model(key="bge-m3", hub_id="BAAI/bge-m3", dimensions=1024),
    ]
}


def create_table(conn: psycopg.Connection, model: Model) -> None:
    """One table per model, because a pgvector column has a fixed width.

    A single table with an unsized `vector` column would take every model, and
    would take no index — which turns each of 5,010 queries into a scan of
    31,304 rows.
    """
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {model.table} (
            passage_id text PRIMARY KEY REFERENCES passages (id) ON DELETE CASCADE,
            embedding  vector({model.dimensions}) NOT NULL
        )
    """)
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS {model.table}_hnsw
        ON {model.table} USING hnsw (embedding vector_cosine_ops)
    """)
    conn.commit()


def load_encoder(model: Model):
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model.hub_id, device=_device())
    return encoder


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def embed_corpus(conn: psycopg.Connection, model: Model, batch: int = 64) -> int:
    """Embeds every passage that does not already have a vector."""
    create_table(conn, model)

    rows = conn.execute(f"""
        SELECT p.id, p.body FROM passages p
        LEFT JOIN {model.table} e ON e.passage_id = p.id
        WHERE e.passage_id IS NULL
        ORDER BY p.id
    """).fetchall()

    if not rows:
        return 0

    encoder = load_encoder(model)
    ids = [r[0] for r in rows]
    texts = [model.passage_prefix + r[1] for r in rows]

    vectors = encoder.encode(
        texts,
        batch_size=batch,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    with conn.cursor() as cur, cur.copy(
        f"COPY {model.table} (passage_id, embedding) FROM STDIN"
    ) as copy:
        for passage_id, vector in zip(ids, vectors, strict=True):
            copy.write_row((passage_id, "[" + ",".join(f"{v:.6f}" for v in vector) + "]"))

    conn.commit()
    return len(ids)


# The benchmark searches exactly, not through the HNSW index.
#
# Measured on 40 passages searched with their own text, e5-small found 26 of
# them through the index and 34 by exact scan. That eight-passage gap is the
# index's approximation, and leaving it in would charge the model for the
# index's error — then compare that total against a lexical baseline that has no
# equivalent handicap.
#
# The index stays, because a deployment needs one at any real corpus size, and
# what it costs is worth measuring. It is worth measuring *separately*.
def search(
    conn: psycopg.Connection,
    model: Model,
    encoder,
    text: str,
    k: int,
    *,
    exact: bool = True,
) -> list[str]:
    vector = encoder.encode(
        model.query_prefix + text, normalize_embeddings=True
    )
    literal = "[" + ",".join(f"{v:.6f}" for v in vector) + "]"

    with conn.cursor() as cur:
        if exact:
            cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute(
            f"SELECT passage_id FROM {model.table}"
            " ORDER BY embedding <=> %s::vector LIMIT %s",
            (literal, k),
        )
        return [row[0] for row in cur.fetchall()]


def rank_gold_set(
    conn: psycopg.Connection,
    model: Model,
    k: int,
    limit: int | None = None,
    *,
    exact: bool = True,
) -> list:
    """Every gold query through one model.

    Queries are encoded in one batch rather than one at a time. Encoding 5,010
    queries individually spends most of its wall clock moving single rows onto
    the GPU, and measures that instead of retrieval.
    """
    from tlr.lexical import Ranking

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, query, gold_id FROM gold_pairs ORDER BY id"
            + (f" LIMIT {int(limit)}" if limit else "")
        )
        pairs = cur.fetchall()

    encoder = load_encoder(model)
    vectors = encoder.encode(
        [model.query_prefix + text for _, text, _ in pairs],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    rankings = []
    with conn.cursor() as cur:
        if exact:
            cur.execute("SET enable_indexscan = off")
        for (query_id, _, gold), vector in zip(pairs, vectors, strict=True):
            literal = "[" + ",".join(f"{v:.6f}" for v in vector) + "]"
            cur.execute(
                f"SELECT passage_id FROM {model.table}"
                " ORDER BY embedding <=> %s::vector LIMIT %s",
                (literal, k),
            )
            rankings.append(
                Ranking(
                    query_id=query_id,
                    gold_id=gold,
                    retrieved=[row[0] for row in cur.fetchall()],
                )
            )

    return rankings
