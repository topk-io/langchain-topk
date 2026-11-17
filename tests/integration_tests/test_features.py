"""Live tests for TopK-specific features of `TopKVectorStore`."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.structured_query import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
)
from topk_sdk.query import field, fn, select

from langchain_topk import RetrievalMode, TopKTranslator, TopKVectorStore

DOCS = [
    Document(
        id="rust",
        page_content="Rust guarantees memory safety without a garbage collector.",
        metadata={"lang": "rust", "year": 2015, "tags": ["systems", "safety"]},
    ),
    Document(
        id="python",
        page_content="Python is known for readable syntax and scientific libraries.",
        metadata={"lang": "python", "year": 1991, "tags": ["scripting", "data"]},
    ),
    Document(
        id="go",
        page_content="Go has a garbage collector and built-in goroutines for concurrency.",  # noqa: E501
        metadata={"lang": "go", "year": 2012, "tags": ["systems", "concurrency"]},
    ),
    Document(
        id="haskell",
        page_content="Haskell is a purely functional language with lazy evaluation.",
        metadata={"lang": "haskell", "year": 1990, "tags": ["functional"]},
    ),
    Document(
        id="sql",
        page_content="SQL queries relational databases with declarative statements.",
        metadata={"lang": "sql", "year": 1974, "tags": ["data"]},
    ),
]


@pytest.fixture(scope="module")
def semantic_store(
    new_collection_name: Callable[[str], str], region: str
) -> TopKVectorStore:
    """Server-side embeddings: created with `from_documents` and no embedding."""
    return TopKVectorStore.from_documents(
        DOCS, collection_name=new_collection_name("semantic"), region=region
    )


def ids(docs: list[Document]) -> list[str | None]:
    return [d.id for d in docs]


def test_semantic_search_without_embedding_model(
    semantic_store: TopKVectorStore,
) -> None:
    assert semantic_store.embeddings is None
    docs = semantic_store.similarity_search("memory safe systems programming", k=2)
    assert docs[0].id == "rust"
    assert docs[0].metadata == DOCS[0].metadata
    scored = semantic_store.similarity_search_with_score("functional programming", k=1)
    assert scored[0][0].id == "haskell"
    assert isinstance(scored[0][1], float)


def test_keyword_search(semantic_store: TopKVectorStore) -> None:
    docs = semantic_store.keyword_search("garbage collector", k=5)
    assert set(ids(docs)) == {"rust", "go"}
    docs = semantic_store.keyword_search(
        "garbage goroutines", k=5, keyword_match_all=True
    )
    assert ids(docs) == ["go"]


def test_hybrid_search_with_filter(semantic_store: TopKVectorStore) -> None:
    docs = semantic_store.similarity_search(
        "garbage collection",
        k=5,
        mode="hybrid",
        weights={"semantic": 1.0, "keyword": 0.2},
        filter={"year": {"$gte": 2010}},
    )
    assert docs, "hybrid search returned nothing"
    assert all(d.metadata["year"] >= 2010 for d in docs)
    assert set(ids(docs)) <= {"rust", "go"}
    # The BM25 term only boosts: documents without the keywords still match.
    docs = semantic_store.hybrid_search("functional programming", k=5)
    assert "haskell" in ids(docs)


def test_hybrid_keyword_required(semantic_store: TopKVectorStore) -> None:
    docs = semantic_store.hybrid_search("goroutines", k=5, keyword_required=True)
    assert ids(docs) == ["go"]


def test_filters(semantic_store: TopKVectorStore) -> None:
    def run(f: object) -> set[str | None]:
        return set(ids(semantic_store.similarity_search("language", k=10, filter=f)))

    assert run({"lang": "go"}) == {"go"}
    assert run({"year": {"$lt": 1991}}) == {"haskell", "sql"}
    assert run({"lang": {"$in": ["rust", "sql"]}}) == {"rust", "sql"}
    assert run({"lang": {"$nin": ["rust", "sql"]}}) == {"python", "go", "haskell"}
    assert run({"tags": {"$contains": "systems"}}) == {"rust", "go"}
    assert run({"$or": [{"lang": "go"}, {"year": 1974}]}) == {"go", "sql"}
    assert run({"$not": {"year": {"$gte": 1991}}}) == {"haskell", "sql"}
    assert run({"lang": {"$starts_with": "py"}}) == {"python"}
    assert run(field("metadata.year") > 2014) == {"rust"}


def test_float_filters(new_collection_name: Callable[[str], str]) -> None:
    store = TopKVectorStore.from_texts(
        ["cheap", "pricey", "free"],
        metadatas=[{"price": 2.5}, {"price": 9.99}, {"price": 0}],
        ids=["cheap", "pricey", "free"],
        collection_name=new_collection_name("floats"),
    )

    def run(f: object) -> set[str | None]:
        return set(ids(store.similarity_search("item", k=10, filter=f)))

    # 9.99 is not exactly representable as f32; it must still match.
    assert run({"price": {"$in": [9.99, 2.5]}}) == {"pricey", "cheap"}
    assert run({"price": {"$in": [9.99, 3]}}) == {"pricey"}
    assert run({"price": {"$nin": [9.99]}}) == {"cheap", "free"}
    assert run({"price": 9.99}) == {"pricey"}
    query, kwargs = TopKTranslator().visit_structured_query(
        StructuredQuery(
            query="item",
            filter=Comparison(
                comparator=Comparator.IN, attribute="price", value=[9.99, 2.5]
            ),
        )
    )
    assert set(ids(store.similarity_search(query, k=10, **kwargs))) == {
        "pricey",
        "cheap",
    }


def test_custom_ranking(semantic_store: TopKVectorStore) -> None:
    # Rank purely by recency among semantic matches.
    docs = semantic_store.similarity_search(
        "programming language",
        k=5,
        ranking=lambda s: field("metadata.year"),
    )
    years = [d.metadata["year"] for d in docs]
    assert years == sorted(years, reverse=True)
    # Blend semantic relevance with a metadata boost.
    docs = semantic_store.similarity_search(
        "programming language",
        k=5,
        mode="hybrid",
        ranking=lambda s: (
            s.semantic
            + s.keyword.coalesce(0.0) * 0.1
            + (field("metadata.lang") == "sql").choose(100.0, 0.0)
        ),
    )
    assert docs[0].id == "sql"


def test_search_with_raw_query(semantic_store: TopKVectorStore) -> None:
    docs = semantic_store.search_with_query(
        select("text", score=fn.semantic_similarity("text", "databases"))
        .filter(field("metadata.year") < 2000)
        .sort(field("score"), asc=False)
        .limit(1)
    )
    assert docs[0].id == "sql"
    assert "score" in docs[0].metadata


def test_get_by_ids(semantic_store: TopKVectorStore) -> None:
    docs = semantic_store.get_by_ids(["go", "nope", "rust"])
    assert ids(docs) == ["go", "rust"]
    assert docs[1].page_content == DOCS[0].page_content


def test_retriever_modes(semantic_store: TopKVectorStore) -> None:
    retriever = semantic_store.as_retriever(search_kwargs={"k": 1, "mode": "keyword"})
    assert ids(retriever.invoke("goroutines")) == ["go"]
    retriever = semantic_store.as_retriever(
        search_kwargs={"k": 2, "filter": {"lang": "python"}}
    )
    assert ids(retriever.invoke("scientific computing")) == ["python"]


def test_self_query_translator(semantic_store: TopKVectorStore) -> None:
    structured = StructuredQuery(
        query="systems programming",
        filter=Operation(
            operator=Operator.AND,
            arguments=[
                Comparison(comparator=Comparator.GT, attribute="year", value=2000),
                Comparison(comparator=Comparator.NE, attribute="lang", value="go"),
            ],
        ),
    )
    query, kwargs = TopKTranslator().visit_structured_query(structured)
    assert ids(semantic_store.similarity_search(query, k=5, **kwargs)) == ["rust"]


def test_partitions_isolate_tenants(
    new_collection_name: Callable[[str], str], region: str
) -> None:
    store = TopKVectorStore(new_collection_name("tenants"), region=region)
    acme = store.with_partition("acme")
    globex = store.with_partition("globex")
    acme.add_texts(["Acme sells rocket skates."], ids=["a1"])
    globex.add_texts(["Globex builds doomsday devices."], ids=["g1"])

    assert ids(acme.similarity_search("rocket", k=5)) == ["a1"]
    assert ids(globex.similarity_search("rocket", k=5)) == ["g1"]
    assert acme.get_by_ids(["g1"]) == []
    assert store.with_partition("initech").similarity_search("anything") == []
    assert {"acme", "globex"} <= set(store.list_partitions())

    globex.delete_partition()


def test_consistency_modes(semantic_store: TopKVectorStore) -> None:
    lsn_store = semantic_store.with_partition(None)
    for consistency in (None, "strong"):
        docs = lsn_store.similarity_search(
            "garbage collector", k=1, consistency=consistency
        )
        assert len(docs) == 1
    # "indexed" reads only data that has finished indexing, so documents
    # written moments ago may not be visible yet.
    assert (
        len(lsn_store.similarity_search("garbage collector", consistency="indexed"))
        <= 4
    )
    strong = TopKVectorStore(
        semantic_store.collection_name,
        client=semantic_store.client,
        consistency="strong",
        read_your_writes=False,
    )
    strong.add_texts(["Zig has comptime."], ids=["zig"])
    assert ids(strong.get_by_ids(["zig"])) == ["zig"]
    # Read a specific write by its LSN.
    lsn_store.add_texts(["Kotlin runs on the JVM."], ids=["kotlin"])
    lsn = lsn_store.last_lsn
    assert lsn is not None
    assert ids(lsn_store.get_by_ids(["kotlin"], lsn=lsn)) == ["kotlin"]
    lsn_store.delete(["zig", "kotlin"])
    assert lsn_store.get_by_ids(["zig", "kotlin"]) == []


def test_delete_by_filter(new_collection_name: Callable[[str], str]) -> None:
    store = TopKVectorStore.from_documents(
        DOCS, collection_name=new_collection_name("delete")
    )
    store.delete(filter={"year": {"$lt": 2000}})
    remaining = store.get_by_ids([d.id for d in DOCS if d.id])
    assert set(ids(remaining)) == {"rust", "go"}


def test_dense_mmr_and_relevance(new_collection_name: Callable[[str], str]) -> None:
    store = TopKVectorStore.from_documents(
        [*DOCS, Document(id="rust2", page_content=DOCS[0].page_content)],
        DeterministicFakeEmbedding(size=16),
        collection_name=new_collection_name("dense"),
    )
    assert store.retrieval_mode == RetrievalMode.DENSE
    plain = store.similarity_search(DOCS[0].page_content, k=2)
    assert set(ids(plain)) == {"rust", "rust2"}
    mmr = store.max_marginal_relevance_search(
        DOCS[0].page_content, k=2, fetch_k=6, lambda_mult=0.1
    )
    assert len(mmr) == 2
    assert {"rust", "rust2"} != set(ids(mmr))
    scored = store.similarity_search_with_relevance_scores(DOCS[0].page_content, k=1)
    assert 0.99 <= scored[0][1] <= 1.0
    # Dense + semantic + BM25 in one hybrid query.
    docs = store.hybrid_search(
        "garbage collector", k=3, weights={"dense": 1.0, "semantic": 1.0}
    )
    assert docs


@pytest.mark.skipif(
    os.environ.get("TOPK_SKIP_FASTEMBED") == "1", reason="fastembed tests disabled"
)
def test_sparse_and_multi_vector(new_collection_name: Callable[[str], str]) -> None:
    pytest.importorskip("fastembed")
    from langchain_topk import FastEmbedLateInteraction, FastEmbedSparse

    store = TopKVectorStore.from_documents(
        DOCS,
        collection_name=new_collection_name("sparse-mv"),
        sparse_embedding=FastEmbedSparse(),
        multi_vector_embedding=FastEmbedLateInteraction(),
        retrieval_mode="sparse",
    )
    assert store.similarity_search("memory safety", k=1)[0].id == "rust"
    assert (
        store.similarity_search("lazy functional", k=1, mode="multi_vector")[0].id
        == "haskell"
    )
    docs = store.similarity_search(
        "concurrency",
        k=2,
        mode="hybrid",
        weights={"sparse": 1.0, "multi_vector": 0.1, "semantic": 1.0},
        filter={"tags": {"$contains": "systems"}},
    )
    assert docs[0].id == "go"


async def test_async_api(new_collection_name: Callable[[str], str]) -> None:
    store = await TopKVectorStore.afrom_texts(
        ["Async Rust uses futures.", "Python has asyncio."],
        ids=["r", "p"],
        metadatas=[{"lang": "rust"}, {"lang": "python"}],
        collection_name=new_collection_name("async"),
    )
    docs = await store.asimilarity_search("event loop", k=1, filter={"lang": "python"})
    assert ids(docs) == ["p"]
    assert ids(await store.aget_by_ids(["r"])) == ["r"]
    await store.adelete(["r"])
    assert await store.aget_by_ids(["r"]) == []
    tenant = store.with_partition(f"t-{uuid.uuid4().hex[:6]}")
    await tenant.aadd_texts(["tenant doc"], ids=["t"])
    assert ids(await tenant.asimilarity_search("tenant", k=5)) == ["t"]
