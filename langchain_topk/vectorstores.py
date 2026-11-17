"""TopK vector store for LangChain."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from topk_sdk import AsyncClient, Client
from topk_sdk.data import f32_sparse_vector as sparse_data
from topk_sdk.data import matrix as matrix_data
from topk_sdk.error import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    PartitionNotFoundError,
)
from topk_sdk.query import (
    FunctionExpr,
    LogicalExpr,
    Query,
    field,
    fn,
    match,
    select,
    should,
)
from topk_sdk.schema import (
    f32_sparse_vector,
    f32_vector,
    keyword_index,
    matrix,
    multi_vector_index,
    semantic_index,
    text,
    vector_index,
)

from langchain_topk._filters import build_filter
from langchain_topk.embeddings import (
    MultiVectorEmbeddings,
    SparseEmbeddings,
    normalize_multi_vector,
    normalize_sparse,
)

if TYPE_CHECKING:
    from topk_sdk import AsyncCollectionClient, CollectionClient

Consistency = Literal["indexed", "strong"]
Metric = Literal["cosine", "dot_product", "euclidean"]

MAX_QUERY_RESULTS = 10_000
"""TopK returns at most this many results from a single query."""

MAX_REQUEST_BYTES = 4 * 1024 * 1024
"""Target upsert payload size, well under TopK's 8MB request limit."""

_METRICS = ("cosine", "dot_product", "euclidean")
_METRIC_NAMES: dict[str, Metric] = {
    "Cosine": "cosine",
    "DotProduct": "dot_product",
    "Euclidean": "euclidean",
}

# Score aliases selected in queries. TopK SQL rejects aliases that start with an
# underscore, so none of these do.
SEMANTIC_SCORE = "semantic_score"
KEYWORD_SCORE = "keyword_score"
DENSE_SCORE = "dense_score"
SPARSE_SCORE = "sparse_score"
MULTI_VECTOR_SCORE = "multi_vector_score"


class RetrievalMode(str, Enum):
    """How `TopKVectorStore` scores documents for a query.

    Pass one as `retrieval_mode=` when creating the store, or per call as
    `mode=` (the enum or its string value) to `similarity_search` and friends.
    """

    SEMANTIC = "semantic"
    """Server-side embeddings: TopK embeds and reranks, no embedding model needed."""
    KEYWORD = "keyword"
    """BM25 keyword search over the text field."""
    DENSE = "dense"
    """Vector search with your own LangChain `Embeddings`."""
    SPARSE = "sparse"
    """Sparse vector search (for example SPLADE), exact with 100% recall."""
    MULTI_VECTOR = "multi_vector"
    """Late-interaction (ColBERT-style) multi-vector search scored with MaxSim."""
    HYBRID = "hybrid"
    """Every configured signal plus BM25, combined in one query."""


class TextIndex(str, Enum):
    """Index created on the text field of a new collection."""

    SEMANTIC = "semantic"
    """`semantic_index()`: server-side embeddings and BM25 (max 32,768 chars)."""
    KEYWORD = "keyword"
    """`keyword_index()`: BM25 only, no length cap beyond the document size."""
    NONE = "none"
    """No index: text is stored and returned but not searchable."""


class TopKVectorStore(VectorStore):
    """TopK vector store.

    [TopK](https://topk.io) is a search database with built-in semantic search
    (embeddings computed and reranked on the server), BM25 keyword search,
    dense, sparse and multi-vector search, metadata filtering and custom
    scoring, all available in a single query.

    Setup:
        Install `langchain-topk` and set the `TOPK_API_KEY` and `TOPK_REGION`
        environment variables. Get an API key from the
        [TopK console](https://console.topk.io).

        ```bash
        pip install -U langchain-topk
        export TOPK_API_KEY="your-api-key"
        export TOPK_REGION="aws-us-east-1-elastica"
        ```

    Key init args — collection:
        collection_name: str
            Name of the TopK collection. Created on first use if it does not
            exist.
        partition: str | None
            Partition (tenant) to read and write. Partitions are isolated
            sub-collections that share a schema.

    Key init args — search:
        embedding: Embeddings | None
            Optional dense embedding model. Leave it out to use TopK's
            server-side semantic search.
        sparse_embedding: SparseEmbeddings | None
            Optional sparse embedding model, for example `FastEmbedSparse`.
        multi_vector_embedding: MultiVectorEmbeddings | None
            Optional late-interaction model, for example
            `FastEmbedLateInteraction`.
        retrieval_mode: RetrievalMode | str
            Default search mode. `"semantic"` when no embedding is given,
            otherwise `"dense"`.

    Key init args — client:
        api_key: str | None
            TopK API key. Defaults to the `TOPK_API_KEY` environment variable.
        region: str | None
            TopK region, for example `"aws-us-east-1-elastica"`. Defaults to
            the `TOPK_REGION` environment variable.
        client: topk_sdk.Client | None
            An existing TopK client to use instead of `api_key` and `region`.

    Instantiate:
        ```python
        from langchain_topk import TopKVectorStore

        vector_store = TopKVectorStore(collection_name="langchain-docs")
        ```

    Add Documents:
        ```python
        from langchain_core.documents import Document

        vector_store.add_documents(
            [
                Document(
                    page_content="Rust guarantees memory safety.",
                    metadata={"lang": "rust"},
                ),
                Document(
                    page_content="Python has readable syntax.",
                    metadata={"lang": "python"},
                ),
            ],
            ids=["rust", "python"],
        )
        ```

    Search:
        ```python
        results = vector_store.similarity_search("memory safe language", k=1)
        ```

    Hybrid search with a filter:
        ```python
        results = vector_store.similarity_search(
            "memory safety",
            k=3,
            mode="hybrid",
            filter={"lang": {"$in": ["rust", "go"]}},
        )
        ```

    Use as Retriever:
        ```python
        retriever = vector_store.as_retriever(search_kwargs={"k": 3, "mode": "keyword"})
        retriever.invoke("garbage collector")
        ```
    """

    def __init__(
        self,
        collection_name: str,
        embedding: Embeddings | None = None,
        *,
        sparse_embedding: SparseEmbeddings | None = None,
        multi_vector_embedding: MultiVectorEmbeddings | None = None,
        retrieval_mode: RetrievalMode | str | None = None,
        partition: str | None = None,
        api_key: str | None = None,
        region: str | None = None,
        host: str = "topk.io",
        https: bool = True,
        client: Client | None = None,
        async_client: AsyncClient | None = None,
        text_field: str = "text",
        metadata_field: str = "metadata",
        metadata_json_field: str = "metadata_json",
        vector_field: str = "dense_vector",
        sparse_vector_field: str = "sparse_vector",
        multi_vector_field: str = "multi_vector",
        text_index: TextIndex | str | None = None,
        distance_metric: Metric | None = None,
        hybrid_weights: Mapping[str, float] | None = None,
        consistency: Consistency | None = None,
        read_your_writes: bool = True,
        create_collection: bool = True,
    ) -> None:
        """Create a TopK vector store.

        Args:
            collection_name: Name of the TopK collection.
            embedding: Optional dense embedding model. When omitted, TopK's
                server-side semantic search is used and no embedding model or
                key is needed.
            sparse_embedding: Optional sparse embedding model (for example
                SPLADE via `FastEmbedSparse`).
            multi_vector_embedding: Optional late-interaction model (for
                example ColBERT via `FastEmbedLateInteraction`).
            retrieval_mode: Default retrieval mode. Defaults to `"semantic"`
                without `embedding` and to `"dense"` with it.
            partition: Partition (tenant) to scope every read and write to.
            api_key: TopK API key. Defaults to `TOPK_API_KEY`.
            region: TopK region. Defaults to `TOPK_REGION`.
            host: TopK host. Change only for private deployments.
            https: Whether to connect over HTTPS.
            client: Existing `topk_sdk.Client` to use.
            async_client: Existing `topk_sdk.AsyncClient` to use for async
                methods.
            text_field: Field that stores `Document.page_content`.
            metadata_field: Struct field that stores `Document.metadata`.
                Filters run against it, for example `metadata.year`.
            metadata_json_field: Field that stores a JSON copy of the
                metadata. TopK queries return only the fields they name, not a
                whole struct, so searches read metadata from this copy in the
                same query. Documents without it (for example, written outside
                LangChain) get their metadata with an extra `get()` by ID.
            vector_field: Field that stores dense vectors.
            sparse_vector_field: Field that stores sparse vectors.
            multi_vector_field: Field that stores multi-vectors.
            text_index: Index for the text field when the collection is
                created: `"semantic"` (default, server-side embeddings and
                BM25), `"keyword"` (BM25 only) or `"none"`. For an existing
                collection it is read from the collection's schema.
            distance_metric: Metric for the dense vector index: `"cosine"`
                (default), `"dot_product"` or `"euclidean"`. For an existing
                collection it is read from the collection's schema.
            hybrid_weights: Default weights for hybrid search, keyed by signal:
                `"semantic"`, `"keyword"`, `"dense"`, `"sparse"`,
                `"multi_vector"`. Every key must be a signal this store is
                configured for.
            consistency: Default read consistency: `None` (balanced, TopK's
                default), `"indexed"` or `"strong"`.
            read_your_writes: Pass the LSN of this store's latest write to
                later reads, so they always see that write. Applies only to
                reads that set neither `lsn` nor `consistency`.
            create_collection: Create the collection if it does not exist,
                or check an existing collection's schema against these
                settings. With `False`, no network call is made.
        """
        if not collection_name:
            msg = "`collection_name` is required."
            raise ValueError(msg)

        self.collection_name = collection_name
        self.partition = partition
        self._embedding = embedding
        self.sparse_embedding = sparse_embedding
        self.multi_vector_embedding = multi_vector_embedding
        self.text_field = text_field
        self.metadata_field = metadata_field
        self.metadata_json_field = metadata_json_field
        self.vector_field = vector_field
        self.sparse_vector_field = sparse_vector_field
        self.multi_vector_field = multi_vector_field
        self._requested_text_index = (
            TextIndex(text_index) if text_index is not None else None
        )
        self.text_index = self._requested_text_index or TextIndex.SEMANTIC
        if distance_metric is not None and distance_metric not in _METRICS:
            msg = (
                "`distance_metric` must be 'cosine', 'dot_product' or "
                f"'euclidean', got {distance_metric!r}."
            )
            raise ValueError(msg)
        self._requested_metric = distance_metric
        self.distance_metric: Metric = distance_metric or "cosine"
        self.consistency = consistency
        self.read_your_writes = read_your_writes
        self.hybrid_weights = dict(hybrid_weights or {})
        self._validate_weights(self.hybrid_weights)
        self._requested_mode = (
            RetrievalMode(retrieval_mode) if retrieval_mode is not None else None
        )
        self.retrieval_mode = RetrievalMode.SEMANTIC
        self._finalize_config()

        self._api_key = api_key
        self._region = region
        self._host = host
        self._https = https
        self._client = client
        self._async_client = async_client
        self.last_lsn: str | None = None
        """LSN of this store's latest write, used for read-your-writes."""

        if create_collection:
            self._ensure_collection()

    def _finalize_config(self) -> None:
        """Resolve the default retrieval mode and validate the configuration."""
        mode = self._requested_mode
        if mode is None:
            mode = (
                RetrievalMode.DENSE
                if self._embedding is not None
                else RetrievalMode.SEMANTIC
            )
        self._check_mode(mode)
        self.retrieval_mode = mode
        if self.hybrid_weights:
            # Fail at construction, not at the first hybrid search.
            self._signals(RetrievalMode.HYBRID, None)

    # ------------------------------------------------------------------
    # Clients and collection management
    # ------------------------------------------------------------------

    def _credentials(self) -> tuple[str, str]:
        api_key = self._api_key or os.environ.get("TOPK_API_KEY")
        region = self._region or os.environ.get("TOPK_REGION")
        if not api_key:
            msg = (
                "TopK API key not found. Pass `api_key=` or set the TOPK_API_KEY "
                "environment variable. Get a key at https://console.topk.io."
            )
            raise ValueError(msg)
        if not region:
            msg = (
                "TopK region not found. Pass `region=` or set the TOPK_REGION "
                "environment variable, for example 'aws-us-east-1-elastica'. "
                "See https://docs.topk.io/regions."
            )
            raise ValueError(msg)
        return api_key, region

    @property
    def client(self) -> Client:
        """The synchronous TopK client."""
        if self._client is None:
            api_key, region = self._credentials()
            self._client = Client(
                api_key=api_key, region=region, host=self._host, https=self._https
            )
        return self._client

    @property
    def async_client(self) -> AsyncClient:
        """The asynchronous TopK client."""
        if self._async_client is None:
            api_key, region = self._credentials()
            self._async_client = AsyncClient(
                api_key=api_key, region=region, host=self._host, https=self._https
            )
        return self._async_client

    def _collection(self) -> CollectionClient:
        return self.client.collection(self.collection_name, self.partition)

    def _acollection(self) -> AsyncCollectionClient:
        return self.async_client.collection(self.collection_name, self.partition)

    def _schema(self, dense_dim: int | None, multi_vector_dim: int | None) -> dict:
        schema: dict[str, Any] = {}
        if self.text_index == TextIndex.SEMANTIC:
            schema[self.text_field] = text().index(semantic_index())
        elif self.text_index == TextIndex.KEYWORD:
            schema[self.text_field] = text().index(keyword_index())
        else:
            schema[self.text_field] = text()
        if dense_dim is not None:
            schema[self.vector_field] = f32_vector(dense_dim).index(
                vector_index(metric=self.distance_metric)
            )
        if self.sparse_embedding is not None:
            schema[self.sparse_vector_field] = f32_sparse_vector().index(
                vector_index(metric="dot_product")
            )
        if multi_vector_dim is not None:
            schema[self.multi_vector_field] = matrix(
                multi_vector_dim, value_type="f32"
            ).index(multi_vector_index(metric="maxsim"))
        return schema

    def _ensure_collection(self) -> None:
        collections = self.client.collections()
        try:
            existing = collections.get(self.collection_name)
        except CollectionNotFoundError:
            existing = None
        if existing is not None:
            self._reconcile_schema(existing.schema)
            return
        dense_dim = (
            len(self._embedding.embed_query("dimension probe"))
            if self._embedding is not None
            else None
        )
        multi_dim = None
        if self.multi_vector_embedding is not None:
            probe = normalize_multi_vector(
                self.multi_vector_embedding.embed_query("dimension probe")
            )
            multi_dim = len(probe[0])
        try:
            collections.create(
                self.collection_name, schema=self._schema(dense_dim, multi_dim)
            )
        except CollectionAlreadyExistsError:
            self._reconcile_schema(collections.get(self.collection_name).schema)

    async def _aensure_collection(self) -> None:
        collections = self.async_client.collections()
        try:
            existing = await collections.get(self.collection_name)
        except CollectionNotFoundError:
            existing = None
        if existing is not None:
            self._reconcile_schema(existing.schema)
            return
        dense_dim = (
            len(await self._embedding.aembed_query("dimension probe"))
            if self._embedding is not None
            else None
        )
        multi_dim = None
        if self.multi_vector_embedding is not None:
            probe = normalize_multi_vector(
                await self.multi_vector_embedding.aembed_query("dimension probe")
            )
            multi_dim = len(probe[0])
        try:
            await collections.create(
                self.collection_name, schema=self._schema(dense_dim, multi_dim)
            )
        except CollectionAlreadyExistsError:
            existing = await collections.get(self.collection_name)
            self._reconcile_schema(existing.schema)

    def _reconcile_schema(self, schema: Mapping[str, Any]) -> None:
        """Check an existing collection's schema against this store's settings.

        The text index and vector metric are read from the schema (a field's
        metric is only visible in its text form). A setting that contradicts
        the schema raises `ValueError` instead of silently returning wrong
        results.
        """
        specs = {name: str(spec) for name, spec in schema.items()}
        where = f"collection {self.collection_name!r}"

        text_spec = specs.get(self.text_field, "")
        if "SemanticIndex" in text_spec:
            actual_index = TextIndex.SEMANTIC
        elif "KeywordIndex" in text_spec:
            actual_index = TextIndex.KEYWORD
        else:
            actual_index = TextIndex.NONE
        requested = self._requested_text_index
        if requested is not None and requested != actual_index:
            msg = (
                f"`text_index={requested.value!r}` does not match {where}, whose "
                f"field {self.text_field!r} has text index {actual_index.value!r}."
            )
            raise ValueError(msg)
        self.text_index = actual_index

        def indexed(field_name: str, kind: str, what: str) -> str:
            spec = specs.get(field_name)
            if spec is None or kind not in spec:
                msg = (
                    f"{what} needs an indexed field {field_name!r}, but {where} "
                    f"has {spec or 'no such field'}. Use a new collection, or set "
                    "the field name to an indexed field of this collection."
                )
                raise ValueError(msg)
            return spec

        if self._embedding is not None:
            spec = indexed(self.vector_field, "VectorIndex", "`embedding`")
            metric = next(
                (m for name, m in _METRIC_NAMES.items() if f"metric: {name}" in spec),
                None,
            )
            if metric is not None:
                if self._requested_metric not in (None, metric):
                    msg = (
                        f"`distance_metric={self._requested_metric!r}` does not "
                        f"match {where}, whose vector index uses {metric!r}."
                    )
                    raise ValueError(msg)
                self.distance_metric = metric
        if self.sparse_embedding is not None:
            indexed(self.sparse_vector_field, "VectorIndex", "`sparse_embedding`")
        if self.multi_vector_embedding is not None:
            indexed(
                self.multi_vector_field,
                "MultiVectorIndex",
                "`multi_vector_embedding`",
            )
        self._finalize_config()

    def delete_collection(self) -> None:
        """Delete the whole TopK collection, including every partition.

        Warning:
            For about a minute after a delete, writes to a collection that is
            recreated under the same name can be lost. Use a new name instead.
        """
        self.client.collections().delete(self.collection_name)
        self.last_lsn = None

    def delete_partition(self, partition: str | None = None) -> None:
        """Delete a partition (tenant) and all of its documents.

        Args:
            partition: Partition to delete. Defaults to this store's partition.
        """
        name = partition or self.partition
        if not name:
            msg = "No partition given and this store is not scoped to one."
            raise ValueError(msg)
        self.client.collection(self.collection_name).delete_partition(name)
        if name == self.partition:
            self.last_lsn = None

    def list_partitions(self) -> list[str]:
        """Return the names of the collection's partitions."""
        return [
            p.name
            for p in self.client.collection(self.collection_name).list_partitions()
        ]

    def with_partition(self, partition: str | None) -> TopKVectorStore:
        """Return a copy of this store scoped to another partition (tenant).

        The copy shares clients and configuration and makes no network calls.
        """
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        clone.partition = partition
        clone.last_lsn = None
        return clone

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> Embeddings | None:
        """The dense embedding model, if any."""
        return self._embedding

    def _validate_weights(self, weights: Mapping[str, float]) -> None:
        allowed = {m.value for m in RetrievalMode} - {"hybrid"}
        unknown = set(weights) - allowed
        if unknown:
            msg = (
                f"Unknown hybrid weight keys {sorted(unknown)}. "
                f"Use any of {sorted(allowed)}."
            )
            raise ValueError(msg)

    def _check_mode(self, mode: RetrievalMode) -> None:
        if mode == RetrievalMode.SEMANTIC and self.text_index != TextIndex.SEMANTIC:
            msg = "Semantic search needs `text_index='semantic'`."
            raise ValueError(msg)
        if mode == RetrievalMode.KEYWORD and self.text_index == TextIndex.NONE:
            msg = "Keyword search needs a semantic or keyword text index."
            raise ValueError(msg)
        if mode == RetrievalMode.DENSE and self._embedding is None:
            msg = "Dense search needs an `embedding` model."
            raise ValueError(msg)
        if mode == RetrievalMode.SPARSE and self.sparse_embedding is None:
            msg = "Sparse search needs a `sparse_embedding` model."
            raise ValueError(msg)
        if mode == RetrievalMode.MULTI_VECTOR and self.multi_vector_embedding is None:
            msg = "Multi-vector search needs a `multi_vector_embedding` model."
            raise ValueError(msg)

    def _resolve_mode(self, mode: RetrievalMode | str | None) -> RetrievalMode:
        resolved = self.retrieval_mode if mode is None else RetrievalMode(mode)
        self._check_mode(resolved)
        return resolved

    def _read_options(
        self, lsn: str | None = None, consistency: Consistency | None = None
    ) -> dict[str, Any]:
        if lsn is not None:
            return {"lsn": lsn}
        if consistency is not None:
            return {"consistency": consistency}
        if self.read_your_writes and self.last_lsn is not None:
            return {"lsn": self.last_lsn}
        if self.consistency is not None:
            return {"consistency": self.consistency}
        return {}

    @staticmethod
    def _check_k(k: int) -> None:
        if k < 1 or k > MAX_QUERY_RESULTS:
            msg = f"`k` must be between 1 and {MAX_QUERY_RESULTS}, got {k}."
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def _build_rows(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict] | None,
        ids: Sequence[str | None] | None,
        dense: Sequence[list[float]] | None,
        sparse: Sequence[Any] | None,
        multi: Sequence[Any] | None,
    ) -> tuple[list[str], list[dict[str, Any]], list[int]]:
        """Build TopK rows. Also returns an estimated payload size per row."""
        n = len(texts)
        if metadatas is not None and len(metadatas) != n:
            msg = f"Got {len(metadatas)} metadatas for {n} texts."
            raise ValueError(msg)
        if ids is not None and len(ids) != n:
            msg = f"Got {len(ids)} ids for {n} texts."
            raise ValueError(msg)
        out_ids = [
            str(i) if i is not None else str(uuid.uuid4())
            for i in (ids if ids is not None else [None] * n)
        ]
        rows: list[dict[str, Any]] = []
        sizes: list[int] = []
        for idx, (doc_id, content) in enumerate(zip(out_ids, texts, strict=True)):
            row: dict[str, Any] = {"_id": doc_id, self.text_field: content}
            size = 64 + len(doc_id) + len(content.encode("utf-8"))
            metadata = dict(metadatas[idx]) if metadatas is not None else {}
            metadata_json = json.dumps(metadata, default=str, ensure_ascii=False)
            row[self.metadata_json_field] = metadata_json
            size += len(metadata_json)
            if metadata:
                row[self.metadata_field] = metadata
                size += len(metadata_json)
            if dense is not None:
                row[self.vector_field] = [float(x) for x in dense[idx]]
                size += 4 * len(row[self.vector_field])
            if sparse is not None:
                sparse_vec = normalize_sparse(sparse[idx])
                row[self.sparse_vector_field] = sparse_data(sparse_vec)
                size += 8 * len(sparse_vec)
            if multi is not None:
                rows_mv = normalize_multi_vector(multi[idx])
                row[self.multi_vector_field] = matrix_data(rows_mv, value_type="f32")
                size += 4 * sum(len(r) for r in rows_mv)
            rows.append(row)
            sizes.append(size)
        return out_ids, rows, sizes

    @staticmethod
    def _size_batches(
        rows: list[dict[str, Any]], sizes: list[int]
    ) -> Iterable[list[dict[str, Any]]]:
        """Split rows into upsert requests that stay under the request limit."""
        batch: list[dict[str, Any]] = []
        total = 0
        for row, size in zip(rows, sizes, strict=True):
            if batch and total + size > MAX_REQUEST_BYTES:
                yield batch
                batch, total = [], 0
            batch.append(row)
            total += size
        if batch:
            yield batch

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        batch_size: int = 256,
        **kwargs: Any,
    ) -> list[str]:
        """Embed (if configured) and upsert texts.

        Documents with an existing ID are replaced. With server-side semantic
        search, no embedding happens on the client.

        Args:
            texts: Texts to add.
            metadatas: Optional metadata dict for each text.
            ids: Optional ID for each text. Random UUIDs are used otherwise.
            batch_size: Number of documents embedded per batch. Upserts are
                further split so that each request stays well under TopK's
                8MB request limit.
            **kwargs: Unused.

        Returns:
            The IDs of the added texts.

        Note:
            Writes are not transactional. If a request fails, documents from
            earlier requests stay written and `last_lsn` points at the last
            successful write. Retrying with the same `ids` is safe, because
            upserts replace documents with the same ID.
        """
        texts = list(texts)
        if not texts:
            return []
        all_ids: list[str] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            chunk_meta = (
                metadatas[start : start + batch_size] if metadatas is not None else None
            )
            chunk_ids = ids[start : start + batch_size] if ids is not None else None
            dense = self._embedding.embed_documents(chunk) if self._embedding else None
            sparse = (
                self.sparse_embedding.embed_documents(chunk)
                if self.sparse_embedding
                else None
            )
            multi = (
                self.multi_vector_embedding.embed_documents(chunk)
                if self.multi_vector_embedding
                else None
            )
            out_ids, rows, sizes = self._build_rows(
                chunk, chunk_meta, chunk_ids, dense, sparse, multi
            )
            for request in self._size_batches(rows, sizes):
                self.last_lsn = self._collection().upsert(request)
            all_ids.extend(out_ids)
        return all_ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        batch_size: int = 256,
        **kwargs: Any,
    ) -> list[str]:
        """Asynchronously embed (if configured) and upsert texts.

        See `add_texts` for details.
        """
        texts = list(texts)
        if not texts:
            return []
        all_ids: list[str] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            chunk_meta = (
                metadatas[start : start + batch_size] if metadatas is not None else None
            )
            chunk_ids = ids[start : start + batch_size] if ids is not None else None
            dense = (
                await self._embedding.aembed_documents(chunk)
                if self._embedding
                else None
            )
            sparse = (
                await self.sparse_embedding.aembed_documents(chunk)
                if self.sparse_embedding
                else None
            )
            multi = (
                await self.multi_vector_embedding.aembed_documents(chunk)
                if self.multi_vector_embedding
                else None
            )
            out_ids, rows, sizes = self._build_rows(
                chunk, chunk_meta, chunk_ids, dense, sparse, multi
            )
            for request in self._size_batches(rows, sizes):
                self.last_lsn = await self._acollection().upsert(request)
            all_ids.extend(out_ids)
        return all_ids

    def _delete_target(self, ids: list[str] | None, filter: Any) -> Any:
        if ids is not None and filter is not None:
            msg = "Pass either `ids` or `filter` to delete, not both."
            raise ValueError(msg)
        if ids is not None:
            return [str(i) for i in ids]
        expr = build_filter(filter, self.metadata_field, allow_text=False)
        if expr is None:
            msg = (
                "Pass `ids` or a `filter` to delete. To remove everything, use "
                "`delete_partition()` or `delete_collection()`."
            )
            raise ValueError(msg)
        return expr

    def delete(
        self,
        ids: list[str] | None = None,
        *,
        filter: dict | Any | None = None,
        **kwargs: Any,
    ) -> bool | None:
        """Delete documents by ID or by metadata filter.

        Args:
            ids: IDs of the documents to delete. Missing IDs are ignored.
            filter: Delete every document matching this filter instead.
            **kwargs: Unused.

        Returns:
            True when the delete was accepted.
        """
        if ids is not None and len(ids) == 0:
            return True
        target = self._delete_target(ids, filter)
        try:
            self.last_lsn = self._collection().delete(target)
        except PartitionNotFoundError:
            pass
        return True

    async def adelete(
        self,
        ids: list[str] | None = None,
        *,
        filter: dict | Any | None = None,
        **kwargs: Any,
    ) -> bool | None:
        """Asynchronously delete documents by ID or by metadata filter."""
        if ids is not None and len(ids) == 0:
            return True
        target = self._delete_target(ids, filter)
        try:
            self.last_lsn = await self._acollection().delete(target)
        except PartitionNotFoundError:
            pass
        return True

    # ------------------------------------------------------------------
    # Reads by ID
    # ------------------------------------------------------------------

    def _docs_from_get(
        self, ids: Sequence[str], found: Mapping[str, Mapping[str, Any]]
    ) -> list[Document]:
        docs = []
        for doc_id in ids:
            row = found.get(doc_id)
            if row is not None:
                docs.append(self._row_to_document({"_id": doc_id, **row}))
        return docs

    def get_by_ids(
        self,
        ids: Sequence[str],
        /,
        *,
        lsn: str | None = None,
        consistency: Consistency | None = None,
    ) -> list[Document]:
        """Get documents by ID, in the order given. Missing IDs are skipped."""
        if not ids:
            return []
        ids = [str(i) for i in ids]
        try:
            found = self._collection().get(
                ids,
                fields=[self.text_field, self.metadata_field],
                **self._read_options(lsn, consistency),
            )
        except PartitionNotFoundError:
            return []
        return self._docs_from_get(ids, found)

    async def aget_by_ids(
        self,
        ids: Sequence[str],
        /,
        *,
        lsn: str | None = None,
        consistency: Consistency | None = None,
    ) -> list[Document]:
        """Asynchronously get documents by ID. Missing IDs are skipped."""
        if not ids:
            return []
        ids = [str(i) for i in ids]
        try:
            found = await self._acollection().get(
                ids,
                fields=[self.text_field, self.metadata_field],
                **self._read_options(lsn, consistency),
            )
        except PartitionNotFoundError:
            return []
        return self._docs_from_get(ids, found)

    # ------------------------------------------------------------------
    # Query building
    # ------------------------------------------------------------------

    def _row_to_document(self, row: Mapping[str, Any]) -> Document:
        metadata = row.get(self.metadata_field)
        if not metadata and row.get(self.metadata_json_field):
            metadata = json.loads(row[self.metadata_json_field])
        return Document(
            id=row.get("_id"),
            page_content=row.get(self.text_field) or "",
            metadata=dict(metadata) if metadata else {},
        )

    def _signals(
        self, mode: RetrievalMode, weights: Mapping[str, float] | None
    ) -> dict[str, float]:
        """Return the scoring signals for `mode` with their weights."""
        if mode != RetrievalMode.HYBRID:
            return {mode.value: 1.0}
        merged = dict(self.hybrid_weights)
        if weights:
            self._validate_weights(weights)
            merged.update(weights)
        available = []
        if self.text_index == TextIndex.SEMANTIC:
            available.append("semantic")
        if self.text_index != TextIndex.NONE:
            available.append("keyword")
        if self._embedding is not None:
            available.append("dense")
        if self.sparse_embedding is not None:
            available.append("sparse")
        if self.multi_vector_embedding is not None:
            available.append("multi_vector")
        unavailable = {k for k, w in merged.items() if w and k not in available}
        if unavailable:
            msg = (
                f"Hybrid weights {sorted(unavailable)} refer to signals this store "
                f"is not configured for. Available signals: {available}."
            )
            raise ValueError(msg)
        if not merged:
            merged = {name: 1.0 for name in available}
        signals = {k: float(v) for k, v in merged.items() if v}
        if not signals:
            msg = "Hybrid search needs at least one non-zero weight."
            raise ValueError(msg)
        return signals

    def _build_query(
        self,
        query: str | None,
        mode: RetrievalMode,
        k: int,
        *,
        filter: Any = None,
        weights: Mapping[str, float] | None = None,
        keyword_query: str | None = None,
        keyword_match_all: bool = False,
        keyword_required: bool = False,
        ranking: Any = None,
        ranking_ascending: bool = False,
        dense_vector: Sequence[float] | None = None,
        sparse_vector: Any = None,
        multi_vector: Any = None,
        fetch_k: int | None = None,
    ) -> tuple[Query, dict[str, float]]:
        """Build a TopK query. Returns the query and the signal weights."""
        self._check_k(k)
        signals = self._signals(mode, weights)
        scores: dict[str, Any] = {}
        text_filter = None
        kw_text = keyword_query if keyword_query is not None else query

        if "semantic" in signals:
            if query is None:
                msg = "Semantic search needs a text query."
                raise ValueError(msg)
            scores[SEMANTIC_SCORE] = fn.semantic_similarity(self.text_field, query)
        if "keyword" in signals:
            if not kw_text:
                msg = "Keyword search needs a text query."
                raise ValueError(msg)
            scores[KEYWORD_SCORE] = fn.bm25_score()
            if mode == RetrievalMode.KEYWORD or keyword_required:
                text_filter = match(
                    kw_text, field=self.text_field, all=keyword_match_all
                )
            else:
                # Boost documents that contain query terms without excluding
                # the ones that only match semantically.
                text_filter = should(kw_text, field=self.text_field)
        if "dense" in signals:
            if dense_vector is None:
                msg = "Dense search needs a query vector."
                raise ValueError(msg)
            scores[DENSE_SCORE] = fn.vector_distance(
                self.vector_field, [float(x) for x in dense_vector]
            )
        if "sparse" in signals:
            if sparse_vector is None:
                msg = "Sparse search needs a sparse query vector."
                raise ValueError(msg)
            scores[SPARSE_SCORE] = fn.vector_distance(
                self.sparse_vector_field, sparse_data(normalize_sparse(sparse_vector))
            )
        if "multi_vector" in signals:
            if multi_vector is None:
                msg = "Multi-vector search needs a query matrix."
                raise ValueError(msg)
            scores[MULTI_VECTOR_SCORE] = fn.multi_vector_distance(
                self.multi_vector_field,
                matrix_data(normalize_multi_vector(multi_vector), value_type="f32"),
            )

        q = select(self.text_field, self.metadata_json_field, **scores)
        if text_filter is not None:
            q = q.filter(text_filter)
        expr = build_filter(filter, self.metadata_field)
        if expr is not None:
            q = q.filter(expr)

        limit = fetch_k or k
        self._check_k(limit)
        if ranking is not None:
            sort_expr = ranking(_ScoreFields()) if callable(ranking) else ranking
            if not isinstance(sort_expr, (LogicalExpr, FunctionExpr)):
                msg = (
                    "`ranking` must be a TopK scoring expression, or a callable "
                    "returning one."
                )
                raise ValueError(msg)
            q = q.sort(sort_expr, asc=ranking_ascending)
        else:
            q = q.sort(self._combined_score(signals), asc=self._ascending(signals))
        return q.limit(limit), signals

    def _signal_sign(self, name: str) -> float:
        # Euclidean distance is lower-is-better; every other score is
        # higher-is-better. Negate it inside hybrid sums.
        if name == "dense" and self.distance_metric == "euclidean":
            return -1.0
        return 1.0

    def _ascending(self, signals: Mapping[str, float]) -> bool:
        return list(signals) == ["dense"] and self.distance_metric == "euclidean"

    def _combined_score(self, signals: Mapping[str, float]) -> Any:
        if len(signals) == 1:
            (name,) = signals
            return field(_SCORE_ALIAS[name])
        expr: Any = None
        for name, weight in signals.items():
            term = field(_SCORE_ALIAS[name]).coalesce(0.0) * (
                weight * self._signal_sign(name)
            )
            expr = term if expr is None else expr + term
        return expr

    def _combined_value(
        self, row: Mapping[str, Any], signals: Mapping[str, float]
    ) -> float:
        if len(signals) == 1:
            (name,) = signals
            value = row.get(_SCORE_ALIAS[name])
            return float(value) if value is not None else 0.0
        total = 0.0
        for name, weight in signals.items():
            value = row.get(_SCORE_ALIAS[name])
            if value is not None:
                total += float(value) * weight * self._signal_sign(name)
        return total

    def _embed_for(self, query: str, signals: Mapping[str, float]) -> dict[str, Any]:
        vectors: dict[str, Any] = {}
        if "dense" in signals and self._embedding is not None:
            vectors["dense_vector"] = self._embedding.embed_query(query)
        if "sparse" in signals and self.sparse_embedding is not None:
            vectors["sparse_vector"] = self.sparse_embedding.embed_query(query)
        if "multi_vector" in signals and self.multi_vector_embedding is not None:
            vectors["multi_vector"] = self.multi_vector_embedding.embed_query(query)
        return vectors

    async def _aembed_for(
        self, query: str, signals: Mapping[str, float]
    ) -> dict[str, Any]:
        vectors: dict[str, Any] = {}
        if "dense" in signals and self._embedding is not None:
            vectors["dense_vector"] = await self._embedding.aembed_query(query)
        if "sparse" in signals and self.sparse_embedding is not None:
            vectors["sparse_vector"] = await self.sparse_embedding.aembed_query(query)
        if "multi_vector" in signals and self.multi_vector_embedding is not None:
            vectors["multi_vector"] = await self.multi_vector_embedding.aembed_query(
                query
            )
        return vectors

    @staticmethod
    def _split_kwargs(kwargs: dict[str, Any]) -> tuple[dict, dict]:
        """Split search kwargs into query-building and read options."""
        read = {
            "lsn": kwargs.pop("lsn", None),
            "consistency": kwargs.pop("consistency", None),
        }
        allowed = {
            "filter",
            "weights",
            "keyword_query",
            "keyword_match_all",
            "keyword_required",
            "ranking",
            "ranking_ascending",
        }
        unknown = set(kwargs) - allowed - {"mode"}
        if unknown:
            msg = f"Unexpected search arguments: {sorted(unknown)}."
            raise TypeError(msg)
        build = {k: v for k, v in kwargs.items() if k in allowed}
        return build, read

    def _needs_metadata(self, rows: list[dict[str, Any]]) -> list[str]:
        return [
            r["_id"]
            for r in rows
            if self.metadata_json_field not in r and self.metadata_field not in r
        ]

    def _merge_metadata(
        self, rows: list[dict[str, Any]], found: Mapping[str, Mapping[str, Any]]
    ) -> None:
        for row in rows:
            metadata = found.get(row["_id"], {}).get(self.metadata_field)
            if metadata is not None:
                row[self.metadata_field] = metadata

    def _run(self, q: Query, read: Mapping[str, Any]) -> list[dict[str, Any]]:
        options = self._read_options(**read)
        try:
            rows = self._collection().query(q, **options)
            missing = self._needs_metadata(rows)
            if missing:
                found = self._collection().get(
                    missing, fields=[self.metadata_field], **options
                )
                self._merge_metadata(rows, found)
            return rows
        except PartitionNotFoundError:
            return []

    async def _arun(self, q: Query, read: Mapping[str, Any]) -> list[dict[str, Any]]:
        options = self._read_options(**read)
        try:
            rows = await self._acollection().query(q, **options)
            missing = self._needs_metadata(rows)
            if missing:
                found = await self._acollection().get(
                    missing, fields=[self.metadata_field], **options
                )
                self._merge_metadata(rows, found)
            return rows
        except PartitionNotFoundError:
            return []

    def _results(
        self, rows: list[dict[str, Any]], signals: Mapping[str, float]
    ) -> list[tuple[Document, float]]:
        return [
            (self._row_to_document(r), self._combined_value(r, signals)) for r in rows
        ]

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        mode: RetrievalMode | str | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search the collection and return documents with their scores.

        Args:
            query: The text query.
            k: Number of documents to return (at most 10,000).
            mode: Retrieval mode for this call. Defaults to the store's
                `retrieval_mode`.
            **kwargs: Additional search options:

                * `filter`: dict filter (for example `{"year": {"$gte": 2020}}`)
                  or a TopK expression.
                * `weights`: hybrid weights, for example
                  `{"semantic": 0.7, "keyword": 0.3}`.
                * `keyword_query`: BM25 query text if it should differ from
                  `query`.
                * `keyword_match_all`: in keyword mode, require every term.
                * `keyword_required`: in hybrid mode, only return documents
                  that contain at least one query term.
                * `ranking`: custom scoring expression used to sort results.
                  Either a TopK expression or a callable that receives the
                  score fields (`s.semantic`, `s.keyword`, `s.dense`,
                  `s.sparse`, `s.multi_vector`) and returns one. Results are
                  sorted by it in descending order (pass
                  `ranking_ascending=True` to reverse). The returned scores are
                  still the signal scores described below, not the ranking
                  value, so they may not be in sorted order.
                * `lsn` / `consistency`: read consistency for this call.

        Returns:
            Documents with scores, most relevant first. Scores are the raw
            TopK scores (for hybrid search, the weighted sum). For
            `distance_metric="euclidean"` dense search, lower is better.
        """
        resolved = self._resolve_mode(mode)
        build, read = self._split_kwargs(kwargs)
        signals = self._signals(resolved, build.get("weights"))
        vectors = self._embed_for(query, signals)
        q, signals = self._build_query(query, resolved, k, **build, **vectors)
        return self._results(self._run(q, read), signals)

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        *,
        mode: RetrievalMode | str | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Asynchronously search and return documents with scores.

        See `similarity_search_with_score` for the arguments.
        """
        resolved = self._resolve_mode(mode)
        build, read = self._split_kwargs(kwargs)
        signals = self._signals(resolved, build.get("weights"))
        vectors = await self._aembed_for(query, signals)
        q, signals = self._build_query(query, resolved, k, **build, **vectors)
        return self._results(await self._arun(q, read), signals)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Search the collection.

        Accepts the same arguments as `similarity_search_with_score`,
        including `mode`, `filter`, `weights` and `ranking`.
        """
        return [doc for doc, _ in self.similarity_search_with_score(query, k, **kwargs)]

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously search the collection."""
        return [
            doc
            for doc, _ in await self.asimilarity_search_with_score(query, k, **kwargs)
        ]

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Dense vector search with a precomputed query embedding."""
        kwargs.pop("mode", None)
        build, read = self._split_kwargs(kwargs)
        if self._embedding is None:
            msg = "Search by vector needs a store created with an `embedding`."
            raise ValueError(msg)
        q, signals = self._build_query(
            None, RetrievalMode.DENSE, k, dense_vector=embedding, **build
        )
        return self._results(self._run(q, read), signals)

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Dense vector search with a precomputed query embedding."""
        return [
            doc
            for doc, _ in self.similarity_search_with_score_by_vector(
                embedding, k, **kwargs
            )
        ]

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously run a dense vector search with a query embedding."""
        kwargs.pop("mode", None)
        build, read = self._split_kwargs(kwargs)
        if self._embedding is None:
            msg = "Search by vector needs a store created with an `embedding`."
            raise ValueError(msg)
        q, signals = self._build_query(
            None, RetrievalMode.DENSE, k, dense_vector=embedding, **build
        )
        return [doc for doc, _ in self._results(await self._arun(q, read), signals)]

    def keyword_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """BM25 keyword search. Shortcut for `similarity_search(mode="keyword")`."""
        return self.similarity_search(query, k, mode=RetrievalMode.KEYWORD, **kwargs)

    def hybrid_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Hybrid search. Shortcut for `similarity_search(mode="hybrid")`."""
        return self.similarity_search(query, k, mode=RetrievalMode.HYBRID, **kwargs)

    def search_with_query(
        self,
        query: Query,
        *,
        lsn: str | None = None,
        consistency: Consistency | None = None,
    ) -> list[Document]:
        """Run any TopK query against this store's collection and partition.

        Use this when you need the full TopK query language. Select the text
        field to get it back as `page_content`; each document's metadata is
        filled in automatically. Any other selected values (for example
        scores) are added to the document's metadata.

        Example:
            ```python
            from topk_sdk.query import field, fn, select

            docs = vector_store.search_with_query(
                select("text", score=fn.semantic_similarity("text", "rust"))
                .filter(field("metadata.year") >= 2020)
                .sort(field("score"), asc=False)
                .limit(5)
            )
            ```
        """
        rows = self._run(query, {"lsn": lsn, "consistency": consistency})
        docs = []
        for row in rows:
            doc = self._row_to_document(row)
            for key, value in row.items():
                if key not in (
                    "_id",
                    self.text_field,
                    self.metadata_field,
                    self.metadata_json_field,
                ):
                    doc.metadata[key] = value
            docs.append(doc)
        return docs

    # ------------------------------------------------------------------
    # Relevance scores
    # ------------------------------------------------------------------

    def _relevance_fn(self, mode: RetrievalMode) -> Callable[[float], float]:
        if mode != RetrievalMode.DENSE:
            msg = (
                "Normalized relevance scores are only defined for dense vector "
                f"search, not for mode {mode.value!r}: semantic, BM25, sparse, "
                "multi-vector and hybrid scores are not bounded. Use "
                "`similarity_search_with_score` instead."
            )
            raise NotImplementedError(msg)
        if self.distance_metric == "euclidean":
            # TopK returns the distance (lower is better); assumes unit vectors.
            return lambda d: _clamp01(1.0 - d / math.sqrt(2))
        # Cosine similarity, and the raw dot product of unit vectors, are in
        # [-1, 1] with higher being better.
        return lambda s: _clamp01((s + 1.0) / 2.0)

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return self._relevance_fn(self.retrieval_mode)

    def _similarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        relevance = self._relevance_fn(self._resolve_mode(kwargs.get("mode")))
        if kwargs.get("ranking") is not None:
            msg = "Relevance scores cannot be computed with a custom `ranking`."
            raise ValueError(msg)
        return [
            (doc, relevance(score))
            for doc, score in self.similarity_search_with_score(query, k, **kwargs)
        ]

    async def _asimilarity_search_with_relevance_scores(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        relevance = self._relevance_fn(self._resolve_mode(kwargs.get("mode")))
        if kwargs.get("ranking") is not None:
            msg = "Relevance scores cannot be computed with a custom `ranking`."
            raise ValueError(msg)
        return [
            (doc, relevance(score))
            for doc, score in await self.asimilarity_search_with_score(
                query, k, **kwargs
            )
        ]

    # ------------------------------------------------------------------
    # Maximal marginal relevance
    # ------------------------------------------------------------------

    def _mmr_candidates(
        self, embedding: list[float], fetch_k: int, read: Mapping[str, Any], build: dict
    ) -> tuple[list[Document], list[list[float]]]:
        q, _ = self._build_query(
            None, RetrievalMode.DENSE, fetch_k, dense_vector=embedding, **build
        )
        rows = self._run(q, read)
        if not rows:
            return [], []
        ids = [r["_id"] for r in rows]
        # A query cannot return an indexed vector field, so fetch the vectors
        # of the candidates by ID.
        found = self._collection().get(
            ids, fields=[self.vector_field], **self._read_options(**read)
        )
        return self._mmr_pairs(rows, found)

    async def _ammr_candidates(
        self, embedding: list[float], fetch_k: int, read: Mapping[str, Any], build: dict
    ) -> tuple[list[Document], list[list[float]]]:
        q, _ = self._build_query(
            None, RetrievalMode.DENSE, fetch_k, dense_vector=embedding, **build
        )
        rows = await self._arun(q, read)
        if not rows:
            return [], []
        ids = [r["_id"] for r in rows]
        found = await self._acollection().get(
            ids, fields=[self.vector_field], **self._read_options(**read)
        )
        return self._mmr_pairs(rows, found)

    def _mmr_pairs(
        self, rows: list[dict[str, Any]], found: Mapping[str, Mapping[str, Any]]
    ) -> tuple[list[Document], list[list[float]]]:
        docs, vectors = [], []
        for row in rows:
            vector = found.get(row["_id"], {}).get(self.vector_field)
            if vector is None:
                continue
            docs.append(self._row_to_document(row))
            vectors.append([float(x) for x in vector])
        return docs, vectors

    def _check_mmr(self) -> Embeddings:
        if self._embedding is None:
            msg = (
                "Maximal marginal relevance needs client-side vectors, so the "
                "store must be created with an `embedding`."
            )
            raise ValueError(msg)
        return self._embedding

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return documents selected by maximal marginal relevance.

        Fetches `fetch_k` candidates by dense vector search, then picks `k` of
        them that balance relevance to the query and diversity.
        """
        self._check_mmr()
        kwargs.pop("mode", None)
        build, read = self._split_kwargs(kwargs)
        docs, vectors = self._mmr_candidates(embedding, max(fetch_k, k), read, build)
        if not docs:
            return []
        picks = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            vectors,
            lambda_mult=lambda_mult,
            k=min(k, len(docs)),
        )
        return [docs[i] for i in picks]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return documents selected by maximal marginal relevance."""
        embedding = self._check_mmr().embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, **kwargs
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously return documents selected by MMR."""
        self._check_mmr()
        kwargs.pop("mode", None)
        build, read = self._split_kwargs(kwargs)
        docs, vectors = await self._ammr_candidates(
            embedding, max(fetch_k, k), read, build
        )
        if not docs:
            return []
        picks = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            vectors,
            lambda_mult=lambda_mult,
            k=min(k, len(docs)),
        )
        return [docs[i] for i in picks]

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Asynchronously return documents selected by MMR."""
        embedding = await self._check_mmr().aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, lambda_mult, **kwargs
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings | None = None,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        collection_name: str | None = None,
        **kwargs: Any,
    ) -> TopKVectorStore:
        """Create a store (and collection, if needed) from texts.

        `embedding` is optional: without it, TopK computes embeddings on the
        server and the store uses semantic search.

        Example:
            ```python
            vector_store = TopKVectorStore.from_texts(
                ["Rust guarantees memory safety.", "Python has readable syntax."],
                collection_name="langchain-docs",
            )
            ```
        """
        if not collection_name:
            msg = "`collection_name` is required."
            raise ValueError(msg)
        batch_size = kwargs.pop("batch_size", 256)
        store = cls(collection_name, embedding, **kwargs)
        store.add_texts(texts, metadatas, ids=ids, batch_size=batch_size)
        return store

    @classmethod
    async def afrom_texts(
        cls,
        texts: list[str],
        embedding: Embeddings | None = None,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        collection_name: str | None = None,
        **kwargs: Any,
    ) -> TopKVectorStore:
        """Asynchronously create a store from texts. See `from_texts`."""
        if not collection_name:
            msg = "`collection_name` is required."
            raise ValueError(msg)
        batch_size = kwargs.pop("batch_size", 256)
        create_collection = kwargs.pop("create_collection", True)
        # Build without network calls, then create or check the collection
        # with the async client so the event loop is never blocked.
        store = cls(collection_name, embedding, create_collection=False, **kwargs)
        if create_collection:
            await store._aensure_collection()
        await store.aadd_texts(texts, metadatas, ids=ids, batch_size=batch_size)
        return store

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings | None = None,
        **kwargs: Any,
    ) -> TopKVectorStore:
        """Create a store from documents. `embedding` is optional."""
        ids = kwargs.pop("ids", None)
        if ids is None and any(d.id for d in documents):
            ids = [d.id or str(uuid.uuid4()) for d in documents]
        return cls.from_texts(
            [d.page_content for d in documents],
            embedding,
            [d.metadata for d in documents],
            ids=ids,
            **kwargs,
        )

    @classmethod
    async def afrom_documents(
        cls,
        documents: list[Document],
        embedding: Embeddings | None = None,
        **kwargs: Any,
    ) -> TopKVectorStore:
        """Asynchronously create a store from documents."""
        ids = kwargs.pop("ids", None)
        if ids is None and any(d.id for d in documents):
            ids = [d.id or str(uuid.uuid4()) for d in documents]
        return await cls.afrom_texts(
            [d.page_content for d in documents],
            embedding,
            [d.metadata for d in documents],
            ids=ids,
            **kwargs,
        )


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


_SCORE_ALIAS = {
    "semantic": SEMANTIC_SCORE,
    "keyword": KEYWORD_SCORE,
    "dense": DENSE_SCORE,
    "sparse": SPARSE_SCORE,
    "multi_vector": MULTI_VECTOR_SCORE,
}


class _ScoreFields:
    """Score fields passed to a `ranking` callable.

    Each attribute is a TopK field expression for one signal's score. Scores of
    signals that did not match a document are null; use `.coalesce(0.0)` when
    combining them.
    """

    semantic = field(SEMANTIC_SCORE)
    keyword = field(KEYWORD_SCORE)
    dense = field(DENSE_SCORE)
    sparse = field(SPARSE_SCORE)
    multi_vector = field(MULTI_VECTOR_SCORE)
