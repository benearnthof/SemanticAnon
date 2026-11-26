from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from semanticsearch.onnx_embedder import ONNXEmbedder
from semanticsearch.semantic.config import SemanticConfig
from semanticsearch.semantic.results import SemanticSearchResult

if TYPE_CHECKING:
    from semanticsearch.lexical.search import LexicalSearcher


class SemanticSearcher:
    """
    Exact semantic retrieval over the precomputed embedding matrix.

    The ONNX model and embedding matrix are loaded once during initialization
    and retained for the lifetime of this object.

    The supplied LexicalSearcher is used to hydrate embedding row indices
    into the common lexical Hit representation.
    """

    def __init__(
        self,
        lexical_searcher: LexicalSearcher,
        config: SemanticConfig | None = None,
    ) -> None:
        self.config = config or SemanticConfig()
        self.config.validate()

        self.lexical_searcher = lexical_searcher
        self._lock = threading.RLock()
        self._closed = False

        self.metadata: dict = {}
        self.passage_ids: list[str] = []

        self.embedder: ONNXEmbedder | None = None
        self.embeddings: np.ndarray | None = None

        self._load_resources()

    def _load_resources(self) -> None:
        """Load and validate all persistent resources."""

        print("Loading semantic search resources...")

        # Load embedding metadata.
        with self.config.metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        if self.metadata.get("dimensions") != self.config.dimensions:
            raise ValueError(
                "Embedding metadata dimension mismatch: "
                f"expected {self.config.dimensions}, "
                f"got {self.metadata.get('dimensions')}."
            )

        if self.metadata.get("matrix_dtype") != "float32":
            raise ValueError(
                "Expected float32 embeddings, got "
                f"{self.metadata.get('matrix_dtype')}."
            )

        if not self.metadata.get("normalized", False):
            raise ValueError(
                "Embedding matrix is not marked as normalized. "
                "Cosine similarity retrieval requires normalized vectors."
            )

        # Load the passage ID mapping.
        with self.config.passage_ids_path.open("r", encoding="utf-8") as f:
            self.passage_ids = json.load(f)

        if not isinstance(self.passage_ids, list):
            raise TypeError("passage_ids.json must contain a JSON list.")

        # Memory-map the embedding matrix. This avoids eagerly copying
        # the approximately 395 MB matrix into another allocation.
        self.embeddings = np.load(
            self.config.embeddings_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        if self.embeddings.ndim != 2:
            raise ValueError(
                f"Expected a 2D embedding matrix, got shape "
                f"{self.embeddings.shape}."
            )

        expected_count = self.metadata.get("passage_count")

        if self.embeddings.shape[0] != len(self.passage_ids):
            raise ValueError(
                "Embedding matrix and passage ID mapping have different "
                f"lengths: {self.embeddings.shape[0]} vs "
                f"{len(self.passage_ids)}."
            )

        if expected_count != self.embeddings.shape[0]:
            raise ValueError(
                "Embedding count does not match metadata: "
                f"{self.embeddings.shape[0]} vs {expected_count}."
            )

        if self.embeddings.shape[1] != self.config.dimensions:
            raise ValueError(
                f"Expected {self.config.dimensions} dimensions, "
                f"got {self.embeddings.shape[1]}."
            )

        # Load the ONNX inference session once.
        self.embedder = ONNXEmbedder(
            model_path=self.config.model_path,
            num_threads=self.config.intra_op_threads,
            max_seq_length=self.config.max_seq_length,
        )

        # Check that the embedding row -> SQLite ordinal mapping is valid.
        self._validate_passage_mapping()

        print("Semantic search ready.")
        print(f"  Model       : {self.metadata.get('model')}")
        print(f"  Dimensions  : {self.embeddings.shape[1]}")
        print(f"  Passages    : {self.embeddings.shape[0]:,}")
        print(f"  Precision   : {self.config.precision.upper()}")
        print(f"  Threads     : {self.config.intra_op_threads}")
        print(f"  Sequence len: {self.config.max_seq_length}")

    def _validate_passage_mapping(self):
        count = len(self.passage_ids)

        if count == 0:
            raise RuntimeError("No passage IDs found.")

        sample_rows = sorted(set([0, count // 2, count - 1]))

        hits = self.lexical_searcher.hydrate(
            sample_rows,
            tier="dense",
        )

        hits_by_ordinal = {hit.ordinal: hit for hit in hits}

        for row in sample_rows:
            hit = hits_by_ordinal.get(row)

            if hit is None:
                raise RuntimeError(
                    f"Could not find SQLite passage for embedding row {row}."
                )

            expected_id = self.passage_ids[row]

            if hit.passage_id != expected_id:
                raise RuntimeError(
                    f"Embedding row / SQLite passage mismatch at row {row}: "
                    f"embeddings contain '{expected_id}', "
                    f"but SQLite contains '{hit.passage_id}'."
                )

    def search(
        self,
        query: str,
        limit: int | None = None,
    ) -> SemanticSearchResult:
        """
        Search for passages semantically related to a natural-language query.

        Returns the top `limit` passages ranked by cosine similarity.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("Query cannot be empty.")

        if limit is None:
            limit = self.config.default_limit

        if limit < 1:
            raise ValueError("limit must be positive.")

        started = time.perf_counter()

        with self._lock:
            if self._closed:
                raise RuntimeError("SemanticSearcher has been closed.")

            if self.embedder is None or self.embeddings is None:
                raise RuntimeError("Semantic search resources are unavailable.")

            # 1. Encode the query using the instructed Qwen query format.
            embedding_started = time.perf_counter()

            query_embedding = self.embedder.encode_query(query)
            query_embedding = np.asarray(query_embedding, dtype=np.float32)

            if query_embedding.shape != (self.config.dimensions,):
                raise RuntimeError(
                    "Unexpected query embedding shape: "
                    f"{query_embedding.shape}."
                )

            embedding_ms = (
                time.perf_counter() - embedding_started
            ) * 1000

            # 2. Exact cosine similarity.
            #
            # Both document and query vectors are L2-normalized, so
            # their dot product is cosine similarity.
            retrieval_started = time.perf_counter()

            scores = self.embeddings @ query_embedding

            k = min(limit, scores.shape[0])

            # Get the k highest-scoring rows without sorting the entire
            # corpus. Then sort only those candidates.
            if k == scores.shape[0]:
                candidate_indices = np.arange(scores.shape[0])
            else:
                candidate_indices = np.argpartition(
                    scores,
                    scores.shape[0] - k,
                )[-k:]

            candidate_scores = scores[candidate_indices]

            # Stable ordering: descending similarity, then ascending row.
            order = np.lexsort((
                candidate_indices,
                -candidate_scores,
            ))

            ordered_indices = candidate_indices[order]
            ordered_scores = candidate_scores[order]

            retrieval_ms = (
                time.perf_counter() - retrieval_started
            ) * 1000

            # 3. Hydrate the embedding row indices into normal Hit objects.
            ordinals = ordered_indices.astype(np.int64).tolist()
            score_list = ordered_scores.astype(np.float32).tolist()

            hits = self.lexical_searcher.hydrate(
                ordinals,
                scores=score_list,
                tier="dense",
            )

        elapsed_ms = (time.perf_counter() - started) * 1000

        return SemanticSearchResult(
            query=query,
            hits=hits,
            elapsed_ms=elapsed_ms,
            embedding_ms=embedding_ms,
            retrieval_ms=retrieval_ms,
        )

    def close(self) -> None:
        """Release the model and embedding matrix references."""

        with self._lock:
            if self._closed:
                return

            self.embedder = None
            self.embeddings = None
            self.passage_ids = []
            self.metadata = {}
            self._closed = True

    def __enter__(self) -> SemanticSearcher:
        if self._closed:
            raise RuntimeError("Cannot re-enter a closed SemanticSearcher.")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()