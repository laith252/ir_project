import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.preprocessing import normalize

from ir_project.config import ARTIFACTS_DIR
from ir_project.services.clustering_service import document_id_digest
from ir_project.services.indexing_service import SearchIndex
from ir_project.services.query_refinement import QueryRefinementService
from ir_project.services.retrieval_service import SearchResult


class VectorStoreService:
    """Persistent local vector retrieval over precomputed LSA embeddings."""

    def __init__(self, index: SearchIndex, metadata: dict, backend_index):
        self.index = index
        self.metadata = metadata
        self.backend_index = backend_index
        self.backend = str(metadata["backend"])
        self.refiner = QueryRefinementService(index.processor)

    @classmethod
    def load(
        cls,
        index: SearchIndex,
        metadata_path: Path = ARTIFACTS_DIR / "vector_store_metadata.json",
    ) -> "VectorStoreService":
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_count = len(index.doc_ids)
        expected_dimensions = int(index.embedding_matrix.shape[1])
        if int(metadata.get("document_count", -1)) != expected_count:
            raise RuntimeError("Vector store document count does not match the search index.")
        if int(metadata.get("dimensions", -1)) != expected_dimensions:
            raise RuntimeError("Vector store dimensions do not match the saved LSA embeddings.")
        if metadata.get("document_id_digest") != document_id_digest(index.doc_ids):
            raise RuntimeError("Vector store document order does not match the search index.")

        artifact_path = metadata_path.parent / str(metadata["artifact_file"])
        if not artifact_path.exists():
            raise RuntimeError(f"Vector store artifact is missing: {artifact_path.name}")
        backend = str(metadata["backend"])
        if backend == "faiss":
            try:
                import faiss
            except ImportError as exc:
                raise RuntimeError("FAISS vector store requires the faiss-cpu package.") from exc
            backend_index = faiss.read_index(str(artifact_path))
        elif backend == "sklearn":
            backend_index = joblib.load(artifact_path)
        else:
            raise RuntimeError(f"Unsupported vector store backend: {backend}")
        return cls(index, metadata, backend_index)

    @property
    def document_count(self) -> int:
        return int(self.metadata["document_count"])

    @property
    def dimensions(self) -> int:
        return int(self.metadata["dimensions"])

    @property
    def display_name(self) -> str:
        return "FAISS IndexFlatIP" if self.backend == "faiss" else "sklearn NearestNeighbors"

    def _query_embedding(self, query: str, refine: bool) -> np.ndarray:
        query = self.refiner.refine(query) if refine else query
        cleaned = self.index.processor.normalize(query)
        query_tfidf = self.index.tfidf_vectorizer.transform([cleaned])
        embedding = normalize(self.index.svd_model.transform(query_tfidf))
        return np.ascontiguousarray(embedding, dtype=np.float32)

    def search(self, query: str, top_k: int = 10, refine: bool = False) -> list[SearchResult]:
        query_embedding = self._query_embedding(query, refine=refine)
        requested = min(max(int(top_k), 1), self.document_count)
        if self.backend == "faiss":
            scores, positions = self.backend_index.search(query_embedding, requested)
            pairs = zip(positions[0], scores[0])
        else:
            distances, positions = self.backend_index.kneighbors(query_embedding, n_neighbors=requested)
            pairs = zip(positions[0], 1.0 - distances[0])

        results = []
        for position, score in pairs:
            position = int(position)
            score = float(score)
            if position < 0 or score <= 0:
                continue
            results.append(
                SearchResult(
                    doc_id=self.index.doc_ids[position],
                    score=score,
                    rank=len(results) + 1,
                    method=f"VectorStore-{self.backend.upper()}",
                )
            )
        return results

    def timed_search(
        self,
        query: str,
        top_k: int = 10,
        refine: bool = False,
    ) -> tuple[list[SearchResult], float]:
        started = perf_counter()
        results = self.search(query, top_k=top_k, refine=refine)
        return results, (perf_counter() - started) * 1000.0
