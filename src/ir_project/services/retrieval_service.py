from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import normalize

from ir_project.config import TOP_K
from ir_project.services.database import DocumentStore
from ir_project.services.indexing_service import SearchIndex
from ir_project.services.query_refinement import QueryRefinementService


@dataclass
class SearchResult:
    doc_id: str
    score: float
    rank: int
    method: str


class RetrievalService:
    def __init__(self, index: SearchIndex, store: DocumentStore | None = None):
        self.index = index
        self.store = store
        self.doc_pos = {doc_id: pos for pos, doc_id in enumerate(index.doc_ids)}
        self.refiner = QueryRefinementService(index.processor)
        self._bert_reranker = None

    def _clean_query(self, query: str) -> str:
        return self.index.processor.normalize(query)

    def tfidf_search(self, query: str, top_k: int = TOP_K) -> list[SearchResult]:
        query_vec = self.index.tfidf_vectorizer.transform([self._clean_query(query)])
        scores = (self.index.tfidf_matrix @ query_vec.T).toarray().ravel()
        return self._top(scores, top_k, "TF-IDF")

    def embedding_search(self, query: str, top_k: int = TOP_K) -> list[SearchResult]:
        query_tfidf = self.index.tfidf_vectorizer.transform([self._clean_query(query)])
        query_embedding = normalize(self.index.svd_model.transform(query_tfidf))
        scores = self.index.embedding_matrix @ query_embedding.ravel()
        return self._top(scores, top_k, "Embedding-LSA")

    def bm25_search(self, query: str, top_k: int = TOP_K, k1: float = 1.5, b: float = 0.75) -> list[SearchResult]:
        tokens = self._clean_query(query).split()
        rows = self.index.bm25.search(tokens, top_k=top_k, k1=k1, b=b)
        return [SearchResult(doc_id, float(score), rank + 1, "BM25") for rank, (doc_id, score) in enumerate(rows)]

    def hybrid_parallel(self, query: str, top_k: int = TOP_K, alpha: float = 0.25, beta: float = 0.60) -> list[SearchResult]:
        tfidf = self._score_map(self.tfidf_search(query, top_k=top_k * 8))
        bm25 = self._score_map(self.bm25_search(query, top_k=top_k * 8))
        emb = self._score_map(self.embedding_search(query, top_k=top_k * 8))
        candidates = set(tfidf) | set(bm25) | set(emb)
        fused = {}
        for doc_id in candidates:
            fused[doc_id] = alpha * tfidf.get(doc_id, 0.0) + beta * bm25.get(doc_id, 0.0) + (1 - alpha - beta) * emb.get(doc_id, 0.0)
        return self._rank_map(fused, top_k, "Hybrid-Parallel")

    def hybrid_serial(self, query: str, top_k: int = TOP_K) -> list[SearchResult]:
        first_stage = self.bm25_search(query, top_k=top_k * 20)
        query_tfidf = self.index.tfidf_vectorizer.transform([self._clean_query(query)])
        query_embedding = normalize(self.index.svd_model.transform(query_tfidf)).ravel()
        scores = {}
        for result in first_stage:
            pos = self.doc_pos[result.doc_id]
            scores[result.doc_id] = float(self.index.embedding_matrix[pos] @ query_embedding) + 0.15 * result.score
        return self._rank_map(scores, top_k, "Hybrid-Serial")

    def bert_rerank(self, query: str, top_k: int = TOP_K, candidate_k: int = 50, k1: float = 1.5, b: float = 0.75) -> list[SearchResult]:
        if self.store is None:
            raise RuntimeError("BERT reranking requires a DocumentStore.")
        from ir_project.services.bert_reranker import BertReranker

        candidates = self.bm25_search(query, top_k=candidate_k, k1=k1, b=b)
        if self._bert_reranker is None:
            self._bert_reranker = BertReranker(self.store)
        return self._bert_reranker.rerank(query, candidates, top_k=top_k)

    def search(
        self,
        query: str,
        method: str,
        top_k: int = TOP_K,
        k1: float = 1.5,
        b: float = 0.75,
        refine: bool = False,
        history: list[str] | None = None,
    ) -> list[SearchResult]:
        query = self.refiner.refine_with_history(query, history) if refine else query
        methods = {
            "tfidf": lambda: self.tfidf_search(query, top_k),
            "bm25": lambda: self.bm25_search(query, top_k, k1=k1, b=b),
            "embedding": lambda: self.embedding_search(query, top_k),
            "hybrid_parallel": lambda: self.hybrid_parallel(query, top_k),
            "hybrid_serial": lambda: self.hybrid_serial(query, top_k),
            "bert_rerank": lambda: self.bert_rerank(query, top_k=top_k, k1=k1, b=b),
        }
        return methods[method]()

    def _top(self, scores: np.ndarray, top_k: int, method: str) -> list[SearchResult]:
        if scores.size == 0:
            return []
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(self.index.doc_ids[int(index)], float(scores[int(index)]), rank + 1, method)
            for rank, index in enumerate(top_indices)
            if scores[int(index)] > 0
        ]

    def _score_map(self, results: list[SearchResult]) -> dict[str, float]:
        if not results:
            return {}
        max_score = max(result.score for result in results) or 1.0
        return {result.doc_id: result.score / max_score for result in results}

    def _rank_map(self, scores: dict[str, float], top_k: int, method: str) -> list[SearchResult]:
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [SearchResult(doc_id, float(score), rank + 1, method) for rank, (doc_id, score) in enumerate(ranked)]
