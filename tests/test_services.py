import os
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".tmp" / "matplotlib"))

from ir_project.services.database import DocumentStore
from ir_project.services.clustering_service import ClusteringService, document_id_digest
from ir_project.services.crawled_retrieval_service import CrawledRetrievalService, CrawledSearchResult
from ir_project.services.evaluation_service import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ir_project.services.query_refinement import QueryRefinementService
from ir_project.services.query_history_service import QueryHistoryService
from ir_project.services.rag_service import RagService
from ir_project.services.retrieval_service import SearchResult
from ir_project.services.search_export_service import SearchExportService
from ir_project.services.text_processing import TextProcessor
from ir_project.services.vector_store_service import VectorStoreService


class FakeRetriever:
    def search(self, query, method, top_k, refine=False):
        return [SearchResult("D1", 1.0, 1, method)]


class ServiceTests(unittest.TestCase):
    def test_document_store_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentStore(Path(directory) / "documents.sqlite")
            store.upsert_many([("D1", "Title", "Original raw text")])
            self.assertEqual(store.count(), 1)
            self.assertEqual(store.get("D1")["body"], "Original raw text")

    def test_text_processing_and_refinement(self):
        processor = TextProcessor()
        self.assertEqual(processor.normalize("The LUNG-cancer Study!"), "lung cancer study")
        refined = QueryRefinementService(processor).refine("treatement cancer")
        self.assertIn("treatment", refined)
        self.assertIn("therapy", refined)
        self.assertIn("tumor", refined)

    def test_query_history_deduplicates_and_keeps_latest_five(self):
        history = []
        for query in ["one", "two", "three", "four", "five", "six"]:
            history = QueryHistoryService.add(history, query, limit=5)
        self.assertEqual(history, ["six", "five", "four", "three", "two"])
        history = QueryHistoryService.add(history, "  FOUR  ", limit=5)
        self.assertEqual(history[0], "FOUR")
        self.assertEqual(sum(item.casefold() == "four" for item in history), 1)
        self.assertEqual(QueryHistoryService.add(history, "   ", limit=5), history)

    def test_ir_metrics(self):
        results = [
            SearchResult("D1", 1.0, 1, "test"),
            SearchResult("D2", 0.5, 2, "test"),
        ]
        relevant = {"D1": 2, "D3": 1}
        self.assertAlmostEqual(average_precision(results, relevant, k=2), 0.5)
        self.assertAlmostEqual(precision_at_k(results, relevant, k=2), 0.5)
        self.assertAlmostEqual(recall_at_k(results, relevant, k=2), 0.5)
        self.assertGreater(ndcg_at_k(results, relevant, k=2), 0.0)

    def test_clustering_groups_results_without_changing_rank(self):
        service = ClusteringService(
            doc_ids=["D1", "D2", "D3"],
            labels=np.array([1, 0, 1]),
            keywords={0: ["diabetes", "insulin"], 1: ["cancer", "egfr"]},
        )
        results = [
            SearchResult("D1", 1.0, 1, "test"),
            SearchResult("D2", 0.8, 2, "test"),
            SearchResult("D3", 0.6, 3, "test"),
        ]
        groups = service.group_results(results)
        self.assertEqual([group.cluster_id for group in groups], [1, 0])
        self.assertEqual([result.rank for result in groups[0].results], [1, 3])
        self.assertEqual(groups[1].results[0].rank, 2)
        self.assertEqual(service.cluster_label(1), "cancer / egfr")

    def test_crawled_retrieval_is_independent_and_ranked(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "crawled.csv"
            fieldnames = ["nct_id", "title", "url", "condition", "summary", "source", "crawl_query", "body"]
            rows = [
                {
                    "nct_id": "NCT1",
                    "title": "Cancer Immunotherapy Trial",
                    "url": "https://clinicaltrials.gov/study/NCT1",
                    "condition": "Cancer",
                    "summary": "Immunotherapy for lung cancer.",
                    "source": "ClinicalTrials.gov",
                    "crawl_query": "cancer",
                    "body": "Cancer immunotherapy trial for lung cancer patients.",
                },
                {
                    "nct_id": "NCT2",
                    "title": "Diabetes Trial",
                    "url": "https://clinicaltrials.gov/study/NCT2",
                    "condition": "Diabetes",
                    "summary": "Insulin treatment study.",
                    "source": "ClinicalTrials.gov",
                    "crawl_query": "diabetes",
                    "body": "Diabetes insulin glucose treatment.",
                },
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            service = CrawledRetrievalService(csv_path)
            results = service.search("lung cancer immunotherapy", top_k=2)
            self.assertEqual(service.document_count, 2)
            self.assertEqual(results[0].nct_id, "NCT1")
            self.assertEqual(results[0].rank, 1)
            self.assertGreater(results[0].score, 0.0)

    def test_vector_store_fallback_uses_saved_lsa_vectors(self):
        processor = TextProcessor()
        doc_ids = ["D1", "D2", "D3"]
        texts = [
            processor.normalize("lung cancer immunotherapy trial"),
            processor.normalize("diabetes insulin glucose study"),
            processor.normalize("heart surgery treatment trial"),
        ]
        vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
        )
        tfidf = vectorizer.fit_transform(texts)
        svd = TruncatedSVD(n_components=2, random_state=42)
        embeddings = normalize(svd.fit_transform(tfidf)).astype(np.float32)
        fake_index = SimpleNamespace(
            doc_ids=doc_ids,
            embedding_matrix=embeddings,
            tfidf_vectorizer=vectorizer,
            svd_model=svd,
            processor=processor,
        )
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            model = NearestNeighbors(metric="cosine", algorithm="brute").fit(embeddings)
            joblib.dump(model, directory_path / "vector_store.joblib")
            metadata = {
                "backend": "sklearn",
                "artifact_file": "vector_store.joblib",
                "document_count": len(doc_ids),
                "dimensions": embeddings.shape[1],
                "document_id_digest": document_id_digest(doc_ids),
            }
            metadata_path = directory_path / "vector_store_metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            service = VectorStoreService.load(fake_index, metadata_path)
            results = service.search("lung cancer", top_k=2)
            self.assertEqual(service.backend, "sklearn")
            self.assertEqual(results[0].doc_id, "D1")
            self.assertEqual(results[0].rank, 1)

    def test_search_export_combines_sources_and_cluster_metadata(self):
        official_results = [SearchResult("NCT1", 0.9, 1, "BM25")]
        official_documents = {
            "NCT1": {
                "title": "Official Cancer Trial",
                "body": "A clinical trial about lung cancer treatment.",
            }
        }
        crawled_results = [
            CrawledSearchResult(
                nct_id="NCT2",
                title="Crawled Diabetes Trial",
                url="https://clinicaltrials.gov/study/NCT2",
                condition="Diabetes",
                summary="An insulin treatment study.",
                source="ClinicalTrials.gov",
                score=0.8,
                rank=1,
            )
        ]
        clustering = ClusteringService(
            doc_ids=["NCT1"],
            labels=np.array([1]),
            keywords={1: ["cancer", "treatment"]},
        )
        rows = SearchExportService.build_rows(
            query="cancer diabetes",
            official_results=official_results,
            official_documents=official_documents,
            crawled_results=crawled_results,
            clustering=clustering,
            clustering_enabled=True,
            query_refinement=False,
            top_k=10,
            bm25_k1=1.5,
            bm25_b=0.75,
        )
        self.assertEqual([row["source"] for row in rows], ["official_dataset", "crawled"])
        self.assertEqual(rows[0]["cluster_id"], 2)
        self.assertEqual(rows[0]["cluster_label"], "cancer / treatment")
        self.assertEqual(rows[1]["cluster_id"], "")
        csv_bytes = SearchExportService.to_csv_bytes(rows)
        self.assertTrue(csv_bytes.startswith(b"\xef\xbb\xbf"))
        self.assertIn(b"official_dataset", csv_bytes)
        self.assertIn(b"crawled", csv_bytes)

    def test_rag_answer_is_grounded_in_database_text(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentStore(Path(directory) / "documents.sqlite")
            store.upsert_many([("D1", "", "EGFR mutations are studied in lung cancer trials.")])
            answer = RagService(FakeRetriever(), store).answer("EGFR lung cancer")
            self.assertIn("EGFR mutations", answer.answer)
            self.assertEqual(answer.sources[0]["doc_id"], "D1")


if __name__ == "__main__":
    unittest.main()
