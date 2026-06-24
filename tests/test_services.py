import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".tmp" / "matplotlib"))

from ir_project.services.database import DocumentStore
from ir_project.services.evaluation_service import (
    average_precision,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from ir_project.services.query_refinement import QueryRefinementService
from ir_project.services.rag_service import RagService
from ir_project.services.retrieval_service import SearchResult
from ir_project.services.text_processing import TextProcessor


class FakeRetriever:
    def search(self, query, method, top_k, refine=False, history=None):
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

    def test_history_based_query_refinement(self):
        refiner = QueryRefinementService(TextProcessor())
        history = [
            "EGFR lung cancer immunotherapy trials",
            "breast cancer hormone therapy",
            "diabetes insulin treatment",
        ]
        refined = refiner.refine_with_history("lung cancer", history)
        suggestions = refiner.suggest_from_history("lung cancer", history)
        self.assertIn("egfr", refined)
        self.assertIn("immunotherapy", refined)
        self.assertEqual(suggestions[0], "EGFR lung cancer immunotherapy trials")

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

    def test_rag_answer_is_grounded_in_database_text(self):
        with tempfile.TemporaryDirectory() as directory:
            store = DocumentStore(Path(directory) / "documents.sqlite")
            store.upsert_many([("D1", "", "EGFR mutations are studied in lung cancer trials.")])
            answer = RagService(FakeRetriever(), store).answer("EGFR lung cancer")
            self.assertIn("EGFR mutations", answer.answer)
            self.assertEqual(answer.sources[0]["doc_id"], "D1")


if __name__ == "__main__":
    unittest.main()
