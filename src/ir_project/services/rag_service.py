from dataclasses import dataclass
import re

from ir_project.services.database import DocumentStore
from ir_project.services.retrieval_service import RetrievalService, SearchResult
from ir_project.services.text_processing import TextProcessor


@dataclass
class RagAnswer:
    answer: str
    sources: list[dict[str, str]]
    evidence: list[str]


class RagService:
    def __init__(self, retriever: RetrievalService, store: DocumentStore):
        self.retriever = retriever
        self.store = store
        self.processor = TextProcessor()

    def answer(self, question: str, method: str = "hybrid_parallel", top_k: int = 5, refine: bool = False) -> RagAnswer:
        results = self.retriever.search(question, method=method, top_k=top_k, refine=refine)
        return self.answer_from_results(question, results)

    def answer_from_results(self, question: str, results: list[SearchResult]) -> RagAnswer:
        documents = self.store.get_many([result.doc_id for result in results])
        sources = []
        evidence = []
        query_tokens = set(self.processor.tokenize(question))
        for result in results:
            doc = documents.get(result.doc_id)
            if not doc:
                continue
            body = " ".join(doc["body"].split())
            snippet = body[:700]
            sources.append(
                {
                    "doc_id": result.doc_id,
                    "rank": str(result.rank),
                    "score": f"{result.score:.4f}",
                    "snippet": snippet,
                }
            )
            sentence = self._best_sentence(body, query_tokens)
            if sentence:
                evidence.append(f"[{result.rank}] {sentence}")

        if not evidence:
            return RagAnswer(
                "Documents were retrieved, but their original text was not found in the local document store. "
                "Check that artifacts/documents.sqlite exists next to search_index.joblib.",
                sources,
                [],
            )

        answer = (
            "Grounded answer based on the retrieved clinical-trial documents:\n\n"
            + "\n\n".join(evidence[:3])
            + "\n\nThe bracketed numbers refer to the source documents listed below."
        )
        return RagAnswer(answer=answer, sources=sources, evidence=evidence[:3])

    def _best_sentence(self, text: str, query_tokens: set[str]) -> str:
        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
        if not sentences:
            return text[:700]

        def score(sentence: str) -> tuple[int, int]:
            tokens = set(self.processor.tokenize(sentence))
            return len(tokens & query_tokens), min(len(sentence), 500)

        best = max(sentences, key=score)
        return best[:700]
