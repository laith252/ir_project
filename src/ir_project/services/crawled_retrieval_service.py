import csv
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ir_project.config import ROOT_DIR
from ir_project.services.text_processing import TextProcessor


@dataclass
class CrawledSearchResult:
    nct_id: str
    title: str
    url: str
    condition: str
    summary: str
    source: str
    score: float
    rank: int

    @property
    def snippet(self) -> str:
        text = " ".join(self.summary.split())
        return text if len(text) <= 420 else text[:417].rstrip() + "..."


class CrawledRetrievalService:
    """Small, independent TF-IDF index over cached crawled documents."""

    def __init__(self, csv_path: Path | None = None):
        self.csv_path = csv_path or ROOT_DIR / "data" / "crawled" / "crawled_trials_clean.csv"
        self.processor = TextProcessor()
        with self.csv_path.open(encoding="utf-8", newline="") as file:
            self.documents = list(csv.DictReader(file))
        if not self.documents:
            raise RuntimeError("The crawled document cache is empty.")
        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            preprocessor=None,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            max_features=8_000,
            norm="l2",
            dtype=np.float32,
        )
        clean_texts = [self.processor.normalize(document.get("body", "")) for document in self.documents]
        self.matrix = self.vectorizer.fit_transform(clean_texts)

    @property
    def document_count(self) -> int:
        return len(self.documents)

    def search(self, query: str, top_k: int = 5) -> list[CrawledSearchResult]:
        query_vector = self.vectorizer.transform([self.processor.normalize(query)])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        positions = np.argsort(scores)[::-1][:top_k]
        results = []
        for position in positions:
            score = float(scores[int(position)])
            if score <= 0:
                continue
            document = self.documents[int(position)]
            results.append(
                CrawledSearchResult(
                    nct_id=document["nct_id"],
                    title=document["title"],
                    url=document["url"],
                    condition=document["condition"],
                    summary=document["summary"],
                    source=document.get("source", "ClinicalTrials.gov"),
                    score=score,
                    rank=len(results) + 1,
                )
            )
        return results

    def timed_search(self, query: str, top_k: int = 5) -> tuple[list[CrawledSearchResult], float]:
        started = perf_counter()
        results = self.search(query, top_k=top_k)
        return results, (perf_counter() - started) * 1000.0
