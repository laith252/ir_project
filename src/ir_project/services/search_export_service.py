import csv
from io import StringIO

from ir_project.services.clustering_service import ClusteringService
from ir_project.services.crawled_retrieval_service import CrawledSearchResult
from ir_project.services.retrieval_service import SearchResult


EXPORT_FIELDS = [
    "query",
    "source",
    "rank",
    "doc_id",
    "retrieval_method",
    "score",
    "title",
    "condition",
    "snippet",
    "url",
    "cluster_id",
    "cluster_label",
    "query_refinement",
    "top_k",
    "bm25_k1",
    "bm25_b",
]


class SearchExportService:
    @staticmethod
    def build_rows(
        query: str,
        official_results: list[SearchResult],
        official_documents: dict[str, dict[str, str]],
        crawled_results: list[CrawledSearchResult],
        clustering: ClusteringService | None,
        clustering_enabled: bool,
        query_refinement: bool,
        top_k: int,
        bm25_k1: float,
        bm25_b: float,
    ) -> list[dict]:
        rows = []
        for result in official_results:
            document = official_documents.get(result.doc_id, {})
            cluster_id = None
            cluster_label = ""
            if clustering_enabled and clustering is not None:
                cluster_id = clustering.cluster_id_for(result.doc_id)
                if cluster_id is not None:
                    cluster_label = clustering.cluster_label(cluster_id)
            rows.append(
                {
                    "query": query,
                    "source": "official_dataset",
                    "rank": result.rank,
                    "doc_id": result.doc_id,
                    "retrieval_method": result.method,
                    "score": f"{result.score:.8f}",
                    "title": document.get("title", ""),
                    "condition": "",
                    "snippet": SearchExportService._snippet(document.get("body", "")),
                    "url": f"https://clinicaltrials.gov/study/{result.doc_id}",
                    "cluster_id": "" if cluster_id is None else cluster_id + 1,
                    "cluster_label": cluster_label,
                    "query_refinement": query_refinement,
                    "top_k": top_k,
                    "bm25_k1": f"{bm25_k1:.2f}",
                    "bm25_b": f"{bm25_b:.2f}",
                }
            )

        for result in crawled_results:
            rows.append(
                {
                    "query": query,
                    "source": "crawled",
                    "rank": result.rank,
                    "doc_id": result.nct_id,
                    "retrieval_method": "Crawled-TFIDF",
                    "score": f"{result.score:.8f}",
                    "title": result.title,
                    "condition": result.condition,
                    "snippet": result.snippet,
                    "url": result.url,
                    "cluster_id": "",
                    "cluster_label": "",
                    "query_refinement": query_refinement,
                    "top_k": top_k,
                    "bm25_k1": f"{bm25_k1:.2f}",
                    "bm25_b": f"{bm25_b:.2f}",
                }
            )
        return rows

    @staticmethod
    def to_csv_bytes(rows: list[dict]) -> bytes:
        stream = StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=EXPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: SearchExportService._excel_safe(row.get(key, "")) for key in EXPORT_FIELDS})
        return ("\ufeff" + stream.getvalue()).encode("utf-8")

    @staticmethod
    def _snippet(text: str, max_length: int = 650) -> str:
        compact = " ".join(str(text or "").split())
        return compact if len(compact) <= max_length else compact[: max_length - 3].rstrip() + "..."

    @staticmethod
    def _excel_safe(value):
        if isinstance(value, str) and value.startswith(("=", "+", "-", "@")):
            return "'" + value
        return value
