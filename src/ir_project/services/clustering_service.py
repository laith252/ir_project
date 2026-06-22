from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import joblib
import numpy as np

from ir_project.config import ARTIFACTS_DIR
from ir_project.services.retrieval_service import SearchResult


@dataclass
class ClusterGroup:
    cluster_id: int
    label: str
    keywords: list[str]
    results: list[SearchResult]


def document_id_digest(doc_ids: list[str]) -> str:
    digest = sha256()
    for doc_id in doc_ids:
        digest.update(doc_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class ClusteringService:
    """Read-only access to precomputed document clusters.

    Cluster labels are aligned with SearchIndex.doc_ids. Grouping never changes
    the retrieval rank or score; it only organizes already retrieved results.
    """

    def __init__(
        self,
        doc_ids: list[str],
        labels: np.ndarray,
        keywords: dict[int, list[str]],
        metrics: dict | None = None,
    ):
        if len(doc_ids) != len(labels):
            raise ValueError("Cluster labels are not aligned with document IDs.")
        self.doc_ids = doc_ids
        self.labels = np.asarray(labels, dtype=np.int32)
        self.keywords = {int(key): list(value) for key, value in keywords.items()}
        self.metrics = metrics or {}
        self.doc_pos = {doc_id: pos for pos, doc_id in enumerate(doc_ids)}

    @classmethod
    def load(
        cls,
        doc_ids: list[str],
        artifact_path: Path = ARTIFACTS_DIR / "document_clusters.joblib",
    ) -> "ClusteringService":
        artifact = joblib.load(artifact_path)
        if int(artifact.get("document_count", -1)) != len(doc_ids):
            raise RuntimeError("Clustering artifact document count does not match the search index.")
        if artifact.get("document_id_digest") != document_id_digest(doc_ids):
            raise RuntimeError("Clustering artifact document order does not match the search index.")
        return cls(
            doc_ids=doc_ids,
            labels=artifact["labels"],
            keywords=artifact["keywords"],
            metrics=artifact.get("metrics", {}),
        )

    @property
    def cluster_count(self) -> int:
        return len(self.keywords)

    def cluster_id_for(self, doc_id: str) -> int | None:
        pos = self.doc_pos.get(doc_id)
        return int(self.labels[pos]) if pos is not None else None

    def cluster_label(self, cluster_id: int) -> str:
        words = self.keywords.get(int(cluster_id), [])
        return " / ".join(words[:4]) if words else f"Cluster {cluster_id + 1}"

    def group_results(self, results: list[SearchResult]) -> list[ClusterGroup]:
        grouped: OrderedDict[int, list[SearchResult]] = OrderedDict()
        for result in results:
            cluster_id = self.cluster_id_for(result.doc_id)
            if cluster_id is None:
                continue
            grouped.setdefault(cluster_id, []).append(result)
        return [
            ClusterGroup(
                cluster_id=cluster_id,
                label=self.cluster_label(cluster_id),
                keywords=self.keywords.get(cluster_id, []),
                results=cluster_results,
            )
            for cluster_id, cluster_results in grouped.items()
        ]
