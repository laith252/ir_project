import json
from dataclasses import dataclass

from ir_project.config import ARTIFACTS_DIR, ROOT_DIR, resolve_artifact_path
from ir_project.services.clustering_service import ClusteringService
from ir_project.services.crawled_retrieval_service import CrawledRetrievalService
from ir_project.services.database import DocumentStore
from ir_project.services.indexing_service import load_index
from ir_project.services.rag_service import RagService
from ir_project.services.retrieval_service import RetrievalService
from ir_project.services.vector_store_service import VectorStoreService


@dataclass
class ServiceContainer:
    metadata: dict
    store: DocumentStore
    retriever: RetrievalService
    rag: RagService
    clustering: ClusteringService | None
    crawled: CrawledRetrievalService | None
    vector_store: VectorStoreService | None


def load_service_container() -> ServiceContainer:
    metadata_path = ARTIFACTS_DIR / "dataset_metadata.json"
    if not metadata_path.exists():
        raise RuntimeError("Dataset metadata is missing. Run scripts/prepare.py first.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    store = DocumentStore(resolve_artifact_path(metadata["db_path"], "documents.sqlite"))
    retriever = RetrievalService(load_index(), store)
    clustering_path = ARTIFACTS_DIR / "document_clusters.joblib"
    clustering = None
    if clustering_path.exists():
        try:
            clustering = ClusteringService.load(retriever.index.doc_ids, clustering_path)
        except (KeyError, OSError, RuntimeError, ValueError):
            clustering = None
    crawled_path = ROOT_DIR / "data" / "crawled" / "crawled_trials_clean.csv"
    crawled = None
    if crawled_path.exists():
        try:
            crawled = CrawledRetrievalService(crawled_path)
        except (KeyError, OSError, RuntimeError, ValueError):
            crawled = None
    vector_store = None
    vector_metadata_path = ARTIFACTS_DIR / "vector_store_metadata.json"
    if vector_metadata_path.exists():
        try:
            vector_store = VectorStoreService.load(retriever.index, vector_metadata_path)
        except (ImportError, KeyError, MemoryError, OSError, RuntimeError, ValueError):
            vector_store = None
    return ServiceContainer(
        metadata=metadata,
        store=store,
        retriever=retriever,
        rag=RagService(retriever, store),
        clustering=clustering,
        crawled=crawled,
        vector_store=vector_store,
    )
