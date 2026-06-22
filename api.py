import os
import sys
import csv
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("IR_DATASETS_HOME", str(ROOT / "data" / "raw" / "ir_datasets"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", str(ROOT / ".tmp"))
sys.path.insert(0, str(ROOT / ".codex_deps"))
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ir_project.services.service_container import ServiceContainer, load_service_container
from ir_project.config import ARTIFACTS_DIR


app = FastAPI(
    title="Information Retrieval Service API",
    version="1.0.0",
    description="REST API gateway for retrieval, RAG, and evaluation artifacts.",
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    method: str = "hybrid_parallel"
    top_k: int = Field(default=10, ge=1, le=100)
    k1: float = Field(default=1.5, ge=0.1, le=5.0)
    b: float = Field(default=0.75, ge=0.0, le=1.0)
    refine: bool = False
    cluster_results: bool = False
    include_crawled: bool = False


class RagRequest(SearchRequest):
    top_k: int = Field(default=5, ge=1, le=20)


@lru_cache(maxsize=1)
def services() -> ServiceContainer:
    return load_service_container()


@app.get("/health")
def health() -> dict:
    container = services()
    return {
        "status": "ok",
        "dataset": container.metadata["dataset_id"],
        "documents": container.store.count(),
        "clustering_ready": container.clustering is not None,
        "crawled_documents": container.crawled.document_count if container.crawled is not None else 0,
        "vector_store_ready": container.vector_store is not None,
        "vector_store_backend": container.vector_store.display_name if container.vector_store is not None else None,
    }


@app.post("/search")
def search(request: SearchRequest) -> dict:
    container = services()
    try:
        if request.method == "vector_store":
            if container.vector_store is None:
                raise RuntimeError("Vector Store Search is unavailable. Build the vector index first.")
            results, vector_latency_ms = container.vector_store.timed_search(
                request.query,
                top_k=request.top_k,
                refine=request.refine,
            )
        else:
            results = container.retriever.search(
                request.query,
                method=request.method,
                top_k=request.top_k,
                k1=request.k1,
                b=request.b,
                refine=request.refine,
            )
            vector_latency_ms = None
    except (KeyError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    docs = container.store.get_many([result.doc_id for result in results])
    cluster_ids = {}
    if request.cluster_results and container.clustering is not None:
        cluster_ids = {
            result.doc_id: container.clustering.cluster_id_for(result.doc_id)
            for result in results
        }
    crawled_results = []
    crawled_latency_ms = None
    if request.include_crawled and container.crawled is not None:
        crawled_rows, crawled_latency_ms = container.crawled.timed_search(
            request.query,
            top_k=min(request.top_k, 10),
        )
        crawled_results = [
            {
                "rank": row.rank,
                "nct_id": row.nct_id,
                "title": row.title,
                "url": row.url,
                "condition": row.condition,
                "score": row.score,
                "snippet": row.snippet,
                "source": row.source,
            }
            for row in crawled_rows
        ]
    return {
        "query": request.query,
        "method": request.method,
        "vector_store_latency_ms": vector_latency_ms,
        "results": [
            {
                "rank": result.rank,
                "doc_id": result.doc_id,
                "score": result.score,
                "text": docs.get(result.doc_id, {}).get("body", ""),
                "cluster_id": cluster_ids.get(result.doc_id),
                "cluster_label": (
                    container.clustering.cluster_label(cluster_ids[result.doc_id])
                    if result.doc_id in cluster_ids and cluster_ids[result.doc_id] is not None
                    else None
                ),
            }
            for result in results
        ],
        "crawled_results": crawled_results,
        "crawled_search_latency_ms": crawled_latency_ms,
    }


@app.post("/rag")
def rag(request: RagRequest) -> dict:
    container = services()
    try:
        if request.method == "vector_store":
            if container.vector_store is None:
                raise RuntimeError("Vector Store Search is unavailable. Build the vector index first.")
            vector_results = container.vector_store.search(
                request.query,
                top_k=request.top_k,
                refine=request.refine,
            )
            answer = container.rag.answer_from_results(request.query, vector_results)
        else:
            answer = container.rag.answer(
                request.query,
                method=request.method,
                top_k=request.top_k,
                refine=request.refine,
            )
    except (KeyError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "query": request.query,
        "answer": answer.answer,
        "evidence": answer.evidence,
        "sources": answer.sources,
    }


@app.get("/metrics")
def metrics() -> dict:
    files = {
        "base": "evaluation_metrics.csv",
        "refined": "evaluation_metrics_refined.csv",
        "bert": "evaluation_metrics_bert.csv",
        "rag": "rag_evaluation_metrics.csv",
        "clustering": "clustering_evaluation_metrics.csv",
        "crawling": "crawling_evaluation_metrics.csv",
        "vector_store": "vector_store_evaluation_metrics.csv",
        "feature_comparison": "feature_comparison.csv",
    }
    payload = {}
    for name, filename in files.items():
        path = ARTIFACTS_DIR / filename
        if path.exists():
            with path.open(encoding="utf-8", newline="") as file:
                payload[name] = list(csv.DictReader(file))
    return payload
