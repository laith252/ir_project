import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean, median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("IR_DATASETS_HOME", str(ROOT / "data" / "raw" / "ir_datasets"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", str(ROOT / ".tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tmp" / "matplotlib"))
sys.path.insert(0, str(ROOT / ".codex_deps"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np

from ir_project.config import ARTIFACTS_DIR, DEFAULT_DATASET_ID, FIGURES_DIR, ensure_dirs, resolve_artifact_path
from ir_project.services.data_service import load_evaluation_data
from ir_project.services.database import DocumentStore
from ir_project.services.evaluation_service import average_precision, ndcg_at_k, precision_at_k, recall_at_k
from ir_project.services.indexing_service import load_index
from ir_project.services.retrieval_service import RetrievalService
from ir_project.services.vector_store_service import VectorStoreService


def summarize(name, all_results, all_relevant, latencies, depth, load_time_ms=None):
    return {
        "method": name,
        f"map_at_{depth}": mean(average_precision(results, relevant, k=depth) for results, relevant in zip(all_results, all_relevant)),
        "ndcg_at_10": mean(ndcg_at_k(results, relevant, k=10) for results, relevant in zip(all_results, all_relevant)),
        "precision_at_10": mean(precision_at_k(results, relevant, k=10) for results, relevant in zip(all_results, all_relevant)),
        f"recall_at_{depth}": mean(recall_at_k(results, relevant, k=depth) for results, relevant in zip(all_results, all_relevant)),
        "average_latency_ms": mean(latencies),
        "median_latency_ms": median(latencies),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "index_load_time_ms": load_time_ms if load_time_ms is not None else "",
    }


def main() -> None:
    ensure_dirs()
    depth = 1000
    metadata = json.loads((ARTIFACTS_DIR / "dataset_metadata.json").read_text(encoding="utf-8"))
    index = load_index()
    store = DocumentStore(resolve_artifact_path(metadata["db_path"], "documents.sqlite"))
    brute = RetrievalService(index, store)
    load_started = perf_counter()
    vector_store = VectorStoreService.load(index)
    vector_load_time_ms = (perf_counter() - load_started) * 1000.0
    queries, qrels = load_evaluation_data(DEFAULT_DATASET_ID, max_queries=0)
    evaluation_queries = [(query_id, text) for query_id, text in queries.items() if qrels.get(query_id)]

    warmup_query = evaluation_queries[0][1]
    for _ in range(3):
        brute.embedding_search(warmup_query, top_k=10)
        vector_store.search(warmup_query, top_k=10)

    brute_results = []
    vector_results = []
    relevant_rows = []
    brute_latencies = []
    vector_latencies = []
    top10_overlaps = []
    for query_id, query_text in evaluation_queries:
        relevant = qrels[query_id]
        started = perf_counter()
        baseline = brute.embedding_search(query_text, top_k=depth)
        brute_latencies.append((perf_counter() - started) * 1000.0)
        vector, latency = vector_store.timed_search(query_text, top_k=depth)
        vector_latencies.append(latency)
        brute_results.append(baseline)
        vector_results.append(vector)
        relevant_rows.append(relevant)
        baseline_top = {result.doc_id for result in baseline[:10]}
        vector_top = {result.doc_id for result in vector[:10]}
        top10_overlaps.append(len(baseline_top & vector_top) / max(len(baseline_top | vector_top), 1))

    rows = [
        summarize("LSA Brute Force", brute_results, relevant_rows, brute_latencies, depth),
        summarize("FAISS Vector Store", vector_results, relevant_rows, vector_latencies, depth, vector_load_time_ms),
    ]
    for row in rows:
        row["average_top10_overlap_with_lsa"] = 1.0 if row["method"] == "LSA Brute Force" else mean(top10_overlaps)

    output = ARTIFACTS_DIR / "vector_store_evaluation_metrics.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure_path = FIGURES_DIR / "vector_store_evaluation.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = [row["method"] for row in rows]
    quality_names = [f"map_at_{depth}", "ndcg_at_10", "precision_at_10", f"recall_at_{depth}"]
    x = np.arange(len(labels))
    width = 0.18
    for offset, metric in enumerate(quality_names):
        axes[0].bar(x + (offset - 1.5) * width, [float(row[metric]) for row in rows], width, label=metric)
    axes[0].set_xticks(x, labels, rotation=12)
    quality_max = max(float(row[metric]) for row in rows for metric in quality_names)
    axes[0].set_ylim(0, max(0.1, quality_max * 1.2))
    axes[0].set_title("Retrieval Quality")
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, [float(row["average_latency_ms"]) for row in rows], color=["#f0ad4e", "#2e74b5"])
    axes[1].set_title("Average Query Latency")
    axes[1].set_ylabel("Milliseconds")
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.2f")
    fig.suptitle("LSA Brute Force vs Local FAISS Vector Store")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print(f"Evaluated {len(evaluation_queries)} queries with qrels.")
    print(f"Saved metrics to {output}")
    print(f"Saved chart to {figure_path}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
