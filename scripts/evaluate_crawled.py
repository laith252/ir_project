import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", str(ROOT / ".tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tmp" / "matplotlib"))
sys.path.insert(0, str(ROOT / ".codex_deps"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt

from ir_project.config import ARTIFACTS_DIR, FIGURES_DIR, ensure_dirs
from ir_project.services.crawled_retrieval_service import CrawledRetrievalService


QUERIES = [
    "cancer immunotherapy",
    "diabetes insulin glucose",
    "cardiovascular heart disease",
    "depression treatment",
    "surgery anesthesia",
]


def main() -> None:
    ensure_dirs()
    metadata_path = ROOT / "data" / "crawled" / "crawling_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    service = CrawledRetrievalService()
    latencies = []
    result_counts = []
    for _ in range(10):
        for query in QUERIES:
            results, latency = service.timed_search(query, top_k=5)
            latencies.append(latency)
            result_counts.append(len(results))

    metrics = {
        "documents_requested": int(metadata["documents_requested"]),
        "successful_documents": int(metadata["successful_documents"]),
        "failed_documents": int(metadata["failed_documents"]),
        "duplicate_documents_removed": int(metadata["duplicate_documents_removed"]),
        "unique_documents": int(metadata["unique_documents"]),
        "average_document_length_words": float(metadata["average_document_length_words"]),
        "average_search_latency_ms": mean(latencies),
        "max_search_latency_ms": max(latencies),
        "average_results_at_5": mean(result_counts),
    }
    output = ARTIFACTS_DIR / "crawling_evaluation_metrics.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)

    metadata["evaluation"] = {
        "benchmark_queries": QUERIES,
        "benchmark_runs": 10,
        "average_search_latency_ms": metrics["average_search_latency_ms"],
        "max_search_latency_ms": metrics["max_search_latency_ms"],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_path = FIGURES_DIR / "crawling_evaluation.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(
        ["Successful", "Failed"],
        [metrics["successful_documents"], metrics["failed_documents"]],
        color=["#2e8b57", "#d9534f"],
    )
    axes[0].set_title("Crawling Success")
    axes[0].set_ylabel("Documents")
    axes[1].bar(
        ["Unique", "Duplicates"],
        [metrics["unique_documents"], metrics["duplicate_documents_removed"]],
        color=["#2e74b5", "#f0ad4e"],
    )
    axes[1].set_title("Deduplication")
    axes[1].set_ylabel("Documents")
    for axis in axes:
        for container in axis.containers:
            axis.bar_label(container)
        axis.set_ylim(bottom=0)
    fig.suptitle("Offline Web Crawling Evaluation")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    print(f"Saved crawling metrics to {output}")
    print(f"Saved crawling chart to {figure_path}")
    print(metrics)


if __name__ == "__main__":
    main()
