import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", str(ROOT / ".tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tmp" / "matplotlib"))
sys.path.insert(0, str(ROOT / ".codex_deps"))
sys.path.insert(0, str(ROOT / "src"))

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from ir_project.config import ARTIFACTS_DIR, FIGURES_DIR, ensure_dirs
from ir_project.services.clustering_service import document_id_digest
from ir_project.services.indexing_service import load_index


def cluster_keywords(index, centroids: np.ndarray, keyword_count: int = 7) -> dict[int, list[str]]:
    approximate_tfidf = index.svd_model.inverse_transform(centroids)
    terms = index.tfidf_vectorizer.get_feature_names_out()
    output = {}
    for cluster_id, weights in enumerate(approximate_tfidf):
        top = np.argsort(weights)[::-1][:keyword_count]
        output[cluster_id] = [str(terms[position]) for position in top if weights[position] > 0]
    return output


def save_distribution_chart(labels: np.ndarray, keywords: dict[int, list[str]], output: Path) -> None:
    cluster_ids, counts = np.unique(labels, return_counts=True)
    names = [f"C{cluster_id + 1}: " + ", ".join(keywords[int(cluster_id)][:3]) for cluster_id in cluster_ids]
    plt.figure(figsize=(13, 7))
    plt.barh(names, counts, color="#ff4b4b")
    plt.xlabel("Number of documents")
    plt.title("Document Distribution Across Clusters")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def save_scatter_chart(embeddings: np.ndarray, labels: np.ndarray, output: Path, sample_size: int) -> None:
    rng = np.random.default_rng(42)
    size = min(sample_size, len(embeddings))
    positions = rng.choice(len(embeddings), size=size, replace=False)
    points = PCA(n_components=2, random_state=42).fit_transform(embeddings[positions])
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels[positions], cmap="tab20", s=8, alpha=0.65)
    plt.colorbar(scatter, label="Cluster ID")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title(f"Document Clusters (PCA sample of {size:,})")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build document clusters from the saved LSA embeddings.")
    parser.add_argument("--clusters", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--evaluation-sample", type=int, default=2500)
    parser.add_argument("--scatter-sample", type=int, default=5000)
    args = parser.parse_args()
    if args.clusters < 2:
        parser.error("--clusters must be at least 2")

    ensure_dirs()
    index = load_index()
    embeddings = np.asarray(index.embedding_matrix, dtype=np.float32)
    print(f"Clustering {len(index.doc_ids):,} saved LSA embeddings into {args.clusters} clusters...", flush=True)
    model = MiniBatchKMeans(
        n_clusters=args.clusters,
        batch_size=args.batch_size,
        n_init=5,
        max_iter=100,
        random_state=42,
        reassignment_ratio=0.01,
    )
    labels = model.fit_predict(embeddings).astype(np.int16)
    keywords = cluster_keywords(index, model.cluster_centers_)

    rng = np.random.default_rng(42)
    evaluation_size = min(args.evaluation_sample, len(embeddings))
    evaluation_positions = rng.choice(len(embeddings), size=evaluation_size, replace=False)
    evaluation_embeddings = embeddings[evaluation_positions]
    evaluation_labels = labels[evaluation_positions]
    metrics = {
        "n_clusters": args.clusters,
        "document_count": len(index.doc_ids),
        "evaluation_sample": evaluation_size,
        "silhouette_cosine": float(silhouette_score(evaluation_embeddings, evaluation_labels, metric="cosine")),
        "davies_bouldin": float(davies_bouldin_score(evaluation_embeddings, evaluation_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(evaluation_embeddings, evaluation_labels)),
        "inertia": float(model.inertia_),
    }

    artifact = {
        "version": 1,
        "algorithm": "MiniBatchKMeans",
        "document_count": len(index.doc_ids),
        "document_id_digest": document_id_digest(index.doc_ids),
        "labels": labels,
        "centroids": np.asarray(model.cluster_centers_, dtype=np.float32),
        "keywords": keywords,
        "metrics": metrics,
    }
    artifact_path = ARTIFACTS_DIR / "document_clusters.joblib"
    joblib.dump(artifact, artifact_path, compress=3)

    metrics_path = ARTIFACTS_DIR / "clustering_evaluation_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)

    distribution_path = FIGURES_DIR / "clustering_distribution.png"
    scatter_path = FIGURES_DIR / "clustering_scatter.png"
    save_distribution_chart(labels, keywords, distribution_path)
    save_scatter_chart(embeddings, labels, scatter_path, args.scatter_sample)

    print(f"Saved clustering artifact to {artifact_path}")
    print(f"Saved evaluation metrics to {metrics_path}")
    print(f"Saved charts to {distribution_path} and {scatter_path}")
    for cluster_id in range(args.clusters):
        count = int(np.sum(labels == cluster_id))
        print(f"Cluster {cluster_id + 1}: {count:,} documents | {', '.join(keywords[cluster_id])}")


if __name__ == "__main__":
    main()
