import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", str(ROOT / ".tmp"))
sys.path.insert(0, str(ROOT / ".codex_deps"))
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from ir_project.config import ARTIFACTS_DIR, ensure_dirs
from ir_project.services.clustering_service import document_id_digest
from ir_project.services.indexing_service import load_index


def build_faiss(vectors: np.ndarray, artifact_path: Path) -> tuple[str, str, str]:
    import faiss

    vector_index = faiss.IndexFlatIP(vectors.shape[1])
    vector_index.add(vectors)
    faiss.write_index(vector_index, str(artifact_path))
    return "faiss", "IndexFlatIP", str(faiss.__version__)


def build_sklearn(vectors: np.ndarray, artifact_path: Path) -> tuple[str, str, str]:
    vector_index = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    vector_index.fit(vectors)
    joblib.dump(vector_index, artifact_path, compress=3)
    return "sklearn", "NearestNeighbors(metric=cosine)", str(sklearn.__version__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local vector store from saved LSA embeddings.")
    parser.add_argument("--backend", choices=["auto", "faiss", "sklearn"], default="auto")
    args = parser.parse_args()
    ensure_dirs()

    index = load_index()
    vectors = normalize(np.asarray(index.embedding_matrix, dtype=np.float32))
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    print(f"Preparing {len(index.doc_ids):,} normalized LSA vectors with {vectors.shape[1]} dimensions...", flush=True)
    started = perf_counter()

    backend = args.backend
    if backend in {"auto", "faiss"}:
        try:
            artifact_path = ARTIFACTS_DIR / "vector_store.index"
            backend_name, index_type, library_version = build_faiss(vectors, artifact_path)
        except (ImportError, OSError, RuntimeError) as exc:
            if backend == "faiss":
                raise
            print(f"FAISS is unavailable ({exc}); using sklearn NearestNeighbors.", flush=True)
            artifact_path = ARTIFACTS_DIR / "vector_store.joblib"
            backend_name, index_type, library_version = build_sklearn(vectors, artifact_path)
    else:
        artifact_path = ARTIFACTS_DIR / "vector_store.joblib"
        backend_name, index_type, library_version = build_sklearn(vectors, artifact_path)

    build_seconds = perf_counter() - started
    metadata = {
        "version": 1,
        "feature": "Local Vector Store Retrieval using Precomputed LSA Embeddings",
        "backend": backend_name,
        "index_type": index_type,
        "library_version": library_version,
        "artifact_file": artifact_path.name,
        "document_count": len(index.doc_ids),
        "dimensions": int(vectors.shape[1]),
        "vector_dtype": "float32",
        "normalized": True,
        "similarity": "cosine via inner product on normalized vectors",
        "document_id_digest": document_id_digest(index.doc_ids),
        "source_artifact": "search_index.joblib",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "build_seconds": build_seconds,
        "artifact_size_bytes": artifact_path.stat().st_size,
    }
    metadata_path = ARTIFACTS_DIR / "vector_store_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {backend_name} vector store to {artifact_path}")
    print(f"Saved metadata to {metadata_path}")
    print(f"Build time: {build_seconds:.2f} seconds; size: {artifact_path.stat().st_size / (1024 ** 2):.2f} MB")


if __name__ == "__main__":
    main()
