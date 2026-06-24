import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("IR_DATASETS_HOME", str(ROOT / "data" / "raw" / "ir_datasets"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", str(ROOT / ".tmp"))

sys.path.insert(0, str(ROOT / ".codex_deps"))
sys.path.insert(0, str(ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ir_project.config import ARTIFACTS_DIR, resolve_artifact_path
from ir_project.services.database import DocumentStore
from ir_project.services.indexing_service import load_index
from ir_project.services.retrieval_service import RetrievalService
from ir_project.services.vector_store_service import VectorStoreService


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a search query from the command line.")
    parser.add_argument("query")
    parser.add_argument("--method", default="hybrid_parallel", choices=["tfidf", "bm25", "embedding", "hybrid_parallel", "hybrid_serial", "bert_rerank", "vector_store"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--refine", action="store_true")
    args = parser.parse_args()

    metadata = json.loads((ARTIFACTS_DIR / "dataset_metadata.json").read_text(encoding="utf-8"))
    store = DocumentStore(resolve_artifact_path(metadata["db_path"], "documents.sqlite"))
    retriever = RetrievalService(load_index(), store)
    try:
        if args.method == "vector_store":
            results = VectorStoreService.load(retriever.index).search(
                args.query,
                top_k=args.top_k,
                refine=args.refine,
            )
        else:
            results = retriever.search(args.query, args.method, top_k=args.top_k, k1=args.k1, b=args.b, refine=args.refine)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from exc
    docs = store.get_many([result.doc_id for result in results])
    for result in results:
        doc = docs.get(result.doc_id, {})
        body = " ".join(doc.get("body", "").split())[:500]
        print(f"{result.rank}. {result.doc_id} | {result.method} | {result.score:.4f}")
        print(body)
        print()


if __name__ == "__main__":
    main()
