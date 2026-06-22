import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("IR_DATASETS_HOME", str(ROOT / "data" / "raw" / "ir_datasets"))
os.environ.setdefault("TMP", str(ROOT / ".tmp"))
os.environ.setdefault("TEMP", str(ROOT / ".tmp"))

sys.path.insert(0, str(ROOT / ".codex_deps"))
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
import pandas as pd

from ir_project.config import ARTIFACTS_DIR
from ir_project.services.query_history_service import QueryHistoryService
from ir_project.services.search_export_service import SearchExportService
from ir_project.services.service_container import load_service_container


st.set_page_config(page_title="IR Search System", layout="wide")

METHOD_EXPLANATIONS = {
    "tfidf": (
        "TF-IDF represents the query and documents as weighted term vectors, "
        "then ranks documents using cosine similarity."
    ),
    "bm25": (
        "BM25 ranks documents using term frequency, inverse document frequency, "
        "and document length normalization."
    ),
    "embedding": (
        "LSA Embedding projects TF-IDF vectors into a 64-dimensional semantic space "
        "and ranks documents using cosine similarity."
    ),
    "hybrid_parallel": (
        "Hybrid Parallel retrieves with TF-IDF, BM25, and LSA independently, "
        "then combines their normalized scores using weighted score fusion."
    ),
    "hybrid_serial": (
        "Hybrid Serial retrieves candidates using BM25, then reranks them using "
        "semantic LSA embeddings."
    ),
    "bert_rerank": (
        "BERT Re-ranking retrieves the top BM25 candidates, then reranks them "
        "using a Sentence-BERT cross-semantic comparison."
    ),
    "vector_store": (
        "Vector Store Search transforms the query into the saved 64D LSA space, "
        "then retrieves exact nearest vectors from a local FAISS IndexFlatIP index."
    ),
}


@st.cache_resource
def load_services():
    try:
        container = load_service_container()
    except RuntimeError:
        return None, None, None, None, None, None
    return (
        container.metadata,
        container.retriever,
        container.rag,
        getattr(container, "clustering", None),
        getattr(container, "crawled", None),
        getattr(container, "vector_store", None),
    )


metadata, retriever, rag, clustering, crawled, vector_store = load_services()

st.session_state.setdefault("search_query", "lung cancer EGFR adult")
st.session_state.setdefault("query_history", [])


def select_recent_query(query_text: str) -> None:
    st.session_state["search_query"] = query_text


def clear_query_history() -> None:
    st.session_state["query_history"] = []


def render_query_history(placeholder) -> None:
    history = st.session_state.get("query_history", [])
    if not history:
        return
    with placeholder.container():
        st.caption("Recent Queries")
        columns = st.columns(2)
        for index, previous_query in enumerate(history):
            columns[index % 2].button(
                previous_query,
                key=f"recent-query-{index}",
                on_click=select_recent_query,
                args=(previous_query,),
                width="stretch",
            )
        st.button(
            "Clear History",
            key="clear-query-history",
            on_click=clear_query_history,
        )

st.title("Information Retrieval System")

if metadata is None:
    st.error(
        "Artifacts are not ready. Run scripts/prepare.py with "
        "--dataset clinicaltrials/2017/trec-pm-2017 --max-docs 0 --max-queries 0"
    )
    st.stop()

with st.sidebar:
    st.subheader("Configuration")
    st.write(f"Dataset: `{metadata['dataset_id']}`")
    st.write(f"Documents: `{metadata['actual_docs']}`")
    retrieval_methods = [
        "hybrid_parallel",
        "hybrid_serial",
        "bm25",
        "tfidf",
        "embedding",
        "bert_rerank",
    ]
    if vector_store is not None:
        retrieval_methods.append("vector_store")
    method = st.selectbox(
        "Retrieval method",
        retrieval_methods,
    )
    st.info(METHOD_EXPLANATIONS[method], icon="ℹ️")
    if method == "vector_store" and vector_store is not None:
        st.caption(
            f"{vector_store.display_name} | Precomputed LSA | "
            f"{vector_store.dimensions} dimensions | {vector_store.document_count:,} vectors"
        )
    elif vector_store is None:
        st.caption("Vector Store Search is unavailable. Build the vector index first.")
    top_k = st.slider("Top K", min_value=5, max_value=20, value=10)
    k1 = st.slider("BM25 k1", min_value=0.2, max_value=3.0, value=1.5, step=0.1)
    b = st.slider("BM25 b", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
    use_refinement = st.checkbox("Query refinement", value=True)
    if use_refinement:
        st.caption("Query Refinement applies spelling correction and synonym expansion before retrieval.")
    use_clustering = st.checkbox(
        "Enable Document Clustering",
        value=False,
        disabled=clustering is None,
        help="Groups retrieved documents by their precomputed LSA cluster without changing rank or score.",
    )
    if clustering is None:
        st.caption("Clustering artifact is not available. The base search remains unchanged.")
    elif use_clustering:
        st.info(
            "Clustering organizes the retrieved results by topic without changing "
            "their original rank or score.",
            icon="🧩",
        )
    include_crawled = st.checkbox(
        "Include Crawled Documents",
        value=False,
        disabled=crawled is None,
        help="Searches a separate offline TF-IDF index. Crawled results are never mixed into official ranking.",
    )
    if crawled is None:
        st.caption("Offline crawled documents are not available.")
    elif include_crawled:
        st.info(
            "Crawling adds a separate offline result section and never mixes crawled scores "
            "with the official Dataset ranking.",
            icon="🌐",
        )

search_tab, rag_tab, metrics_tab = st.tabs(["Search", "RAG Chat", "Evaluation"])

with search_tab:
    query = st.text_input("Search query", key="search_query")
    query_history_placeholder = st.empty()
    if st.button("Search", type="primary"):
        try:
            crawled_results = []
            vector_latency = None
            if method == "vector_store" and vector_store is not None:
                results, vector_latency = vector_store.timed_search(
                    query,
                    top_k=top_k,
                    refine=use_refinement,
                )
                st.info(
                    f"Vector Store Retrieval: {vector_store.display_name} | "
                    f"{vector_store.dimensions}D LSA | Search latency: {vector_latency:.2f} ms"
                )
            else:
                results = retriever.search(query, method=method, top_k=top_k, k1=k1, b=b, refine=use_refinement)
            docs = rag.store.get_many([result.doc_id for result in results])
            if use_clustering and clustering is not None:
                groups = clustering.group_results(results)
                st.info(
                    f"Document Clustering is ON — {len(results)} results are organized into "
                    f"{len(groups)} topic groups. Original ranks and scores are preserved."
                )
                for group in groups:
                    title = f"Cluster {group.cluster_id + 1}: {group.label} ({len(group.results)} results)"
                    with st.expander(title, expanded=True):
                        st.caption("Keywords: " + ", ".join(group.keywords))
                        for result in group.results:
                            doc = docs.get(result.doc_id, {})
                            st.markdown(f"#### {result.rank}. `{result.doc_id}`")
                            st.caption(f"{result.method} score: {result.score:.4f}")
                            st.text_area(
                                "Original document",
                                value=doc.get("body", ""),
                                height=320,
                                disabled=True,
                                key=f"cluster-document-{group.cluster_id}-{result.rank}-{result.doc_id}",
                            )
            else:
                for result in results:
                    doc = docs.get(result.doc_id, {})
                    st.markdown(f"#### {result.rank}. `{result.doc_id}`")
                    st.caption(f"{result.method} score: {result.score:.4f}")
                    st.text_area(
                        "Original document",
                        value=doc.get("body", ""),
                        height=320,
                        disabled=True,
                        key=f"search-document-{result.rank}-{result.doc_id}",
                    )
            if include_crawled and crawled is not None:
                crawled_results, crawled_latency = crawled.timed_search(query, top_k=min(top_k, 10))
                st.divider()
                st.subheader("Crawled Source Results")
                st.caption(
                    f"Independent offline TF-IDF search over {crawled.document_count} cached documents "
                    f"({crawled_latency:.2f} ms). These results are not part of the official ranking."
                )
                if not crawled_results:
                    st.info("No crawled documents matched this query.")
                for crawled_result in crawled_results:
                    st.markdown(f"#### {crawled_result.rank}. [Crawled Source] `{crawled_result.nct_id}`")
                    st.markdown(f"**{crawled_result.title}**")
                    st.caption(
                        f"TF-IDF score: {crawled_result.score:.4f} | "
                        f"Condition: {crawled_result.condition or 'Not specified'}"
                    )
                    st.write(crawled_result.snippet)
                    st.link_button(
                        "Open on ClinicalTrials.gov",
                        crawled_result.url,
                        key=f"crawled-link-{crawled_result.nct_id}",
                    )
            export_rows = SearchExportService.build_rows(
                query=query,
                official_results=results,
                official_documents=docs,
                crawled_results=crawled_results,
                clustering=clustering,
                clustering_enabled=use_clustering,
                query_refinement=use_refinement,
                top_k=top_k,
                bm25_k1=k1,
                bm25_b=b,
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state["last_search_export"] = {
                "data": SearchExportService.to_csv_bytes(export_rows),
                "filename": f"search_results_{timestamp}.csv",
                "query": query,
                "official_count": len(results),
                "crawled_count": len(crawled_results),
            }
            st.session_state["query_history"] = QueryHistoryService.add(
                st.session_state.get("query_history", []),
                query,
                limit=5,
            )
        except RuntimeError as exc:
            st.error(str(exc))

    render_query_history(query_history_placeholder)

    last_export = st.session_state.get("last_search_export")
    if last_export:
        st.divider()
        st.download_button(
            "Download Search Results as CSV",
            data=last_export["data"],
            file_name=last_export["filename"],
            mime="text/csv; charset=utf-8",
        )
        st.caption(
            f"Last search: {last_export['query']} | "
            f"Official: {last_export['official_count']} | Crawled: {last_export['crawled_count']}"
        )

with rag_tab:
    question = st.text_input("Ask a question", value="Which clinical trials discuss lung cancer and EGFR?")
    if st.button("Generate grounded answer"):
        try:
            if method == "vector_store" and vector_store is not None:
                rag_results = vector_store.search(question, top_k=min(top_k, 8), refine=use_refinement)
                answer = rag.answer_from_results(question, rag_results)
            else:
                answer = rag.answer(question, method=method, top_k=min(top_k, 8), refine=use_refinement)
            st.markdown(answer.answer)
            st.subheader("Sources")
            for source in answer.sources:
                st.markdown(f"**{source['rank']}. `{source['doc_id']}`** score `{source['score']}`")
                st.write(source["snippet"])
        except RuntimeError as exc:
            st.error(str(exc))

with metrics_tab:
    st.header("Feature Comparison: Before & After")
    comparison_path = ARTIFACTS_DIR / "feature_comparison.csv"
    if comparison_path.exists():
        st.dataframe(
            pd.read_csv(comparison_path),
            width="stretch",
            hide_index=True,
        )

    rag_metrics_path = ARTIFACTS_DIR / "rag_evaluation_metrics.csv"
    clustering_metrics_path = ARTIFACTS_DIR / "clustering_evaluation_metrics.csv"
    crawling_metrics_path = ARTIFACTS_DIR / "crawling_evaluation_metrics.csv"
    vector_metrics_path = ARTIFACTS_DIR / "vector_store_evaluation_metrics.csv"
    if all(
        path.exists()
        for path in [rag_metrics_path, clustering_metrics_path, crawling_metrics_path, vector_metrics_path]
    ):
        rag_metrics = pd.read_csv(rag_metrics_path).iloc[0]
        clustering_metrics = pd.read_csv(clustering_metrics_path).iloc[0]
        crawling_metrics = pd.read_csv(crawling_metrics_path).iloc[0]
        vector_metrics = pd.read_csv(vector_metrics_path)
        brute_metrics = vector_metrics.loc[vector_metrics["method"] == "LSA Brute Force"].iloc[0]
        faiss_metrics = vector_metrics.loc[vector_metrics["method"] == "FAISS Vector Store"].iloc[0]
        speedup = (1.0 - faiss_metrics["average_latency_ms"] / brute_metrics["average_latency_ms"]) * 100.0

        rag_col, cluster_col, crawl_col, vector_col = st.columns(4)
        rag_col.metric("RAG Groundedness", f"{rag_metrics['groundedness']:.0%}")
        cluster_col.metric("Clustering Silhouette", f"{clustering_metrics['silhouette_cosine']:.4f}")
        crawl_col.metric(
            "Crawling Success",
            f"{int(crawling_metrics['successful_documents'])}/{int(crawling_metrics['documents_requested'])}",
        )
        vector_col.metric(
            "FAISS Speed-up",
            f"{speedup:.1f}%",
            delta=f"-{brute_metrics['average_latency_ms'] - faiss_metrics['average_latency_ms']:.2f} ms",
            delta_color="inverse",
        )

        rag_detail, cluster_detail, crawl_detail, vector_detail = st.columns(4)
        rag_detail.metric("Citation Coverage", f"{rag_metrics['citation_coverage']:.0%}")
        cluster_detail.metric("Davies-Bouldin", f"{clustering_metrics['davies_bouldin']:.4f}")
        crawl_detail.metric("Offline Search Latency", f"{crawling_metrics['average_search_latency_ms']:.2f} ms")
        vector_detail.metric("Top-10 Overlap", f"{faiss_metrics['average_top10_overlap_with_lsa']:.0%}")

    st.caption(
        "Each feature is evaluated with metrics appropriate to its purpose. "
        "Clustering does not change official ranking, and crawled documents do not have qrels."
    )
    st.divider()
    st.header("Detailed Evaluation Artifacts")
    chart_path = Path("reports") / "figures" / "evaluation_metrics.png"
    metric_files = [
        ("Base retrieval evaluation", ARTIFACTS_DIR / "evaluation_metrics.csv"),
        ("Query refinement evaluation", ARTIFACTS_DIR / "evaluation_metrics_refined.csv"),
        ("BERT reranking evaluation", ARTIFACTS_DIR / "evaluation_metrics_bert.csv"),
        ("RAG quality evaluation", ARTIFACTS_DIR / "rag_evaluation_metrics.csv"),
        ("Document clustering evaluation", ARTIFACTS_DIR / "clustering_evaluation_metrics.csv"),
        ("Web crawling evaluation", ARTIFACTS_DIR / "crawling_evaluation_metrics.csv"),
        ("Vector store retrieval evaluation", ARTIFACTS_DIR / "vector_store_evaluation_metrics.csv"),
    ]
    for title, metrics_path in metric_files:
        if metrics_path.exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(metrics_path), width="stretch")
    if chart_path.exists():
        st.image(str(chart_path))
    clustering_charts = [
        Path("reports") / "figures" / "clustering_distribution.png",
        Path("reports") / "figures" / "clustering_scatter.png",
    ]
    if all(path.exists() for path in clustering_charts):
        st.subheader("Document clustering charts")
        left, right = st.columns(2)
        left.image(str(clustering_charts[0]), caption="Cluster document distribution")
        right.image(str(clustering_charts[1]), caption="PCA visualization of document clusters")
    crawling_chart = Path("reports") / "figures" / "crawling_evaluation.png"
    if crawling_chart.exists():
        st.subheader("Offline web crawling chart")
        st.image(str(crawling_chart), caption="Collection success and duplicate removal")
    vector_chart = Path("reports") / "figures" / "vector_store_evaluation.png"
    if vector_chart.exists():
        st.subheader("Local vector store comparison")
        st.image(str(vector_chart), caption="LSA brute force compared with FAISS Vector Store")
