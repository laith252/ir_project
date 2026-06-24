import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "crawled"
API_URL = "https://clinicaltrials.gov/api/v2/studies"
DEFAULT_QUERIES = ["cancer", "diabetes", "cardiovascular disease", "depression", "surgery"]


def fetch_studies(query: str, page_size: int, timeout: int) -> tuple[str, dict]:
    params = urlencode({"format": "json", "pageSize": page_size, "query.cond": query})
    url = f"{API_URL}?{params}"
    request = Request(
        url,
        headers={
            "User-Agent": "IR-Project-2026-Offline-Crawler/1.0",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        return url, json.loads(response.read().decode("utf-8"))


def clean_study(study: dict, crawl_query: str) -> dict | None:
    protocol = study.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})
    description = protocol.get("descriptionModule", {})
    conditions_module = protocol.get("conditionsModule", {})
    nct_id = str(identification.get("nctId", "")).strip()
    title = str(identification.get("briefTitle", "")).strip()
    conditions = [str(item).strip() for item in conditions_module.get("conditions", []) if str(item).strip()]
    summary = str(description.get("briefSummary") or description.get("detailedDescription") or "").strip()
    if not nct_id or not title or not summary:
        return None
    condition = "; ".join(conditions)
    body = f"{title}. Conditions: {condition}. {summary}".strip()
    return {
        "nct_id": nct_id,
        "title": title,
        "url": f"https://clinicaltrials.gov/study/{nct_id}",
        "condition": condition,
        "summary": summary,
        "source": "ClinicalTrials.gov",
        "crawl_query": crawl_query,
        "body": body,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and cache ClinicalTrials.gov studies for offline search.")
    parser.add_argument("--per-query", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    args = parser.parse_args()
    if args.per_query < 1 or args.per_query > 100:
        parser.error("--per-query must be between 1 and 100")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    raw_requests = []
    clean_documents = []
    seen_ids = set()
    duplicate_count = 0
    invalid_count = 0
    failed_requests = 0

    for query in args.queries:
        print(f"Crawling condition: {query}", flush=True)
        try:
            request_url, payload = fetch_studies(query, args.per_query, args.timeout)
            studies = payload.get("studies", [])
            raw_requests.append(
                {
                    "query": query,
                    "request_url": request_url,
                    "status": "success",
                    "received": len(studies),
                    "response": payload,
                }
            )
            for study in studies:
                cleaned = clean_study(study, query)
                if cleaned is None:
                    invalid_count += 1
                    continue
                if cleaned["nct_id"] in seen_ids:
                    duplicate_count += 1
                    continue
                seen_ids.add(cleaned["nct_id"])
                clean_documents.append(cleaned)
        except Exception as exc:
            failed_requests += 1
            raw_requests.append(
                {
                    "query": query,
                    "request_url": f"{API_URL}?query.cond={query}",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    requested = len(args.queries) * args.per_query
    duration = time.perf_counter() - started
    raw_payload = {
        "crawl_started_at_utc": started_at,
        "source": "ClinicalTrials.gov API v2",
        "api_url": API_URL,
        "requests": raw_requests,
    }
    raw_path = OUTPUT_DIR / "crawled_trials_raw.json"
    raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    clean_path = OUTPUT_DIR / "crawled_trials_clean.csv"
    fieldnames = ["nct_id", "title", "url", "condition", "summary", "source", "crawl_query", "body"]
    with clean_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_documents)

    word_counts = [len(document["body"].split()) for document in clean_documents]
    metadata = {
        "crawl_mode": "offline_cached",
        "source": "ClinicalTrials.gov API v2",
        "source_url": API_URL,
        "crawl_started_at_utc": started_at,
        "queries": list(args.queries),
        "documents_requested": requested,
        "documents_received": sum(item.get("received", 0) for item in raw_requests),
        "successful_documents": len(clean_documents),
        "failed_documents": max(requested - len(clean_documents), 0),
        "duplicate_documents_removed": duplicate_count,
        "invalid_documents_removed": invalid_count,
        "unique_documents": len(clean_documents),
        "successful_requests": len(args.queries) - failed_requests,
        "failed_requests": failed_requests,
        "average_document_length_words": (sum(word_counts) / len(word_counts)) if word_counts else 0.0,
        "crawl_duration_seconds": duration,
        "raw_file": str(raw_path.relative_to(ROOT)),
        "clean_file": str(clean_path.relative_to(ROOT)),
    }
    metadata_path = OUTPUT_DIR / "crawling_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved raw responses to {raw_path}")
    print(f"Saved {len(clean_documents)} unique searchable documents to {clean_path}")
    print(f"Saved crawl metadata to {metadata_path}")
    print(f"Duplicates removed: {duplicate_count}; invalid: {invalid_count}; failed requests: {failed_requests}")


if __name__ == "__main__":
    main()
