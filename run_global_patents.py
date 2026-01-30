# ================== CONFIG ==================
import os
import json
import requests
from datetime import datetime
from serpapi import GoogleSearch


SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

if not SERPAPI_KEY:
    raise RuntimeError("SERPAPI_API_KEY missing")
TECH_FIELDS = {
    "artificial_intelligence": [
        "artificial intelligence",
        "machine learning",
        "deep learning",
        "generative ai"
    ],
    "semiconductors": [
        "semiconductor",
        "chip manufacturing",
        "integrated circuits"
    ],
    "robotics": [
        "robotics",
        "autonomous systems"
    ],
    "space": [
        "space technology",
        "satellite",
        "launch vehicle"
    ],
    "defence": [
        "defence technology",
        "military technology"
    ]
}

# ================== SERPAPI WRAPPER ==================
def serpapi_search(params):
    params["api_key"] = SERPAPI_KEY
    return GoogleSearch(params).get_dict()

# ================== HELPERS ==================
def extract_patent_pdf(patent):
    """
    Extract PDF link if available (often missing).
    Always safe to return None.
    """
    for r in (patent.get("resources") or []):
        if r.get("file_format", "").lower() == "pdf":
            return r.get("link")
    return None



# ================== FETCH: GLOBAL PATENTS ==================
def fetch_global_patents(num_per_field=10):
    records = []

    for field, keywords in TECH_FIELDS.items():
        query = " OR ".join(keywords)

        params = {
            "engine": "google_patents",
            "q": query,
            "num": num_per_field,
            "sort": "new"
        }

        try:
            results = serpapi_search(params)
            patents = results.get("organic_results", []) or []

            for p in patents:
                link = p.get("patent_link")

                # Drop patents without a canonical link
                if not link:
                    continue

                year = None
                if p.get("publication_date"):
                    year = p["publication_date"][:4]

                records.append({
                    "title": p.get("title"),
                    "field": field,
                    "signal_type": "patent",
                    "year": year,
                    "assignee": p.get("assignee"),
                    "link": link,              # always present
                    "pdf_link": extract_patent_pdf(p),  # optional
                })


        except Exception as e:
            print(f"SerpAPI error for field '{field}':", e,file=sys.stderr)

    return records

# ================== PIPELINE ==================
def run_global_patent_pipeline():
    print("Fetching global patents...")

    patents = fetch_global_patents()

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "counts": len(patents),
        "signals": patents
    }

# ================== EXPORT ==================
def export_global_patents_json():
    result = run_global_patent_pipeline()
    
    print(json.dumps(result, ensure_ascii=False))
    print("global patents dumped from ML -> backend", file=sys.stderr)

# ================== ENTRY POINT ==================
if __name__ == "__main__":
    export_global_patents_json()
