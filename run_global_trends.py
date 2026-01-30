# ================== CONFIG ==================
import os
import json
from datetime import datetime
from serpapi import GoogleSearch


SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

if not SERPAPI_KEY:
    raise RuntimeError("SERPAPI_API_KEY missing")
TECH_FIELDS = {
    "artificial_intelligence": [
        "artificial intelligence",
        "machine learning",
        "generative ai"
    ],
    "semiconductors": [
        "semiconductor",
        "chip manufacturing"
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

# ================== FETCH: GLOBAL NEWS ==================
def fetch_global_trends(num_per_field=10, days=7):
    """
    Fetch recent technology news grouped by field.
    Drops articles without title or link.
    """
    records = []

    for field, keywords in TECH_FIELDS.items():
        query = " OR ".join(keywords)

        params = {
            "engine": "google_news",
            "q": query,
            "num": num_per_field,
            "when": f"{days}d",
            "hl": "en",
            "gl": "us"
        }

        try:
            results = serpapi_search(params)
            news = results.get("news_results", []) or []

            for n in news:
                title = n.get("title")
                link = n.get("link")

                #Drop useless entries
                if not title or not link:
                    continue

                records.append({
                    "title": title,
                    "field": field,
                    "signal_type": "news",
                    "source": (n.get("source") or {}).get("name"),
                    "date": n.get("date"),
                    "snippet": n.get("snippet"),
                    "link": link
                })

        except Exception as e:
            print(f"News fetch error for '{field}':", e,file=sys.stderr)

    return records

# ================== PIPELINE ==================
def run_global_trends_pipeline():
    print("Fetching global technology trends...",file=sys.stderr)

    trends = fetch_global_trends()

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "counts": len(trends),
        "signals": trends
    }

# ================== EXPORT ==================
def export_global_trends_json():
    result = run_global_trends_pipeline()

    
    print(json.dumps(result, ensure_ascii=False))
    print("global trends dumped from ML -> backend",file=sys.stderr)

# ================== ENTRY POINT ==================
if __name__ == "__main__":
    export_global_trends_json()
