# =================== CONFIG ===================
import os
import re
import json
import pandas as pd
from serpapi import GoogleSearch

SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

if not SERPAPI_KEY:
    raise RuntimeError("SERPAPI_API_KEY missing")



# ================== SERPAPI ==================
def serpapi_search(params):
    params["api_key"] = SERPAPI_KEY
    search = GoogleSearch(params)
    return search.get_dict()


# ================== COUNTRY + TECH CONFIG ==================
COUNTRIES = {
    "United Kingdom": {"key": "uk", "gl": "gb"},
    "India": {"key": "india", "gl": "in"},
    "United States": {"key": "usa", "gl": "us"},
    "Japan": {"key": "japan", "gl": "jp"},
    "Germany": {"key": "germany", "gl": "de"},
}

TECH_QUERIES = {
    "ai": "AI funding",
    "semiconductors": "semiconductor funding",
    "biotech": "biotech funding",
    "clean_energy": "clean energy funding",
    "quantum": "quantum computing funding",
}


# ================== CONFIDENCE WEIGHTING ==================
def article_confidence_weight(article):
    text = " ".join(
        str(article.get(k, "")).lower()
        for k in ["title", "snippet"]
    )

    weight = 1.0  # baseline

    if any(k in text for k in ["funding", "invests", "investment", "backs"]):
        weight += 0.5

    if "series a" in text:
        weight += 0.8
    elif "series b" in text or "series c" in text:
        weight += 1.2

    if any(k in text for k in ["acquires", "acquisition", "merger"]):
        weight += 1.0

    if re.search(r"\$|\bmillion\b|\bbillion\b", text):
        weight += 1.5

    return weight


# ================== FETCHERS ==================
def fetch_country_investment_news(country_name, country_cfg):
    gl = country_cfg["gl"]

    country_block = {
        "total_signals": 0,
        "total_score": 0.0,
        "technologies": {}
    }

    print(f"\n📍 Fetching investment news for {country_name}")

    for tech, base_query in TECH_QUERIES.items():
        params = {
            "engine": "google_news",
            "q": base_query,
            "gl": gl,
            "hl": "en",
            "num": 25,
            "when": "6m"
        }

        try:
            results = serpapi_search(params)
            articles = results.get("news_results", []) or []

            tech_score = 0.0

            processed_articles = []
            for a in articles:
                w = article_confidence_weight(a)
                tech_score += w

                processed_articles.append({
                    "title": a.get("title"),
                    "source": (a.get("source") or {}).get("name"),
                    "link": a.get("link"),
                    "date": a.get("date"),
                    "confidence_weight": round(w, 2),
                })

            country_block["technologies"][tech] = {
                "signal_count": len(articles),
                "signal_score": round(tech_score, 2),
                "investment_percent": 0.0,  # filled later
                "articles": processed_articles,
            }

            country_block["total_signals"] += len(articles)
            country_block["total_score"] += tech_score

            print(f"  • {tech}: {len(articles)} articles | score {tech_score:.2f}")

        except Exception as e:
            print(f"  ⚠ Failed for {tech}: {e}")
            country_block["technologies"][tech] = {
                "signal_count": 0,
                "signal_score": 0.0,
                "investment_percent": 0.0,
                "articles": [],
            }

    return country_block


# ================== NORMALIZATION ==================
def normalize_country_investments(country_block):
    total_score = country_block["total_score"]
    if total_score == 0:
        return

    for tech in country_block["technologies"].values():
        tech["investment_percent"] = round(
            (tech["signal_score"] / total_score) * 100, 2
        )


# ================== PIPELINE ==================
def run_global_pulse():
    print("🚀 Running Global Investment Pulse pipeline...")

    output = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "countries": {}
    }

    for country_name, cfg in COUNTRIES.items():
        country_data = fetch_country_investment_news(country_name, cfg)
        normalize_country_investments(country_data)
        output["countries"][cfg["key"]] = country_data

    return output


# ================== EXPORT ==================
def export_global_pulse_json():
    result = run_global_pulse()
    
    json.dump(result, f, indent=2, ensure_ascii=False)
    print("json transferred from ML to Backend")
    


# ================== ENTRY ==================
if __name__ == "__main__":
    export_global_pulse_json()
