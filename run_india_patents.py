import os
import json
import re
from serpapi import GoogleSearch
from datetime import datetime


SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

if not SERPAPI_KEY:
    raise RuntimeError("SERPAPI_API_KEY missing")
OUTPUT_DIR = "data/india"
MAX_RESULTS_PER_INSTITUTE = 50

# ================= INDIAN INSTITUTES (FULL NAMES ONLY) =================

INDIAN_INSTITUTES = [
    "Indian Institute of Technology Delhi",
    "Indian Institute of Technology Bombay",
    "Indian Institute of Technology Madras",
    "Indian Institute of Technology Kanpur",
    "Indian Institute of Technology Kharagpur",
    "Indian Institute of Technology Roorkee",
    "Indian Institute of Science Bangalore",
    "Council of Scientific and Industrial Research",
    "Defence Research and Development Organisation",
    "Indian Space Research Organisation",
    "Bhabha Atomic Research Centre",
    "All India Institute of Medical Sciences"
]

# ================= HELPERS =================

def extract_year(*texts):
    for t in texts:
        if t:
            m = re.search(r"(19|20)\d{2}", str(t))
            if m:
                return int(m.group())
    return None


def extract_country(patent_id):
    if not patent_id:
        return None
    m = re.search(r"patent/([A-Z]{2})", patent_id)
    return m.group(1) if m else None


def build_institute_query(institute):
    return f'assignee:"{institute}"'


# ================= SERPAPI =================

def fetch_patents_for_institute(institute):
    params = {
        "engine": "google_patents",
        "q": build_institute_query(institute),
        "num": MAX_RESULTS_PER_INSTITUTE,
        "api_key": SERPAPI_KEY
    }

    res = GoogleSearch(params).get_dict()
    return res.get("organic_results", [])


# ================= MAIN PIPELINE =================

def export_patents_by_institute():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_patents = []
    by_institute = {}

    for institute in INDIAN_INSTITUTES:
        print(f"🏛️ Fetching patents for: {institute}")
        results = fetch_patents_for_institute(institute)

        records = []
        for r in results:
            pid = r.get("patent_id")

            records.append({
                "patent_id": pid,
                "title": r.get("title"),
                "link": r.get("patent_link"),
                "year": extract_year(
                    r.get("publication_date"),
                    r.get("filing_date"),
                    r.get("priority_date")
                ),
                "assignee": r.get("assignee"),
                "country_code": extract_country(pid),
                "institute": institute
            })

        print(f"   → {len(records)} patents collected")

        by_institute[institute] = records
        all_patents.extend(records)

    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_patents": len(all_patents),
        "institutes": by_institute,
        "patents": all_patents
    }

    
    print(json.dumps(output, ensure_ascii=False)))
    print("indian patents dumped from ML -> backend")

    print("📦 Total patents:", len(all_patents))


# ================= ENTRY =================

if __name__ == "__main__":
    export_patents_by_institute()
