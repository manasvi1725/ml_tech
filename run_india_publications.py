import os
import json
import requests
from collections import defaultdict

# ================== CONFIG ==================

CROSSREF_URL = "https://api.crossref.org/works"
OUTPUT_DIR = "data/india"

FIELDS = {
    "artificial_intelligence": [
        "artificial intelligence", "machine learning",
        "deep learning", "computer vision"
    ],
    "semiconductors": [
        "semiconductor", "vlsi", "integrated circuits",
        "chip fabrication"
    ],
    "cybersecurity": [
        "cybersecurity", "network security",
        "cryptography", "malware"
    ],
    "space_technology": [
        "space technology", "satellite",
        "launch vehicle", "indian space research organisation"
    ]
}

# Extended full-name institute list
INDIA_KEYWORDS = [
    # Country / nationality
    "india",
    "indian",

    # Major academic institutions
    "indian institute of technology",
    "indian institute of information technology",
    "indian institute of science",
    "indian statistical institute",
    "jawaharlal nehru university",
    "delhi technological university",
    "anna university",
    "jadavpur university",
    "banaras hindu university",

    # Research councils & labs
    "council of scientific and industrial research",
    "defence research and development organisation",
    "indian council of medical research",
    "indian council of agricultural research",

    # Space / atomic / strategic bodies
    "indian space research organisation",
    "department of atomic energy",
    "bhaba atomic research centre",
    "vikram sarabhai space centre",
    "satish dhawan space centre",

    # Medical & health institutes
    "all india institute of medical sciences",
    "national institute of mental health and neurosciences",

    # Government & public sector
    "ministry of electronics and information technology",
    "ministry of science and technology",
    "government of india"
]


# ================== HELPERS ==================

def find_india_matches(text):
    t = str(text).lower()
    return [k for k in INDIA_KEYWORDS if k in t]

def extract_year(item):
    for key in ("published-print", "published-online", "issued"):
        if key in item and "date-parts" in item[key]:
            return item[key]["date-parts"][0][0]
    return None

def build_trend(items):
    counter = defaultdict(int)
    for it in items:
        if it.get("year"):
            counter[int(it["year"])] += 1
    return [
        {"year": y, "count": c}
        for y, c in sorted(counter.items())
    ]

# ================== CROSSREF ==================

def fetch_crossref(query, rows=100):
    params = {
        "query": query,
        "filter": "type:journal-article",
        "rows": rows,
        "mailto": "techintel@example.com"
    }
    r = requests.get(CROSSREF_URL, params=params, timeout=20)
    r.raise_for_status()
    return r.json()["message"]["items"]

# ================== FIELD FETCH ==================

def fetch_field_publications(field, keywords):
    query = " OR ".join(keywords)
    items = fetch_crossref(query)

    records = []
    for item in items:
        title = " ".join(item.get("title", []))
        abstract = item.get("abstract", "")
        authors = item.get("author", [])

        matched_institutes = []

        # Check title & abstract
        matched_institutes.extend(find_india_matches(title))
        matched_institutes.extend(find_india_matches(abstract))

        # Check author affiliations
        for a in authors:
            for aff in (a.get("affiliation") or []):
                matched_institutes.extend(
                    find_india_matches(aff.get("name", ""))
                )

        if not matched_institutes:
            continue

        doi = item.get("DOI")

        records.append({
            "title": title,
            "year": extract_year(item),
            "doi": doi,
            "citations": item.get("is-referenced-by-count", 0),
            "link": f"https://doi.org/{doi}" if doi else None,
            "field": field,
            "source": "Crossref",
            "matched_institute": matched_institutes[0]  # 👈 simple version
        })

    return records

# ================== EXPORT ==================

def export_india_fields_json():

    output = {
        "country": "India",
        "fields": {}
    }

    for field, keywords in FIELDS.items():
        publications = fetch_field_publications(field, keywords)

        output["fields"][field] = {
            "publications": publications,
            "trends": {
                "publications_year": build_trend(publications)
            }
        }

    
    print(json.dumps(output, indent=2, ensure_ascii=False))


# ================== ENTRY ==================

if __name__ == "__main__":
    export_india_fields_json()
