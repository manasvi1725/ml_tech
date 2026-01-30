import os
import requests
import subprocess

BACKEND_API = os.getenv("MONGODB_URI")
ML_TOKEN = os.getenv("ML_INTERNAL_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {ML_TOKEN}"
}

def get_all_technologies():
    resp = requests.get(
        f"{BACKEND_API}/internal/technologies",
        headers=HEADERS,
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()["technologies"]

def refresh_technology(tech):
    print(f"🔄 Refreshing {tech}", flush=True)
    subprocess.run(
        ["python", "run_pipeline.py", tech],
        check=True
    )

def main():
    print("🌙 Daily refresh started", flush=True)

    techs = get_all_technologies()

    for tech in techs:
        try:
            refresh_technology(tech)
        except Exception as e:
            print(f"❌ Failed {tech}: {e}", flush=True)

    print("Daily refresh completed", flush=True)

if __name__ == "__main__":
    main()
