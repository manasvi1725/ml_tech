print("🔥🔥🔥 ML APP STARTED - VERSION 2026-01-26 🔥🔥🔥", flush=True)

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os
import subprocess
import json
import sys

app = FastAPI()

INTERNAL_TOKEN = os.getenv("ML_INTERNAL_TOKEN")


class RunRequest(BaseModel):
    tech: str


@app.get("/health")
def health():
    return {"ok": True, "msg": "ML service running ✅"}


@app.post("/run")
def run_pipeline(payload: RunRequest, x_internal_token: str = Header(default="")):
    if INTERNAL_TOKEN and x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # 🚨 DO NOT re-normalize — backend already did
    tech = payload.tech

    try:
        # 1️⃣ run pipeline
        result = subprocess.run(
            ["python", "run_pipeline.py", tech],
            capture_output=True,
            text=True,
        )

        # 2️⃣ DEBUG OUTPUT
        print("STDOUT:", result.stdout, file=sys.stderr)
        print("STDERR:", result.stderr, file=sys.stderr)
        print("RETURN CODE:", result.returncode, file=sys.stderr)

        if result.returncode != 0:
            raise RuntimeError("Pipeline failed")

        # 3️⃣ parse JSON from stdout
        ml_output = json.loads(result.stdout)

        # 4️⃣ RETURN RESULT — DO NOT STORE
        return {
            "ok": True,
            "tech": tech,
            "data": ml_output,
        }

    except Exception as e:
        print(f"[ML ERROR] {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))





def run_script(script_name: str):
    script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        script_name
    )

    result = subprocess.run(
        ["python", script_path],
        capture_output=True,
        text=True,
    )

    print("🧪 SCRIPT:", script_name, file=sys.stderr)
    print("🧪 RETURN CODE:", result.returncode, file=sys.stderr)
    print("🧪 STDOUT LEN:", len(result.stdout), file=sys.stderr)
    print("🧪 STDERR:", result.stderr[:500], file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError("Script crashed")

    if not result.stdout.strip():
        raise RuntimeError("Script produced NO stdout")

    return json.loads(result.stdout)


@app.post("/internal/run-global-investments")
def run_global_investments(authorization: str = Header(None)):
    if authorization != f"Bearer {INTERNAL_TOKEN}":
        raise HTTPException(status_code=403)

    data = run_script("run_global_investments.py")
    return { "investments": data }

@app.post("/internal/run-global-patents")
def run_global_patents(authorization: str = Header(None)):
    if authorization != f"Bearer {INTERNAL_TOKEN}":
        raise HTTPException(status_code=403)

    data = run_script("run_global_patents.py")
    return { "patents": data }

@app.post("/internal/run-global-trends")
def run_global_trends(authorization: str = Header(None)):
    if authorization != f"Bearer {INTERNAL_TOKEN}":
        raise HTTPException(status_code=403)

    data = run_script("run_global_trends.py")
    return { "trends": data }


@app.post("/internal/run-india")
def run_india(authorization: str = Header(None)):
    if authorization != f"Bearer {INTERNAL_TOKEN}":
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        patents = run_script("run_india_patents.py")
        publications = run_script("run_india_publications.py")

        return {
            "india": {
                "patents": patents,
                "publications": publications,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
