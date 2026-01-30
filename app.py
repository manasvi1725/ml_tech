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


@app.post("/internal/refresh-all")
def refresh_all(token: str = Header(None)):
    if token != os.getenv("ML_INTERNAL_TOKEN"):
        raise HTTPException(status_code=401, detail="Unauthorized")

    subprocess.run(["python", "refresh_all.py"], check=True)
    return {"status": "ok"}




def run_script(script_name: str):
    result = subprocess.run(
        ["python", script_name],
        capture_output=True,
        text=True,
        check=True
    )
    return json.loads(result.stdout)

@app.post("/internal/run-global")
def run_global(authorization: str = Header(None)):
    if authorization != f"Bearer {ML_TOKEN}":
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        trends = run_script("run_global_trends.py")
        patents = run_script("run_global_patents.py")
        investments = run_script("run_global_investments.py")

        return {
            "global": {
                "trends": trends,
                "patents": patents,
                "investments": investments,
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
