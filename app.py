from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os
import subprocess

app = FastAPI()

INTERNAL_TOKEN = os.getenv("ML_INTERNAL_TOKEN")


class RunRequest(BaseModel):
    tech: str


@app.get("/health")
def health():
    return {"ok": True, "msg": "ML service running ✅"}


@app.post("/run")
def run_pipeline(payload: RunRequest, x_internal_token: str = Header(default="")):
    # ✅ protect service
    if INTERNAL_TOKEN and x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    tech = payload.tech.strip().lower().replace(" ", "_")

    # ✅ run your pipeline
    subprocess.run(["python", "run_pipeline.py", tech], check=True)

    return {"ok": True, "tech": tech}
