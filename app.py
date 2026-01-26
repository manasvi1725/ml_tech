from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os
import subprocess
import json
from pymongo import MongoClient

app = FastAPI()

INTERNAL_TOKEN = os.getenv("ML_INTERNAL_TOKEN")

MONGODB_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGODB_DB", "techintel")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "technology")


class RunRequest(BaseModel):
    tech: str


def get_collection():
    if not MONGODB_URI:
        raise RuntimeError("MONGODB_URI missing in ML service")

    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    return db[COLLECTION_NAME]


@app.get("/health")
def health():
    return {"ok": True, "msg": "ML service running ✅"}


@app.post("/run")
def run_pipeline(
    payload: RunRequest,
    x_internal_token: str = Header(default="")
):
    # 🔐 protect service
    if INTERNAL_TOKEN and x_internal_token != INTERNAL_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    tech = payload.tech.strip().lower().replace(" ", "_")

    try:
        # 1️⃣ mark ML as running
        collection = get_collection()
        collection.update_one(
            {"name": tech},
            {"$set": {"ml_status": "running"}},
            upsert=True,
        )

        # 2️⃣ run pipeline and CAPTURE output
        result = subprocess.run(
    ["python", "run_pipeline.py", tech],
    capture_output=True,
    text=True,
)

print("STDOUT:", result.stdout, file=sys.stderr)
print("STDERR:", result.stderr, file=sys.stderr)
print("RETURN CODE:", result.returncode, file=sys.stderr)

if result.returncode != 0:
    raise RuntimeError("Pipeline failed")


        # 3️⃣ parse JSON output
        ml_output = json.loads(result.stdout)

        # 4️⃣ store result in MongoDB
        collection.update_one(
            {"name": tech},
            {
                "$set": {
                    "latest_json": ml_output,
                    "ml_status": "done",
                }
            },
            upsert=True,
        )

        return {
            "ok": True,
            "tech": tech,
            "status": "stored",
        }

    except Exception as e:
        # mark ML as failed
        collection.update_one(
            {"name": tech},
            {"$set": {"ml_status": "failed", "error": str(e)}},
            upsert=True,
        )
        raise HTTPException(status_code=500, detail=str(e))
