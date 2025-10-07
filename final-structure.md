Perfect! Let’s create a **complete ready-to-run modular folder structure** for your in-house chatbot with **FastAPI backend + Gradio frontend + scheduler + scripts + MongoDB integration**. I’ll provide all the files and explain how they work together.

---

# **1️⃣ Folder Structure**

```
supportbot/
│
├── backend/
│   ├── api.py              # FastAPI endpoints (query, feedback, admin)
│   ├── scheduler.py        # APScheduler daily jobs (tag enrichment + evaluation)
│   └── main.py             # Start FastAPI + Scheduler
│
├── frontend/
│   └── app.py              # Gradio UI calling backend APIs
│
├── core/
│   ├── config.py           # Config & feature flags
│   ├── mongo_client.py     # MongoDB connection manager
│   ├── db.py               # Collections shortcut
│   ├── search_engine.py    # Hybrid search (BM25 + optional embeddings)
│   ├── feedback_handler.py # Feedback storage + ranking adjustments
│   ├── tag_enricher.py     # Tag enrichment logic
│   ├── evaluator.py        # Feedback evaluation + reporting
│   ├── paraphraser.py      # Query paraphrasing
│   └── synonym_map.py      # Domain-specific synonym mapping
│
├── scripts/
│   ├── load_pages_to_db.py    # Load Confluence JSON pages to MongoDB
│   ├── run_tag_enrichment.py  # Run tag enrichment manually
│   └── run_feedback_update.py # Evaluate feedback manually
│
├── logs/
│   └── scheduler.log       # Scheduler logs
│
└── data/
    ├── synonym_map.json    # Optional external synonyms
    └── config.yaml         # Optional global configuration
```

---

# **2️⃣ Core Modules**

### **core/config.py**

```python
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "supportbot")

# Feature flags
USE_EMBEDDINGS = False
USE_PARAPHRASING = True
USE_TAG_FILTER = True

# Hybrid search weights
WEIGHTS = {
    "bm25": 0.6,
    "embedding": 0.4
}
```

---

### **core/mongo_client.py**

```python
from pymongo import MongoClient
from core.config import MONGO_URI, DB_NAME

class MongoClientManager:
    def __init__(self, uri=MONGO_URI, db_name=DB_NAME):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]

    def get_collection(self, name):
        return self.db[name]

mongo_manager = MongoClientManager()
```

---

### **core/db.py**

```python
from core.mongo_client import mongo_manager

pages_collection = mongo_manager.get_collection("pages")
feedback_collection = mongo_manager.get_collection("feedback")
evaluation_collection = mongo_manager.get_collection("evaluation_reports")
```

---

### **core/synonym_map.py**

```python
DOMAIN_SYNONYMS = {
    "trade": ["deal", "transaction", "order"],
    "margin": ["collateral", "exposure", "haircut"],
    "batch": ["job", "process", "schedule"],
    "fail": ["error", "exception", "issue"],
    "equity": ["stock", "share", "security"]
}

def apply_synonyms(query: str) -> str:
    for word, syns in DOMAIN_SYNONYMS.items():
        for s in syns:
            if s in query.lower():
                query = query.replace(s, word)
    return query
```

---

### **core/paraphraser.py**

```python
def generate_paraphrases(query: str):
    templates = [
        "how to resolve {}",
        "steps to fix {}",
        "troubleshooting {}",
        "resolution for {}"
    ]
    return [t.format(query) for t in templates]
```

---

### **core/tag_enricher.py**

```python
import re
from core.db import pages_collection

KEYWORD_TAGS = {
    "trade": "TradeOps",
    "margin": "MarginOps",
    "equity": "EquityMarkets",
    "fail": "Incident",
    "job": "BatchProcess",
    "restart": "Recovery"
}

def enrich_tags(page_ids=None):
    query = {} if not page_ids else {"page_id": {"$in": page_ids}}
    docs = pages_collection.find(query)
    for doc in docs:
        tags = set(doc.get("tags", []))
        text = (doc.get("content") or "").lower()
        for k, v in KEYWORD_TAGS.items():
            if re.search(rf"\b{k}\b", text):
                tags.add(v)
        pages_collection.update_one({"_id": doc["_id"]}, {"$set": {"tags": list(tags)}})

```

---

### **core/feedback_handler.py**

```python
from datetime import datetime
from core.db import feedback_collection, pages_collection

def save_feedback(query, page_id, feedback_value):
    feedback_collection.insert_one({
        "query": query,
        "page_id": page_id,
        "feedback": feedback_value,
        "timestamp": datetime.utcnow()
    })
    adjust_ranking_weights(page_id, feedback_value)

def adjust_ranking_weights(page_id, feedback_value):
    doc = pages_collection.find_one({"page_id": page_id})
    current = doc.get("score_boost", 0)
    delta = 0.05 if feedback_value == "good" else -0.05
    pages_collection.update_one({"page_id": page_id}, {"$set": {"score_boost": current + delta}})
```

---

### **core/search_engine.py**

```python
from rank_bm25 import BM25Okapi
from core.db import pages_collection
from core.synonym_map import apply_synonyms
from core.paraphraser import generate_paraphrases
from core.config import USE_PARAPHRASING, WEIGHTS
import numpy as np

# Optional embedding import (if available)
try:
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    model = None

def get_all_pages():
    return list(pages_collection.find({}, {"title":1, "content":1, "tags":1, "url":1, "score_boost":1, "page_id":1}))

def hybrid_search(query, top_k=5):
    query = apply_synonyms(query)
    queries = [query]

    if USE_PARAPHRASING:
        queries.extend(generate_paraphrases(query))

    docs = get_all_pages()
    contents = [doc["content"] for doc in docs]

    # BM25 scoring
    bm25 = BM25Okapi([c.split() for c in contents])
    bm25_scores = np.mean([bm25.get_scores(q.split()) for q in queries], axis=0)

    # Embedding similarity
    if model:
        query_emb = np.mean(model.encode(queries), axis=0)
        doc_embs = model.encode(contents)
        emb_scores = util.cos_sim(query_emb, doc_embs).cpu().numpy()[0]
    else:
        emb_scores = np.zeros(len(contents))

    scores = WEIGHTS["bm25"]*bm25_scores + WEIGHTS["embedding"]*emb_scores

    # Apply score boost from feedback
    for i, doc in enumerate(docs):
        scores[i] += doc.get("score_boost", 0)

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [docs[i] | {"score": float(scores[i])} for i in top_indices]
```

---

### **core/evaluator.py**

```python
from core.db import feedback_collection, evaluation_collection
import pandas as pd
from datetime import datetime

def evaluate_quality(save_to_db=False):
    fb = list(feedback_collection.find())
    if not fb:
        return "No feedback yet."

    df = pd.DataFrame(fb)
    summary = df.groupby("feedback").size().reset_index(name="count")
    total = df.shape[0]
    good_ratio = df[df["feedback"]=="good"].shape[0]/total

    result = {
        "timestamp": datetime.utcnow(),
        "total_feedback": total,
        "good_feedback": int(df[df["feedback"]=="good"].shape[0]),
        "bad_feedback": int(df[df["feedback"]=="bad"].shape[0]),
        "good_ratio": round(good_ratio,3),
        "summary": summary.to_dict(orient="records")
    }

    if save_to_db:
        evaluation_collection.insert_one(result)

    print("Feedback summary:", summary)
    print(f"Good feedback ratio: {good_ratio:.2%}")
    return result
```

---

# **3️⃣ Scripts**

### **scripts/load_pages_to_db.py**

```python
# scripts/load_pages_to_db.py
import json
import sys
from core.mongo_client import mongo_manager

def load_pages(json_file_path, overwrite=False):
    pages_col = mongo_manager.get_collection("pages")
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages = data.get("pages", [])
    print(f"Found {len(pages)} pages in JSON file.")

    if overwrite:
        pages_col.delete_many({})
        print("Existing pages cleared.")

    for page in pages:
        doc = {
            "page_id": page.get("page_id"),
            "title": page.get("title"),
            "content": page.get("content"),
            "url": page.get("url"),
            "space_key": page.get("space_key"),
            "tags": [],
            "score_boost": 0
        }
        pages_col.update_one({"page_id": doc["page_id"]}, {"$set": doc}, upsert=True)

    print(f"Loaded {len(pages)} pages into MongoDB.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_pages_to_db.py <json_file_path> [overwrite]")
        sys.exit(1)

    json_file_path = sys.argv[1]
    overwrite_flag = False

    if len(sys.argv) > 2:
        arg = sys.argv[2].lower()
        if arg in ("true", "1", "yes"):
            overwrite_flag = True

    load_pages(json_file_path, overwrite_flag)

```

### **scripts/run_tag_enrichment.py**

```python
# scripts/run_tag_enrichment.py
import sys
import argparse
from core.tag_enricher import enrich_tags

def main():
    parser = argparse.ArgumentParser(description="Run tag enrichment on pages.")
    parser.add_argument("--pages", nargs="*", help="Optional list of page IDs to enrich. Enrich all if not provided.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs")

    args = parser.parse_args()

    if args.pages:
        if args.verbose:
            print(f"Enriching tags for {len(args.pages)} specified pages...")
        enrich_tags(page_ids=args.pages)
    else:
        if args.verbose:
            print("Enriching tags for all pages...")
        enrich_tags()

    print("Tag enrichment completed.")

if __name__ == "__main__":
    main()

```

### **scripts/run_feedback_update.py**

```python
# scripts/run_feedback_update.py
import argparse
from core.evaluator import evaluate_quality

def main():
    parser = argparse.ArgumentParser(description="Evaluate feedback quality.")
    parser.add_argument("--save", type=bool, default=True, help="Save evaluation results to MongoDB")
    parser.add_argument("--summary", action="store_true", help="Only print summary without saving")

    args = parser.parse_args()

    save_flag = args.save and not args.summary
    print("Running feedback evaluation...")
    evaluate_quality(save_to_db=save_flag)
    print("Evaluation completed.")

if __name__ == "__main__":
    main()

```

---

# **4️⃣ Backend FastAPI + Scheduler**

Already covered in the previous answer with `backend/api.py`, `backend/scheduler.py`, `backend/main.py`.

---

# **5️⃣ Frontend Gradio UI**

Already covered in previous answer (`frontend/app.py`) — interacts with FastAPI APIs.

---

# **6️⃣ How to run everything**

1. Load pages into MongoDB:

```bash
python scripts/load_pages_to_db.py data/confluence_pages.json True
```

2. Start backend (FastAPI + Scheduler):

```bash
python backend/main.py
```

3. Start frontend (Gradio):

```bash
python frontend/app.py
```

---

✅ This structure includes:

* MongoDB client module
* Scripts to load pages, enrich tags, and evaluate feedback
* Core modules for search, feedback handling, tagging, paraphrasing
* FastAPI backend + APScheduler
* Gradio frontend

---

Perfect! Let’s update **`run_tag_enrichment.py`** and **`run_feedback_update.py`** so they accept command-line parameters for flexibility. This way, both scripts can be run manually with options or scheduled automatically.

---

## **1️⃣ scripts/run_tag_enrichment.py**

We’ll allow:

* `--pages` → optional, specify a subset of page IDs to enrich
* `--verbose` → optional, print detailed output

```python
# scripts/run_tag_enrichment.py
import sys
import argparse
from core.tag_enricher import enrich_tags

def main():
    parser = argparse.ArgumentParser(description="Run tag enrichment on pages.")
    parser.add_argument("--pages", nargs="*", help="Optional list of page IDs to enrich. Enrich all if not provided.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs")

    args = parser.parse_args()

    if args.pages:
        if args.verbose:
            print(f"Enriching tags for {len(args.pages)} specified pages...")
        enrich_tags(page_ids=args.pages)
    else:
        if args.verbose:
            print("Enriching tags for all pages...")
        enrich_tags()

    print("Tag enrichment completed.")

if __name__ == "__main__":
    main()
```

**Update `core/tag_enricher.py`** to accept optional `page_ids`:

```python
def enrich_tags(page_ids=None):
    query = {} if not page_ids else {"page_id": {"$in": page_ids}}
    docs = pages_collection.find(query)
    for doc in docs:
        tags = set(doc.get("tags", []))
        text = (doc.get("content") or "").lower()
        for k, v in KEYWORD_TAGS.items():
            if re.search(rf"\b{k}\b", text):
                tags.add(v)
        pages_collection.update_one({"_id": doc["_id"]}, {"$set": {"tags": list(tags)}})
```

**Now you can run:**

```bash
# Enrich all pages
python scripts/run_tag_enrichment.py --verbose

# Enrich specific pages
python scripts/run_tag_enrichment.py --pages 123 456 --verbose
```

---

## **2️⃣ scripts/run_feedback_update.py**

We’ll allow:

* `--save` → save evaluation results to MongoDB (default True)
* `--summary` → only print summary, do not save

```python
# scripts/run_feedback_update.py
import argparse
from core.evaluator import evaluate_quality

def main():
    parser = argparse.ArgumentParser(description="Evaluate feedback quality.")
    parser.add_argument("--save", type=bool, default=True, help="Save evaluation results to MongoDB")
    parser.add_argument("--summary", action="store_true", help="Only print summary without saving")

    args = parser.parse_args()

    save_flag = args.save and not args.summary
    print("Running feedback evaluation...")
    evaluate_quality(save_to_db=save_flag)
    print("Evaluation completed.")

if __name__ == "__main__":
    main()
```

**Run examples:**

```bash
# Evaluate feedback and save results
python scripts/run_feedback_update.py

# Evaluate feedback and only print summary
python scripts/run_feedback_update.py --summary
```

---

✅ **Now all scripts are fully parameterized and flexible:**

* `load_pages_to_db.py` → load JSON, optional overwrite
* `run_tag_enrichment.py` → enrich all or specific pages, optional verbose logs
* `run_feedback_update.py` → evaluate feedback, optional save or summary only

This makes them fully compatible with:

* **Scheduler jobs**
* **Manual admin runs**
* **Future automated pipelines**

---

Perfect! Let’s update both **scheduler** and **FastAPI admin APIs** so they can **call the scripts using flags**, rather than hardcoding function calls. This makes it consistent, modular, and easier for future expansion.

---

# **1️⃣ Updated Scheduler (`backend/scheduler.py`)**

We’ll use **subprocess** to run the scripts with optional flags:

```python
import logging
import subprocess
from apscheduler.schedulers.background import BackgroundScheduler

logging.basicConfig(
    filename="logs/scheduler.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

def run_script(script_path, *args):
    """Run a Python script with optional command-line arguments."""
    cmd = ["python", script_path] + list(args)
    logging.info(f"Running script: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logging.info(f"Script output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        logging.exception(f"Script failed: {e.stderr}")

def job_tag_enrichment():
    logging.info("🏷️ Scheduled tag enrichment started.")
    run_script("scripts/run_tag_enrichment.py", "--verbose")
    logging.info("🏷️ Scheduled tag enrichment finished.")

def job_feedback_evaluation():
    logging.info("📊 Scheduled feedback evaluation started.")
    run_script("scripts/run_feedback_update.py")
    logging.info("📊 Scheduled feedback evaluation finished.")

def start_scheduler():
    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
    scheduler.add_job(job_tag_enrichment, "cron", hour=1, id="tag_enrichment")
    scheduler.add_job(job_feedback_evaluation, "cron", hour=2, id="feedback_evaluation")
    scheduler.start()
    logging.info("🚀 Scheduler started with daily tag & evaluation jobs.")
```

✅ Now the scheduler **runs the actual scripts** with proper logging.

---

# **2️⃣ Updated FastAPI Admin APIs (`backend/api.py`)**

We’ll add endpoints that **trigger scripts with flags**, so L2/support staff can manage without touching the server directly.

```python
from fastapi import FastAPI, Query
import subprocess

app = FastAPI(title="Confluence Chatbot API")

def run_script(script_path, *args):
    cmd = ["python", script_path] + list(args)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "error": e.stderr}

@app.get("/query")
def query_bot(q: str = Query(...)):
    from core.search_engine import hybrid_search
    results = hybrid_search(q)
    return {"query": q, "results": results}

@app.post("/feedback")
def feedback(q: str, page_id: str, feedback: str):
    from core.feedback_handler import save_feedback
    save_feedback(q, page_id, feedback)
    return {"status": "success"}

@app.post("/admin/enrich_tags")
def admin_enrich_tags(pages: str = None, verbose: bool = True):
    """Trigger tag enrichment manually via API."""
    args = []
    if pages:
        args += pages.split(",")
    if verbose:
        args.append("--verbose")
    return run_script("scripts/run_tag_enrichment.py", *args)

@app.post("/admin/evaluate_feedback")
def admin_evaluate_feedback(save: bool = True, summary: bool = False):
    """Trigger feedback evaluation manually via API."""
    args = []
    if not save:
        args.append("--summary")
    return run_script("scripts/run_feedback_update.py", *args)
```

**Usage examples (FastAPI endpoints):**

* **Trigger full tag enrichment with verbose logs:**

```
POST /admin/enrich_tags
```

* **Enrich specific pages (comma-separated page_ids):**

```
POST /admin/enrich_tags?pages=123,456
```

* **Run feedback evaluation without saving to DB (summary only):**

```
POST /admin/evaluate_feedback?save=false
```

---

# **✅ Benefits of this approach**

1. **Centralized execution**: Both scheduler and admin API use the **same scripts**, so logic is not duplicated.
2. **Flag-driven**: Easily add new command-line flags in scripts, scheduler, and API.
3. **Junior developer-friendly**: To add a new admin operation, just create a script + API endpoint.
4. **Logging & Debugging**: Scheduler logs all outputs to `logs/scheduler.log` and API returns stdout/stderr.

---

