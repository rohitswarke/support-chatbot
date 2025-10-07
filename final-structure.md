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

def enrich_tags():
    docs = pages_collection.find()
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
import json
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
```

### **scripts/run_tag_enrichment.py**

```python
from core.tag_enricher import enrich_tags

if __name__ == "__main__":
    print("Running tag enrichment...")
    enrich_tags()
    print("Tag enrichment completed.")
```

### **scripts/run_feedback_update.py**

```python
from core.evaluator import evaluate_quality

if __name__ == "__main__":
    print("Running feedback evaluation...")
    evaluate_quality(save_to_db=True)
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

If you want, I can **also add an automatic feedback-based ranking update + admin APIs** to allow **manual trigger of tag enrichment, evaluation, or weight adjustments** directly from FastAPI, so L2 support can manage without touching MongoDB.

Do you want me to do that next?
