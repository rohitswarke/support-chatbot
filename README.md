# support-chatbot
A simple chatbot with Confluence pages as knowledge-base.


**full extension** of the chatbot system that includes:

✅ **1. One-click feedback UI** (Good / Bad) directly in Gradio
✅ **2. Feedback persistence** in MongoDB
✅ **3. Feedback-based ranking update logic**
✅ **4. Evaluation script** to periodically assess search quality using feedback data
✅ **5. Clean modular package layout** so your junior devs can easily expand

Let’s go step-by-step 👇

---

## 🧩 Project Layout

```
chatbot/
│
├── app.py                       # Gradio chatbot UI + logic
│
├── core/
│   ├── config.py                 # Config (Mongo, flags, etc.)
│   ├── db.py                     # MongoDB connection helpers
│   ├── search_engine.py          # Hybrid search + ranking
│   ├── feedback_handler.py       # Feedback persistence + ranking adjustment
│   ├── tag_enricher.py           # Auto-tagging job for pages
│   ├── synonym_map.py            # Domain synonym map
│   ├── paraphraser.py            # Optional query paraphrasing
│   └── evaluator.py              # Evaluation script
│
└── data/
    ├── synonym_map.json          # Domain synonyms (Equity & Margin Ops)
    └── config.yaml               # Optional global config file
```

---

## ⚙️ 1. config.py

```python
# core/config.py
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "confluence_bot")

# Feature flags
USE_EMBEDDINGS = True
USE_PARAPHRASING = True
USE_TAG_FILTER = True

# Scoring weights
WEIGHTS = {
    "bm25": 0.4,
    "embedding": 0.6
}
```

---

## 🧠 2. db.py

```python
# core/db.py
from pymongo import MongoClient
from core.config import MONGO_URI, DB_NAME

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

pages_collection = db["pages"]
feedback_collection = db["feedback"]
```

---

## 🔍 3. search_engine.py

```python
# core/search_engine.py
from rank_bm25 import BM25Okapi
from core.db import pages_collection
from core.synonym_map import apply_synonyms
from core.paraphraser import generate_paraphrases
from core.config import USE_PARAPHRASING, WEIGHTS
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load model if available
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    model = None

def get_all_pages():
    return list(pages_collection.find({}, {"title": 1, "content": 1, "tags": 1, "url": 1}))

def hybrid_search(query, top_k=5):
    # Apply synonyms
    query = apply_synonyms(query)
    
    # Optionally paraphrase
    queries = [query]
    if USE_PARAPHRASING:
        queries.extend(generate_paraphrases(query))
    
    docs = get_all_pages()
    contents = [doc["content"] for doc in docs]
    
    # BM25
    bm25 = BM25Okapi([c.split() for c in contents])
    bm25_scores = np.mean([bm25.get_scores(q.split()) for q in queries], axis=0)
    
    # Embedding similarity (if available)
    if model:
        query_emb = np.mean(model.encode(queries), axis=0)
        doc_embs = model.encode(contents)
        emb_scores = util.cos_sim(query_emb, doc_embs).cpu().numpy()[0]
    else:
        emb_scores = np.zeros(len(contents))
    
    # Weighted hybrid score
    scores = WEIGHTS["bm25"] * bm25_scores + WEIGHTS["embedding"] * emb_scores
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [docs[i] | {"score": float(scores[i])} for i in top_indices]
```

---

## 🏷️ 4. tag_enricher.py

```python
# core/tag_enricher.py
import re
from core.db import pages_collection

# Very simple heuristic tag enricher for Equity Markets L1/L2 runbooks
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

## 💬 5. feedback_handler.py

```python
# core/feedback_handler.py
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
    """Reward pages with positive feedback by slightly boosting their future rank"""
    doc = pages_collection.find_one({"page_id": page_id})
    current = doc.get("score_boost", 0)
    delta = 0.05 if feedback_value == "good" else -0.05
    pages_collection.update_one({"page_id": page_id}, {"$set": {"score_boost": current + delta}})
```

---

## 🧩 6. paraphraser.py

```python
# core/paraphraser.py
import itertools

def generate_paraphrases(query):
    """Basic paraphrasing by rewording common ops phrases"""
    templates = [
        "how to resolve {}",
        "steps to fix {}",
        "troubleshooting {}",
        "resolution for {}"
    ]
    return [t.format(query) for t in templates]
```

---

## 📚 7. synonym_map.py

```python
# core/synonym_map.py
DOMAIN_SYNONYMS = {
    "trade": ["deal", "transaction", "order"],
    "margin": ["collateral", "exposure", "haircut"],
    "batch": ["job", "process", "schedule"],
    "fail": ["error", "exception", "issue"],
    "equity": ["stock", "share", "security"]
}

def apply_synonyms(query):
    for word, syns in DOMAIN_SYNONYMS.items():
        for s in syns:
            if s in query.lower():
                query = query.replace(s, word)
    return query
```

---

## 🎨 8. app.py (Gradio UI)

```python
# app.py
import gradio as gr
from core.search_engine import hybrid_search
from core.feedback_handler import save_feedback

def chat(query):
    results = hybrid_search(query)
    display = "\n\n".join([
        f"**{r['title']}**\nScore: {r['score']:.2f}\n{r['url']}\n---"
        for r in results
    ])
    return display, results

def record_feedback(query, result_idx, feedback, results_state):
    if results_state and 0 <= result_idx < len(results_state):
        page = results_state[result_idx]
        save_feedback(query, page["page_id"], feedback)
        return f"✅ Feedback '{feedback}' saved for: {page['title']}"
    return "⚠️ Invalid feedback target."

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Confluence Support Bot (Equity Margin Ops)")
    
    query = gr.Textbox(label="Ask your question")
    output = gr.Markdown()
    results_state = gr.State()
    
    submit = gr.Button("Search")
    submit.click(fn=chat, inputs=query, outputs=[output, results_state])
    
    with gr.Row():
        result_idx = gr.Number(label="Result index (0-based)", value=0)
        good_btn = gr.Button("👍 Good")
        bad_btn = gr.Button("👎 Bad")
    
    feedback_msg = gr.Textbox(label="Feedback log", interactive=False)
    
    good_btn.click(fn=lambda q, i, r: record_feedback(q, i, "good", r),
                   inputs=[query, result_idx, results_state],
                   outputs=feedback_msg)
    bad_btn.click(fn=lambda q, i, r: record_feedback(q, i, "bad", r),
                  inputs=[query, result_idx, results_state],
                  outputs=feedback_msg)

demo.launch()
```

---

## 📈 9. evaluator.py

```python
# core/evaluator.py
from core.db import feedback_collection
import pandas as pd

def evaluate_quality():
    fb = list(feedback_collection.find())
    df = pd.DataFrame(fb)
    if df.empty:
        print("No feedback yet.")
        return
    
    summary = df.groupby("feedback").size().reset_index(name="count")
    total = df.shape[0]
    good_ratio = df[df["feedback"] == "good"].shape[0] / total
    print("Feedback summary:")
    print(summary)
    print(f"\n✅ Good feedback ratio: {good_ratio:.2%}")

if __name__ == "__main__":
    evaluate_quality()
```

---

## 🚀 10. How to Expand

* Add **more paraphrasing templates** → `core/paraphraser.py`
* Add **domain synonyms** → `core/synonym_map.py`
* Extend **tag logic** for new spaces → `core/tag_enricher.py`
* Improve feedback → aggregate by query patterns in `feedback_handler.py`

---

✅ integrate **automatic daily tag enrichment & evaluation scheduler (using APScheduler)**,
so it runs `tag_enricher` and `evaluator` automatically every night

1. **Enrich tags** for new or modified Confluence pages (auto-labeling job)
2. **Evaluate feedback quality** and print/report insights daily
3. Optionally log the results to MongoDB for dashboarding

We’ll use **APScheduler** (a lightweight, production-safe Python scheduler) to automate this.

---

## 🧩 Updated Project Layout

```
chatbot/
│
├── app.py
├── scheduler.py              ← 🔥 New automatic scheduler
│
├── core/
│   ├── config.py
│   ├── db.py
│   ├── search_engine.py
│   ├── feedback_handler.py
│   ├── tag_enricher.py
│   ├── synonym_map.py
│   ├── paraphraser.py
│   ├── evaluator.py
│   └── utils.py              ← (optional helper for logging)
│
└── logs/
    └── scheduler.log
```

---

## ⚙️ 1. Install dependency

```bash
pip install apscheduler
```

---

## 🕓 2. scheduler.py — Daily Tag Enrichment & Evaluation

```python
# scheduler.py
import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from core.tag_enricher import enrich_tags
from core.evaluator import evaluate_quality
from core.db import db

# --- Setup logging ---
logging.basicConfig(
    filename="logs/scheduler.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

def job_tag_enrichment():
    logging.info("🏷️ Tag enrichment started.")
    try:
        enrich_tags()
        logging.info("✅ Tag enrichment completed successfully.")
    except Exception as e:
        logging.exception(f"❌ Tag enrichment failed: {e}")

def job_evaluation():
    logging.info("📊 Evaluation started.")
    try:
        summary = evaluate_quality(save_to_db=True)
        logging.info(f"✅ Evaluation summary: {summary}")
    except Exception as e:
        logging.exception(f"❌ Evaluation failed: {e}")

def start_scheduler():
    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")

    # Tag enrichment every day at 1:00 AM
    scheduler.add_job(job_tag_enrichment, "cron", hour=1, minute=0, id="tag_enrichment")

    # Feedback evaluation every day at 2:00 AM
    scheduler.add_job(job_evaluation, "cron", hour=2, minute=0, id="evaluation")

    scheduler.start()
    logging.info("🚀 Scheduler started. Jobs registered:")
    for job in scheduler.get_jobs():
        logging.info(f"  -> {job}")

    try:
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logging.info("🛑 Scheduler stopped.")

if __name__ == "__main__":
    start_scheduler()
```

---

## 🧠 3. Updated evaluator.py — Save Evaluation Results in MongoDB

```python
# core/evaluator.py
from core.db import feedback_collection, db
import pandas as pd
from datetime import datetime

def evaluate_quality(save_to_db=False):
    fb = list(feedback_collection.find())
    if not fb:
        return "No feedback yet."

    df = pd.DataFrame(fb)
    summary = df.groupby("feedback").size().reset_index(name="count")
    total = df.shape[0]
    good_ratio = df[df["feedback"] == "good"].shape[0] / total

    result = {
        "timestamp": datetime.utcnow(),
        "total_feedback": total,
        "good_feedback": int(df[df["feedback"] == "good"].shape[0]),
        "bad_feedback": int(df[df["feedback"] == "bad"].shape[0]),
        "good_ratio": round(good_ratio, 3),
        "summary": summary.to_dict(orient="records"),
    }

    if save_to_db:
        db["evaluation_reports"].insert_one(result)

    print("Feedback summary:")
    print(summary)
    print(f"\n✅ Good feedback ratio: {good_ratio:.2%}")

    return result
```

---

## 🔧 4. Optional: Run Together with Chatbot

You can start the chatbot **and scheduler together** if you want it all in one process.

Example combined launcher:

```python
# run_all.py
import threading
from app import demo
from scheduler import start_scheduler

def start_gradio():
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    t1 = threading.Thread(target=start_scheduler)
    t1.daemon = True
    t1.start()

    start_gradio()
```

This way:

* Gradio chatbot runs on port `7860`
* Scheduler runs silently in the background
* All logs go to `logs/scheduler.log`

---

## 🧪 5. Verification

Once you start it, you’ll see logs like:

```
2025-10-07 01:00:00 | INFO | 🏷️ Tag enrichment started.
2025-10-07 01:00:03 | INFO | ✅ Tag enrichment completed successfully.
2025-10-07 02:00:00 | INFO | 📊 Evaluation started.
2025-10-07 02:00:00 | INFO | ✅ Evaluation summary: {'total_feedback': 42, 'good_ratio': 0.67}
```

---

## 🧩 6. What You Can Add Next (Optional Enhancements)

1. **Email or Slack notifications** if evaluation score drops below threshold
2. **Daily dashboard** for feedback & tag analytics
3. **Dynamic job discovery** → auto-tag new pages without restart
4. **Auto-synonym discovery** → from high-frequency words in successful queries

---

Excellent decision ✅ — keeping **FastAPI as the backend** will future-proof your chatbot system and make it enterprise-ready.

Let’s summarize what your final architecture will look like (with FastAPI + Gradio + Scheduler), and then I’ll show you the complete base setup that your team can run and extend right away.

---

## 🧩 Final Architecture Overview

```
chatbot/
│
├── backend/
│   ├── api.py              # FastAPI routes (query, feedback, admin)
│   ├── scheduler.py        # APScheduler daily jobs (tag + evaluation)
│   ├── main.py             # Starts FastAPI + Scheduler threads
│
├── frontend/
│   └── app.py              # Gradio UI connecting to backend APIs
│
├── core/
│   ├── config.py
│   ├── db.py
│   ├── search_engine.py
│   ├── feedback_handler.py
│   ├── tag_enricher.py
│   ├── evaluator.py
│   ├── synonym_map.py
│   └── paraphraser.py
│
└── logs/
    └── scheduler.log
```

---

## 🧠 `backend/api.py`

```python
from fastapi import FastAPI, Query
from core.search_engine import hybrid_search
from core.feedback_handler import save_feedback
from core.evaluator import evaluate_quality
from core.tag_enricher import enrich_tags

app = FastAPI(title="Confluence Chatbot API")

@app.get("/query")
def query_bot(q: str = Query(..., description="User query text")):
    results = hybrid_search(q)
    return {"query": q, "results": results}

@app.post("/feedback")
def feedback(q: str, page_id: str, feedback: str):
    save_feedback(q, page_id, feedback)
    return {"status": "success"}

@app.get("/admin/summary")
def admin_summary():
    """Check latest evaluation and tags status."""
    summary = evaluate_quality(save_to_db=False)
    return {"evaluation": summary}

@app.post("/admin/enrich_tags")
def admin_enrich_tags():
    """Trigger manual tag enrichment."""
    enrich_tags()
    return {"status": "tag enrichment complete"}
```

---

## 🕓 `backend/scheduler.py`

```python
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from core.tag_enricher import enrich_tags
from core.evaluator import evaluate_quality

logging.basicConfig(
    filename="logs/scheduler.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

def job_tag_enrichment():
    logging.info("🏷️ Tag enrichment started.")
    try:
        enrich_tags()
        logging.info("✅ Tag enrichment completed successfully.")
    except Exception as e:
        logging.exception(f"❌ Tag enrichment failed: {e}")

def job_evaluation():
    logging.info("📊 Evaluation started.")
    try:
        summary = evaluate_quality(save_to_db=True)
        logging.info(f"✅ Evaluation summary: {summary}")
    except Exception as e:
        logging.exception(f"❌ Evaluation failed: {e}")

def start_scheduler():
    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
    scheduler.add_job(job_tag_enrichment, "cron", hour=1, id="tag_enrichment")
    scheduler.add_job(job_evaluation, "cron", hour=2, id="evaluation")
    scheduler.start()
    logging.info("🚀 Scheduler started with daily tag & evaluation jobs.")
```

---

## 🚀 `backend/main.py`

```python
import threading
import uvicorn
from backend.api import app
from backend.scheduler import start_scheduler

def start_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    t = threading.Thread(target=start_scheduler, daemon=True)
    t.start()
    start_fastapi()
```

---

## 💬 `frontend/app.py` (Gradio UI)

```python
import gradio as gr
import requests

API_URL = "http://localhost:8000"

def chat(query):
    r = requests.get(f"{API_URL}/query", params={"q": query}).json()
    results = r.get("results", [])
    display = "\n\n".join([
        f"**{r['title']}**\nScore: {r['score']:.2f}\n{r['url']}\n---"
        for r in results
    ]) if results else "No relevant results found."
    return display, results

def record_feedback(query, result_idx, feedback, results_state):
    if results_state and 0 <= result_idx < len(results_state):
        page = results_state[result_idx]
        requests.post(f"{API_URL}/feedback", params={
            "q": query,
            "page_id": page["page_id"],
            "feedback": feedback
        })
        return f"✅ Feedback '{feedback}' saved for: {page['title']}"
    return "⚠️ Invalid feedback target."

with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Equity & Margin Support Bot")

    query = gr.Textbox(label="Ask your question")
    output = gr.Markdown()
    results_state = gr.State()

    submit = gr.Button("Search")
    submit.click(fn=chat, inputs=query, outputs=[output, results_state])

    with gr.Row():
        idx = gr.Number(label="Result index", value=0)
        good_btn = gr.Button("👍 Good")
        bad_btn = gr.Button("👎 Bad")

    feedback_msg = gr.Textbox(interactive=False)
    good_btn.click(lambda q, i, r: record_feedback(q, i, "good", r),
                   [query, idx, results_state], feedback_msg)
    bad_btn.click(lambda q, i, r: record_feedback(q, i, "bad", r),
                  [query, idx, results_state], feedback_msg)

demo.launch(server_port=7860)
```

---

## 🧪 Run Instructions

1. **Start backend (FastAPI + Scheduler):**

   ```bash
   python backend/main.py
   ```

   * Backend API → [http://localhost:8000](http://localhost:8000)
   * Scheduler logs → `logs/scheduler.log`

2. **Start Gradio frontend:**

   ```bash
   python frontend/app.py
   ```

   * UI → [http://localhost:7860](http://localhost:7860)
   * Queries/Feedback → Sent via backend APIs

---

## 🧭 Next Steps (for Long-Term Scalability)

1. **Add Auth Middleware** → Azure AD / SSO in FastAPI
2. **Add Domain Configs** → For multi-space Confluence content
3. **Add “/train” endpoint** → For embedding updates when LLM access is available
4. **Add Slack/Email alerts** when feedback quality < threshold
5. **Dashboard** for daily tag & evaluation summary (Grafana or Streamlit)

---
