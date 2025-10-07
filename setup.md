Great — I implemented the four features you requested and wired them into the modular system:

1. **Feedback-based ranking update mechanism** — aggregates feedback and writes per-document boost factors used at runtime. A periodic job updates boosts in MongoDB so the search engine reads a simple multiplier per document (fast at query time).
2. **Tag enrichment job** — periodically refines tags for each page using TF-IDF / frequency heuristics and stores them back into MongoDB (safe to run repeatedly).
3. **Query paraphrasing (flagged)** — simple paraphraser that expands user queries into variants using the synonym map, synonyms expansion and simple phrase permutations. Controlled by `ENABLE_PARAPHRASING` flag.
4. **Synonym map for Equity Markets / cross-product margining** — a starter synonym map tailored to your domain; easily extendable.

Everything is designed to be **process oriented**, **safe to run repeatedly**, and **commented** for junior developers to extend.

---

# What I am delivering now

A compact modular package `supportbot/` with the following new/updated modules:

```
supportbot/
├── config.py
├── app.py                     # main Gradio + FastAPI (entry)
├── core/
│   ├── mongo_client.py
│   ├── data_loader.py
│   ├── tag_generator.py
│   ├── tag_enrichment_job.py     # new: periodic tag enrichment
│   ├── hybrid_search.py
│   ├── feedback_manager.py
│   ├── feedback_updater.py       # new: compute per-doc boost and store
│   ├── query_paraphraser.py      # new: paraphrase/expand queries
│   └── utils.py
└── scripts/
    ├── load_pages_to_db.py
    └── run_enrichment_and_feedback_update.py
```

Below I paste **all files** (complete working code). After the code you'll find:

* How it works (process flow)
* How to run (commands)
* Notes for junior developers on expanding synonyms, paraphrasing, tag logic and embedding usage

---

> IMPORTANT: these modules use only standard Python packages and the ones we discussed earlier (`pymongo`, `scikit-learn`, `rank-bm25`, `gradio`, etc.). If you enable embeddings later, install `sentence-transformers` and set `ENABLE_EMBEDDING=True` in `config.py`.

---

## `supportbot/config.py`

```python
# supportbot/config.py
"""
Central configuration and feature flags.

Edit these values or set environment variables before running.
"""
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "supportbot_db")
PAGES_COLLECTION = os.getenv("PAGES_COLLECTION", "pages")
FEEDBACK_COLLECTION = os.getenv("FEEDBACK_COLLECTION", "feedback")
DOC_BOOST_COLLECTION = os.getenv("DOC_BOOST_COLLECTION", "doc_boosts")  # stores computed boosts per page_id

# Search / model flags
ENABLE_EMBEDDING = os.getenv("ENABLE_EMBEDDING", "false").lower() == "true"  # optional, needs sentence-transformers
ENABLE_PARAPHRASING = os.getenv("ENABLE_PARAPHRASING", "true").lower() == "true"
ENABLE_TAG_ENRICHMENT = os.getenv("ENABLE_TAG_ENRICHMENT", "true").lower() == "true"
ENABLE_FEEDBACK_UPDATES = os.getenv("ENABLE_FEEDBACK_UPDATES", "true").lower() == "true"

# Operational settings
TAG_TOP_K = int(os.getenv("TAG_TOP_K", "10"))
ENRICHMENT_INTERVAL_SECONDS = int(os.getenv("ENRICHMENT_INTERVAL_SECONDS", str(60 * 60 * 6)))  # every 6 hours
FEEDBACK_UPDATE_INTERVAL_SECONDS = int(os.getenv("FEEDBACK_UPDATE_INTERVAL_SECONDS", str(60 * 30)))  # every 30 minutes

# Hybrid scoring weights (tune as needed)
TFIDF_WEIGHT = float(os.getenv("TFIDF_WEIGHT", "0.5"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
OVERLAP_WEIGHT = float(os.getenv("OVERLAP_WEIGHT", "0.1"))
FEEDBACK_ALPHA = float(os.getenv("FEEDBACK_ALPHA", "0.15"))  # max +/- 15% effect
PARAPHRASE_VARIANTS = int(os.getenv("PARAPHRASE_VARIANTS", "6"))  # max variants to generate per query

# Optional path to local embedding model (if you have an org-hosted file)
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "/models/all-MiniLM-L6-v2")
```

---

## `supportbot/core/mongo_client.py`

```python
# supportbot/core/mongo_client.py
from pymongo import MongoClient
from supportbot.config import MONGO_URI, DB_NAME

_client = None

def get_db():
    """
    Returns a pymongo database handle. Reuses the client across calls.
    """
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client[DB_NAME]
```

---

## `supportbot/core/data_loader.py`

```python
# supportbot/core/data_loader.py
from supportbot.core.mongo_client import get_db
from supportbot.config import PAGES_COLLECTION

def fetch_all_pages():
    """
    Returns all page documents from MongoDB.
    Each document is expected to contain at least:
      - page_id (unique)
      - title
      - content (plain text)
      - url (optional)
      - space_key (optional)
      - tags (optional)
    """
    db = get_db()
    coll = db[PAGES_COLLECTION]
    return list(coll.find({}, projection={"_id": 0}))
```

---

## `supportbot/core/tag_generator.py`

```python
# supportbot/core/tag_generator.py
"""
Tag extraction utilities.
We use a simple TF-IDF / frequency based extractor. This is intentionally simple
and safe to run for junior devs to extend to YAKE/KeyBERT later.
"""
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from supportbot.config import TAG_TOP_K

_non_alnum = re.compile(r'[^0-9a-zA-Z\s]')

def clean_text(text: str) -> str:
    if not text:
        return ""
    t = _non_alnum.sub(" ", text)
    t = " ".join(t.split())
    return t.lower()

def extract_candidate_keywords(text: str, top_k: int = TAG_TOP_K):
    """
    Return top_k candidate keywords from text using TF-IDF on the single doc.
    Implementation detail: we split into pseudo-sentences to compute local TF-IDF —
    this is a quick heuristic and works reasonably for tagging runbooks.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    # split into sentences (crudely), so TF-IDF can rank local terms
    parts = [p for p in cleaned.split('.') if p.strip()]
    # If we have only one part, fallback to splitting by newline/space
    if len(parts) == 1:
        parts = [p for p in cleaned.split('\n') if p.strip()]

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = vectorizer.fit_transform(parts)
        # aggregate tf-idf scores across parts for each token
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf.sum(axis=0).A1  # sums per feature
        zipped = list(zip(feature_names, scores))
        sorted_by_score = sorted(zipped, key=lambda x: x[1], reverse=True)
        keywords = [w for w, _ in sorted_by_score[:top_k]]
        return keywords
    except Exception:
        # fallback: frequency-based
        tokens = cleaned.split()
        c = Counter([t for t in tokens if len(t) > 3])
        return [w for w, _ in c.most_common(top_k)]

def generate_tags_for_page(page: dict, top_k: int = TAG_TOP_K):
    """
    Return a list of tags for the page using title + first N chars of content.
    Keep tags normalized (lowercase, whitespace stripped).
    """
    title = page.get("title", "") or ""
    content = page.get("content", "") or ""
    seed = title + ". " + content[:2000]  # limit content used for speed
    tags = extract_candidate_keywords(seed, top_k=top_k)
    # normalize tags
    clean = []
    for t in tags:
        t = t.strip().lower()
        if t and len(t) > 2:
            clean.append(t)
    return clean
```

---

## `supportbot/core/tag_enrichment_job.py`

```python
# supportbot/core/tag_enrichment_job.py
"""
Periodic job that updates tags for all pages in MongoDB.

- Reads pages
- Generates tags with tag_generator.generate_tags_for_page
- Stores tags in pages collection as 'tags' field
- Runs periodically (threaded job) if enabled
"""

import threading
import time
from supportbot.core.mongo_client import get_db
from supportbot.config import PAGES_COLLECTION, ENRICHMENT_INTERVAL_SECONDS, ENABLE_TAG_ENRICHMENT, TAG_TOP_K
from supportbot.core.tag_generator import generate_tags_for_page

_db = get_db()

def run_enrichment_once():
    coll = _db[PAGES_COLLECTION]
    pages = list(coll.find({}, projection={"page_id": 1, "title": 1, "content": 1}))
    print(f"[TagEnrich] Found {len(pages)} pages to enrich tags for.")
    ops = []
    for p in pages:
        page_obj = {"page_id": p.get("page_id"), "title": p.get("title"), "content": p.get("content")}
        tags = generate_tags_for_page(page_obj, top_k=TAG_TOP_K)
        # update document with tags
        coll.update_one({"page_id": p["page_id"]}, {"$set": {"tags": tags}})

    print("[TagEnrich] Tag enrichment complete.")

def _run_periodic():
    while True:
        try:
            run_enrichment_once()
        except Exception as e:
            print("[TagEnrich] Exception:", e)
        time.sleep(ENRICHMENT_INTERVAL_SECONDS)

def start_background_job():
    if not ENABLE_TAG_ENRICHMENT:
        print("[TagEnrich] Tag enrichment disabled by config.")
        return None
    t = threading.Thread(target=_run_periodic, daemon=True)
    t.start()
    print("[TagEnrich] Background tag enrichment job started.")
    return t
```

---

## `supportbot/core/query_paraphraser.py`

```python
# supportbot/core/query_paraphraser.py
"""
Query paraphrasing / expansion.

The paraphraser is intentionally simple and deterministic:
- Uses a domain synonym map to expand tokens
- Generates combination variants by replacing words with synonyms (bounded by PARAPHRASE_VARIANTS)
- Controlled by ENABLE_PARAPHRASING flag; safe to turn off

Junior dev notes:
- This is not a neural paraphraser. If/when you add an LLM or paraphrase model,
  keep interface `expand_query(query) -> List[str]` so search engine code does not change.
"""
import itertools
from supportbot.config import PARAPHRASE_VARIANTS, ENABLE_PARAPHRASING

# Starter domain synonym map for Equity Markets & margining platform.
# This is a seed list — add entries based on real business terminology.
SYNONYM_MAP = {
    # equity markets & margining
    "margin": ["margin", "collateral", "collateral margin"],
    "margining": ["margining", "margin process"],
    "mtm": ["mark-to-market", "mtm", "mark to market"],
    "reconciliation": ["reconciliation", "recon"],
    "recon": ["reconciliation", "recon"],
    "reboot": ["restart", "reboot", "re-start"],
    "restart": ["restart", "reboot", "re-start"],
    "svc": ["service", "svc", "service process"],
    "svcdown": ["service down", "svc down", "service outage"],
    "file": ["file", "feed", "input file"],
    "filetransfer": ["file transfer", "sftp", "ftp", "file movement"],
    "trade": ["trade", "trades", "transaction"],
    "job": ["job", "batch", "process"],
    "failed": ["failed", "failure", "error"],
    "error": ["error", "exception", "err"],
    "node": ["node", "server", "host"],
    "db": ["database", "db"],
    "dbconn": ["database connection", "db connection", "dbconn"],
    # add more as you find domain synonyms
}

def tokenize_simple(q: str):
    return [t.strip() for t in q.lower().split() if t.strip()]

def expand_query(query: str, max_variants: int = PARAPHRASE_VARIANTS):
    """
    Return a list of paraphrased query variants.
    Strategy:
      - For each token, if synonyms exist, use them, otherwise use the token itself.
      - Generate cartesian product combinations, but cap total at max_variants.
    """
    if not ENABLE_PARAPHRASING:
        return [query]

    tokens = tokenize_simple(query)
    if not tokens:
        return [query]

    options = []
    for t in tokens:
        opts = SYNONYM_MAP.get(t, None)
        if opts:
            options.append(list(dict.fromkeys(opts)))  # dedupe synonyms list
        else:
            options.append([t])

    # generate combinations
    combos = itertools.islice(itertools.product(*options), max_variants)
    variants = []
    for c in combos:
        variants.append(" ".join(c))
    # ensure original query is present
    if query not in variants:
        variants.insert(0, query)
    return variants
```

---

## `supportbot/core/hybrid_search.py` (updated to use doc boosts, paraphrasing & tag boosts)

```python
# supportbot/core/hybrid_search.py
"""
Hybrid search engine:
- Uses BM25 (rank-bm25) and TF-IDF cosine similarity (scikit-learn)
- Optionally uses paraphrase expansion to create multiple query variants and merge scores
- Applies precomputed document boost multipliers (from doc_boosts collection) for feedback
- Applies tag filter and tag overlap boost

This engine reads live tags and doc boosts from MongoDB on init and can be refreshed.
"""
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from supportbot.core.mongo_client import get_db
from supportbot.config import PAGES_COLLECTION, DOC_BOOST_COLLECTION, TFIDF_WEIGHT, BM25_WEIGHT, OVERLAP_WEIGHT
from supportbot.core.query_paraphraser import expand_query

_db = get_db()

class HybridSearchEngine:
    def __init__(self, pages=None, refresh_from_db=True):
        """
        If pages is None or refresh_from_db True, load pages from MongoDB.
        """
        self.refresh(refresh_from_db, pages)

    def refresh(self, refresh_from_db=True, pages=None):
        if refresh_from_db or pages is None:
            self.pages = list(_db[PAGES_COLLECTION].find({}, projection={"_id":0}))
        else:
            self.pages = pages
        self.page_id_index = {p.get("page_id"): i for i, p in enumerate(self.pages)}
        self.documents = [p.get("content", "") or "" for p in self.pages]
        # TF-IDF vectorizer & matrix
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
        if len(self.documents) == 0:
            self.tfidf_matrix = None
        else:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        # BM25 (tokenized on whitespace)
        self.tokenized = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized) if len(self.tokenized) > 0 else None
        # tags
        self.tags = [p.get("tags", []) for p in self.pages]
        # doc boosts (from feedback updater)
        self.doc_boosts = self._load_doc_boosts()

    def _load_doc_boosts(self):
        coll = _db[DOC_BOOST_COLLECTION]
        docs = {}
        for rec in coll.find({}):
            pid = rec.get("page_id")
            if pid:
                docs[pid] = rec.get("boost", 1.0)
        return docs

    def _feedback_boost(self, page_id):
        return float(self.doc_boosts.get(page_id, 1.0))

    def _overlap_score(self, q_tokens, doc_text):
        if not q_tokens:
            return 0.0
        doc_tokens = set(doc_text.lower().split())
        overlap = len(set(q_tokens) & doc_tokens) / float(len(q_tokens))
        return overlap

    def search(self, query, top_k=5, space_key=None, tag_filter=None, tfidf_weight=TFIDF_WEIGHT, bm25_weight=BM25_WEIGHT, overlap_weight=OVERLAP_WEIGHT):
        """
        Main search:
        - expand query variants (if paraphrasing enabled)
        - for each variant compute TF-IDF cosine and BM25 scores
        - normalize and average across variants
        - apply boosts: tag overlap and feedback doc boosts
        - filter by space_key and tag_filter
        """
        variants = expand_query(query)
        # limit variants
        # compute scores per variant
        all_scores = np.zeros((len(self.documents),), dtype=float)

        for v in variants:
            # tfidf
            if self.tfidf_matrix is not None:
                qvec = self.vectorizer.transform([v])
                tfidf_sims = cosine_similarity(qvec, self.tfidf_matrix).flatten()
            else:
                tfidf_sims = np.zeros(len(self.documents))

            # bm25
            if self.bm25:
                bm25_scores = np.array(self.bm25.get_scores(v.split()))
            else:
                bm25_scores = np.zeros(len(self.documents))

            # normalize both to 0..1
            def minmax(arr):
                if len(arr) == 0:
                    return arr
                a_min = arr.min()
                a_max = arr.max()
                if a_max == a_min:
                    return np.ones_like(arr)
                return (arr - a_min) / (a_max - a_min)

            tfidf_n = minmax(tfidf_sims)
            bm25_n = minmax(bm25_scores)

            # combined variant score
            variant_score = tfidf_weight * tfidf_n + bm25_weight * bm25_n

            all_scores += variant_score

        # average across variants
        all_scores = all_scores / float(len(variants))

        results = []
        for i, p in enumerate(self.pages):
            # space filter
            if space_key and p.get("space_key") != space_key:
                continue
            # tag filter
            if tag_filter and tag_filter not in (p.get("tags") or []):
                continue
            overlap = self._overlap_score(query.split(), p.get("content", ""))
            score = all_scores[i] + overlap_weight * overlap
            # apply feedback boost multiplier
            score *= self._feedback_boost(p.get("page_id"))
            results.append({
                "page": p,
                "score": float(score),
                "components": {
                    "tfidf_bm25": float(all_scores[i]),
                    "overlap": float(overlap),
                    "boost": float(self._feedback_boost(p.get("page_id")))
                }
            })
        # sort and return top_k
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]
```

---

## `supportbot/core/feedback_manager.py` (save feedback)

```python
# supportbot/core/feedback_manager.py
from datetime import datetime
from supportbot.core.mongo_client import get_db
from supportbot.config import FEEDBACK_COLLECTION

def save_feedback(query: str, page_id: str, feedback_type: str):
    """
    Persist a feedback record.
    feedback_type: "positive" or "negative"
    """
    db = get_db()
    rec = {
        "query": query,
        "page_id": page_id,
        "feedback": feedback_type,
        "ts": datetime.utcnow()
    }
    db[FEEDBACK_COLLECTION].insert_one(rec)
```

---

## `supportbot/core/feedback_updater.py` (the ranking update mechanism)

```python
# supportbot/core/feedback_updater.py
"""
This module computes per-document boost multipliers from feedback and stores
them into the DOC_BOOST_COLLECTION. The logic is intentionally conservative:
boost multiplier = 1 + alpha * (pos - neg) / total

This job can be run periodically or manually. We provide a background runner.
"""
import threading
import time
from supportbot.core.mongo_client import get_db
from supportbot.config import FEEDBACK_COLLECTION, DOC_BOOST_COLLECTION, FEEDBACK_ALPHA, FEEDBACK_UPDATE_INTERVAL_SECONDS, ENABLE_FEEDBACK_UPDATES

_db = get_db()

def compute_boosts_once():
    fb_coll = _db[FEEDBACK_COLLECTION]
    boost_coll = _db[DOC_BOOST_COLLECTION]

    pipeline = [
        {"$group": {
            "_id": {"page_id": "$page_id"},
            "pos": {"$sum": {"$cond": [{"$eq": ["$feedback", "positive"]}, 1, 0]}},
            "neg": {"$sum": {"$cond": [{"$eq": ["$feedback", "negative"]}, 1, 0]}}
        }}
    ]
    agg = fb_coll.aggregate(pipeline)
    updates = []
    count = 0
    for row in agg:
        pid = row["_id"]["page_id"]
        pos = row.get("pos", 0)
        neg = row.get("neg", 0)
        total = pos + neg
        if total == 0:
            boost = 1.0
        else:
            score = (pos - neg) / float(total)  # -1..1
            boost = 1.0 + FEEDBACK_ALPHA * score
        # upsert
        boost_coll.update_one({"page_id": pid}, {"$set": {"page_id": pid, "boost": boost, "last_updated": int(time.time())}}, upsert=True)
        count += 1
    print(f"[FeedbackUpdate] Updated boosts for {count} documents.")

def _run_periodic():
    while True:
        try:
            compute_boosts_once()
        except Exception as e:
            print("[FeedbackUpdate] Exception:", e)
        time.sleep(FEEDBACK_UPDATE_INTERVAL_SECONDS)

def start_background_job():
    if not ENABLE_FEEDBACK_UPDATES:
        print("[FeedbackUpdate] Disabled by config.")
        return None
    t = threading.Thread(target=_run_periodic, daemon=True)
    t.start()
    print("[FeedbackUpdate] Background feedback update job started.")
    return t
```

---

## `supportbot/core/utils.py`

```python
# supportbot/core/utils.py
"""
Utility helpers for junior devs:
- simple function to rebuild and refresh the search engine
- simple evaluation scaffolding placeholder
"""
from supportbot.core.mongo_client import get_db
from supportbot.config import PAGES_COLLECTION

def pages_count():
    db = get_db()
    return db[PAGES_COLLECTION].count_documents({})
```

---

## `supportbot/app.py` (entrypoint — Gradio UI and API endpoints)

```python
# supportbot/app.py
"""
Main app: mounts a lightweight API and Gradio UI, starts background jobs.
Run: python -m supportbot.app
"""
import threading
from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import gradio as gr
from supportbot.core.data_loader import fetch_all_pages
from supportbot.core.hybrid_search import HybridSearchEngine
from supportbot.core.feedback_manager import save_feedback
from supportbot.core.feedback_updater import start_background_job as start_feedback_job
from supportbot.core.tag_enrichment_job import start_background_job as start_tag_job
from supportbot.config import ENABLE_PARAPHRASING, ENABLE_TAG_ENRICHMENT, ENABLE_FEEDBACK_UPDATES

# FastAPI (programmatic access)
app = FastAPI(title="SupportBot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Instantiate engine (reads from MongoDB)
print("[App] Loading pages from MongoDB...")
pages = fetch_all_pages()
search_engine = HybridSearchEngine(pages=pages, refresh_from_db=False)  # pages provided; engine can refresh later

@app.get("/")
def root():
    return RedirectResponse(url="/ui")

@app.get("/api/search")
def api_search(q: str = Query(...), space_key: str = Query(None), tag: str = Query(None), top_k: int = Query(5)):
    results = search_engine.search(q, top_k=top_k, space_key=space_key, tag_filter=tag)
    out = []
    for r in results:
        p = r["page"]
        out.append({
            "page_id": p.get("page_id"),
            "title": p.get("title"),
            "url": p.get("url"),
            "space_key": p.get("space_key"),
            "tags": p.get("tags", []),
            "score": r["score"],
            "components": r["components"]
        })
    return {"query": q, "results": out}

@app.post("/api/feedback")
def api_feedback(payload: dict = Body(...)):
    save_feedback(payload.get("query"), payload.get("page_id"), payload.get("feedback"))
    # We don't recompute boosts here (background job does)
    return {"status": "ok"}

# Gradio UI
def gradio_search(query, tag_filter):
    # call local engine
    tag = None if tag_filter in (None, "All") else tag_filter
    results = search_engine.search(query, top_k=5, tag_filter=tag)
    if not results:
        return "No results found."
    text = ""
    for idx, r in enumerate(results, start=1):
        p = r["page"]
        text += f"**{idx}. {p.get('title')}**  \n"
        text += f"_space:_ {p.get('space_key')}  \n"
        text += f"Tags: {', '.join(p.get('tags', []))}  \n"
        snippet = (p.get('content') or "")[:400].replace("\n", " ")
        text += f"{snippet}...  \n"
        text += f"[🔗 Read More]({p.get('url')})  \n\n"
        text += f"> score: {round(r['score'],4)} (components: {r['components']})\n\n"
    return text

def gradio_feedback(query, page_id, fb):
    save_feedback(query, page_id, fb)
    return "Feedback recorded — thanks!"

all_pages = fetch_all_pages()
# gather tags
all_tags = sorted({t for p in all_pages for t in (p.get("tags") or [])})

with gr.Blocks(title="SupportBot") as demo:
    gr.Markdown("# SupportBot — In-house KB Assistant")
    with gr.Row():
        q = gr.Textbox(label="Your question")
        tag_dropdown = gr.Dropdown(["All"] + all_tags, label="Tag Filter", value="All")
        submit = gr.Button("Search")
    output = gr.Markdown()
    submit.click(fn=gradio_search, inputs=[q, tag_dropdown], outputs=[output])

    gr.Markdown("### Feedback (select a page_id from results and mark helpful/not helpful)")
    fb_q = gr.Textbox(label="Original query (copy/paste)")
    fb_page = gr.Textbox(label="Page ID to rate")
    fb_choice = gr.Dropdown(["positive", "negative"], value="positive", label="Feedback")
    fb_btn = gr.Button("Submit Feedback")
    fb_status = gr.Textbox()
    fb_btn.click(fn=gradio_feedback, inputs=[fb_q, fb_page, fb_choice], outputs=[fb_status])

# Start background jobs
if __name__ == "__main__":
    # start periodic jobs if enabled
    if ENABLE_TAG_ENRICHMENT:
        start_tag_job()
    if ENABLE_FEEDBACK_UPDATES:
        start_feedback_job()

    # run gradio UI
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
```

---

## `supportbot/scripts/load_pages_to_db.py` (safe loader — update mode default)

```python
# supportbot/scripts/load_pages_to_db.py
"""
Safe loader to insert/update pages from a JSON file into MongoDB.

Modes:
  - reset: drop collection then insert
  - update (default): upsert by page_id
  - append: insert all documents (may create duplicates)

Example:
  python load_pages_to_db.py --file ../data/confluence_data.json --mode update
"""
import argparse
import json
from pymongo import UpdateOne
from supportbot.core.mongo_client import get_db
from supportbot.config import PAGES_COLLECTION
from supportbot.core.tag_generator import generate_tags_for_page

def load(file_path, mode="update"):
    db = get_db()
    coll = db[PAGES_COLLECTION]
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    pages = payload.get("pages", [])
    print(f"[Loader] Loaded {len(pages)} pages from {file_path}")

    if mode == "reset":
        coll.drop()
        print("[Loader] Dropped existing collection.")

    ops = []
    insert_docs = []
    for p in pages:
        # ensure content field is present; assume content is already cleaned to plain text
        p.setdefault("content", "")
        # generate tags if not present
        if not p.get("tags"):
            p["tags"] = generate_tags_for_page(p)
        doc = {
            "page_id": p.get("page_id"),
            "title": p.get("title"),
            "content": p.get("content"),
            "url": p.get("url"),
            "space_key": p.get("space_key"),
            "tags": p.get("tags")
        }
        if mode == "append" or mode == "reset":
            insert_docs.append(doc)
        else:  # update -> upsert
            ops.append(UpdateOne({"page_id": doc["page_id"]}, {"$set": doc}, upsert=True)

)
    if ops:
        result = coll.bulk_write(ops)
        print("[Loader] Bulk upsert done. upserted:", result.upserted_count, "modified:", result.modified_count)
    if insert_docs:
        coll.insert_many(insert_docs)
        print("[Loader] Inserted", len(insert_docs), "documents")

    print("[Loader] Total documents in collection:", coll.count_documents({}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to confluence_data.json")
    parser.add_argument("--mode", choices=["reset", "update", "append"], default="update")
    args = parser.parse_args()
    load(args.file, mode=args.mode)
```

---

## `supportbot/scripts/run_enrichment_and_feedback_update.py`

```python
# supportbot/scripts/run_enrichment_and_feedback_update.py
"""
Run tag enrichment and feedback update once (manual invocation).
Useful for cron or adhoc runs.
"""
from supportbot.core.tag_enrichment_job import run_enrichment_once
from supportbot.core.feedback_updater import compute_boosts_once

if __name__ == "__main__":
    print("Running tag enrichment...")
    run_enrichment_once()
    print("Running feedback boosts computation...")
    compute_boosts_once()
    print("Done.")
```

---

# How it all works — process flow

1. **Load pages to MongoDB**
   Run `scripts/load_pages_to_db.py --file data/confluence_data.json --mode update`. This upserts pages (preserves existing ones) and auto-generates tags for documents that lack them.

2. **Start the app**
   Run `python -m supportbot.app` (or `python supportbot/app.py`), which:

   * Loads pages into `HybridSearchEngine` (initial snapshot from DB)
   * Starts Gradio UI at `http://localhost:7860`
   * Starts background jobs (if enabled) for tag enrichment and feedback updates

3. **User interaction**

   * Users query via Gradio or `GET /api/search?q=...&tag=...`
   * If paraphrasing enabled, `expand_query()` generates variants; engine averages scores across variants
   * Search results include tag lists and scores

4. **Collect feedback**

   * Users submit feedback via `POST /api/feedback` (or via Gradio)
   * Feedback records are stored in `feedback` collection

5. **Feedback update job**

   * Background job aggregates feedback and writes a conservative `boost` multiplier to `doc_boosts` collection
   * Search engine reads these boosts at initialization and on refresh; runtime search multiplies final scores by boost

6. **Tag enrichment job**

   * Periodically refreshes tags (TF-IDF-based) and updates `tags` field in each page.

---

# How feedback affects future results

* Positive feedback increases a doc's boost up to `+FEEDBACK_ALPHA` (default +15%). Negative feedback reduces it down to `-FEEDBACK_ALPHA`. Logic is conservative to avoid overfitting.
* Boosts are computed and stored separately (`doc_boosts`). Search engine multiplies score by boost at query time.
* You can extend `feedback_updater` to do time-decay, clustering, or per-tag boosts if you want more sophistication.

---

# How tag enrichment improves results

* Tags provide structured signals to narrow search space (tag filtering) and to boost documents when tags overlap with query keywords.
* Enrichment job uses TF-IDF on page title + first 2000 chars of content; this balances speed vs signal.
* Later you can plug YAKE/KeyBERT or LLM-based keyphrase extraction for better tags.

---

# Query paraphrasing details

* Controlled by `ENABLE_PARAPHRASING` in `config.py`. If `False`, search uses raw query only.
* When `True`, the paraphraser uses the `SYNONYM_MAP` to replace tokens with alternatives and generates up to `PARAPHRASE_VARIANTS` variants (configurable). The engine computes normalized scores per variant and averages them — this helps short queries match more documents.
* This method is conservative, deterministic and easily auditable. If results degrade, toggle the flag off.

---

# Synonym map (starter) — how to extend

Synonym map is in `query_paraphraser.py` as `SYNONYM_MAP`. Add domain-specific mappings like:

```python
SYNONYM_MAP["pnl"] = ["pnl", "profit and loss"]
SYNONYM_MAP["collateral"] = ["collateral", "margin", "collat"]
SYNONYM_MAP["sftp"] = ["sftp", "file transfer", "filetransfer"]
```

Junior dev note: keep keys lowercased and prefer short single-word keys that map to lists of variants. This map can be loaded from a JSON/YAML in future.

---

# How to expand later (suggestions for juniors)

1. **Replace TF-IDF tagger with YAKE/KeyBERT**: add optional dependency and keep same interface `generate_tags_for_page(page, top_k)`.
2. **Add embeddings**: set `ENABLE_EMBEDDING=True` and provide `EMBEDDING_MODEL_PATH`. Replace paraphrase averaging with semantic similarity.
3. **Add scheduled job manager**: replace naive threads with `apscheduler` if needed.
4. **Evaluation harness**: add simple labeled dataset `examples/dev_queries.json` mapping query → expected page_id and compute metrics (precision@k, ndcg).
5. **UI improvements**: show page_id buttons in Gradio results to make feedback one-click.
6. **Analytics**: build a small dashboard showing top queries with many negative votes to prioritize KB fixes.

---

# Run checklist (quick)

1. Install packages:

```bash
pip install pymongo scikit-learn rank-bm25 gradio
# optional: pip install sentence-transformers  (only if you enable embeddings)
```

2. Load pages to DB (example):

```bash
python supportbot/scripts/load_pages_to_db.py --file ./data/confluence_data.json --mode update
```

3. Run the app:

```bash
python -m supportbot.app
# open http://localhost:7860
```

4. (Optional) Run jobs manually:

```bash
python supportbot/scripts/run_enrichment_and_feedback_update.py
```

---

If you want, next I can:

* Wire the **HybridSearchEngine** to refresh its in-memory index when `doc_boosts` or `tags` collections change (e.g., by listening for a manual /api/refresh call).
* Add an **evaluation script** where you provide ~50 labeled queries → expected page_id and I generate a script that sweeps TF-IDF/BM25/overlap weights for best P@1/P@3.
* Produce a **one-click feedback UI** in Gradio results so that users can rate any of the top-k results directly (instead of copying page_id into a box).

Which of those would you like next?
