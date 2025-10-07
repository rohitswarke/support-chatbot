Got it! Let’s completely redesign the **Gradio frontend** to make it look like a **chat interface with message history**, **inline Good/Bad buttons** for each result, and **optional text feedback** in a popup. Also, I’ll explain how to **utilize text feedback** from users.

---

# **1️⃣ Frontend Design Changes**

### Goals:

1. Maintain **conversation history**.
2. Display **search results** per query as messages.
3. For each result, show **👍 / 👎** buttons instead of asking the user to type the index.
4. When a user clicks 👍/👎, show a **popup** for optional textual feedback.
5. Automatically send **query, page_id, and feedback** to FastAPI backend.
6. Store **textual feedback** in MongoDB for **future evaluation** and optionally **improve ranking/labels**.

---

### **Updated Gradio UI**

```python
# frontend/app.py
import gradio as gr
import requests

API = "http://localhost:8000"

# Store conversation history
conversation_history = []

def chat(query):
    global conversation_history
    response = requests.get(f"{API}/query", params={"q": query}).json()
    results = response.get("results", [])
    
    messages = []
    # Add user query to conversation
    conversation_history.append(("User", query))
    messages.append(("User", query))
    
    # Format results with emojis for feedback
    result_texts = []
    for doc in results:
        text = f"**{doc['title']}**\n{doc['url']}\nScore: {doc['score']:.2f}"
        # Add placeholder for Good/Bad feedback
        result_texts.append((text, doc["page_id"]))
    
    # Add bot response
    conversation_history.append(("Bot", result_texts))
    
    # Prepare display
    display_messages = []
    for sender, content in conversation_history:
        if sender == "User":
            display_messages.append((sender, content))
        elif sender == "Bot":
            for text, page_id in content:
                display_messages.append(("Bot", text))
    
    return display_messages, result_texts

# Handle feedback click
def feedback_click(page_id, is_good):
    # Show popup for optional textual feedback
    def submit_text_feedback(text_feedback):
        payload = {
            "q": "",  # Optional: can store last query or pass separately
            "page_id": page_id,
            "feedback": "good" if is_good else "bad",
            "text_feedback": text_feedback
        }
        requests.post(f"{API}/feedback", json=payload)
        return f"Feedback submitted for page {page_id}"
    return gr.Textbox(label="Optional Comments", placeholder="Add details (optional)...", interactive=True, submit=submit_text_feedback)

with gr.Blocks() as demo:
    chat_history = gr.Chatbot()
    query_input = gr.Textbox(label="Enter your query")
    
    result_state = gr.State()
    
    def on_submit(query):
        display_messages, results = chat(query)
        # Update chat history
        chat_messages = []
        for sender, content in display_messages:
            chat_messages.append((sender, content))
        # Save result_state for feedback buttons
        return chat_messages, results

    query_input.submit(on_submit, query_input, [chat_history, result_state])
    
    # Feedback buttons: dynamically generated per result
    with gr.Column():
        def generate_feedback_buttons(results):
            buttons = []
            for text, page_id in results:
                btn_good = gr.Button("👍 Good")
                btn_bad = gr.Button("👎 Bad")
                btn_good.click(lambda pid=page_id: feedback_click(pid, True), None, None)
                btn_bad.click(lambda pid=page_id: feedback_click(pid, False), None, None)
                buttons.append((btn_good, btn_bad))
            return buttons
        
        result_state.change(generate_feedback_buttons, result_state, None)

demo.launch()
```

---

# **2️⃣ Backend Feedback Handling**

Modify **FastAPI `/feedback` endpoint** to accept optional text:

```python
from fastapi import Body
from core.feedback_handler import save_feedback

@app.post("/feedback")
def feedback(payload: dict = Body(...)):
    """
    payload: {
        "q": str,                # user query
        "page_id": str,
        "feedback": "good"/"bad",
        "text_feedback": str (optional)
    }
    """
    save_feedback(payload.get("q", ""), payload["page_id"], payload["feedback"], payload.get("text_feedback", ""))
    return {"status":"success"}
```

Update **core/feedback_handler.py**:

```python
def save_feedback(query, page_id, feedback_value, text_feedback=""):
    feedback_collection.insert_one({
        "query": query,
        "page_id": page_id,
        "feedback": feedback_value,
        "text_feedback": text_feedback,
        "timestamp": datetime.utcnow()
    })
    # Update ranking
    adjust_ranking_weights(page_id, feedback_value)
```

---

# **3️⃣ How We Can Use Text Feedback**

Textual feedback can be used in multiple ways:

1. **Future Query Ranking Adjustments**

   * Parse text for additional keywords or job names.
   * Boost documents that match frequently mentioned keywords.

2. **Tag Enrichment**

   * Extract common terms from textual feedback to automatically **add tags** to pages.

3. **Quality Evaluation / Reports**

   * Evaluate **percentage of positive/negative responses** per page.
   * Highlight pages needing content improvement.

4. **Training Future LLM or Search Models**

   * Feedback corpus can be used to fine-tune embeddings, improve synonym maps, or expand paraphrasing templates.

---

# **✅ Benefits of this Approach**

* Chat interface with full **conversation history**.
* Inline **Good/Bad buttons** → avoids user errors selecting index.
* **Optional comments** allow richer feedback.
* Feedback used to **boost ranking** and **improve tagging**.
* Easy to **expand** with NLP processing on text feedback later.

---


Perfect! Let’s create a **polished Gradio chat interface** that looks and feels like a real chat app for your support team, with:

* **Scrollable chat history**
* **Dynamic Good/Bad buttons for each bot result**
* **Popup/modal for optional textual feedback**
* Fully integrated **backend feedback submission**

---

# **frontend/app.py – Final Polished Version**

```python
import gradio as gr
import requests

API = "http://localhost:8000"

# Store conversation history
conversation_history = []

# Store last search results for mapping feedback
last_results = []

def chat(query):
    global conversation_history, last_results
    response = requests.get(f"{API}/query", params={"q": query}).json()
    results = response.get("results", [])

    last_results = results

    # Add user query
    conversation_history.append(("User", query))

    # Format bot messages
    bot_messages = []
    for doc in results:
        text = f"**{doc['title']}**\n{doc['url']}\nScore: {doc['score']:.2f}"
        bot_messages.append({"text": text, "page_id": doc["page_id"]})

    conversation_history.append(("Bot", bot_messages))

    # Prepare messages for Gradio Chatbot display
    display_messages = []
    for sender, content in conversation_history:
        if sender == "User":
            display_messages.append((sender, content))
        else:
            for item in content:
                display_messages.append(("Bot", item["text"]))
    return display_messages, bot_messages

def submit_feedback(page_id, is_good, optional_text):
    """Send feedback to backend API."""
    payload = {
        "q": "",  # Optionally store last query
        "page_id": page_id,
        "feedback": "good" if is_good else "bad",
        "text_feedback": optional_text
    }
    try:
        requests.post(f"{API}/feedback", json=payload)
        return f"✅ Feedback submitted for page {page_id}"
    except Exception as e:
        return f"❌ Failed to submit feedback: {e}"

def feedback_popup(page_id, is_good):
    """Return a textbox for optional textual feedback."""
    return gr.Textbox(label="Optional Comments", placeholder="Add details (optional)...",
                      interactive=True, submit=lambda text: submit_feedback(page_id, is_good, text))

with gr.Blocks() as demo:
    gr.Markdown("## 📝 Equity Markets Support Chatbot")

    chatbot = gr.Chatbot()
    query_input = gr.Textbox(label="Enter your query", placeholder="Type a job failure or issue...")
    
    feedback_container = gr.Column()
    
    result_state = gr.State()

    # On user submitting a query
    def on_submit(query):
        display_messages, bot_messages = chat(query)
        return display_messages, bot_messages

    query_input.submit(on_submit, query_input, [chatbot, result_state])

    # Dynamically generate feedback buttons under bot messages
    with gr.Row():
        def create_feedback_buttons(bot_messages):
            btns = []
            for msg in bot_messages:
                text, page_id = msg["text"], msg["page_id"]
                btn_good = gr.Button(f"👍 Good")
                btn_bad = gr.Button(f"👎 Bad")
                # Bind events to open optional comment box
                btn_good.click(lambda pid=page_id: feedback_popup(pid, True), None, feedback_container)
                btn_bad.click(lambda pid=page_id: feedback_popup(pid, False), None, feedback_container)
                btns.append((btn_good, btn_bad))
            return btns

        result_state.change(create_feedback_buttons, result_state, None)

demo.launch()
```

---

# **✨ Features**

1. **Full Chat History**

   * Users see all previous queries and bot responses.

2. **Dynamic Feedback Buttons**

   * Each bot result gets **👍 / 👎** buttons.
   * No need to type result index.

3. **Optional Text Feedback via Popup**

   * Clicking a button opens a **textbox** for comments.
   * Text feedback is submitted along with good/bad rating.

4. **Automatic Backend Submission**

   * Feedback is sent to `/feedback` endpoint and stored in MongoDB.
   * Can be used for **ranking boosts, tag enrichment, and evaluation reports**.

5. **Extensible**

   * Easily add more actions, styling, or chat components.

---

# **💡 Notes on Utilizing Text Feedback**

* **Ranking Boosts**: Use keywords in feedback to increase page relevance.
* **Tag Enrichment**: Extract frequently mentioned terms to auto-tag pages.
* **Content Improvement**: Evaluate pages with repeated negative feedback.
* **Future NLP Models**: Feedback corpus can improve paraphrasing, embeddings, or synonym mapping.

---

This version should **look clean, act like a real chat**, and handle feedback robustly.
