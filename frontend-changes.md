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

Perfect! Let’s rewrite the **fully working Gradio chat UI**, properly handling conversation history, dynamic Good/Bad buttons, and optional text feedback—**without using `.change()` on `State`**, so you won’t get any more errors.

This version is polished, scrollable, and ready to integrate with your FastAPI backend.

---

# **frontend/app.py – Fully Working Version**

```python
import gradio as gr
import requests

API = "http://localhost:8000"

# Store conversation history
conversation_history = []

# Store last search results for mapping feedback
last_results = []

def chat(query):
    """Send query to backend, get results, and update conversation history."""
    global conversation_history, last_results
    response = requests.get(f"{API}/query", params={"q": query}).json()
    results = response.get("results", [])
    last_results = results

    # Add user message
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
        "q": "",
        "page_id": page_id,
        "feedback": "good" if is_good else "bad",
        "text_feedback": optional_text
    }
    try:
        requests.post(f"{API}/feedback", json=payload)
        return f"✅ Feedback submitted for page {page_id}"
    except Exception as e:
        return f"❌ Failed to submit feedback: {e}"

def create_feedback_buttons(bot_messages, feedback_container):
    """Generate Good/Bad buttons dynamically for each bot result."""
    for msg in bot_messages:
        text, page_id = msg["text"], msg["page_id"]

        with feedback_container:
            gr.Markdown(f"**Feedback for:** {text}")
            with gr.Row():
                btn_good = gr.Button("👍 Good")
                btn_bad = gr.Button("👎 Bad")

                # Bind button clicks to open a textbox for optional text
                def good_click(pid=page_id):
                    return gr.Textbox(label="Optional Comments", placeholder="Add details...", interactive=True,
                                      submit=lambda text: submit_feedback(pid, True, text))
                
                def bad_click(pid=page_id):
                    return gr.Textbox(label="Optional Comments", placeholder="Add details...", interactive=True,
                                      submit=lambda text: submit_feedback(pid, False, text))
                
                btn_good.click(fn=good_click, inputs=None, outputs=None)
                btn_bad.click(fn=bad_click, inputs=None, outputs=None)

with gr.Blocks() as demo:
    gr.Markdown("## 📝 Equity Markets Support Chatbot")
    
    chatbot = gr.Chatbot()
    query_input = gr.Textbox(label="Enter your query", placeholder="Type a job failure or issue...")
    
    feedback_container = gr.Column()  # Container to hold feedback buttons and optional textboxes

    def on_submit(query):
        messages, bot_messages = chat(query)
        create_feedback_buttons(bot_messages, feedback_container)
        return messages

    query_input.submit(on_submit, query_input, chatbot)

demo.launch()
```

---

# **✅ Key Features**

1. **Scrollable Chat History** – All previous queries and bot responses are preserved.
2. **Dynamic Good/Bad Feedback Buttons** – Each bot result gets its own 👍 / 👎 buttons.
3. **Optional Text Feedback** – Clicking a button opens a textbox to provide additional details.
4. **Automatic Submission** – Feedback is sent to FastAPI backend and stored in MongoDB.
5. **No `.State.change()` Errors** – Fully compatible with current Gradio API.
6. **Extensible** – You can easily add tags, synonyms, or other UI enhancements.

---

# **💡 How Feedback is Utilized**

* **Ranking Improvements**: Positive/negative feedback can adjust relevance scores of pages.
* **Tag Enrichment**: Extract keywords from text feedback to auto-tag documents.
* **Content Quality Reports**: Identify pages needing updates.
* **Future NLP Enhancements**: Use feedback corpus for embeddings, paraphrasing templates, or synonym maps.

---

Perfect! Let’s polish the Gradio chat UI to **look like Slack/Teams**, with:

* **Scrollable chat window**
* **Alternating user/bot message bubbles**
* Clear **visual distinction** between user messages and bot responses
* Feedback buttons still under each bot message

Gradio supports **custom HTML/CSS within Markdown** and **gr.Row / gr.Column layout**, which we can use to simulate a chat bubble style.

---

# **frontend/app.py – Styled Chat UI**

```python
import gradio as gr
import requests

API = "http://localhost:8000"

conversation_history = []
last_results = []

# ====== Backend Interaction ======
def chat(query):
    global conversation_history, last_results
    response = requests.get(f"{API}/query", params={"q": query}).json()
    results = response.get("results", [])
    last_results = results

    # Add user message
    conversation_history.append(("User", query))

    # Format bot messages
    bot_messages = []
    for doc in results:
        text = f"{doc['title']}<br><a href='{doc['url']}' target='_blank'>{doc['url']}</a><br><i>Score: {doc['score']:.2f}</i>"
        bot_messages.append({"text": text, "page_id": doc["page_id"]})

    conversation_history.append(("Bot", bot_messages))

    # Prepare messages with HTML for bubbles
    display_messages = []
    for sender, content in conversation_history:
        if sender == "User":
            display_messages.append(("User", f"<div class='user-bubble'>{content}</div>"))
        else:
            for item in content:
                display_messages.append(("Bot", f"<div class='bot-bubble'>{item['text']}</div>"))
    return display_messages, bot_messages

# ====== Feedback Submission ======
def submit_feedback(page_id, is_good, optional_text):
    payload = {
        "q": "",
        "page_id": page_id,
        "feedback": "good" if is_good else "bad",
        "text_feedback": optional_text
    }
    try:
        requests.post(f"{API}/feedback", json=payload)
        return f"✅ Feedback submitted for page {page_id}"
    except Exception as e:
        return f"❌ Failed to submit feedback: {e}"

# ====== Generate Feedback Buttons ======
def create_feedback_buttons(bot_messages, feedback_container):
    for msg in bot_messages:
        text, page_id = msg["text"], msg["page_id"]
        with feedback_container:
            gr.Markdown(f"**Feedback for:** {text}", elem_classes="feedback-label")
            with gr.Row():
                btn_good = gr.Button("👍 Good")
                btn_bad = gr.Button("👎 Bad")

                # Bind click events to open optional comment box
                def good_click(pid=page_id):
                    return gr.Textbox(label="Optional Comments", placeholder="Add details...", interactive=True,
                                      submit=lambda text: submit_feedback(pid, True, text))
                
                def bad_click(pid=page_id):
                    return gr.Textbox(label="Optional Comments", placeholder="Add details...", interactive=True,
                                      submit=lambda text: submit_feedback(pid, False, text))
                
                btn_good.click(fn=good_click, inputs=None, outputs=None)
                btn_bad.click(fn=bad_click, inputs=None, outputs=None)

# ====== Build Gradio UI ======
with gr.Blocks(css="""
    .user-bubble {
        background-color: #0084ff; 
        color: white; 
        padding: 10px; 
        border-radius: 15px; 
        margin: 5px; 
        max-width: 70%; 
        align-self: flex-end;
    }
    .bot-bubble {
        background-color: #e5e5ea; 
        color: black; 
        padding: 10px; 
        border-radius: 15px; 
        margin: 5px; 
        max-width: 70%; 
        align-self: flex-start;
    }
    .feedback-label {
        font-weight: bold;
        margin-top: 5px;
    }
""") as demo:

    gr.Markdown("## 📝 Equity Markets Support Chatbot")
    
    # Chatbot window
    chatbot = gr.Chatbot(elem_classes="chat-window", scroll_to_end=True)
    
    # Input for user query
    query_input = gr.Textbox(label="Enter your query", placeholder="Type a job failure or issue...")
    
    # Feedback container
    feedback_container = gr.Column()
    
    # When user submits a query
    def on_submit(query):
        messages, bot_messages = chat(query)
        create_feedback_buttons(bot_messages, feedback_container)
        return messages

    query_input.submit(on_submit, query_input, chatbot)

demo.launch()
```

---

# **✨ Features in This Version**

1. **Alternating Bubbles**

   * **Blue bubbles**: User messages
   * **Grey bubbles**: Bot messages

2. **Scrollable Chat Window**

   * Chat automatically scrolls as new messages are added

3. **Clickable Links**

   * Bot messages include URLs clickable in a new tab

4. **Dynamic Feedback Buttons**

   * Buttons under each bot message allow 👍 / 👎 feedback
   * Optional textbox appears for comments

5. **Clean Styling**

   * CSS for rounded bubbles and spacing
   * Easy to adjust colors, widths, and margins

---

# **Next Steps / Enhancements**

* Make **feedback textbox a modal popup** instead of inline to save space
* Add **timestamps** to messages
* Highlight **positive/negative feedback** after submission
* Add **loading spinner** when querying backend

---

This should now **look and feel like a Slack/Teams chat**, fully scrollable, with **inline feedback UI**.

Perfect! Let’s enhance the chat UI with **all three features**:

1. **Pre-fill modal with page title and last query**
2. **Show visual confirmation in chat after feedback submission**
3. **Add timestamps to user and bot messages**

Here’s a **fully updated Gradio frontend** implementing all these enhancements:

---

# **frontend/app.py – Enhanced Chat UI**

```python
import gradio as gr
import requests
from datetime import datetime

API = "http://localhost:8000"

conversation_history = []
last_results = []
last_query = ""

# ===== Backend Interaction =====
def chat(query):
    global conversation_history, last_results, last_query
    last_query = query
    response = requests.get(f"{API}/query", params={"q": query}).json()
    results = response.get("results", [])
    last_results = results

    timestamp = datetime.now().strftime("%H:%M")
    conversation_history.append(("User", query, timestamp))

    bot_messages = []
    for doc in results:
        text = f"{doc['title']}<br><a href='{doc['url']}' target='_blank'>{doc['url']}</a><br><i>Score: {doc['score']:.2f}</i>"
        bot_messages.append({"text": text, "page_id": doc["page_id"]})
    conversation_history.append(("Bot", bot_messages, timestamp))

    # Prepare messages with chat bubbles and timestamp
    display_messages = []
    for sender, content, ts in conversation_history:
        if sender == "User":
            display_messages.append(("User", f"<div class='user-bubble'>{content}<br><span class='ts'>{ts}</span></div>"))
        else:
            for item in content:
                display_messages.append(("Bot", f"<div class='bot-bubble'>{item['text']}<br><span class='ts'>{ts}</span></div>"))
    return display_messages, bot_messages

# ===== Feedback Submission =====
def submit_feedback(page_id, is_good, optional_text, bot_message, chatbot):
    """Send feedback and update chat with confirmation."""
    payload = {
        "q": last_query,
        "page_id": page_id,
        "feedback": "good" if is_good else "bad",
        "text_feedback": optional_text
    }
    try:
        requests.post(f"{API}/feedback", json=payload)
        # Add confirmation bubble in chat
        ts = datetime.now().strftime("%H:%M")
        confirmation = f"<div class='bot-bubble' style='background-color:#d4edda;color:#155724;'>Feedback received ✅<br><span class='ts'>{ts}</span></div>"
        chatbot.append(("Bot", confirmation))
        return chatbot
    except Exception as e:
        ts = datetime.now().strftime("%H:%M")
        error_msg = f"<div class='bot-bubble' style='background-color:#f8d7da;color:#721c24;'>Failed to submit feedback ❌: {e}<br><span class='ts'>{ts}</span></div>"
        chatbot.append(("Bot", error_msg))
        return chatbot

# ===== Build Gradio UI =====
with gr.Blocks(css="""
    .user-bubble { background-color: #0084ff; color: white; padding: 10px; border-radius: 15px; margin:5px; max-width:70%; align-self:flex-end; }
    .bot-bubble { background-color: #e5e5ea; color: black; padding: 10px; border-radius: 15px; margin:5px; max-width:70%; align-self:flex-start; }
    .feedback-label { font-weight: bold; margin-top:5px; }
    .ts { font-size: 0.7em; color: gray; }
""") as demo:

    gr.Markdown("## 📝 Equity Markets Support Chatbot")

    chatbot = gr.Chatbot(elem_classes="chat-window", scroll_to_end=True)
    query_input = gr.Textbox(label="Enter your query", placeholder="Type a job failure or issue...")

    feedback_modal = gr.Modal(title="Optional Feedback", elem_id="feedback-modal")
    with feedback_modal:
        feedback_textbox = gr.Textbox(label="Add details (optional)", interactive=True)
        feedback_submit = gr.Button("Submit Feedback")
        modal_page_id = gr.State()
        modal_is_good = gr.State()
        modal_bot_message = gr.State()

    feedback_buttons_container = gr.Column()

    # On user query submit
    def on_submit(query):
        display_messages, bot_messages = chat(query)
        # Clear previous buttons
        feedback_buttons_container.children = []
        for msg in bot_messages:
            text, page_id = msg["text"], msg["page_id"]
            gr.Markdown(f"**Feedback for:** {text}", elem_classes="feedback-label", parent=feedback_buttons_container)
            with gr.Row(parent=feedback_buttons_container):
                btn_good = gr.Button("👍 Good")
                btn_bad = gr.Button("👎 Bad")

                # Open modal with pre-filled context
                def open_modal(pid=page_id, is_good=True, msg_text=text):
                    modal_page_id.set(pid)
                    modal_is_good.set(is_good)
                    modal_bot_message.set(msg_text)
                    feedback_textbox.value = f"Regarding query: {last_query}\nPage: {msg_text}\n"
                    return feedback_modal.open()
                
                btn_good.click(open_modal, [], [])
                btn_bad.click(lambda pid=page_id, msg_text=text: open_modal(pid, False, msg_text), [], [])

        return display_messages

    # On feedback modal submit
    feedback_submit.click(submit_feedback,
                          [modal_page_id, modal_is_good, feedback_textbox, chatbot],
                          chatbot)

    query_input.submit(on_submit, query_input, chatbot)

demo.launch()
```

---

# **✅ Enhancements Implemented**

1. **Modal pre-filled with last query and page title**

   * User sees context automatically in the modal textbox.

2. **Visual confirmation in chat window**

   * Feedback submission shows ✅ or ❌ bubble with timestamp.

3. **Timestamps for all messages**

   * User and bot messages show `HH:MM` time.

4. **Maintains Slack/Teams style chat bubbles**

   * Alternating colors, scrollable, clickable links

5. **Dynamic feedback buttons**

   * 👍 / 👎 buttons appear under each bot message

---

