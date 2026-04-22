# AutoStream AI Agent
### ServiceHive x Inflx — Machine Learning Intern Assignment

A conversational AI agent built with LangGraph that handles product inquiries and captures qualified leads for AutoStream, a fictional SaaS video editing platform.

---
## Demo Video

[Watch the demo on YouTube](youtu.be/8r6bV5lNm2s)

## How to Run Locally

### Prerequisites
- Python 3.9+
- A Google AI Studio API key (Gemini 2.5 Flash)

### 1. Clone and set up

```bash
git clone <your-repo-url>
cd autostream-agent
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Set your API key

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your-key-here
```

### 3. Run the agent

```bash
python agent.py
```

You'll enter an interactive CLI chat. Type `exit` to quit.

---

## Architecture

### Why LangGraph?

LangGraph was chosen over AutoGen because it gives explicit control over the conversation flow through a state machine. For a lead-capture workflow, this matters — we need to guarantee that `mock_lead_capture()` fires only after all three fields are collected, and never before. LangGraph's conditional edges make this easy to enforce and easy to read.

### How the Agent Works

The graph has three nodes:

**classify_intent** — Every user message enters here first. The LLM classifies it as `greeting`, `inquiry`, or `high_intent`. This is a focused, single-purpose call with a constrained prompt so the output is always predictable.

**respond** — Handles greetings and product questions. The full knowledge base (pricing, policies, FAQ) is injected into the system prompt, giving the LLM accurate, grounded answers. For a knowledge base this small (under 2KB), in-context RAG is faster and more reliable than chunking with a vector database.

**collect_lead** — Triggered when high intent is detected, or when lead collection is already in progress. A secondary LLM call extracts name, email, and platform from the user's free-form text and stores them in the agent state. Once all three fields are present, `mock_lead_capture()` is called exactly once.

### State Management

`AgentState` is a TypedDict that travels through every node in the graph. It holds the full message history, the current intent, lead fields, and status flags. This means the agent retains context across 5-6+ conversation turns without any external memory store.

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, the following setup is needed:

**1. WhatsApp Business API** — Register a phone number on Meta for Developers and enable the WhatsApp Business API.

**2. Webhook Endpoint** — Set up a FastAPI or Flask server with a `POST /webhook` route. Meta sends incoming messages to this URL.

```python
@app.post("/webhook")
async def whatsapp_webhook(payload: dict):
    sender = payload["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
    text   = payload["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]
    
    # Load existing state for this user from Redis or DB
    state = get_user_state(sender)
    state["messages"].append(HumanMessage(content=text))
    
    # Run the agent
    state = agent.invoke(state)
    
    # Save updated state back
    save_user_state(sender, state)
    
    # Send reply via WhatsApp API
    reply = get_last_ai_message(state)
    send_whatsapp_message(sender, reply)
```

**3. Session Persistence** — The in-memory AgentState dict needs to be replaced with a Redis or PostgreSQL store keyed by the sender's WhatsApp number. This keeps conversation state alive across messages and server restarts.

**4. Webhook Verification** — WhatsApp requires a `GET /webhook` handler that echoes back the `hub.challenge` token during initial setup.

**5. Hosting** — Deploy on any platform that supports HTTPS (Railway, Render, AWS EC2). Meta requires a valid SSL certificate on the webhook URL.

**6. Lead Capture** — Replace `mock_lead_capture()` with a real CRM call such as HubSpot or Salesforce, passing the same `name`, `email`, and `platform` fields collected by the agent.

---

## Project Structure

```
autostream-agent/
├── agent.py                   # Agent logic, graph definition, chat loop
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── knowledge_base/
    └── autostream_kb.json     # Product data used for RAG
```

---

## Evaluation Criteria

| Criterion | Implementation |
|-----------|---------------|
| Agent reasoning and intent detection | classify_intent node with constrained LLM prompt |
| Correct use of RAG | Knowledge base injected into system prompt at respond node |
| Clean state management | AgentState TypedDict carries full history and lead fields |
| Proper tool calling logic | collect_lead node gates mock_lead_capture() on all 3 fields |
| Code clarity and structure | Single-file agent with clearly labelled sections |
| Real-world deployability | WhatsApp webhook design documented above |

---

## Model Used

Gemini 2.5 Flash via Google AI Studio — chosen for its strong reasoning capability and generous free tier, which makes it practical for development and testing without upfront cost.

---

## Sample Conversation

```
You: Hi there
infix: Hey! I am infix, AutoStream's assistant. I help content creators
      automate their video editing. What can I help you with today?

You: What is the difference between Basic and Pro?
infix: Basic is $29/month and gives you 10 videos per month at 720p.
      Pro is $79/month with unlimited videos, 4K resolution, AI captions,
      and 24/7 support. If you are creating content regularly, Pro is
      the better fit.

You: That sounds good, I want to try the Pro plan for my YouTube channel.
infix: Great, let me get you set up. Could I start with your name?

You: I am Piyush Gupta
infix: Nice to meet you Piyush. What is your email address?

You: piyush@example.com
infix: And just to confirm, YouTube is your main creator platform?

You: Yes

==================================================
Lead captured: Piyush Gupta | piyush@example.com | YouTube
==================================================

infix: All done Piyush. Our team will be in touch shortly. Welcome to AutoStream.
```
