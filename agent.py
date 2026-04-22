
import json
import os
import re
from typing import Annotated, TypedDict, Optional
from pathlib import Path

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from dotenv import load_dotenv
load_dotenv()

# KNOWLEDGE BASE LOADER


def load_knowledge_base() -> str:
    """Load the AutoStream knowledge base and return it as a formatted string."""
    kb_path = Path(__file__).parent / "knowledge_base" / "autostream_kb.json"
    with open(kb_path, "r") as f:
        kb = json.load(f)

    kb_text = f"""
=== AutoStream Knowledge Base ===

Product: {kb['product']}
Description: {kb['description']}

--- PRICING PLANS ---

Basic Plan: ${kb['plans']['basic']['price_monthly']}/month
  - {kb['plans']['basic']['videos_per_month']} videos/month
  - {kb['plans']['basic']['resolution']} resolution
  - Features: {', '.join(kb['plans']['basic']['features'])}
  - Support: {kb['plans']['basic']['support']}

Pro Plan: ${kb['plans']['pro']['price_monthly']}/month
  - {kb['plans']['pro']['videos_per_month']} videos
  - {kb['plans']['pro']['resolution']} resolution
  - Features: {', '.join(kb['plans']['pro']['features'])}
  - Support: {kb['plans']['pro']['support']}

--- COMPANY POLICIES ---
  - Refund Policy: {kb['policies']['refund']}
  - Support: {kb['policies']['support']}
  - Free Trial: {kb['policies']['free_trial']}
  - Cancellation: {kb['policies']['cancellation']}

--- FAQ ---
"""
    for faq in kb["faq"]:
        kb_text += f"  Q: {faq['question']}\n  A: {faq['answer']}\n\n"

    return kb_text.strip()


KNOWLEDGE_BASE = load_knowledge_base()


# MOCK LEAD CAPTURE TOOL


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Mock API call to capture a qualified lead."""
    print(f"\n{'='*50}")
    print(f"Lead captured successfully!")
    print(f"   Name     : {name}")
    print(f"   Email    : {email}")
    print(f"   Platform : {platform}")
    print(f"{'='*50}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"



# AGENT STATE


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: Optional[str]          # "greeting" | "inquiry" | "high_intent"
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    collecting_lead: bool



# LLM SETUP


def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key)



#GRAPH NODES


def classify_intent(state: AgentState) -> AgentState:
    """
    Node 1: Classify the user's latest message into one of three intents.
    Uses the LLM for nuanced understanding.
    """
    llm = get_llm()
    last_message = state["messages"][-1].content

    prompt = f"""You are an intent classifier for AutoStream, a video editing SaaS.

Classify the following user message into EXACTLY ONE of:
- "greeting"       → Simple hello, how are you, or off-topic chat
- "inquiry"        → Questions about product features, pricing, policies, or comparisons
- "high_intent"    → User explicitly wants to sign up, buy, try, or subscribe

User message: "{last_message}"

Respond with ONLY one word: greeting, inquiry, or high_intent"""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip().lower()

    # Safely parse intent
    if "high_intent" in raw or "high intent" in raw:
        intent = "high_intent"
    elif "inquiry" in raw:
        intent = "inquiry"
    else:
        intent = "greeting"

    return {**state, "intent": intent}


def route_after_intent(state: AgentState) -> str:
    """
    Conditional edge: decide which node to run next based on intent and lead state.
    """
    if state.get("lead_captured"):
        return "respond"
    if state.get("collecting_lead"):
        return "collect_lead"
    if state["intent"] == "high_intent":
        return "collect_lead"
    return "respond"


def respond(state: AgentState) -> AgentState:
    """
    Node 2: Generate a response using RAG (knowledge base) for inquiry/greeting intents.
    """
    llm = get_llm()

    system_prompt = f"""You are infix, the friendly AI assistant for AutoStream — a SaaS platform for automated video editing.

You help content creators understand our product. Always be helpful, concise, and warm.

Use the following knowledge base to answer questions accurately:

{KNOWLEDGE_BASE}

Rules:
- Only answer based on the knowledge base. Don't invent features or prices.
- If the user seems interested in signing up or trying the Pro plan, gently encourage them.
- Keep responses under 150 words unless a detailed comparison is needed.
- Never ask for personal information unless you've identified high buying intent."""

    messages_for_llm = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages_for_llm)

    return {**state, "messages": state["messages"] + [AIMessage(content=response.content)]}


def collect_lead(state: AgentState) -> AgentState:
    """
    Node 3: Progressively collect lead details (name → email → platform),
    then trigger mock_lead_capture() once all three are gathered.
    """
    llm = get_llm()
    last_message = state["messages"][-1].content

    name = state.get("lead_name")
    email = state.get("lead_email")
    platform = state.get("lead_platform")

    # Extract info from latest message 
    extract_prompt = f"""The user is signing up for AutoStream. Extract any of the following from their message if present.
Return JSON only with keys: name, email, platform (use null if not found).

Message: "{last_message}"

Examples of platforms: YouTube, Instagram, TikTok, Twitter, LinkedIn, Facebook

JSON:"""

    extract_response = llm.invoke([HumanMessage(content=extract_prompt)])
    extracted = {}
    try:
        raw = extract_response.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        extracted = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    # Update state with newly extracted values
    if extracted.get("name") and not name:
        name = extracted["name"]
    if extracted.get("email") and not email:
        email = extracted["email"]
    if extracted.get("platform") and not platform:
        platform = extracted["platform"]

    # Check if all details collected
    if name and email and platform:
        result = mock_lead_capture(name, email, platform)
        confirmation = (
            f"🎉 You're all set, **{name}**! We've captured your details and our team will reach out to your {platform} account shortly.\n\n"
            f"In the meantime, your **Pro Plan free trial** starts today. Welcome to AutoStream! "
        )
        return {
            **state,
            "lead_name": name,
            "lead_email": email,
            "lead_platform": platform,
            "lead_captured": True,
            "collecting_lead": False,
            "messages": state["messages"] + [AIMessage(content=confirmation)],
        }

    #  Ask for the next missing field 
    system_prompt = f"""You are infix from AutoStream. The user wants to sign up for the Pro Plan — great!

Collected so far:
- Name: {name or '(not yet collected)'}
- Email: {email or '(not yet collected)'}
- Creator Platform: {platform or '(not yet collected)'}

Ask ONLY for the next missing detail. Be friendly and brief (1–2 sentences).
Do NOT ask for something already collected.
Do NOT ask for multiple things at once."""

    ask_response = llm.invoke(
        [SystemMessage(content=system_prompt)] + state["messages"]
    )

    return {
        **state,
        "lead_name": name,
        "lead_email": email,
        "lead_platform": platform,
        "collecting_lead": True,
        "messages": state["messages"] + [AIMessage(content=ask_response.content)],
    }


# 6. BUILD THE GRAPH


def build_agent() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("respond", respond)
    graph.add_node("collect_lead", collect_lead)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Edges
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "respond": "respond",
            "collect_lead": "collect_lead",
        },
    )
    graph.add_edge("respond", END)
    graph.add_edge("collect_lead", END)

    return graph.compile()



# 7. CHAT LOOP (CLI)


def run_chat():
    """Run the agent as an interactive CLI chatbot."""
    agent = build_agent()

    print("\n" + "="*55)
    print(" AutoStream AI Assistant  (type 'exit' to quit)")
    print("="*55 + "\n")

    state: AgentState = {
        "messages": [],
        "intent": None,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "collecting_lead": False,
    }

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit", "bye"):
            print("\ninfix: Thanks for chatting! See you on AutoStream.\n")
            break
        if not user_input:
            continue

        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]
        state = agent.invoke(state)

        # Print last AI message
        last_ai = [m for m in state["messages"] if isinstance(m, AIMessage)]
        if last_ai:
            print(f"\ninfix: {last_ai[-1].content}\n")


if __name__ == "__main__":
    run_chat()

import json
import os
import re
from typing import Annotated, TypedDict, Optional
from pathlib import Path

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from dotenv import load_dotenv
load_dotenv()

# KNOWLEDGE BASE LOADER


def load_knowledge_base() -> str:
    """Load the AutoStream knowledge base and return it as a formatted string."""
    kb_path = Path(__file__).parent / "knowledge_base" / "autostream_kb.json"
    with open(kb_path, "r") as f:
        kb = json.load(f)

    kb_text = f"""
=== AutoStream Knowledge Base ===

Product: {kb['product']}
Description: {kb['description']}

--- PRICING PLANS ---

Basic Plan: ${kb['plans']['basic']['price_monthly']}/month
  - {kb['plans']['basic']['videos_per_month']} videos/month
  - {kb['plans']['basic']['resolution']} resolution
  - Features: {', '.join(kb['plans']['basic']['features'])}
  - Support: {kb['plans']['basic']['support']}

Pro Plan: ${kb['plans']['pro']['price_monthly']}/month
  - {kb['plans']['pro']['videos_per_month']} videos
  - {kb['plans']['pro']['resolution']} resolution
  - Features: {', '.join(kb['plans']['pro']['features'])}
  - Support: {kb['plans']['pro']['support']}

--- COMPANY POLICIES ---
  - Refund Policy: {kb['policies']['refund']}
  - Support: {kb['policies']['support']}
  - Free Trial: {kb['policies']['free_trial']}
  - Cancellation: {kb['policies']['cancellation']}

--- FAQ ---
"""
    for faq in kb["faq"]:
        kb_text += f"  Q: {faq['question']}\n  A: {faq['answer']}\n\n"

    return kb_text.strip()


KNOWLEDGE_BASE = load_knowledge_base()


# MOCK LEAD CAPTURE TOOL


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Mock API call to capture a qualified lead."""
    print(f"\n{'='*50}")
    print(f"Lead captured successfully!")
    print(f"   Name     : {name}")
    print(f"   Email    : {email}")
    print(f"   Platform : {platform}")
    print(f"{'='*50}\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"



# AGENT STATE


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: Optional[str]          # "greeting" | "inquiry" | "high_intent"
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    collecting_lead: bool



# LLM SETUP


def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)



#GRAPH NODES


def classify_intent(state: AgentState) -> AgentState:
    """
    Node 1: Classify the user's latest message into one of three intents.
    Uses the LLM for nuanced understanding.
    """
    llm = get_llm()
    last_message = state["messages"][-1].content

    prompt = f"""You are an intent classifier for AutoStream, a video editing SaaS.

Classify the following user message into EXACTLY ONE of:
- "greeting"       → Simple hello, how are you, or off-topic chat
- "inquiry"        → Questions about product features, pricing, policies, or comparisons
- "high_intent"    → User explicitly wants to sign up, buy, try, or subscribe

User message: "{last_message}"

Respond with ONLY one word: greeting, inquiry, or high_intent"""

    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip().lower()

    # Safely parse intent
    if "high_intent" in raw or "high intent" in raw:
        intent = "high_intent"
    elif "inquiry" in raw:
        intent = "inquiry"
    else:
        intent = "greeting"

    return {**state, "intent": intent}


def route_after_intent(state: AgentState) -> str:
    """
    Conditional edge: decide which node to run next based on intent and lead state.
    """
    if state.get("lead_captured"):
        return "respond"
    if state.get("collecting_lead"):
        return "collect_lead"
    if state["intent"] == "high_intent":
        return "collect_lead"
    return "respond"


def respond(state: AgentState) -> AgentState:
    """
    Node 2: Generate a response using RAG (knowledge base) for inquiry/greeting intents.
    """
    llm = get_llm()

    system_prompt = f"""You are infix, the friendly AI assistant for AutoStream — a SaaS platform for automated video editing.

You help content creators understand our product. Always be helpful, concise, and warm.

Use the following knowledge base to answer questions accurately:

{KNOWLEDGE_BASE}

Rules:
- Only answer based on the knowledge base. Don't invent features or prices.
- If the user seems interested in signing up or trying the Pro plan, gently encourage them.
- Keep responses under 150 words unless a detailed comparison is needed.
- Never ask for personal information unless you've identified high buying intent."""

    messages_for_llm = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages_for_llm)

    return {**state, "messages": state["messages"] + [AIMessage(content=response.content)]}


def collect_lead(state: AgentState) -> AgentState:
    """
    Node 3: Progressively collect lead details (name → email → platform),
    then trigger mock_lead_capture() once all three are gathered.
    """
    llm = get_llm()
    last_message = state["messages"][-1].content

    name = state.get("lead_name")
    email = state.get("lead_email")
    platform = state.get("lead_platform")

    # Extract info from latest message 
    extract_prompt = f"""The user is signing up for AutoStream. Extract any of the following from their message if present.
Return JSON only with keys: name, email, platform (use null if not found).

Message: "{last_message}"

Examples of platforms: YouTube, Instagram, TikTok, Twitter, LinkedIn, Facebook

JSON:"""

    extract_response = llm.invoke([HumanMessage(content=extract_prompt)])
    extracted = {}
    try:
        raw = extract_response.content.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        extracted = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    # Update state with newly extracted values
    if extracted.get("name") and not name:
        name = extracted["name"]
    if extracted.get("email") and not email:
        email = extracted["email"]
    if extracted.get("platform") and not platform:
        platform = extracted["platform"]

    # Check if all details collected
    if name and email and platform:
        result = mock_lead_capture(name, email, platform)
        confirmation = (
            f"🎉 You're all set, **{name}**! We've captured your details and our team will reach out to your {platform} account shortly.\n\n"
            f"In the meantime, your **Pro Plan free trial** starts today. Welcome to AutoStream! "
        )
        return {
            **state,
            "lead_name": name,
            "lead_email": email,
            "lead_platform": platform,
            "lead_captured": True,
            "collecting_lead": False,
            "messages": state["messages"] + [AIMessage(content=confirmation)],
        }

    #  Ask for the next missing field 
    system_prompt = f"""You are Aria from AutoStream. The user wants to sign up for the Pro Plan — great!

Collected so far:
- Name: {name or '(not yet collected)'}
- Email: {email or '(not yet collected)'}
- Creator Platform: {platform or '(not yet collected)'}

Ask ONLY for the next missing detail. Be friendly and brief (1–2 sentences).
Do NOT ask for something already collected.
Do NOT ask for multiple things at once."""

    ask_response = llm.invoke(
        [SystemMessage(content=system_prompt)] + state["messages"]
    )

    return {
        **state,
        "lead_name": name,
        "lead_email": email,
        "lead_platform": platform,
        "collecting_lead": True,
        "messages": state["messages"] + [AIMessage(content=ask_response.content)],
    }


# 6. BUILD THE GRAPH


def build_agent() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("respond", respond)
    graph.add_node("collect_lead", collect_lead)

    # Entry point
    graph.set_entry_point("classify_intent")

    # Edges
    graph.add_conditional_edges(
        "classify_intent",
        route_after_intent,
        {
            "respond": "respond",
            "collect_lead": "collect_lead",
        },
    )
    graph.add_edge("respond", END)
    graph.add_edge("collect_lead", END)

    return graph.compile()



# 7. CHAT LOOP (CLI)


def run_chat():
    """Run the agent as an interactive CLI chatbot."""
    agent = build_agent()

    print("\n" + "="*55)
    print("  🎬  AutoStream AI Assistant  (type 'exit' to quit)")
    print("="*55 + "\n")

    state: AgentState = {
        "messages": [],
        "intent": None,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "collecting_lead": False,
    }

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit", "bye"):
            print("\nAria: Thanks for chatting! See you on AutoStream. 👋\n")
            break
        if not user_input:
            continue

        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]
        state = agent.invoke(state)

        # Print last AI message
        last_ai = [m for m in state["messages"] if isinstance(m, AIMessage)]
        if last_ai:
            print(f"\nAria: {last_ai[-1].content}\n")


if __name__ == "__main__":
    run_chat()
