"""
pipeline_utils.py

Shared functions for the AI Customer Support Ticket Intelligence System.
Consolidates Stage 2 (customer history), Stage 3 (similar ticket search),
and Stage 4 (ticket intelligence) so later stages can import them directly
instead of re-pasting code into every notebook.
"""

import re

import ollama
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# Shared connections / models (initialized once on import)
# ---------------------------------------------------------------------------

engine = create_engine("mysql+pymysql://root:database.777@localhost:3306/customer_support")

_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

_chroma_client = chromadb.PersistentClient(path="chroma_db")
_collection = _chroma_client.get_or_create_collection(name="ticket_descriptions")


# ---------------------------------------------------------------------------
# Stage 2 — Customer history
# ---------------------------------------------------------------------------

def get_customer_history(customer_id, current_ticket_id):

    query = text("""
        SELECT `Ticket ID`, `Ticket Type`, `Product Purchased`, `Ticket Status`,
               `Resolution`, `Customer Satisfaction Rating`, `Date of Purchase`
        FROM tickets
        WHERE `Customer ID` = :customer_id
          AND `Ticket ID` != :current_ticket_id
    """)
    customer_tickets = pd.read_sql(
        query, con=engine,
        params={"customer_id": customer_id, "current_ticket_id": current_ticket_id}
    )

    current_query = text("""
        SELECT `Product Purchased`
        FROM tickets
        WHERE `Ticket ID` = :current_ticket_id
    """)
    current_result = pd.read_sql(current_query, con=engine, params={"current_ticket_id": current_ticket_id})
    current_product = current_result["Product Purchased"].iloc[0]

    previous_products = customer_tickets["Product Purchased"].dropna().unique().tolist()
    previously_purchased = current_product in previous_products

    previous_purchase_dates = (
        customer_tickets[customer_tickets["Product Purchased"] == current_product]["Date of Purchase"]
        .dropna().astype(str).unique().tolist()
    )

    previous_tickets = [
        {"ticket_id": int(r["Ticket ID"]), "status": r["Ticket Status"]}
        for _, r in customer_tickets.iterrows()
    ]

    return {
        "previous_ticket_count": len(customer_tickets),
        "unresolved_ticket_count": int((customer_tickets["Ticket Status"] != "Closed").sum()),
        "previous_tickets": previous_tickets,
        "previous_ticket_types": customer_tickets["Ticket Type"].dropna().unique().tolist(),
        "previous_products": previous_products,
        "previous_resolutions": customer_tickets["Resolution"].dropna().tolist(),
        "previous_satisfaction": customer_tickets["Customer Satisfaction Rating"].dropna().tolist(),
        "current_product": current_product,
        "previously_purchased_current_product": previously_purchased,
        "previous_purchase_dates_for_product": previous_purchase_dates,
    }


def get_full_customer_history(customer_id):
    """
    Like get_customer_history, but for chatbot queries that aren't about
    any specific ticket (e.g. "what did Allison order?") -- pulls ALL of
    this customer's tickets, nothing excluded.
    """
    query = text("""
        SELECT `Ticket ID`, `Ticket Type`, `Product Purchased`, `Ticket Status`,
               `Ticket Description`, `Resolution`, `Customer Satisfaction Rating`,
               `Date of Purchase`
        FROM tickets
        WHERE `Customer ID` = :customer_id
        ORDER BY `Date of Purchase`
    """)
    customer_tickets = pd.read_sql(query, con=engine, params={"customer_id": customer_id})

    tickets = [
        {"ticket_id": int(r["Ticket ID"]), "status": r["Ticket Status"]}
        for _, r in customer_tickets.iterrows()
    ]

    return {
        "ticket_count": len(customer_tickets),
        "unresolved_ticket_count": int((customer_tickets["Ticket Status"] != "Closed").sum()),
        "tickets": tickets,
        "products_purchased": customer_tickets["Product Purchased"].dropna().unique().tolist(),
        "ticket_types": customer_tickets["Ticket Type"].dropna().unique().tolist(),
        "descriptions": customer_tickets["Ticket Description"].dropna().tolist(),
        "resolutions": customer_tickets["Resolution"].dropna().tolist(),
        "satisfaction_ratings": customer_tickets["Customer Satisfaction Rating"].dropna().tolist(),
    }


# ---------------------------------------------------------------------------
# Stage 3 — Similar previous ticket search
# ---------------------------------------------------------------------------

def clean_placeholders(text_value):
    if pd.isna(text_value):
        return text_value
    cleaned = re.sub(r"\{.*?\}", "", text_value, flags=re.DOTALL)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def find_similar_ticket(ticket_description, current_ticket_id, customer_id=None, similarity_threshold=0.6):

    ticket_description = clean_placeholders(ticket_description)

    def _search(where_filter):
        query_embedding = _embedding_model.encode([ticket_description]).tolist()

        results = _collection.query(
            query_embeddings=query_embedding,
            n_results=5,
            where=where_filter
        )

        for distance, document, metadata in zip(
            results["distances"][0], results["documents"][0], results["metadatas"][0]
        ):
            if metadata["ticket_id"] == current_ticket_id:
                continue

            similarity = 1 / (1 + distance)

            if similarity >= similarity_threshold:
                return {
                    "found": True,
                    "similar_ticket_id": metadata["ticket_id"],
                    "customer_id": metadata["customer_id"],
                    "similarity": round(similarity, 3),
                    "previous_description": document,
                    "search_scope": "same_customer" if where_filter else "global",
                }
        return None

    if customer_id is not None:
        result = _search({"customer_id": customer_id})
        if result:
            return result

    return _search(None) or {"found": False}


# ---------------------------------------------------------------------------
# Stage 4 — Ticket intelligence (type / priority / urgency)
# ---------------------------------------------------------------------------

def classify_urgency(ticket_description, ticket_priority=None):

    context = f"Ticket Priority (if known): {ticket_priority}\n" if ticket_priority else ""

    prompt = f"""You are classifying a customer support ticket's urgency.

{context}Ticket description:
"{ticket_description}"

Classify this ticket's urgency as exactly one word: Urgent or Normal.

Consider it Urgent if the customer indicates: something has completely stopped working,
they are blocked from using the product/service, financial harm (e.g. double charges),
time-sensitive language (e.g. "immediately", "urgent", "as soon as possible"), or high emotional distress.

Consider it Normal for general questions, minor issues, feature requests, or informational inquiries.

Respond with ONLY the single word: Urgent or Normal. No explanation, no punctuation."""

    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    result = response["message"]["content"].strip()

    if "urgent" in result.lower():
        return "Urgent"
    return "Normal"


def analyze_ticket(ticket_id):

    query = text("""
        SELECT `Ticket ID`, `Ticket Type`, `Ticket Priority`, `Ticket Description`
        FROM tickets
        WHERE `Ticket ID` = :ticket_id
    """)
    row = pd.read_sql(query, con=engine, params={"ticket_id": ticket_id}).iloc[0]

    urgency = classify_urgency(row["Ticket Description"], row["Ticket Priority"])

    return {
        "ticket_id": int(row["Ticket ID"]),
        "type": row["Ticket Type"],
        "priority": row["Ticket Priority"],
        "urgency": urgency,
    }


# ---------------------------------------------------------------------------
# Stage 5 — AI recommendation (internal, agent-facing)
# ---------------------------------------------------------------------------

def generate_recommendation(ticket_id):

    ticket_query = text("""
        SELECT `Ticket ID`, `Customer ID`, `Product Purchased`, `Ticket Description`, `Ticket Priority`
        FROM tickets
        WHERE `Ticket ID` = :ticket_id
    """)
    ticket_row = pd.read_sql(ticket_query, con=engine, params={"ticket_id": ticket_id}).iloc[0]

    customer_id = ticket_row["Customer ID"]
    description = ticket_row["Ticket Description"]

    analysis = analyze_ticket(ticket_id)
    history = get_customer_history(customer_id, ticket_id)
    similar = find_similar_ticket(description, ticket_id, customer_id=customer_id)

    if similar["found"]:
        similar_section = f"""Found: Yes
Similarity: {similar['similarity']}
Previous issue: {similar['previous_description']}
Search scope: {similar['search_scope']}"""
    else:
        similar_section = "Found: No similar previous ticket"

    prompt = f"""You are assisting a customer support agent. Based ONLY on the context below, provide:
1. A brief summary of the customer's issue
2. A recommended action for the agent to take
3. A short reason explaining why you recommend that action

CURRENT TICKET
---------------
Product: {ticket_row['Product Purchased']}
Description: {description}
Priority: {analysis['priority']}
Urgency: {analysis['urgency']}

CUSTOMER HISTORY
----------------
Previous tickets: {history['previous_ticket_count']}
Unresolved tickets: {history['unresolved_ticket_count']}
Previous ticket IDs and status: {history['previous_tickets']}
Previous products: {history['previous_products']}
Previous satisfaction ratings: {history['previous_satisfaction']}
Previously purchased this product: {history['previously_purchased_current_product']}

SIMILAR PREVIOUS TICKET
------------------------
{similar_section}

STRICT RULES:
- Only reference facts that appear explicitly above. Do not mention policies, refund rules, warranty terms, wholesale/retail distinctions, or resolutions that are not written above.
- If no similar previous ticket was found, do not invent one or reference a "previous resolution."
- If the context does not clearly support a specific action, recommend that the agent gather more information from the customer rather than guessing a resolution.
- Keep the response short: 1-2 sentences per section."""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )

    return {
        "ticket_id": ticket_id,
        "recommendation_text": response["message"]["content"].strip(),
        "context_used": {
            "analysis": analysis,
            "history": history,
            "similar_ticket": similar,
        },
    }


# ---------------------------------------------------------------------------
# Stage 6 — Suggested customer response
# ---------------------------------------------------------------------------

def generate_customer_response(ticket_id):

    ticket_query = text("""
        SELECT `Ticket ID`, `Customer ID`, `Customer Name`, `Product Purchased`, `Ticket Description`
        FROM tickets
        WHERE `Ticket ID` = :ticket_id
    """)
    ticket_row = pd.read_sql(ticket_query, con=engine, params={"ticket_id": ticket_id}).iloc[0]

    customer_id = ticket_row["Customer ID"]
    description = ticket_row["Ticket Description"]

    history = get_customer_history(customer_id, ticket_id)
    similar = find_similar_ticket(description, ticket_id, customer_id=customer_id)

    if similar["found"]:
        if similar["search_scope"] == "same_customer":
            similar_section = f"""This customer has a similar previous issue on file:
Previous issue: {similar['previous_description']}"""
        else:
            similar_section = f"""A similar issue has been reported by another customer (not this customer's own history):
Previous issue: {similar['previous_description']}"""
    else:
        similar_section = "No similar previous ticket was found."

    prompt = f"""Write a short, professional customer support reply to the customer below.

CUSTOMER'S ISSUE
-----------------
Product: {ticket_row['Product Purchased']}
Description: {description}

CUSTOMER CONTEXT
------------------
Previously purchased this product before: {history['previously_purchased_current_product']}
Unresolved tickets on file: {history['unresolved_ticket_count']}

RELATED HISTORY
----------------
{similar_section}

STRICT RULES:
- Do not promise a specific resolution (refund, replacement, repair) unless one is explicitly stated above.
- Do not invent policy details, timelines, or outcomes not present above.
- Only say "your previous case" or refer to the customer's own history if the related history above is explicitly from this same customer. If the similar issue is from another customer, refer to it only as "a known issue" or "similar cases we've seen," never as belonging to this customer.
- Acknowledge the issue, reassure the customer it's being looked into, and if related history exists, you may mention it using the framing above.
- Keep it to 3-4 sentences, professional and empathetic tone, no signature/closing needed.
- Do not use the customer's name if uncertain of it; a generic greeting is fine."""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3}
    )

    return {
        "ticket_id": ticket_id,
        "suggested_response": response["message"]["content"].strip(),
    }


# ---------------------------------------------------------------------------
# Stage 7 — Chatbot: intent routing, name resolution, conversational answers
# ---------------------------------------------------------------------------

def resolve_customer_by_name(name_query, max_display=6):
    query = text("""
        SELECT DISTINCT `Customer ID`, `Customer Name`, `Customer Email`
        FROM tickets
        WHERE `Customer Name` LIKE :pattern
    """)
    matches = pd.read_sql(query, con=engine, params={"pattern": f"%{name_query}%"})

    if len(matches) == 0:
        return {"status": "no_match"}
    elif len(matches) == 1:
        return {"status": "single_match", "customer": matches.iloc[0].to_dict()}
    elif len(matches) <= max_display:
        return {"status": "ambiguous", "candidates": matches.to_dict("records")}
    else:
        return {"status": "too_many", "count": len(matches)}


def format_disambiguation_message(resolution_result, name_query):
    if resolution_result["status"] == "no_match":
        return f"I couldn't find any customer matching \"{name_query}\". Could you check the spelling or try their email instead?"

    elif resolution_result["status"] == "too_many":
        return (f"There are {resolution_result['count']} customers matching \"{name_query}\" — "
                f"that's too many to list. Could you narrow it down with a full name, "
                f"email address, or a product they mentioned?")

    elif resolution_result["status"] == "ambiguous":
        lines = [f"I found a few customers matching \"{name_query}\":"]
        for i, c in enumerate(resolution_result["candidates"], 1):
            lines.append(f"{i}. {c['Customer Name']} ({c['Customer Email']})")
        lines.append("Which one did you mean? You can reply with a number or their email.")
        return "\n".join(lines)


def resolve_disambiguation_reply(reply, candidates):
    reply = reply.strip().lower()

    if reply.isdigit():
        index = int(reply) - 1
        if 0 <= index < len(candidates):
            return candidates[index]
        return None

    for c in candidates:
        if c["Customer Email"].lower() == reply:
            return c

    for c in candidates:
        if reply in c["Customer Email"].lower():
            return c

    return None


def extract_ticket_id(user_message):
    match = re.search(r'\b(\d{1,6})\b', user_message)
    if match:
        return int(match.group(1))
    return None


def classify_intent(user_message, awaiting_disambiguation=False):
    if awaiting_disambiguation:
        return {"intent": "disambiguation_reply", "name": None}

    prompt = f"""Analyze this customer support agent's message.

Message: "{user_message}"

Determine the intent - exactly one of:
- ticket_id: the message references a specific ticket number
- customer_name: the message asks about a specific customer by name
- unclear: neither applies clearly

Separately, check whether the message mentions any person's name at all (first name,
last name, or full name) -- extract it even if the overall intent is unclear or ticket_id.

Respond in EXACTLY this format, nothing else:
Intent: <ticket_id, customer_name, or unclear>
Name: <any person name mentioned anywhere in the message, or None if no name is present>"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )
    result = response["message"]["content"].strip()

    intent = "unclear"
    name = None
    for line in result.splitlines():
        if line.lower().startswith("intent:"):
            value = line.split(":", 1)[1].strip().lower()
            if "ticket_id" in value:
                intent = "ticket_id"
            elif "customer_name" in value:
                intent = "customer_name"
        elif line.lower().startswith("name:"):
            value = line.split(":", 1)[1].strip()
            if value.lower() != "none":
                name = value

    return {"intent": intent, "name": name}


_DETAIL_REQUEST_PHRASES = [
    "more detail", "full breakdown", "full details", "tell me more",
    "recommendation", "suggested response", "show everything", "more info",
]


def is_detail_request(user_message):
    lowered = user_message.lower()
    return any(phrase in lowered for phrase in _DETAIL_REQUEST_PHRASES)


def format_ticket_summary(ticket_id):
    """Short, templated (non-LLM) summary of a single ticket -- fast, no hallucination risk."""
    query = text("""
        SELECT `Ticket ID`, `Product Purchased`, `Ticket Description`
        FROM tickets
        WHERE `Ticket ID` = :ticket_id
    """)
    row = pd.read_sql(query, con=engine, params={"ticket_id": ticket_id})
    if row.empty:
        return f"I couldn't find a ticket with ID {ticket_id}."

    row = row.iloc[0]
    analysis = analyze_ticket(ticket_id)
    short_desc = row["Ticket Description"][:150].strip()
    for ch in ["\\", "*", "_", "#", "`", "[", "]", "(", ")", ">", "~"]:
        short_desc = short_desc.replace(ch, "\\" + ch)

    return (
        f"Ticket {ticket_id} — {row['Product Purchased']} ({analysis['type']}, "
        f"{analysis['priority']} priority, {analysis['urgency']}).\n"
        f"\"{short_desc}...\"\n\n"
        f"Want the full breakdown (history, similar tickets, AI recommendation)?"
    )


def answer_customer_query(customer_id, customer_name, original_question):
    """
    Answers a specific natural-language question about a customer
    (e.g. "what did they order?", "have they complained before?")
    grounded strictly in that customer's actual ticket history.
    """
    history = get_full_customer_history(customer_id)

    prompt = f"""Answer the support agent's question using ONLY the customer data below.

Agent's question: "{original_question}"

CUSTOMER: {customer_name}
Total tickets: {history['ticket_count']}
Unresolved tickets: {history['unresolved_ticket_count']}
Ticket IDs and status: {history['tickets']}
Products purchased: {history['products_purchased']}
Ticket types: {history['ticket_types']}
Satisfaction ratings given: {history['satisfaction_ratings']}

STRICT RULES:
- Only state facts present above. Do not invent products, dates, or outcomes.
- If the data doesn't answer the question, say so plainly.
- Keep it to 1-3 sentences, conversational tone (you're answering a colleague, not writing a report).
- End by mentioning the agent can ask for the full breakdown if they want ticket-by-ticket detail."""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}
    )

    return response["message"]["content"].strip()


def generate_suggestions(user_message, extracted_name):
    """
    When intent classification comes back unclear, build a short list of
    concrete follow-up questions the agent can pick from instead of a
    generic "please rephrase" -- grounded in whatever name/number we
    could still pull out of the original message.
    """
    suggestions = []

    if extracted_name:
        suggestions.append(f"What did {extracted_name} order?")
        suggestions.append(f"Does {extracted_name} have any unresolved tickets?")
        suggestions.append(f"Show {extracted_name}'s ticket history")

    ticket_id = extract_ticket_id(user_message)
    if ticket_id:
        suggestions.append(f"Show me ticket {ticket_id}")

    if not suggestions:
        suggestions = [
            "Show me ticket 4491",
            "What did a specific customer order? (try: 'what did <name> order')",
        ]

    return suggestions


def route_chat_message(user_message, session_state):

    # Follow-up: agent wants the full breakdown for whatever was last resolved
    if is_detail_request(user_message):
        if session_state.get("last_ticket_id"):
            return {"action": "show_ticket_detail", "ticket_id": session_state["last_ticket_id"]}
        elif session_state.get("last_customer_id"):
            return {
                "action": "show_customer_detail",
                "customer_id": session_state["last_customer_id"],
                "customer_name": session_state.get("last_customer_name"),
            }
        else:
            return {"action": "reply", "message": "I don't have a ticket or customer in context yet — ask about one first (e.g. 'show me ticket 4491')."}

    # If we're mid-disambiguation, treat this message as a reply, not a new query
    if session_state.get("awaiting_disambiguation"):
        candidates = session_state["disambiguation_candidates"]
        resolved = resolve_disambiguation_reply(user_message, candidates)

        if resolved is None:
            return {
                "action": "reply",
                "message": "I didn't recognize that choice. Please reply with a number from the list, or an email address."
            }

        session_state["awaiting_disambiguation"] = False
        session_state["disambiguation_candidates"] = None
        session_state["last_customer_id"] = resolved["Customer ID"]
        session_state["last_customer_name"] = resolved["Customer Name"]
        original_question = session_state.get("pending_question", "What can you tell me about this customer?")

        answer = answer_customer_query(resolved["Customer ID"], resolved["Customer Name"], original_question)
        return {"action": "reply", "message": answer}

    # Otherwise, classify intent fresh
    intent_result = classify_intent(user_message)
    intent = intent_result["intent"]

    if intent == "ticket_id":
        ticket_id = extract_ticket_id(user_message)
        if ticket_id is None:
            return {"action": "reply", "message": "I couldn't find a ticket number in that message. Could you specify the ticket ID?"}
        session_state["last_ticket_id"] = ticket_id
        session_state["last_customer_id"] = None
        return {"action": "reply", "message": format_ticket_summary(ticket_id)}

    elif intent == "customer_name":
        name_query = intent_result["name"] or user_message
        resolution = resolve_customer_by_name(name_query)

        if resolution["status"] == "single_match":
            c = resolution["customer"]
            session_state["last_customer_id"] = c["Customer ID"]
            session_state["last_customer_name"] = c["Customer Name"]
            session_state["last_ticket_id"] = None
            answer = answer_customer_query(c["Customer ID"], c["Customer Name"], user_message)
            return {"action": "reply", "message": answer}

        elif resolution["status"] == "ambiguous":
            session_state["awaiting_disambiguation"] = True
            session_state["disambiguation_candidates"] = resolution["candidates"]
            session_state["pending_question"] = user_message
            return {"action": "reply", "message": format_disambiguation_message(resolution, name_query)}

        else:
            return {"action": "reply", "message": format_disambiguation_message(resolution, name_query)}

    else:
        suggestions = generate_suggestions(user_message, intent_result["name"])
        return {
            "action": "suggestions",
            "message": "I wasn't sure what you meant. Did you mean one of these?",
            "suggestions": suggestions,
        }