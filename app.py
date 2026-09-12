"""
app.py
 
Stage 7 — AI Customer Support Assistant interface.
Two tabs:
  - Dashboard: direct ticket-ID lookup, full analysis in one view
  - Chatbot: conversational lookup by ticket ID or customer name
             (with disambiguation when a name matches multiple customers)
 
Run with: streamlit run app.py
"""
 
import streamlit as st
 
from pipeline_utils import (
    analyze_ticket,
    get_customer_history,
    find_similar_ticket,
    generate_recommendation,
    generate_customer_response,
    route_chat_message,
    format_ticket_summary,
    answer_customer_query,
    get_full_customer_history,
    engine,
)
from sqlalchemy import text
import pandas as pd
 
st.set_page_config(page_title="AI Customer Support Assistant", layout="wide")
 
st.markdown("""
<style>
div[class*="st-key-dashboard_search"],
div[class*="st-key-chatbot_header"] {
    position: sticky;
    top: 0;
    z-index: 999;
    background-color: var(--background-color, white);
    padding: 0.75rem 0 1rem 0;
    border-bottom: 1px solid rgba(128, 128, 128, 0.3);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)
 
 
def escape_markdown(raw_text):
    """
    Raw ticket descriptions sometimes contain stray Markdown control characters
    (leftover template artifacts -- see Stage 3 notes on placeholder/code-fragment
    junk in this dataset). Escaping them prevents Streamlit from misrendering
    things like a leading '#' as a giant heading.
    """
    if raw_text is None:
        return ""
    for ch in ["\\", "*", "_", "#", "`", "[", "]", "(", ")", ">", "~"]:
        raw_text = raw_text.replace(ch, "\\" + ch)
    return raw_text
 
 
def format_ticket_list(ticket_entries):
    """Renders a list of {'ticket_id': ..., 'status': ...} dicts as 'ID (status)' pairs."""
    if not ticket_entries:
        return "None"
    return ", ".join(f"#{t['ticket_id']} ({t['status']})" for t in ticket_entries)
 
 
dashboard_tab, chatbot_tab = st.tabs(["📊 Dashboard", "💬 Chatbot"])
 
 
# ---------------------------------------------------------------------------
# Dashboard tab
# ---------------------------------------------------------------------------
 
with dashboard_tab:
    st.title("AI Customer Support Assistant")
 
    with st.container(key="dashboard_search"):
        ticket_id_input = st.number_input("Ticket ID", min_value=1, step=1, value=4491)
        analyze_clicked = st.button("Analyze Ticket", type="primary")
 
    if analyze_clicked:
 
        ticket_query = text("""
            SELECT `Ticket ID`, `Customer ID`, `Customer Name`, `Product Purchased`, `Ticket Description`
            FROM tickets
            WHERE `Ticket ID` = :ticket_id
        """)
        ticket_row_df = pd.read_sql(ticket_query, con=engine, params={"ticket_id": int(ticket_id_input)})
 
        if ticket_row_df.empty:
            st.error(f"No ticket found with ID {ticket_id_input}.")
        else:
            ticket_row = ticket_row_df.iloc[0]
            customer_id = ticket_row["Customer ID"]
            description = ticket_row["Ticket Description"]
 
            st.subheader("Ticket")
            st.write(f"**Product:** {ticket_row['Product Purchased']}")
            st.write(f"**Description:** {escape_markdown(description)}")
 
            with st.spinner("Analyzing ticket type, priority, urgency..."):
                analysis = analyze_ticket(int(ticket_id_input))
 
            col1, col2 = st.columns(2)
 
            with col1:
                st.subheader("Ticket Analysis")
                st.write(f"**Type:** {analysis['type']}")
                st.write(f"**Priority:** {analysis['priority']}")
                st.write(f"**Urgency:** {analysis['urgency']}")
 
                with st.spinner("Loading customer history..."):
                    history = get_customer_history(customer_id, int(ticket_id_input))
 
                st.subheader("Customer History")
                st.write(f"**Previous tickets:** {history['previous_ticket_count']}")
                st.write(f"**Unresolved tickets:** {history['unresolved_ticket_count']}")
                st.write(f"**Ticket IDs & status:** {format_ticket_list(history['previous_tickets'])}")
                st.write(f"**Previous products:** {', '.join(history['previous_products']) or 'None'}")
                st.write(f"**Previous satisfaction:** {history['previous_satisfaction']}")
                st.write(f"**Previously purchased this product:** {history['previously_purchased_current_product']}")
 
                with st.spinner("Searching for similar previous tickets..."):
                    similar = find_similar_ticket(description, int(ticket_id_input), customer_id=customer_id)
 
                st.subheader("Similar Previous Ticket")
                if similar["found"]:
                    st.write(f"**Found:** Yes ({similar['search_scope']} match, similarity {similar['similarity']})")
                    st.write(f"**Previous issue:** {escape_markdown(similar['previous_description'])}")
                else:
                    st.write("**Found:** No similar previous ticket")
 
            with col2:
                st.subheader("AI Recommendation")
                with st.spinner("Generating recommendation..."):
                    recommendation = generate_recommendation(int(ticket_id_input))
                st.write(recommendation["recommendation_text"])
 
                st.subheader("Suggested Response")
                with st.spinner("Drafting suggested response..."):
                    response = generate_customer_response(int(ticket_id_input))
                st.write(response["suggested_response"])
 
 
# ---------------------------------------------------------------------------
# Chatbot tab
# ---------------------------------------------------------------------------
 
with chatbot_tab:
    with st.container(key="chatbot_header"):
        st.title("Support Assistant Chat")
        st.caption("Ask about a ticket number or a customer by name — e.g. \"show me ticket 4491\" or \"what did Allison order?\"")
 
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_session_state" not in st.session_state:
        st.session_state.chat_session_state = {}
 
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)
 
    user_input = st.chat_input("Type your question...")
 
    # A suggestion button click is treated exactly like the agent typing that text
    if not user_input and st.session_state.get("pending_suggestion_click"):
        user_input = st.session_state.pending_suggestion_click
        st.session_state.pending_suggestion_click = None
 
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)
 
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = route_chat_message(user_input, st.session_state.chat_session_state)
 
            if result["action"] == "reply":
                st.markdown(result["message"])
                st.session_state.chat_history.append(("assistant", result["message"]))
 
            elif result["action"] == "show_ticket_detail":
                ticket_id = result["ticket_id"]
 
                with st.spinner("Pulling full ticket breakdown..."):
                    analysis = analyze_ticket(ticket_id)
 
                    ticket_query = text("""
                        SELECT `Customer ID`, `Product Purchased`, `Ticket Description`
                        FROM tickets WHERE `Ticket ID` = :ticket_id
                    """)
                    row = pd.read_sql(ticket_query, con=engine, params={"ticket_id": ticket_id}).iloc[0]
 
                    history = get_customer_history(row["Customer ID"], ticket_id)
                    similar = find_similar_ticket(row["Ticket Description"], ticket_id, customer_id=row["Customer ID"])
                    recommendation = generate_recommendation(ticket_id)
                    response = generate_customer_response(ticket_id)
 
                detail_text = f"""**Ticket {ticket_id} — Full Breakdown**
 
**Analysis:** {analysis['type']}, {analysis['priority']} priority, {analysis['urgency']}
 
**Customer History:** {history['previous_ticket_count']} previous tickets, {history['unresolved_ticket_count']} unresolved.
**Ticket IDs & status:** {format_ticket_list(history['previous_tickets'])}
Previous products: {', '.join(history['previous_products']) or 'None'}.
 
**Similar Ticket:** {"Found — " + escape_markdown(similar['previous_description']) if similar['found'] else "None found"}
 
**AI Recommendation:**
{recommendation['recommendation_text']}
 
**Suggested Customer Response:**
{response['suggested_response']}"""
 
                st.markdown(detail_text)
                st.session_state.chat_history.append(("assistant", detail_text))
 
            elif result["action"] == "show_customer_detail":
                customer_id = result["customer_id"]
                customer_name = result.get("customer_name", customer_id)
 
                with st.spinner("Pulling full customer history..."):
                    history = get_full_customer_history(customer_id)
 
                detail_text = f"""**{customer_name} — Full History**
 
**Total tickets:** {history['ticket_count']}
**Unresolved:** {history['unresolved_ticket_count']}
**Ticket IDs & status:** {format_ticket_list(history['tickets'])}
**Products purchased:** {', '.join(history['products_purchased']) or 'None'}
**Ticket types:** {', '.join(history['ticket_types']) or 'None'}
**Satisfaction ratings:** {history['satisfaction_ratings']}"""
 
                st.markdown(detail_text)
                st.session_state.chat_history.append(("assistant", detail_text))
 
            elif result["action"] == "suggestions":
                st.markdown(result["message"])
                st.session_state.chat_history.append(("assistant", result["message"]))
 
                for i, suggestion in enumerate(result["suggestions"]):
                    if st.button(suggestion, key=f"suggestion_{len(st.session_state.chat_history)}_{i}"):
                        st.session_state.pending_suggestion_click = suggestion
                        st.rerun()
 
