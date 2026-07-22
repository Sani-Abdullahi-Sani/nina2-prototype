"""
app.py — Nina 2.0 prototype (3-day Build-to-Learn version).

Run with:  streamlit run app.py

This proves the pattern: Intent Router -> Multi-source Retrieval ->
Confidence-aware Synthesis -> Logged for governance — using tools that
let us iterate fast. See README.md for what maps to production and how.

PERMISSION-AWARE RETRIEVAL:
When AZURE_SEARCH_QUERY_KEY is set, the Ask tab queries the real Azure
AI Search index with a security filter built from the selected user's
Entra ID group membership (see retrieval.py). The permission-aware path
does NOT fall back to local mock retrieval on empty results — an empty
result there almost always means the user's access correctly excluded
everything relevant, and falling back would leak unfiltered content.

NETWORK RESILIENCE:
If the Azure AI Search call fails due to a connectivity issue (flaky
network, VPN, DNS), retrieval.py catches it, sets
retrieval.SEARCH_TEMPORARILY_UNAVAILABLE, and returns no chunks rather
than letting the exception propagate. run_pipeline() checks that flag
and returns a clear "temporarily unavailable, please retry" result
instead of calling the LLM on empty context or letting Streamlit crash
with a raw traceback — important for demo day, where one bad Wi-Fi
moment shouldn't take down the whole app.
"""

import streamlit as st

import db
import llm
import router
import retrieval

import os
for k, v in st.secrets.items():
    os.environ[k] = str(v)

st.set_page_config(page_title="Nina 2.0 Prototype", page_icon="🧭", layout="wide")
db.init_db()

# ---- Real Nina 1.0 failures, captured verbatim from live testing ----
# Currently unused in the UI (Replay tab removed — see module docstring).
# Kept here for reference / easy reintroduction later.
REPLAY_CASES = [
    {
        "question": "Who is the IT Manager in NEPL",
        "nina1_answer": "The requested information is not available in the retrieved data. "
                         "Please try another query or topic.",
    },
    {
        "question": "Do you know the new NEPL organogram?",
        "nina1_answer": "The requested information is not available in the retrieved data. "
                         "Please try another query or topic.",
    },
    {
        "question": "Tell me what you know about project sigma",
        "nina1_answer": "The requested information is not available in the retrieved data. "
                         "Please try another query or topic.",
    },
    {
        "question": "Do I accrue annual leave while on sick leave?",
        "nina1_answer": "There is no explicit mention of whether annual leave continues to "
                         "accrue while an employee is on sick leave... it is recommended to "
                         "consult the NNPC Limited HCM team.",
    },
]


def run_pipeline(question: str, user_email: str = None):
    """
    Routes and retrieves for a question.

    Deliberately does NOT fall back to the local mock retrieval when the
    permission-aware search returns nothing — an empty result there means
    the user's access correctly excluded everything relevant, and falling
    back would leak unfiltered content. The local mock retrieval is only
    used when no user is selected or the search connection isn't
    configured (e.g. AZURE_SEARCH_QUERY_KEY unset).

    If the search call failed due to a network issue (as opposed to a
    genuine "no results"), returns a distinct "temporarily unavailable"
    result instead of treating it as "not found" or crashing.
    """
    category, confidence, matched = router.classify(question)

    if user_email and retrieval.SEARCH_AVAILABLE:
        chunks = retrieval.retrieve_permission_aware(question, user_email)
        if retrieval.SEARCH_TEMPORARILY_UNAVAILABLE:
            result = {
                "tier": "not_found",
                "text": "⚠️ I couldn't reach the knowledge base just now - this looks like a "
                        "temporary network issue, "
                        "Please try asking again in a moment.",
                "sources": [],
            }
            db.log_question(question, category, confidence, result["tier"], result["sources"])
            return category, confidence, result
    else:
        chunks = retrieval.retrieve(question, [category])

    result = llm.answer(question, chunks)
    db.log_question(question, category, confidence, result["tier"], result["sources"])
    return category, confidence, result


tab_chat, tab_admin = st.tabs(["💬 Ask Nina 2.0", "📊 Admin Dashboard"])

# ============================================================
# TAB 1 — CHAT
# ============================================================
with tab_chat:
    st.subheader("Nina 2.0")
    st.caption(
        #"Prototype knowledge base: HR policy · IT procedures & current initiatives · "
        #"Org directory "
    )

    selected_user = st.selectbox(
        "Signed in as:",
        options=list(retrieval.USER_GROUPS.keys()),
        format_func=lambda x: x.split("@")[0].replace(".", " ").title(),
        help="Switch identities to demonstrate permission-aware retrieval — "
             "each user only sees results from their permitted knowledge areas.",
    )

    if retrieval.SEARCH_AVAILABLE:
        st.success(
            "Connected to live Azure AI Search index with permission-aware "
            "retrieval (Entra ID group-based access control).",
            icon="🔒",
        )
    else:
        st.info(
            "AZURE_SEARCH_QUERY_KEY not set — running on local mock retrieval "
            "(no live permission filtering). Set the environment variable and "
            "restart to enable the permission-aware demo.",
            icon="ℹ️",
        )

    if not llm.USE_REAL_LLM:
        st.info(
            "Running in **offline mode** (no AZURE_OPENAI_API_KEY set) — answers are "
            "assembled directly from retrieved text rather than LLM-synthesized. "
            "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT to use real synthesis.",
            icon="ℹ️",
        )

    # ---- Chat history, kept in session state so it persists across turns ----
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of dicts: user, question, category, confidence, result

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()

    # ---- Render chat history, oldest first, most recent just above the
    # input — the standard chat-thread reading order. ----
    tier_style = {
        "answered": ("✅", "green"),
        "hedged": ("⚠️", "orange"),
        "not_found": ("⛔", "red"),
    }

    for turn in st.session_state.chat_history:
        display_name = turn["user"].split("@")[0].replace(".", " ").title()

        with st.chat_message("user"):
            st.markdown(f"**{display_name}:** {turn['question']}")

        with st.chat_message("assistant"):
            icon, color = tier_style[turn["result"]["tier"]]
            st.markdown(f"**Routed to:** `{turn['category']}`")
            st.markdown(f"##### {icon} :{color}[{turn['result']['tier'].replace('_', ' ').title()}]")
            st.write(turn["result"]["text"])
            if turn["result"]["sources"]:
                st.caption(f"Sources: {', '.join(turn['result']['sources'])}")
            st.caption("🤖 Nina 2.0 is AI and can make mistakes. Please verify sensitive information.")

    # ---- Input, pinned to the bottom of the page by Streamlit automatically.
    # st.chat_input clears itself on submit — no manual reset needed. ----
    question = st.chat_input("Type a new question...")

    if question and question.strip():
        category, confidence, result = run_pipeline(question, selected_user)
        st.session_state.chat_history.append(
            {
                "user": selected_user,
                "question": question,
                "category": category,
                "confidence": confidence,
                "result": result,
            }
        )
        st.rerun()

# ============================================================
# TAB 2 — ADMIN DASHBOARD
# ============================================================
with tab_admin:
    st.subheader("Admin Dashboard")
    st.caption(
        "Prototype version of what LangFuse would provide in production, "
        "visibility into what Nina can't answer, as a permanent feature "
        "instead of a one-off manual benchmark."
    )

    stats = db.fetch_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total questions logged", stats["total"])
    c2.metric("✅ Answered", stats["answered"])
    c3.metric("⚠️ Hedged", stats["hedged"])
    c4.metric("⛔ Not found", stats["not_found"])

    st.markdown("#### Question Log")
    rows = db.fetch_all()
    if rows:
        #st.dataframe(rows, use_container_width=True, hide_index=True)
        st.dataframe(rows, width='stretch', hide_index=True)
    else:
        st.write("No questions logged yet — try the Ask tab.")
