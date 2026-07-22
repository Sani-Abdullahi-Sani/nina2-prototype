"""
retrieval.py — Knowledge source retrieval for Nina 2.0 prototype.

PROTOTYPE NOTE: Loads 3 small local mock files and does simple keyword /
substring matching to find the most relevant chunk. This proves the
"route to the right source, then retrieve" pattern without needing a
real vector index.

PRODUCTION RECOMMENDATION: Replace the load/search functions here with
Foundry IQ queries against live SharePoint (see pitch deck Appendix E).
The return shape (chunks with source + score) is kept stable so the rest
of the app doesn't need to change when this is swapped out.
"""

import csv
import os
import re
from azure.core.exceptions import AzureError
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

SOURCE_FILES = {
    "policy": ["hr_policy.md"],
    "procedure": ["it_procedures.md"],
    "organization": ["org_directory.csv", "it_procedures.md"],  # directory (Graph-style)
                                                                  # + narrative org knowledge (SharePoint-style)
}


def _chunk_markdown(path):
    """Split a markdown file into chunks by ## headings."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    parts = re.split(r"\n(?=##\s)", text)
    return [p.strip() for p in parts if p.strip()]


def _load_org_directory(path):
    """Load the CSV as one chunk per row, formatted as readable text."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chunk = (
                f"{row['name']} is the {row['role']} in {row['department']} "
                f"({row['business_unit']}), reporting to {row['reports_to']}."
            )
            chunks.append(chunk)
    return chunks


def _load_source(category):
    chunks = []
    for filename in SOURCE_FILES[category]:
        path = os.path.join(DATA_DIR, filename)
        if filename.endswith(".csv"):
            chunks.extend(_load_org_directory(path))
        else:
            chunks.extend(_chunk_markdown(path))
    return chunks


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "i", "you", "he", "she", "it", "we", "they", "me", "my", "your",
    "what", "who", "whom", "which", "when", "where", "why", "how",
    "do", "does", "did", "doing", "done",
    "of", "in", "on", "at", "to", "for", "with", "about", "as", "by",
    "and", "or", "but", "if", "so", "than", "that", "this", "these", "those",
    "know", "tell", "can", "could", "would", "should", "will", "shall",
    "have", "has", "had", "get", "got", "please", "want",
}


def _content_words(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def _score_chunk(question, chunk):
    """
    Overlap score using only meaningful content words (stopwords stripped),
    so generic phrasing ("tell me what you know about...") doesn't inflate
    the score, and unrelated questions don't accidentally "match" on
    common words. Requires at least 2 content-word overlaps OR 1 overlap
    that's a rare/distinctive term (>=6 chars) to register at all.
    """
    q_words = _content_words(question)
    c_words = _content_words(chunk)
    if not q_words:
        return 0.0
    overlap = q_words & c_words
    if not overlap:
        return 0.0
    strong_overlap = {w for w in overlap if len(w) >= 6}
    if len(overlap) < 2 and not strong_overlap:
        return 0.0
    return len(overlap) / len(q_words)


def retrieve(question: str, categories):
    """
    Search across one or more categories (e.g. ["policy"] or ["policy","procedure"]).
    Returns a list of dicts: {source, text, score}, sorted best-first.
    """
    results = []
    for cat in categories:
        if cat == "hybrid":
            search_cats = ["policy", "organization", "procedure"]
        else:
            search_cats = [cat]
        for c in search_cats:
            for chunk in _load_source(c):
                score = _score_chunk(question, chunk)
                if score > 0:
                    results.append({"source": c, "text": chunk, "score": round(score, 3)})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:3]  # top-3 chunks


# ---- Permission-aware retrieval against real Azure AI Search index ----
# This is the "production-shaped" path described in the module docstring
# above — Foundry IQ / SharePoint would replace this same interface later.

SEARCH_ENDPOINT = "https://nina2-search.search.windows.net"
SEARCH_INDEX = "rag-1784386244487"

SEARCH_QUERY_KEY = os.environ.get("AZURE_SEARCH_QUERY_KEY", "")

_search_client = None
if SEARCH_QUERY_KEY:
    _search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX,
        credential=AzureKeyCredential(SEARCH_QUERY_KEY),
    )

SEARCH_AVAILABLE = _search_client is not None

# Set once a search call fails due to network/connectivity issues, so the
# UI can show a distinct "temporarily unavailable" message rather than a
# generic "not found" or, worse, an uncaught traceback crashing the app.
# Reset to False at the start of every retrieve_permission_aware() call.
SEARCH_TEMPORARILY_UNAVAILABLE = False

USER_GROUPS = {
    "fatima.hr@saniabdullahiranogmail.onmicrosoft.com": [
        "05a711cb-882e-480b-bf74-c6264764f0d5",
        "94e108b2-850b-44ba-9f36-a4cc96c32bdb",
    ],
    "aminu.it@saniabdullahiranogmail.onmicrosoft.com": [
        "1f8f24cc-bf65-4f0f-8a6a-f532cd9e52ea",
        "94e108b2-850b-44ba-9f36-a4cc96c32bdb",
    ],
    "mustapha.finance@saniabdullahiranogmail.onmicrosoft.com": [
        "ac2b2c00-730a-48a6-a389-c40621cae7f3",
        "94e108b2-850b-44ba-9f36-a4cc96c32bdb",
    ],
    "abba.general@saniabdullahiranogmail.onmicrosoft.com": [
        "94e108b2-850b-44ba-9f36-a4cc96c32bdb",
    ],
}


def _build_access_filter(user_email: str) -> str:
    groups = USER_GROUPS.get(user_email, [])
    if not groups:
        return "access_group eq 'no-access'"
    return " or ".join([f"access_group eq '{g}'" for g in groups])


def retrieve_permission_aware(question: str, user_email: str, top: int = 3):
    """
    Retrieves from the real Azure AI Search index, scoped to only the
    access_group values the given user's Entra ID group membership permits.
    Returns the same {source, text, score} shape as retrieve(), so the
    rest of the app treats it identically to the mock retrieval path.

    NETWORK RESILIENCE: the default Azure SDK connection timeout is 300
    seconds — on a flaky connection this means the whole Streamlit app
    hangs for 5 minutes before crashing with an uncaught traceback. Here
    the timeout is cut to a few seconds and any connectivity failure is
    caught, so a bad network moment fails fast with a clear message
    instead of freezing or crashing the demo.
    """
    global SEARCH_TEMPORARILY_UNAVAILABLE
    SEARCH_TEMPORARILY_UNAVAILABLE = False

    if not SEARCH_AVAILABLE:
        return []

    filter_expr = _build_access_filter(user_email)
    try:
        results = _search_client.search(
            search_text=question,
            filter=filter_expr,
            select=["title", "chunk"],
            top=top,
            connection_timeout=8,   # fail fast instead of hanging up to 300s
            read_timeout=20,
        )
        return [
            {"source": r["title"], "text": r["chunk"], "score": 1.0}
            for r in results
        ]
    except AzureError:
        # Network/connectivity failure talking to Azure AI Search — not a
        # code bug. Surface it as "temporarily unavailable" rather than
        # crashing the app or silently returning zero results.
        SEARCH_TEMPORARILY_UNAVAILABLE = True
        return []