"""
llm.py — Answer synthesis for Nina 2.0 prototype.


"""

import os
import re

USE_REAL_LLM = bool(os.environ.get("AZURE_OPENAI_API_KEY")) and bool(
    os.environ.get("AZURE_OPENAI_ENDPOINT")
)

CONFIDENCE_ANSWER = 0.35   # above this: answer confidently
CONFIDENCE_HEDGE = 0.12    # above this but below ANSWER: hedge / partial match
# below CONFIDENCE_HEDGE: escalate, don't guess

NINA_IDENTITY = (
    "I'm Nina 2.0 — NNPC's internal AI knowledge assistant. I help employees find "
    "answers from company policies, SOPs, HR guidance, and organizational knowledge, "
    "grounded in what's actually in our knowledge base, with citations so you know "
    "where each answer came from. Ask me anything about company policy, procedures, "
    "or where to find something, and I'll only answer from documents you're "
    "authorized to see."
)

NINA_GREETING = (
    "Hello! How can I help you today?"
    )

# Meta/identity questions — matched loosely so common phrasings all hit this
# path instead of going through document retrieval, where they'd find nothing.
_IDENTITY_PATTERNS = [
    r"\bwho are you\b",
    r"\bwhat are you\b",
    r"\bwhat is nina\b",
    r"\bwho is nina\b",
    r"\bwhat can you (do|help)\b",
    r"\btell me about yourself\b",
    r"\bintroduce yourself\b",
    r"\bwhat do you do\b",
]
_identity_re = re.compile("|".join(_IDENTITY_PATTERNS), re.IGNORECASE)

# Plain greetings with no real question content — matched as a whole-message
# pattern (allowing trailing punctuation/whitespace) so this doesn't
# accidentally swallow a real question that merely starts with "hi, ...".
_GREETING_PATTERNS = [
    r"^hi+!?$",
    r"^hello!?$",
    r"^hey!?$",
    r"^hiya!?$",
    r"^good (morning|afternoon|evening)!?$",
    r"^how (are you|are you doing|is it going)\??$",
    r"^what'?s up\??$",
]
_greeting_re = re.compile("|".join(_GREETING_PATTERNS), re.IGNORECASE)


def _is_identity_question(question: str) -> bool:
    return bool(_identity_re.search(question.strip()))


def _is_greeting(question: str) -> bool:
    return bool(_greeting_re.match(question.strip()))


def _offline_synthesize(question, chunks):
    """Very simple offline synthesis: stitches the best chunk(s) into an answer."""
    if not chunks:
        return None
    top = chunks[0]
    extra = f" Related: {chunks[1]['text']}" if len(chunks) > 1 else ""
    return f"{top['text']}{extra}"


def _real_llm_synthesize(question, chunks):
    """Calls Azure OpenAI. Only runs if credentials are configured."""
    from openai import AzureOpenAI  # import here so it's optional at install time

    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-08-01-preview",
    )
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
    prompt = (
        "You are Nina, NNPC's internal assistant. Answer the employee's question "
        "using ONLY the context below. If the context doesn't fully answer the "
        "question, say what you found and what's missing. Cite the source tag "
        "in brackets.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    # NOTE: newer model families (e.g. gpt-5-mini) renamed max_tokens to
    # max_completion_tokens, and some reject a custom temperature entirely
    # (only the default of 1 is accepted) — temperature is omitted so this
    # works across both older (gpt-4o-mini) and newer model families.
    #
    # Reasoning-tier models also spend part of the completion token budget
    # on internal reasoning before producing visible output. A budget of
    # ~300 can be entirely consumed by reasoning with nothing left for the
    # actual answer, resulting in an empty content string. 1500 gives
    # enough headroom for reasoning plus a full answer.
    resp = client.chat.completions.create(
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=1500,
    )
    content = resp.choices[0].message.content
    if not content:
        # Safety net: never show a blank "Answered" result to the user.
        # Falls back to the raw retrieved text if the model returned
        # nothing (e.g. reasoning budget exhausted, content filtered).
        return _offline_synthesize(question, chunks)
    return content


def answer(question: str, chunks: list):
    """
    Returns a dict: {tier, text, sources}
    tier is one of: "answered" | "hedged" | "not_found"
    This is the 3-tier confidence behavior from the benchmark taxonomy.
    """
    # Greetings and identity/meta questions bypass retrieval entirely — see
    # module docstring for why. These run before the "no chunks" check
    # below, since both should answer even though retrieval would find
    # nothing relevant for either (which is the normal, expected case).
    if _is_greeting(question):
        return {"tier": "answered", "text": NINA_GREETING, "sources": []}

    if _is_identity_question(question):
        return {"tier": "answered", "text": NINA_IDENTITY, "sources": []}

    if not chunks:
        return {
            "tier": "not_found",
            "text": "The requested information is not available in the retrieved data. "
                    "Would you like me to route this to the relevant department?",
            "sources": [],
        }

    top_score = chunks[0]["score"]
    sources = sorted(set(c["source"] for c in chunks))

    if top_score < CONFIDENCE_HEDGE:
        return {
            "tier": "not_found",
            "text": "The requested information is not available in the retrieved data. "
                    "Would you like me to route this to the relevant department?",
            "sources": [],
        }

    if USE_REAL_LLM:
        text = _real_llm_synthesize(question, chunks)
    else:
        text = _offline_synthesize(question, chunks)

    if top_score < CONFIDENCE_ANSWER:
        return {
            "tier": "hedged",
            "text": f"I found related information, but I'm not fully confident it "
                    f"directly answers your question: {text}",
            "sources": sources,
        }

    return {"tier": "answered", "text": text, "sources": sources}
