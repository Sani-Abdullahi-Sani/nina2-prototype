"""
router.py — Intent classification for Nina 2.0 prototype.

PROTOTYPE NOTE: This uses simple keyword rules to classify intent in milliseconds
with zero cost, so the prototype is fast and demoable offline.

PRODUCTION RECOMMENDATION: Replace this with a small fine-tuned classifier
(MiniLM / DistilBERT) as described in the pitch deck (Appendix B). The interface
below (classify() returning a label + confidence) is deliberately kept the same
shape so swapping the implementation later doesn't require changing app.py.
"""

CATEGORIES = ["policy", "organization", "procedure", "hybrid"]

KEYWORDS = {
    "policy": [
        "leave", "sick", "annual leave", "maternity", "study leave", "disciplinary",
        "warning", "policy", "benefit", "entitled", "entitlement",
    ],
    "organization": [
        "who is", "manager", "director", "reports to", "organogram", "org chart",
        "department", "business unit", "subsidiary", "role", "head of", "ceo", "cto",
    ],
    "procedure": [
        "how do i", "how to", "process for", "procurement", "vendor", "onboarding",
        "steps", "workflow", "apply for", "request", "project", "initiative", "sigma",
    ],
}


def classify(question: str):
    """
    Returns (label, confidence, matched_keywords).
    confidence is a crude 0-1 score based on keyword hit density —
    good enough for a 3-day prototype; a real classifier would replace this.
    """
    q = question.lower()
    scores = {cat: 0 for cat in KEYWORDS}
    matched = {cat: [] for cat in KEYWORDS}

    for cat, words in KEYWORDS.items():
        for w in words:
            if w in q:
                scores[cat] += 1
                matched[cat].append(w)

    total_hits = sum(scores.values())
    if total_hits == 0:
        return "hybrid", 0.0, []

    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]

    # If two categories are close, it's genuinely ambiguous -> hybrid (search both)
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] > 0 and sorted_scores[1] >= sorted_scores[0] * 0.6:
        return "hybrid", 0.5, matched[best_cat]

    confidence = min(1.0, best_score / 3)  # crude normalization
    return best_cat, confidence, matched[best_cat]
