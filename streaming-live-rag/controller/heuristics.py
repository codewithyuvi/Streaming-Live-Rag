"""
controller/heuristics.py — Stability heuristic filter (H4).

Acts as a cheap filter to prevent thrashing the LLM controller on
incomplete clauses or dangling determiners/prepositions.
"""

import re

DANGLING = {
    "the", "a", "an", "of", "to", "for", "in", "on", "and", "or",
    "but", "with", "about", "that", "is", "are", "what", "how"
}


def is_stable_enough(text: str) -> bool:
    """
    Returns True if the text is 'stable enough' to evaluate with the LLM classifier.
    Rejects text ending in dangling prepositions/determiners or incomplete clauses.
    """
    t = text.strip()
    if not t:
        return False

    # Terminal punctuation is always stable
    if re.search(r"[.?!]$", t):
        return True

    words = t.rstrip(",;:").split()
    # At least 4 words and last word must not be a dangling connector/determiner
    return len(words) >= 4 and words[-1].lower() not in DANGLING


def get_stable_query_prefix(text: str) -> str | None:
    """
    Returns the text if it is stable, or strips trailing dangling connectors/prepositions
    if the resulting prefix is stable enough for evaluation.
    Returns None if no stable prefix exists.
    """
    if is_stable_enough(text):
        return text
    words = text.rstrip(",;:").split()
    while words and words[-1].lower() in DANGLING:
        words.pop()
    stripped = " ".join(words)
    if is_stable_enough(stripped):
        return stripped
    return None

