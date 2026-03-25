"""Shared entropy computation utilities.

Single implementations used by info_gain.py, q_values.py, and run scripts.
"""

import math
from collections import Counter


def string_entropy(predictions: list[str]) -> float:
    """Shannon entropy over a list of prediction strings."""
    if not predictions:
        return 0.0
    counts = Counter(predictions)
    total = len(predictions)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def logprob_token_entropy(token_logprobs: list[dict]) -> float:
    """Mean per-token entropy from logprob data.

    Args:
        token_logprobs: list of dicts with 'top_logprobs' key mapping
                        token strings to log probabilities.
    """
    entropies = []
    for tok in token_logprobs:
        top_lps = tok.get("top_logprobs", {})
        if not top_lps:
            continue
        probs = [math.exp(lp) for lp in top_lps.values()]
        total = sum(probs)
        if total <= 0:
            continue
        probs = [p / total for p in probs]
        ent = -sum(p * math.log2(p) for p in probs if p > 0)
        entropies.append(ent)
    return sum(entropies) / len(entropies) if entropies else 0.0
