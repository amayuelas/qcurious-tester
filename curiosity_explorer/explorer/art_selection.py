"""Adaptive Random Testing (ART) selection.

Select the candidate that is most different from all previously executed tests.
Maximizes input space coverage by maintaining diversity in the test suite.
"""

import logging
from difflib import SequenceMatcher

log = logging.getLogger(__name__)


def select_most_distant(candidates: list[str], test_history: list) -> str:
    """Select the candidate most different from all previous tests.

    Uses string similarity as a proxy for input space distance.
    For each candidate, compute its minimum similarity to any previous test.
    Pick the candidate with the lowest minimum similarity (most distant).
    """
    if not test_history:
        # No history — pick randomly
        import random
        return random.choice(candidates)

    previous_tests = [test_code for test_code, _ in test_history]

    best_candidate = candidates[0]
    best_min_distance = -1

    for cand in candidates:
        # Distance = 1 - similarity to the nearest previous test
        min_distance = min(
            1.0 - SequenceMatcher(None, cand, prev).ratio()
            for prev in previous_tests
        )
        if min_distance > best_min_distance:
            best_min_distance = min_distance
            best_candidate = cand

    return best_candidate


def select_art_with_entropy(candidates: list[str], test_history: list,
                            entropy_scores: dict[str, float],
                            alpha: float = 0.5) -> str:
    """Combine ART distance with entropy scoring.

    score = alpha * normalized_distance + (1-alpha) * normalized_entropy
    """
    if not test_history:
        # No history — pick by entropy alone
        return max(candidates, key=lambda c: entropy_scores.get(c, 0))

    previous_tests = [test_code for test_code, _ in test_history]

    # Compute distances
    distances = {}
    for cand in candidates:
        distances[cand] = min(
            1.0 - SequenceMatcher(None, cand, prev).ratio()
            for prev in previous_tests
        )

    # Normalize both to [0, 1]
    dist_vals = list(distances.values())
    ent_vals = [entropy_scores.get(c, 0) for c in candidates]

    dist_min, dist_max = min(dist_vals), max(dist_vals)
    ent_min, ent_max = min(ent_vals), max(ent_vals)

    dist_range = dist_max - dist_min if dist_max > dist_min else 1
    ent_range = ent_max - ent_min if ent_max > ent_min else 1

    best_candidate = candidates[0]
    best_score = -1

    for cand in candidates:
        norm_dist = (distances[cand] - dist_min) / dist_range
        norm_ent = (entropy_scores.get(cand, 0) - ent_min) / ent_range
        score = alpha * norm_dist + (1 - alpha) * norm_ent
        if score > best_score:
            best_score = score
            best_candidate = cand

    return best_candidate
