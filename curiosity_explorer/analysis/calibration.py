"""Calibration analysis: entropy-coverage correlation."""

import statistics


def analyze_calibration(diagnostics: list[dict]) -> dict:
    """Key question: does output prediction entropy correlate with
    actual coverage gain?
    """
    entropies = []
    actual_gains = []

    for diag in diagnostics:
        for cand in diag["candidates"]:
            entropies.append(cand["entropy"])
            actual_gains.append(cand["actual_new_branches"])

    if len(entropies) < 5:
        return {"status": "insufficient_data"}

    # Spearman rank correlation
    n = len(entropies)
    rank_e = _rank(entropies)
    rank_g = _rank(actual_gains)
    d_sq = sum((rank_e[i] - rank_g[i]) ** 2 for i in range(n))
    spearman = 1 - (6 * d_sq) / (n * (n**2 - 1)) if n > 1 else 0

    median_e = statistics.median(entropies)
    high_entropy_gains = [g for e, g in zip(entropies, actual_gains) if e > median_e]
    low_entropy_gains = [g for e, g in zip(entropies, actual_gains) if e <= median_e]

    return {
        "n_datapoints": n,
        "spearman_correlation": round(spearman, 3),
        "mean_entropy": round(statistics.mean(entropies), 3),
        "mean_actual_gain": round(statistics.mean(actual_gains), 3),
        "high_entropy_mean_gain": round(statistics.mean(high_entropy_gains), 3) if high_entropy_gains else 0,
        "low_entropy_mean_gain": round(statistics.mean(low_entropy_gains), 3) if low_entropy_gains else 0,
    }


def _rank(values):
    """Simple ranking (average rank for ties)."""
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_indices):
        j = i
        while j < len(sorted_indices) and values[sorted_indices[j]] == values[sorted_indices[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1
        for k in range(i, j):
            ranks[sorted_indices[k]] = avg_rank
        i = j
    return ranks
