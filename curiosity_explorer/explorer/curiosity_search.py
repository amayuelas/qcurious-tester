"""Curiosity-guided exploration strategy.

Supports multiple estimators:
  - sampling_entropy: S predictions, Shannon entropy (original, has zero-entropy floor)
  - q_value: Q-value with planning lookahead (Sun et al. Proposition 1)
"""

import logging

from ..runner.coverage import CoverageRunner
from .candidate_gen import generate_test_candidates
from .info_gain import estimate_output_entropy
from .q_values import compute_q_values

log = logging.getLogger(__name__)


def run_curiosity_strategy(func_name: str, source_code: str,
                           runner: CoverageRunner, budget: int,
                           K: int = 5, S: int = 8, gamma: float = 0.5,
                           estimator: str = "q_value",
                           code_visible: bool = True) -> tuple[list[dict], list[dict]]:
    """Select tests by information gain estimation.

    Args:
        estimator: "sampling_entropy" or "q_value"
        gamma: discount factor for q_value (0 = greedy, >0 = planning)
    """
    runner.reset()
    steps = []
    test_history = []
    diagnostics = []

    for step in range(budget):
        candidates = generate_test_candidates(
            func_name, source_code, test_history=test_history,
            K=K, code_visible=code_visible
        )
        if not candidates:
            steps.append({
                "step": step, "test": None, "score": 0.0,
                "new_branches": 0,
                "cumulative": runner.get_cumulative_coverage(),
            })
            continue

        # Score candidates based on estimator
        if estimator == "q_value":
            q_results = compute_q_values(
                func_name, source_code, test_history, candidates,
                gamma=gamma, future_K=3, code_visible=code_visible,
            )
            scored = [
                (cand, q_results.get(cand, {}).get("q_value", 0))
                for cand in candidates
            ]
        else:
            # Default: sampling entropy
            scored = []
            for cand in candidates:
                entropy = estimate_output_entropy(
                    func_name, source_code, test_history,
                    cand, S=S, code_visible=code_visible
                )
                scored.append((cand, entropy))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_test, best_score = scored[0]

        result = runner.run_test(best_test)
        test_history.append((best_test, result))

        steps.append({
            "step": step,
            "test": best_test,
            "score": best_score,
            "new_branches": result.new_branches,
            "cumulative": runner.get_cumulative_coverage(),
            "output": result.output,
            "exception": result.exception,
            "all_candidates": [(c, s) for c, s in scored],
        })

        # Counterfactual diagnostics
        diag_entry = {
            "step": step,
            "candidates": [],
            "selected": best_test,
            "selected_score": best_score,
            "selected_new_branches": result.new_branches,
        }

        temp_runner = CoverageRunner(func_name, source_code)
        temp_runner.cumulative_branches = set(runner.cumulative_branches)
        for cand, score in scored[1:]:
            temp_result = temp_runner.run_test(cand)
            diag_entry["candidates"].append({
                "test": cand,
                "score": score,
                "actual_new_branches": temp_result.new_branches,
            })
        diag_entry["candidates"].append({
            "test": best_test,
            "score": best_score,
            "actual_new_branches": result.new_branches,
        })
        diagnostics.append(diag_entry)
        temp_runner.cleanup()

        print(f"    Step {step+1}: {best_test[:50]}... → "
              f"score={best_score:.2f}, "
              f"new_branches={result.new_branches}, "
              f"cumulative={runner.get_cumulative_coverage()}")

    return steps, diagnostics
