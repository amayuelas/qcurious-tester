"""Curiosity-guided exploration strategy."""

import logging

from ..runner.coverage import CoverageRunner
from .candidate_gen import generate_test_candidates
from .info_gain import estimate_output_entropy

log = logging.getLogger(__name__)


def run_curiosity_strategy(func_name: str, source_code: str,
                           runner: CoverageRunner, budget: int,
                           K: int = 5, S: int = 8,
                           code_visible: bool = True) -> tuple[list[dict], list[dict]]:
    """Select tests by output prediction entropy (information gain proxy).

    Returns (steps, diagnostics) where steps include per-step details
    and diagnostics include counterfactual analysis for calibration.
    """
    runner.reset()
    steps = []
    test_history = []
    diagnostics = []

    for step in range(budget):
        # Generate K candidates
        candidates = generate_test_candidates(
            func_name, source_code, test_history=test_history,
            K=K, code_visible=code_visible
        )
        if not candidates:
            steps.append({
                "step": step, "test": None, "entropy": 0.0,
                "new_branches": 0,
                "cumulative": runner.get_cumulative_coverage(),
            })
            continue

        # Score each by output prediction entropy
        scored = []
        for cand in candidates:
            entropy = estimate_output_entropy(
                func_name, source_code, test_history,
                cand, S=S, code_visible=code_visible
            )
            scored.append((cand, entropy))

        # Select highest entropy candidate
        scored.sort(key=lambda x: x[1], reverse=True)
        best_test, best_entropy = scored[0]

        # Execute it
        result = runner.run_test(best_test)
        test_history.append((best_test, result))

        steps.append({
            "step": step,
            "test": best_test,
            "entropy": best_entropy,
            "new_branches": result.new_branches,
            "cumulative": runner.get_cumulative_coverage(),
            "output": result.output,
            "exception": result.exception,
            "all_candidates": [(c, e) for c, e in scored],
        })

        # Counterfactual diagnostics: run remaining candidates to measure
        # what we WOULD have gotten (for calibration analysis)
        diag_entry = {
            "step": step,
            "candidates": [],
            "selected": best_test,
            "selected_entropy": best_entropy,
            "selected_new_branches": result.new_branches,
        }

        temp_runner = CoverageRunner(func_name, source_code)
        temp_runner.cumulative_branches = set(runner.cumulative_branches)
        for cand, entropy in scored[1:]:
            temp_result = temp_runner.run_test(cand)
            diag_entry["candidates"].append({
                "test": cand,
                "entropy": entropy,
                "actual_new_branches": temp_result.new_branches,
            })
        diag_entry["candidates"].append({
            "test": best_test,
            "entropy": best_entropy,
            "actual_new_branches": result.new_branches,
        })
        diagnostics.append(diag_entry)

        print(f"    Step {step+1}: {best_test[:50]}... → "
              f"entropy={best_entropy:.2f}, "
              f"new_branches={result.new_branches}, "
              f"cumulative={runner.get_cumulative_coverage()}")

    return steps, diagnostics
