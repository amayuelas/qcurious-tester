"""Baseline exploration strategies: random and greedy coverage."""

import logging

from ..runner.coverage import CoverageRunner
from .candidate_gen import generate_test_candidates

log = logging.getLogger(__name__)


def run_random_strategy(func_name: str, source_code: str,
                        runner: CoverageRunner, budget: int) -> list[dict]:
    """Baseline: LLM generates tests with no feedback."""
    runner.reset()
    steps = []

    for step in range(budget):
        candidates = generate_test_candidates(
            func_name, source_code, test_history=[], K=1
        )
        if not candidates:
            steps.append({
                "step": step, "test": None, "new_branches": 0,
                "cumulative": runner.get_cumulative_coverage(),
            })
            continue

        result = runner.run_test(candidates[0])
        steps.append({
            "step": step,
            "test": candidates[0],
            "new_branches": result.new_branches,
            "cumulative": runner.get_cumulative_coverage(),
            "output": result.output,
            "exception": result.exception,
        })

        print(f"    Step {step+1}: {candidates[0][:60]}... → "
              f"new_branches={result.new_branches}, "
              f"cumulative={runner.get_cumulative_coverage()}")

    return steps


def run_greedy_coverage_strategy(func_name: str, source_code: str,
                                  runner: CoverageRunner, budget: int) -> list[dict]:
    """Baseline: LLM sees coverage feedback, greedily targets uncovered code."""
    runner.reset()
    steps = []
    test_history = []

    for step in range(budget):
        candidates = generate_test_candidates(
            func_name, source_code, test_history=test_history, K=1
        )
        if not candidates:
            steps.append({
                "step": step, "test": None, "new_branches": 0,
                "cumulative": runner.get_cumulative_coverage(),
            })
            continue

        result = runner.run_test(candidates[0])
        test_history.append((candidates[0], result))
        steps.append({
            "step": step,
            "test": candidates[0],
            "new_branches": result.new_branches,
            "cumulative": runner.get_cumulative_coverage(),
            "output": result.output,
            "exception": result.exception,
        })

        print(f"    Step {step+1}: {candidates[0][:60]}... → "
              f"new_branches={result.new_branches}, "
              f"cumulative={runner.get_cumulative_coverage()}")

    return steps
