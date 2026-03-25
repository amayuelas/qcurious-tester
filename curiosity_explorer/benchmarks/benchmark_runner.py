"""Reusable benchmark runner for comparing exploration strategies.

Runs multiple strategies on the same functions with the same seeds,
tracking step-by-step coverage curves for statistical comparison.
"""

import logging
import random
import re
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import config
from ..llm import generate_with_model, batch_generate
from ..runner.coverage import CoverageRunner
from ..explorer.candidate_gen import generate_test_candidates
from ..explorer.diverse_gen import generate_diverse_candidates
from ..explorer.info_gain import estimate_output_entropy
from ..explorer.q_values import compute_q_values
from ..explorer.parse_utils import parse_candidate

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of running one strategy on one function with one seed."""
    func_key: str
    strategy: str
    seed: int
    coverage_curve: list[int] = field(default_factory=list)
    gate_passage_steps: dict = field(default_factory=dict)
    api_calls: int = 0
    elapsed_seconds: float = 0


def run_single(func_name, source, strategy, budget=30, K=5, S=8,
               gamma=0.5, seed=42, code_visible=True):
    """Run a single strategy on a single function, return coverage curve.

    Strategies:
      - random: standard gen + random pick
      - greedy: standard gen + LLM picks best
      - curiosity_entropy: diverse gen + sampling entropy
      - curiosity_qvalue: diverse gen + Q-value
      - reflective_qvalue: learnings-guided gen + Q-value + reflection
      - oracle: standard gen + execute all, pick best
    """
    random.seed(seed)
    runner = CoverageRunner(func_name, source)
    test_history = []
    curve = []
    gate_steps = {}  # gate_error_code -> first step that passed it
    learnings = ""
    prev_max_gate = -1

    for step in range(budget):
        # Generation
        if strategy == "reflective_qvalue":
            candidates = _generate_reflective(func_name, source, test_history,
                                              learnings, K, code_visible)
        elif strategy in ("curiosity_entropy", "curiosity_qvalue"):
            candidates = generate_diverse_candidates(
                func_name, source, test_history=test_history, K=K,
                code_visible=code_visible)
        else:
            candidates = generate_test_candidates(
                func_name, source, test_history=test_history, K=K,
                code_visible=code_visible)

        if not candidates:
            curve.append(runner.get_cumulative_coverage())
            continue

        # Selection
        if strategy == "random":
            selected = random.choice(candidates)

        elif strategy == "greedy":
            selected = _select_greedy(candidates, func_name, source,
                                      test_history, runner, code_visible)

        elif strategy == "curiosity_entropy":
            scores = {}
            for c in candidates:
                scores[c] = estimate_output_entropy(
                    func_name, source, test_history, c, S=S,
                    code_visible=code_visible)
            selected = max(candidates, key=lambda c: scores.get(c, 0))

        elif strategy in ("curiosity_qvalue", "reflective_qvalue"):
            q_results = compute_q_values(
                func_name, source, test_history, candidates,
                gamma=gamma, future_K=3, code_visible=code_visible)
            selected = max(candidates,
                           key=lambda c: q_results.get(c, {}).get("q_value", 0))

        elif strategy == "oracle":
            selected = _select_oracle(candidates, func_name, source, runner)

        else:
            selected = candidates[0]

        # Predict (for reflective)
        predicted = None
        if strategy == "reflective_qvalue":
            predicted = _predict(func_name, source, test_history, selected,
                                 code_visible)

        # Execute
        result = runner.run_test(selected)
        test_history.append((selected, result))
        curve.append(runner.get_cumulative_coverage())

        # Track gate passage (output < 100 = error code = gate number)
        if result.output is not None:
            try:
                code = int(result.output)
                if code >= 100 and prev_max_gate < 100:
                    # First time reaching deep logic
                    gate_steps["deep_logic"] = step
                if code > prev_max_gate:
                    gate_steps[f"gate_{code}"] = step
                    prev_max_gate = code
            except (ValueError, TypeError):
                pass

        # Reflect (for reflective)
        if strategy == "reflective_qvalue" and predicted:
            actual = result.output or result.exception or "None"
            learnings = _reflect(func_name, source, test_history, selected,
                                 predicted, actual, learnings, code_visible)

    runner.cleanup()
    return curve, gate_steps


def run_benchmark(programs, strategies, budget=30, K=5, S=8, gamma=0.5,
                  seeds=None, code_visible=True):
    """Run all strategies on all functions with all seeds.

    Args:
        programs: dict mapping key -> {func_name, source, metadata}
        strategies: list of strategy names
        seeds: list of random seeds (default: [42, 123, 456])

    Returns:
        list of RunResult
    """
    seeds = seeds or [42, 123, 456]
    results = []
    total = len(programs) * len(strategies) * len(seeds)
    done = 0

    for func_key, prog in programs.items():
        func_name = prog["func_name"]
        source = prog["source"]
        difficulty = prog.get("metadata", {}).get("difficulty", "unknown")

        for seed in seeds:
            for strategy in strategies:
                start = time.time()
                curve, gate_steps = run_single(
                    func_name, source, strategy, budget=budget,
                    K=K, S=S, gamma=gamma, seed=seed,
                    code_visible=code_visible,
                )
                elapsed = time.time() - start
                done += 1

                rr = RunResult(
                    func_key=func_key,
                    strategy=strategy,
                    seed=seed,
                    coverage_curve=curve,
                    gate_passage_steps=gate_steps,
                    elapsed_seconds=round(elapsed, 1),
                )
                results.append(rr)

                final = curve[-1] if curve else 0
                print(f"  [{done}/{total}] {func_key} seed={seed} "
                      f"{strategy:<22} final={final} ({elapsed:.1f}s)",
                      flush=True)

    return results


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def _select_greedy(candidates, func_name, source, test_history, runner,
                   code_visible):
    if code_visible:
        code_section = f"```python\n{source[:2000]}\n```"
    else:
        from ..runner.trace_parser import extract_function_signature
        code_section = f"```python\n{extract_function_signature(source)}\n```"

    history_str = ""
    if test_history:
        for tc, res in test_history[-5:]:
            history_str += f"  {tc} → {res.output or res.exception}\n"

    cand_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidates))
    prompt = (f"Function:\n{code_section}\n\n"
              f"Previous results:\n{history_str}\n\n"
              f"Which test discovers the most NEW behavior? "
              f"Respond with ONLY the number.\n\nCandidates:\n{cand_list}")
    resp = generate_with_model(config.MODEL, prompt, 0.3, 20)
    for n in re.findall(r'\d+', resp):
        idx = int(n) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    return candidates[0]


def _select_oracle(candidates, func_name, source, runner):
    best, bg = candidates[0], -1
    for c in candidates:
        tr = CoverageRunner(func_name, source)
        tr.cumulative_branches = set(runner.cumulative_branches)
        r = tr.run_test(c)
        if r.new_branches > bg:
            best, bg = c, r.new_branches
        tr.cleanup()
    return best


def _predict(func_name, source, test_history, selected, code_visible):
    if code_visible:
        code_section = f"```python\n{source[:2000]}\n```"
    else:
        from ..runner.trace_parser import extract_function_signature
        code_section = f"```python\n{extract_function_signature(source)}\n```"

    history_str = ""
    if test_history:
        for tc, res in test_history[-5:]:
            history_str += f"  {tc} → {res.output or res.exception}\n"

    prompt = (f"Function:\n{code_section}\n\nPrevious results:\n{history_str}\n\n"
              f"What will be the output of: {selected}\n\n"
              f"Respond with ONLY the expected output.")
    return generate_with_model(config.MODEL, prompt, 0.3, 100)


def _reflect(func_name, source, test_history, selected, predicted, actual,
             learnings, code_visible):
    if code_visible:
        code_section = f"```python\n{source[:2000]}\n```"
    else:
        from ..runner.trace_parser import extract_function_signature
        code_section = f"```python\n{extract_function_signature(source)}\n```"

    prompt = (f"Function:\n{code_section}\n\n"
              f"You predicted: {predicted[:200]}\n"
              f"Actual result: {actual[:200]}\n\n"
              f"{'Previous learnings: ' + learnings if learnings else ''}\n\n"
              f"In 2 sentences: What did you learn? What should you test next?")
    reflection = generate_with_model(config.MODEL, prompt, 0.3, 200)
    return (learnings + "\n" + reflection)[-500:]


def _generate_reflective(func_name, source, test_history, learnings, K,
                         code_visible):
    if code_visible:
        code_section = f"```python\n{source[:2000]}\n```"
    else:
        from ..runner.trace_parser import extract_function_signature
        code_section = f"```python\n{extract_function_signature(source)}\n```"

    history_str = ""
    if test_history:
        for tc, res in test_history[-8:]:
            history_str += f"  {tc} → {res.output or res.exception}\n"

    prompt = (f"Function:\n{code_section}\n\nPrevious results:\n{history_str}\n\n"
              f"{'LEARNINGS: ' + learnings if learnings else ''}\n\n"
              f"Based on what you've learned, generate a test that will "
              f"discover NEW behavior. Focus on passing validation gates "
              f"you haven't passed yet.\n"
              f"Respond with ONLY the function call: {func_name}(...)")

    responses = batch_generate([prompt] * K, temperature=0.9, max_tokens=256)
    candidates = []
    for resp in responses:
        call = parse_candidate(resp, func_name)
        if call and call not in candidates:
            candidates.append(call)
    return candidates
