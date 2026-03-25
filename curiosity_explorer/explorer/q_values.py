"""Curiosity Q-values: planning-aware test selection.

Implements Sun et al. Proposition 1:
    q(a|h) = ḡ(a|h) + γ · E_o[v(h')]

where:
    ḡ(a|h) = immediate information gain (output prediction entropy)
    h' = history after executing a and observing o
    v(h') = max over future actions of their information gain from h'
    γ = discount factor (how much to value future vs immediate)

The key insight: a test that passes validation gates has low ḡ
(model can predict the output) but high γ·E[v(h')] (the updated
history enables much better future candidates).
"""

import logging
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..llm import batch_generate, generate_with_model
from ..runner.trace_parser import extract_function_signature

log = logging.getLogger(__name__)


def compute_q_values(func_name: str, source_code: str, test_history: list,
                     candidates: list[str], gamma: float = 0.5,
                     S: int = 8, future_K: int = 3,
                     code_visible: bool = True) -> dict[str, dict]:
    """Compute curiosity Q-values for each candidate.

    Returns dict mapping candidate -> {q_value, immediate_ig, future_value,
    predicted_output}.
    """
    if code_visible:
        code_section = f"```python\n{source_code}\n```"
    else:
        sig = extract_function_signature(source_code)
        code_section = f"```python\n{sig}\n```"

    history_str = _format_history(test_history, code_visible)

    results = {}

    # Score all candidates in parallel
    with ThreadPoolExecutor(max_workers=min(len(candidates), 5)) as executor:
        futures = {
            executor.submit(
                _compute_single_q, func_name, source_code, code_section,
                history_str, test_history, cand, gamma, S, future_K,
                code_visible
            ): cand
            for cand in candidates
        }
        for future in as_completed(futures):
            cand = futures[future]
            try:
                results[cand] = future.result()
            except Exception as e:
                log.warning(f"Q-value computation failed for {cand}: {e}")
                results[cand] = {"q_value": 0, "immediate_ig": 0,
                                 "future_value": 0, "predicted_output": ""}

    return results


def _compute_single_q(func_name, source_code, code_section, history_str,
                      test_history, candidate, gamma, S, future_K,
                      code_visible):
    """Compute Q-value for a single candidate."""

    # Step 1: Immediate information gain ḡ(a|h)
    predict_prompt = f"""Given this function:
{code_section}

{history_str}

What will be the output of: {candidate}

Respond with ONLY the expected output value, nothing else."""

    predictions = batch_generate([predict_prompt] * S, temperature=0.9,
                                 max_tokens=100)
    predictions = [p.strip().lower()[:100] for p in predictions if p]
    immediate_ig = _entropy(predictions)

    # Get the most common prediction as the "expected output"
    if predictions:
        predicted_output = Counter(predictions).most_common(1)[0][0]
    else:
        predicted_output = "unknown"

    # Step 2: Simulate the future — E_o[v(h')]
    # Create simulated history with the predicted result
    future_value = _estimate_future_value(
        func_name, source_code, code_section, test_history,
        candidate, predicted_output, S, future_K, code_visible
    )

    # Step 3: Q-value
    q_value = immediate_ig + gamma * future_value

    return {
        "q_value": round(q_value, 4),
        "immediate_ig": round(immediate_ig, 4),
        "future_value": round(future_value, 4),
        "predicted_output": predicted_output,
    }


def _estimate_future_value(func_name, source_code, code_section,
                           test_history, candidate, predicted_output,
                           S, future_K, code_visible):
    """Estimate v(h') = max over future candidates of their info gain.

    Simulates: what if we ran `candidate` and got `predicted_output`?
    Then generates future candidates from that state and measures
    the best future information gain.
    """
    # Build simulated updated history
    sim_history_str = _format_history(test_history, code_visible)
    sim_history_str += f"  {candidate} → {predicted_output}\n"

    # Generate future candidate prompts
    if code_visible:
        gen_prompt = f"""You are testing this Python function:
{code_section}

{sim_history_str}

Generate a single test call that would reveal NEW behavior not seen above.
Respond with ONLY the function call, nothing else. Example:
{func_name}(arg1, arg2)"""
    else:
        sig = f"```python\n{code_section}\n```" if "```" not in code_section else code_section
        gen_prompt = f"""Function:
{code_section}

{sim_history_str}

Generate a single test call that would reveal NEW behavior.
Respond with ONLY the function call. Example:
{func_name}(arg1, arg2)"""

    # Generate future_K candidates from simulated state
    future_responses = batch_generate([gen_prompt] * future_K,
                                      temperature=0.9, max_tokens=256)
    future_candidates = []
    for resp in future_responses:
        call = _parse_candidate(resp, func_name)
        if call:
            future_candidates.append(call)

    if not future_candidates:
        return 0.0

    # Score each future candidate's information gain from the simulated state
    future_igs = []
    for fc in future_candidates:
        predict_prompt = f"""Given this function:
{code_section}

{sim_history_str}

What will be the output of: {fc}

Respond with ONLY the expected output value, nothing else."""

        # Use fewer samples for future predictions (cheaper)
        future_preds = batch_generate([predict_prompt] * max(3, S // 2),
                                      temperature=0.9, max_tokens=100)
        future_preds = [p.strip().lower()[:100] for p in future_preds if p]
        future_igs.append(_entropy(future_preds))

    # v(h') = max future information gain
    return max(future_igs) if future_igs else 0.0


def _format_history(test_history, code_visible=True):
    """Format test history for prompts."""
    if not test_history:
        return ""
    recent = test_history[-5:]
    lines = ["Known test results:\n"]
    for test_code, result in recent:
        output = result.output or result.exception or "None"
        lines.append(f"  {test_code} → {output}\n")
    return "".join(lines)


def _entropy(predictions):
    """Shannon entropy over prediction strings."""
    if not predictions:
        return 0.0
    counts = Counter(predictions)
    total = len(predictions)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _parse_candidate(response, func_name):
    if not response:
        return None
    call = response.strip().split("\n")[0].strip()
    if call.startswith("```"):
        call = call.strip("`").strip()
    if not call.startswith(func_name + "("):
        return None
    if call.count("(") != call.count(")"):
        return None
    return call
