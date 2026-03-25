"""Curiosity Q-values: planning-aware test selection.

Implements Sun et al. Proposition 1:
    q(a|h) = ḡ(a|h) + γ · E_o[v(h')]

where:
    ḡ(a|h) = immediate information gain (logprob entropy from single prediction)
    h' = history after executing a and observing o
    v(h') = max over future actions of their information gain from h'
    γ = discount factor (how much to value future vs immediate)

Fixes applied:
  - Immediate IG uses logprob entropy (1 call, no zero-entropy floor)
    instead of sampling entropy (S calls, 79-91% floor)
  - Future value simulates multiple outcome branches (success/failure)
    instead of one-sided most-likely prediction
  - Falls back to sampling entropy when logprobs unavailable
"""

import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from ..llm import batch_generate, generate_with_logprobs
from ..runner.trace_parser import extract_function_signature
from .parse_utils import parse_candidate
from .entropy_utils import string_entropy, logprob_token_entropy

log = logging.getLogger(__name__)


def compute_q_values(func_name: str, source_code: str, test_history: list,
                     candidates: list[str], gamma: float = 0.5,
                     future_K: int = 3, code_visible: bool = True,
                     logprob_model: str = None) -> dict[str, dict]:
    """Compute curiosity Q-values for each candidate.

    Args:
        logprob_model: Model for logprob-based IG (default: config.LOGPROB_MODEL
                       if available, else falls back to sampling)

    Returns dict mapping candidate -> {q_value, immediate_ig, future_value,
    predicted_output}.
    """
    logprob_model = logprob_model or getattr(config, 'LOGPROB_MODEL', None)

    if code_visible:
        code_section = f"```python\n{source_code}\n```"
    else:
        sig = extract_function_signature(source_code)
        code_section = f"```python\n{sig}\n```"

    history_str = _format_history(test_history)

    results = {}

    with ThreadPoolExecutor(max_workers=min(len(candidates), 5)) as executor:
        futures = {
            executor.submit(
                _compute_single_q, func_name, code_section,
                history_str, test_history, cand, gamma, future_K,
                code_visible, logprob_model
            ): cand
            for cand in candidates
        }
        for future in as_completed(futures):
            cand = futures[future]
            try:
                results[cand] = future.result()
            except Exception as e:
                log.warning(f"Q-value failed for {cand}: {e}")
                results[cand] = {"q_value": 0, "immediate_ig": 0,
                                 "future_value": 0, "predicted_output": ""}

    return results


def _compute_single_q(func_name, code_section, history_str,
                      test_history, candidate, gamma, future_K,
                      code_visible, logprob_model):
    """Compute Q-value for a single candidate."""

    predict_prompt = f"""Given this function:
{code_section}

{history_str}

What will be the output of: {candidate}

Respond with ONLY the expected output value, nothing else."""

    # Step 1: Immediate information gain ḡ(a|h) via logprob entropy
    predicted_output = "unknown"
    if logprob_model:
        result = generate_with_logprobs(logprob_model, predict_prompt,
                                        temperature=0.3, max_tokens=100,
                                        top_logprobs=5)
        if result and result["token_logprobs"]:
            immediate_ig = logprob_token_entropy(result["token_logprobs"])
            predicted_output = result["text"].strip()[:100]
        else:
            immediate_ig = 0.0
    else:
        # Fallback: sampling entropy (S=6)
        predictions = batch_generate([predict_prompt] * 6, temperature=0.9,
                                     max_tokens=100)
        predictions = [p.strip().lower()[:100] for p in predictions if p]
        immediate_ig = string_entropy(predictions)
        if predictions:
            predicted_output = Counter(predictions).most_common(1)[0][0]

    if gamma == 0:
        return {
            "q_value": round(immediate_ig, 4),
            "immediate_ig": round(immediate_ig, 4),
            "future_value": 0.0,
            "predicted_output": predicted_output,
        }

    # Step 2: Simulate multiple outcome branches (Fix #7)
    # Instead of one-sided simulation with most-likely output,
    # branch on success vs failure and compute E_o[v(h')]
    future_value = _estimate_future_value_branched(
        func_name, code_section, test_history, candidate,
        predicted_output, future_K, code_visible, logprob_model,
    )

    q_value = immediate_ig + gamma * future_value

    return {
        "q_value": round(q_value, 4),
        "immediate_ig": round(immediate_ig, 4),
        "future_value": round(future_value, 4),
        "predicted_output": predicted_output,
    }


def _estimate_future_value_branched(func_name, code_section, test_history,
                                     candidate, predicted_output, future_K,
                                     code_visible, logprob_model):
    """Estimate E_o[v(h')] by branching on outcome type.

    Simulates two branches (success/failure) in parallel.
    For each branch, generate future candidates and score their IG.
    """
    base_history = _format_history(test_history)
    outcomes = [
        ("succeeds and returns: " + predicted_output[:80], 0.5),
        ("fails with an error or unexpected result", 0.5),
    ]

    def _score_branch(outcome_desc, weight):
        sim_history = base_history + f"  {candidate} → {outcome_desc}\n"

        gen_prompt = f"""Given this function:
{code_section}

{sim_history}

Generate a single test call that would reveal NEW behavior not seen above.
Respond with ONLY the function call. Example: {func_name}(arg1, arg2)"""

        future_responses = batch_generate(
            [gen_prompt] * future_K, temperature=0.9, max_tokens=256
        )
        future_candidates = []
        for resp in future_responses:
            call = parse_candidate(resp, func_name)
            if call and call not in future_candidates:
                future_candidates.append(call)

        if not future_candidates:
            return 0.0

        # Score future candidates in parallel
        def _score_future(fc):
            fc_prompt = f"""Given this function:
{code_section}

{sim_history}

What will be the output of: {fc}

Respond with ONLY the expected output value, nothing else."""

            if logprob_model:
                fr = generate_with_logprobs(logprob_model, fc_prompt,
                                            temperature=0.3, max_tokens=100,
                                            top_logprobs=5)
                if fr and fr["token_logprobs"]:
                    return logprob_token_entropy(fr["token_logprobs"])
                return 0.0
            else:
                preds = batch_generate([fc_prompt] * 3, temperature=0.9,
                                       max_tokens=100)
                preds = [p.strip().lower()[:100] for p in preds if p]
                return string_entropy(preds)

        with ThreadPoolExecutor(max_workers=len(future_candidates)) as ex:
            igs = list(ex.map(_score_future, future_candidates))

        return weight * max(igs) if igs else 0.0

    # Run both branches in parallel
    with ThreadPoolExecutor(max_workers=2) as ex:
        branch_futures = [
            ex.submit(_score_branch, desc, w) for desc, w in outcomes
        ]
        return sum(f.result() for f in branch_futures)


def _format_history(test_history):
    """Format test history for prompts."""
    if not test_history:
        return ""
    recent = test_history[-8:]
    lines = ["Known test results:\n"]
    for test_code, result in recent:
        output = result.output or result.exception or "None"
        lines.append(f"  {test_code} → {output}\n")
    return "".join(lines)
