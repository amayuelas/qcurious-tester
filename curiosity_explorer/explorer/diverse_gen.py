"""Diverse candidate generation strategies.

Instead of K identical prompts at temp=0.9, use different prompting
strategies to generate structurally diverse candidates.
"""

import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..llm import llm_generate
from ..runner.trace_parser import format_test_history, extract_function_signature

log = logging.getLogger(__name__)

# Different prompting strategies for diversity
STRATEGIES = [
    {
        "name": "edge_case",
        "instruction": "Generate a test that exercises EDGE CASES — boundary values, empty inputs, zero, None, maximum/minimum values.",
    },
    {
        "name": "error_path",
        "instruction": "Generate a test that is likely to trigger an ERROR or EXCEPTION — invalid types, missing fields, out-of-range values.",
    },
    {
        "name": "deep_logic",
        "instruction": "Generate a test with VALID, well-formed inputs that will reach the DEEPEST logic in the function — pass all validation gates.",
    },
    {
        "name": "typical_usage",
        "instruction": "Generate a test that represents TYPICAL, normal usage of the function — the kind of input it was designed for.",
    },
    {
        "name": "adversarial",
        "instruction": "Generate a test that is ADVERSARIAL — unusual combinations, unexpected types, inputs the developer probably didn't anticipate.",
    },
    {
        "name": "mutation",
        "instruction": "Take the most recent test from the history and MUTATE it — change one argument slightly to explore a nearby code path.",
    },
    {
        "name": "cot_untested",
        "instruction": "First, identify which code paths have NOT been tested yet based on the history. Then generate an input that specifically exercises one of those untested paths.",
    },
]


def generate_diverse_candidates(func_name: str, source_code: str,
                                test_history: list = None, K: int = 5,
                                code_visible: bool = True) -> list[str]:
    """Generate K candidates using different prompting strategies for diversity."""
    if code_visible:
        code_section = f"```python\n{source_code}\n```"
    else:
        sig = extract_function_signature(source_code)
        code_section = f"```python\n{sig}\n```"

    history_str = format_test_history(test_history or [])

    # Pick K strategies (cycle if K > len(STRATEGIES))
    selected = []
    available = list(STRATEGIES)
    random.shuffle(available)
    while len(selected) < K:
        selected.extend(available)
    selected = selected[:K]

    prompts = []
    for strategy in selected:
        prompt = f"""You are testing this Python function to discover all its behaviors:

{code_section}

{history_str}

{strategy["instruction"]}

Generate a single test call. Respond with ONLY the function call, nothing else. Example format:
{func_name}(arg1, arg2)
"""
        prompts.append(prompt)

    # Generate all in parallel
    from ..llm import batch_generate
    responses = batch_generate(prompts, temperature=0.9)

    candidates = []
    for response in responses:
        call = _parse_candidate(response, func_name)
        if call and call not in candidates:  # deduplicate
            candidates.append(call)

    return candidates


def _parse_candidate(response: str, func_name: str) -> str | None:
    """Parse an LLM response into a valid function call, or None."""
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
