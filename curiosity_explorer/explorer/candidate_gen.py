"""LLM test input generation with retries and batch support."""

import logging

from ..llm import batch_generate
from ..runner.trace_parser import format_test_history, extract_function_signature

log = logging.getLogger(__name__)


def generate_test_candidates(func_name: str, source_code: str,
                              test_history: list = None, K: int = 5,
                              code_visible: bool = True) -> list[str]:
    """Generate K candidate test inputs using batch API calls."""
    if code_visible:
        code_section = f"```python\n{source_code}\n```"
    else:
        sig = extract_function_signature(source_code)
        code_section = f"```python\n{sig}\n```"

    history_str = format_test_history(test_history or [])

    prompt = f"""You are testing this Python function to discover all its behaviors:

{code_section}

{history_str}

Generate a single test call that would reveal NEW behavior not seen in previous tests.
Respond with ONLY the function call, nothing else. Example format:
{func_name}(arg1, arg2)
"""

    # Batch generate K responses in parallel
    prompts = [prompt] * K
    responses = batch_generate(prompts, temperature=0.9)

    candidates = []
    for response in responses:
        call = _parse_candidate(response, func_name)
        if call:
            candidates.append(call)

    # Retry if we got too few valid candidates
    if len(candidates) < max(1, K // 2):
        retry_count = K - len(candidates)
        log.debug(f"Retrying {retry_count} candidates (got {len(candidates)}/{K})")
        retry_responses = batch_generate([prompt] * retry_count, temperature=0.95)
        for response in retry_responses:
            call = _parse_candidate(response, func_name)
            if call:
                candidates.append(call)

    return candidates


def _parse_candidate(response: str, func_name: str) -> str | None:
    """Parse an LLM response into a valid function call, or None."""
    if not response:
        return None
    call = response.strip().split("\n")[0].strip()
    if call.startswith("```"):
        call = call.strip("`").strip()
    # Must start with the function name (reject garbage prefixed text)
    if not call.startswith(func_name + "("):
        return None
    # Basic balanced-parens check
    if call.count("(") != call.count(")"):
        return None
    return call
