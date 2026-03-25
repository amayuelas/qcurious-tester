"""LLM test input generation with retries and batch support."""

import logging

from ..llm import batch_generate
from ..runner.trace_parser import format_test_history, extract_function_signature
from .parse_utils import parse_candidate as _shared_parse

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
        call = _shared_parse(response, func_name)
        if call:
            candidates.append(call)

    # Retry if we got too few valid candidates
    if len(candidates) < max(1, K // 2):
        retry_count = K - len(candidates)
        log.debug(f"Retrying {retry_count} candidates (got {len(candidates)}/{K})")
        retry_responses = batch_generate([prompt] * retry_count, temperature=0.95)
        for response in retry_responses:
            call = _shared_parse(response, func_name)
            if call:
                candidates.append(call)

    return candidates


# _parse_candidate moved to parse_utils.py
