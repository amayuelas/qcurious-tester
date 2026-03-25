"""Shared candidate parsing utility.

Single implementation used by all modules that need to extract
function calls from LLM responses.
"""

import re


def parse_candidate(response: str, func_name: str) -> str | None:
    """Parse an LLM response into a valid function call, or None.

    Handles common LLM output patterns:
    - Direct function call: func_name(args)
    - With prefix text: "Here's a test: func_name(args)"
    - With markdown: ```python\\nfunc_name(args)\\n```
    - With trailing comments: func_name(args)  # test edge case
    """
    if not response:
        return None

    text = response.strip()

    # Remove markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    # Try to find func_name(...) anywhere in the text
    # Use regex to match balanced parens (simple version)
    pattern = rf'{re.escape(func_name)}\('
    match = re.search(pattern, text)
    if not match:
        return None

    # Extract from the match to the end, find balanced closing paren
    start = match.start()
    depth = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    else:
        return None  # no balanced closing paren

    return text[start:end]
