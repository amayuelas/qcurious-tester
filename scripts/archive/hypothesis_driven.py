"""Hypothesis-driven exploration — Schmidhuber-inspired approach.

Instead of implicit curiosity (entropy), make the world model explicit:
1. After each test, ask the LLM to update its beliefs about the program
2. Before selecting, ask it to formulate hypotheses about unexplored behavior
3. Select tests that discriminate between competing hypotheses

This directly implements the Bayesian update cycle:
  - Prior: LLM's beliefs about program behavior
  - Action: test input
  - Observation: execution result
  - Posterior: updated beliefs
  - Next action: test that maximally reduces remaining uncertainty
"""

import logging
import random
import re

from ..llm import generate_with_model, batch_generate
from ..runner.trace_parser import extract_function_signature

log = logging.getLogger(__name__)


def generate_with_hypotheses(func_name: str, source_code: str,
                              test_history: list, K: int = 5,
                              code_visible: bool = True,
                              model: str = None) -> tuple[list[str], dict]:
    """Generate K candidates guided by explicit hypothesis testing.

    Returns (candidates, metadata) where metadata includes the model's
    current beliefs and hypotheses.
    """
    import config
    model = model or config.MODEL

    if code_visible:
        code_section = f"```python\n{source_code}\n```"
    else:
        sig = extract_function_signature(source_code)
        code_section = f"```python\n{sig}\n```"

    # Step 1: Ask the model to summarize what it's learned so far
    beliefs = ""
    if test_history:
        history_str = ""
        for tc, res in test_history[-8:]:
            out = res.output or res.exception or "None"
            history_str += f"  {tc} → {out}\n"

        belief_prompt = f"""Here is a function and the test results so far:

{code_section}

Test results:
{history_str}

Based on these results, briefly summarize:
1. What you now know about this function's behavior (what input patterns work, what causes errors)
2. What you're still UNCERTAIN about (what behaviors haven't been tested yet)
3. What you think the function does overall

Be concise — 3-5 sentences total."""

        beliefs = generate_with_model(model, belief_prompt, temperature=0.3,
                                      max_tokens=300)

    # Step 2: Generate hypotheses about unexplored behavior
    if beliefs:
        hypo_prompt = f"""Here is a function:

{code_section}

Your current understanding:
{beliefs}

List 3 SPECIFIC HYPOTHESES about behaviors you haven't confirmed yet.
Each hypothesis should be testable with a single function call.

Format:
H1: [hypothesis] → Test: {func_name}(...)
H2: [hypothesis] → Test: {func_name}(...)
H3: [hypothesis] → Test: {func_name}(...)"""
    else:
        hypo_prompt = f"""Here is a function you've never tested:

{code_section}

List 3 SPECIFIC HYPOTHESES about what this function does and how to test each one.
Start with the most basic: what type of input does it expect?

Format:
H1: [hypothesis] → Test: {func_name}(...)
H2: [hypothesis] → Test: {func_name}(...)
H3: [hypothesis] → Test: {func_name}(...)"""

    hypo_response = generate_with_model(model, hypo_prompt, temperature=0.7,
                                        max_tokens=500)

    # Parse hypothesis tests
    candidates = []
    hypotheses = []
    for line in hypo_response.split("\n"):
        # Look for Test: func_name(...) patterns
        match = re.search(rf'{func_name}\([^)]*\)', line)
        if match:
            call = match.group(0)
            if call.count("(") == call.count(")"):
                candidates.append(call)
                # Extract hypothesis text
                h_match = re.match(r'H\d+:\s*(.+?)→', line)
                if h_match:
                    hypotheses.append(h_match.group(1).strip())

    # Fill remaining slots with standard diverse generation
    if len(candidates) < K:
        fill_prompts = []
        strategies = [
            "Generate an input that would CONFIRM your current understanding.",
            "Generate an input that might SURPRISE you — something you're uncertain about.",
            "Generate an input as DIFFERENT as possible from all previous tests.",
        ]
        random.shuffle(strategies)

        history_str = ""
        if test_history:
            for tc, res in test_history[-5:]:
                out = res.output or res.exception or "None"
                history_str += f"  {tc} → {out}\n"

        for strat in strategies[:K - len(candidates)]:
            prompt = f"""{code_section}

{"Current understanding: " + beliefs if beliefs else ""}

Previous tests:
{history_str if history_str else "  (none yet)"}

{strat}
Respond with ONLY the function call:
{func_name}(...)"""
            fill_prompts.append(prompt)

        if fill_prompts:
            responses = batch_generate(fill_prompts, temperature=0.9)
            for resp in responses:
                call = _parse(resp, func_name)
                if call and call not in candidates:
                    candidates.append(call)

    metadata = {
        "beliefs": beliefs,
        "hypotheses": hypotheses,
        "n_hypothesis_tests": min(len(hypotheses), len(candidates)),
    }

    return candidates[:K], metadata


def select_hypothesis_discriminating(candidates: list[str], func_name: str,
                                      source_code: str, test_history: list,
                                      beliefs: str = "",
                                      code_visible: bool = True,
                                      model: str = None) -> str:
    """Select the candidate that best discriminates between hypotheses.

    Ask the model: "For which candidate are you LEAST confident about
    the outcome?" This directly targets information gain.
    """
    import config
    model = model or config.MODEL

    if code_visible:
        code_section = f"```python\n{source_code}\n```"
    else:
        sig = extract_function_signature(source_code)
        code_section = f"```python\n{sig}\n```"

    cand_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(candidates))

    prompt = f"""{code_section}

{"Current understanding: " + beliefs if beliefs else ""}

For which of these tests are you LEAST SURE what the output will be?
The test where you're most uncertain is the most informative to run.

Candidates:
{cand_list}

Respond with ONLY the number of the test you're MOST UNCERTAIN about."""

    resp = generate_with_model(model, prompt, temperature=0.3, max_tokens=20)
    for n in re.findall(r'\d+', resp):
        idx = int(n) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    return candidates[0]


def _parse(response, func_name):
    if not response:
        return None
    call = response.strip().split("\n")[0].strip()
    if call.startswith("```"):
        call = call.strip("`").strip()
    match = re.search(rf'{func_name}\([^)]*\)', call)
    if match:
        call = match.group(0)
        if call.count("(") == call.count(")"):
            return call
    if call.startswith(func_name + "(") and call.count("(") == call.count(")"):
        return call
    return None
