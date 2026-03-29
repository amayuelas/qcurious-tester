"""Coverage-map-guided exploration with trajectory planning.

Direct instantiation of Sun et al. (2011) for code exploration:

    Θ = program's branch reachability function: input → set of branches hit
    p(Θ|h) = coverage map after running tests h (which branches are known reachable)
    ḡ(a|h) = I(O;Θ|h,a) = expected new branches discovered by test a
    q(a|h) = ḡ(a|h) + γ·E[v(h')] = immediate discovery + future reachability

The coverage map IS the Bayesian posterior. Coverage growth IS learning progress.
Frontier branches (adjacent to covered code) are where information gain is highest.

Strategies:
    coverage_greedy: Show coverage map, target nearest uncovered branches (CoverUp-style)
    coverage_planned: Show coverage map, plan trajectory of 3 tests to reach deep branches
    coverage_qvalue: Generate K plans, score each by Q-value, execute the best plan
"""

import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor

from ..llm import generate_with_model, batch_generate

log = logging.getLogger(__name__)


class CoverageMap:
    """Tracks the posterior p(Θ|h) — what we know about branch reachability."""

    def __init__(self):
        self.covered_branches = set()   # branches we've confirmed are reachable
        self.total_branches = 0         # total branches in the program (if known)
        self.test_to_branches = {}      # which test covered which branches
        self.step_coverage = []         # coverage at each step

    def update(self, test_code, branches_hit, new_branches):
        """Bayesian update: incorporate observation from running a test."""
        self.covered_branches.update(branches_hit)
        self.test_to_branches[test_code[:80]] = len(branches_hit)
        self.step_coverage.append(len(self.covered_branches))

    def coverage_summary(self):
        """Format the posterior for the LLM prompt."""
        n = len(self.covered_branches)
        parts = [f"COVERAGE MAP (Bayesian posterior — what you know about the program):"]
        parts.append(f"  Branches discovered: {n}")
        if self.total_branches:
            parts.append(f"  Estimated total branches: {self.total_branches}")
            parts.append(f"  Coverage: {100*n/self.total_branches:.0f}%")

        # Learning progress (coverage growth rate)
        if len(self.step_coverage) >= 2:
            recent_growth = self.step_coverage[-1] - self.step_coverage[-2]
            avg_growth = self.step_coverage[-1] / len(self.step_coverage)
            parts.append(f"  Last step discovered: {recent_growth} new branches")
            parts.append(f"  Average per step: {avg_growth:.1f} branches")

            if recent_growth == 0 and len(self.step_coverage) >= 3:
                stagnant = sum(1 for i in range(max(0, len(self.step_coverage)-3),
                                                 len(self.step_coverage))
                               if i > 0 and self.step_coverage[i] == self.step_coverage[i-1])
                if stagnant >= 2:
                    parts.append(f"  WARNING: Coverage has stagnated for {stagnant} steps")
                    parts.append(f"  You may need a fundamentally different approach")

        # Which tests were most informative
        if self.test_to_branches:
            best = sorted(self.test_to_branches.items(), key=lambda x: -x[1])[:3]
            parts.append(f"  Most informative tests:")
            for test, count in best:
                parts.append(f"    {test} → {count} branches")

        return "\n".join(parts)


def generate_coverage_greedy(source, module_name, test_history, coverage_map,
                              K=3):
    """Generate tests targeting uncovered branches (CoverUp-style).

    Shows the coverage map and asks for tests targeting gaps.
    No planning — just immediate coverage maximization.
    """
    code_section = f"```python\n{source[:2500]}\n```"
    cov_summary = coverage_map.coverage_summary()

    history_str = ""
    if test_history:
        for tc, res in test_history[-3:]:
            out = (res.output or res.exception or "None")[:50]
            history_str += f"  {tc.strip()[:80]} → {out} (new branches: {res.new_branches})\n"

    prompt = f"""Module: {module_name}
{code_section}

{cov_summary}

Previous tests:
{history_str}

Write a test script (5-10 lines) that covers branches NOT YET discovered.
Look at the code and target specific uncovered code paths.
Import from the module and print results.

IMPORTANT: Django is already configured. Do NOT call settings.configure() or django.setup().
Respond with ONLY executable Python code.

```python
"""

    responses = batch_generate([prompt] * K, temperature=0.9, max_tokens=500)
    scripts = []
    for resp in responses:
        script = _parse_script(resp)
        if script and script not in scripts:
            scripts.append(script)
    return scripts


def generate_coverage_planned(source, module_name, test_history, coverage_map,
                               K=3, plan_length=3):
    """Generate a PLAN of tests to reach deep uncovered branches.

    Sun et al.'s Q-value: value tests for their FUTURE coverage potential,
    not just immediate discovery. The plan navigates through corridors
    (setup steps that enable access to deep logic).
    """
    code_section = f"```python\n{source[:2500]}\n```"
    cov_summary = coverage_map.coverage_summary()

    history_str = ""
    if test_history:
        for tc, res in test_history[-3:]:
            out = (res.output or res.exception or "None")[:50]
            history_str += f"  {tc.strip()[:80]} → {out} (new branches: {res.new_branches})\n"

    prompt = f"""Module: {module_name}
{code_section}

{cov_summary}

Previous tests:
{history_str}

PLAN a sequence of {plan_length} test scripts that TOGETHER will reach the
DEEPEST uncovered code paths. Think about what setup is needed:

Step 1: What basic setup/import is needed to reach deeper code?
Step 2: Building on step 1's result, what exercises the next layer?
Step 3: Now target the deepest uncovered branches.

For each step, write a separate test script (5-10 lines each).
Import from the module and print results.

IMPORTANT: Django is already configured. Do NOT call settings.configure() or django.setup().

Format your response as:
### TEST 1
```python
[code]
```
### TEST 2
```python
[code]
```
### TEST 3
```python
[code]
```
"""

    # Generate one plan (not K plans — the plan is the unit)
    response = generate_with_model("gemini-3-flash-preview", prompt,
                                    temperature=0.7, max_tokens=1500)

    # Parse multiple test scripts from the plan
    scripts = _parse_plan(response)

    if not scripts:
        # Fallback: generate individual scripts
        return generate_coverage_greedy(source, module_name, test_history,
                                        coverage_map, K)

    return scripts


def generate_coverage_qvalue(source, module_name, test_history, coverage_map,
                              K=3, plan_length=3, gamma=0.5):
    """Generate K plans, score each by Q-value, return the best plan.

    This is the full Sun et al. algorithm:
    1. Generate K candidate trajectories (plans)
    2. For each plan, estimate Q-value = ḡ(plan|h) + γ·E[v(h')]
       where ḡ = expected immediate branches, E[v(h')] = future reachability
    3. Select and return the plan with highest Q-value

    The Q-value scoring is done by the LLM estimating both terms
    given the source code, coverage map, and plan contents.
    """
    # Generate K plans in parallel
    plans = _generate_k_plans(source, module_name, test_history, coverage_map,
                               K=K, plan_length=plan_length)

    if not plans:
        # Fallback to greedy single-script generation
        return generate_coverage_greedy(source, module_name, test_history,
                                        coverage_map, K)

    if len(plans) == 1:
        return plans[0]

    # Score each plan by Q-value
    best_plan = _score_and_select_plan(plans, source, module_name,
                                        coverage_map, gamma=gamma)
    return best_plan


def _generate_k_plans(source, module_name, test_history, coverage_map,
                       K=3, plan_length=3):
    """Generate K diverse trajectory plans in parallel."""
    code_section = f"```python\n{source[:2500]}\n```"
    cov_summary = coverage_map.coverage_summary()

    history_str = ""
    if test_history:
        for tc, res in test_history[-3:]:
            out = (res.output or res.exception or "None")[:50]
            history_str += f"  {tc.strip()[:80]} → {out} (new branches: {res.new_branches})\n"

    # Different diversity prompts for each plan
    diversity_hints = [
        "Focus on the MAIN functionality — constructors, primary methods.",
        "Focus on ERROR HANDLING — invalid inputs, edge cases, exceptions.",
        "Focus on INTERACTIONS — create objects, pass them to each other, chain calls.",
        "Focus on CONFIGURATION — different parameter combinations, options, flags.",
        "Focus on RARELY-USED features — optional arguments, deprecated paths, callbacks.",
    ]

    prompts = []
    for i in range(K):
        hint = diversity_hints[i % len(diversity_hints)]
        prompt = f"""Module: {module_name}
{code_section}

{cov_summary}

Previous tests:
{history_str}

PLAN a sequence of {plan_length} test scripts that TOGETHER will reach
UNCOVERED code paths. {hint}

Think about what setup is needed:
Step 1: What basic setup/import is needed to reach deeper code?
Step 2: Building on step 1's result, what exercises the next layer?
Step 3: Now target the deepest uncovered branches.

For each step, write a separate test script (5-10 lines each).
Import from the module and print results.

IMPORTANT: Django is already configured. Do NOT call settings.configure() or django.setup().

Format your response as:
### TEST 1
```python
[code]
```
### TEST 2
```python
[code]
```
### TEST 3
```python
[code]
```
"""
        prompts.append(prompt)

    # Generate K plans in parallel via batch_generate
    responses = batch_generate(prompts, temperature=0.9, max_tokens=1500)

    plans = []
    for resp in responses:
        scripts = _parse_plan(resp)
        if scripts:
            plans.append(scripts)

    return plans


def _score_and_select_plan(plans, source, module_name, coverage_map, gamma=0.5):
    """Score each plan by Q-value and return the best one.

    For each plan, we ask the LLM to estimate:
      ḡ(plan|h) = total new branches the plan will likely discover
      E[v(h')]  = additional branches that become reachable AFTER this plan

    Q(plan|h) = ḡ(plan|h) + γ · E[v(h')]
    """
    code_section = f"```python\n{source[:2000]}\n```"
    cov_summary = coverage_map.coverage_summary()

    def score_plan(plan_idx, plan):
        # Format the plan for scoring
        plan_str = ""
        for i, script in enumerate(plan):
            plan_str += f"\nStep {i+1}:\n```python\n{script[:200]}\n```\n"

        prompt = f"""Module: {module_name}
{code_section}

{cov_summary}

Consider this TEST PLAN (a sequence of {len(plan)} scripts to execute in order):
{plan_str}

Evaluate this plan by answering TWO questions with just numbers:

1. IMMEDIATE GAIN: How many total NEW branches (not yet covered) will
   this sequence of tests likely discover? Consider that later steps
   build on earlier ones. (0-50)

2. FUTURE VALUE: After executing this plan, how many ADDITIONAL branches
   become reachable by future tests that weren't reachable before?
   Consider what state/objects/setup the plan leaves behind. (0-50)

Format: immediate, future
Example: 15, 25"""

        resp = generate_with_model("gemini-3-flash-preview", prompt,
                                    temperature=0.3, max_tokens=50)
        immediate, future = _parse_scores(resp)
        q = immediate + gamma * future
        log.info(f"Plan {plan_idx}: ḡ={immediate}, γE[v]={gamma*future:.1f}, Q={q:.1f}")
        return q

    # Score all plans in parallel
    scores = {}
    with ThreadPoolExecutor(max_workers=len(plans)) as ex:
        futures = {ex.submit(score_plan, i, p): i for i, p in enumerate(plans)}
        for future in futures:
            idx = futures[future]
            try:
                scores[idx] = future.result()
            except Exception as e:
                log.warning(f"Plan {idx} scoring failed: {e}")
                scores[idx] = 0

    best_idx = max(scores, key=scores.get)
    log.info(f"Selected plan {best_idx} with Q={scores[best_idx]:.1f} "
             f"(scores: {[f'{scores[i]:.1f}' for i in range(len(plans))]})")
    return plans[best_idx]


def generate_plans_for_exec_selection(source, module_name, test_history,
                                       coverage_map, K=3, plan_length=3):
    """Generate K diverse plans and return them for execution-based selection.

    Instead of LLM-estimated Q-values, the RUNNER executes step 1 of each
    plan and selects the plan whose step 1 discovered the most branches.
    This is a 1-step lookahead with real observation — the truest form of
    Sun et al.'s framework.

    Returns:
        list of plans, where each plan is a list of S scripts.
        The runner should:
        1. Execute plan[0] for each plan → observe actual coverage
        2. Pick the plan with highest step-1 coverage
        3. Execute plan[1:] of the winning plan
    """
    plans = _generate_k_plans(source, module_name, test_history, coverage_map,
                               K=K, plan_length=plan_length)
    if not plans:
        scripts = generate_coverage_greedy(source, module_name, test_history,
                                            coverage_map, K)
        return [[s] for s in scripts] if scripts else []

    return plans


def select_by_coverage_qvalue(scripts, source, module_name, test_history,
                                coverage_map, gamma=0.5):
    """Select the script with highest Q-value based on coverage potential.

    q(a|h) = estimated_new_branches(a) + γ × estimated_future_reachability(a)

    Uses the LLM to estimate both terms based on code analysis + coverage map.
    """
    code_section = f"```python\n{source[:2000]}\n```"
    cov_summary = coverage_map.coverage_summary()

    def score_script(script):
        short = script.strip()[:200]
        prompt = f"""Module: {module_name}
{code_section}

{cov_summary}

Consider this test script:
```python
{short}
```

Answer TWO questions with just numbers:
1. How many NEW branches (not yet covered) will this test likely hit? (0-20)
2. After running this test, how many ADDITIONAL branches become reachable
   by future tests that weren't reachable before? (0-30)

Format: immediate, future
Example: 5, 12"""

        resp = generate_with_model("gemini-3-flash-preview", prompt,
                                    temperature=0.3, max_tokens=50)
        immediate, future = _parse_scores(resp)
        return immediate + gamma * future

    # Score all scripts in parallel
    scores = {}
    with ThreadPoolExecutor(max_workers=len(scripts)) as ex:
        futures = {ex.submit(score_script, s): id(s) for s in scripts}
        for future in futures:
            sid = futures[future]
            try:
                scores[sid] = future.result()
            except Exception:
                scores[sid] = 0

    return max(scripts, key=lambda s: scores.get(id(s), 0))


def _parse_scores(resp):
    """Parse 'immediate, future' from LLM response."""
    numbers = re.findall(r'\d+', resp)
    if len(numbers) >= 2:
        return int(numbers[0]), int(numbers[1])
    elif len(numbers) == 1:
        return int(numbers[0]), 0
    return 0, 0


def _parse_plan(response):
    """Parse multiple test scripts from a trajectory plan."""
    if not response:
        return []

    scripts = []
    # Split on ### TEST or ```python markers
    parts = re.split(r'###\s*TEST\s*\d+', response)

    for part in parts:
        # Extract code blocks
        code_match = re.search(r'```python\s*\n(.*?)```', part, re.DOTALL)
        if code_match:
            script = code_match.group(1).strip()
            if script and len(script) > 10:
                # Clean up
                lines = script.split("\n")
                lines = [l for l in lines if "django.setup()" not in l
                         and l.strip() != "import django"
                         and "settings.configure" not in l]
                script = "\n".join(lines).strip()
                if script:
                    scripts.append(script)

    return scripts


def _parse_script(response):
    """Parse a single script from LLM response."""
    if not response:
        return None
    code = response.strip()
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()
    if not code or len(code) < 10:
        return None
    if "subprocess" in code or "os.system" in code:
        return None
    # Remove Django setup if present
    lines = code.split("\n")
    lines = [l for l in lines if "django.setup()" not in l
             and l.strip() != "import django"
             and "settings.configure" not in l]
    code = "\n".join(lines).strip()
    # Reject non-code
    first_line = code.split("\n")[0].strip() if code else ""
    if not any(first_line.startswith(kw) for kw in
               ["import ", "from ", "#", "class ", "def ", "print(",
                "try:", "with ", "for ", "if ", "assert "]) \
       and "=" not in first_line and "(" not in first_line:
        return None
    return code
