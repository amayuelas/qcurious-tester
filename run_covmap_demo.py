"""Demo: show exactly what each method generates for one file."""

import random as _random
import re
import logging

import config
from curiosity_explorer.llm import generate_with_model, batch_generate
from curiosity_explorer.runner.docker_coverage import DockerCoverageRunner
from curiosity_explorer.explorer.coverage_exploration import (
    CoverageMap, _parse_script,
)

logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

_random.seed(42)


def demo_random(ex, steps=3):
    """Show what random does."""
    module = ex["code_file"].replace("/", ".").replace(".py", "")
    code = ex["code_src"]
    ctx = f"Module: {module}\n```python\n{code[:2000]}\n```"

    runner = DockerCoverageRunner(
        image=f"aorwall/swe-bench-django_django-testbed:4.0",
        source_module=module,
        setup_code="import django; django.setup()",
        working_dir="/opt/django__django",
        env={"DJANGO_SETTINGS_MODULE": "tests.test_sqlite"},
    )

    hist = []
    for step in range(steps):
        h = ""
        if hist:
            h = "\nPrevious:\n"
            for s, r in hist[-3:]:
                out = (r.output or r.exception or "None")[:50]
                h += f"  {s.strip()[:80]} -> {out}\n"
        prompt = (f"{ctx}\n{h}\nWrite a test script (5-10 lines). "
                  f"Import from the module and print results.\n"
                  f"IMPORTANT: Django is already configured. "
                  f"Do NOT call settings.configure() or django.setup().\n\n```python\n")

        if step == 0:
            print("=" * 70)
            print("RANDOM — Prompt (step 1):")
            print("=" * 70)
            print(prompt[:500])
            print("...\n")

        responses = batch_generate([prompt] * 3, temperature=0.9, max_tokens=400)
        scripts = [_parse_script(r) for r in responses if _parse_script(r)]
        if not scripts:
            continue

        selected = _random.choice(scripts)
        print(f"--- Random step {step+1}: selected script ---")
        print(selected[:300])
        print()

        result = runner.run_test(selected)
        hist.append((selected, result))
        out = result.output or result.exception or "None"
        print(f"  -> new_branches={result.new_branches}, "
              f"cumulative={runner.get_cumulative_coverage()}")
        print(f"  -> output: {out[:100]}")
        print()

    print(f"RANDOM FINAL: {runner.get_cumulative_coverage()} branches\n")


def demo_cov_planned(ex, steps=3):
    """Show what cov_planned does."""
    module = ex["code_file"].replace("/", ".").replace(".py", "")
    code = ex["code_src"]

    runner = DockerCoverageRunner(
        image=f"aorwall/swe-bench-django_django-testbed:4.0",
        source_module=module,
        setup_code="import django; django.setup()",
        working_dir="/opt/django__django",
        env={"DJANGO_SETTINGS_MODULE": "tests.test_sqlite"},
    )

    hist = []
    cov_map = CoverageMap()

    for step in range(steps):
        code_section = f"```python\n{code[:2500]}\n```"
        cov_summary = cov_map.coverage_summary()
        history_str = ""
        if hist:
            for tc, res in hist[-3:]:
                out = (res.output or res.exception or "None")[:50]
                history_str += f"  {tc.strip()[:80]} -> {out} (new branches: {res.new_branches})\n"

        prompt = f"""Module: {module}
{code_section}

{cov_summary}

Previous tests:
{history_str}

PLAN a sequence of 3 test scripts that TOGETHER will reach the
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

        print("=" * 70)
        print(f"COV_PLANNED — Plan {step+1} prompt (coverage map section):")
        print("=" * 70)
        print(cov_summary)
        if history_str:
            print(f"\nPrevious tests:\n{history_str}")
        print()

        response = generate_with_model("gemini-3-flash-preview", prompt,
                                        temperature=0.7, max_tokens=1500)

        # Parse plan
        scripts = []
        parts = re.split(r'###\s*TEST\s*\d+', response)
        for part in parts:
            code_match = re.search(r'```python\s*\n(.*?)```', part, re.DOTALL)
            if code_match:
                script = code_match.group(1).strip()
                if script and len(script) > 10:
                    lines = script.split("\n")
                    lines = [l for l in lines if "django.setup()" not in l
                             and l.strip() != "import django"
                             and "settings.configure" not in l]
                    script = "\n".join(lines).strip()
                    if script:
                        scripts.append(script)

        if not scripts:
            print("  (no scripts parsed from plan)\n")
            continue

        for i, plan_script in enumerate(scripts):
            print(f"--- Plan {step+1}, Test {i+1}/{len(scripts)} ---")
            print(plan_script[:300])
            print()

            result = runner.run_test(plan_script)
            hist.append((plan_script, result))
            cov_map.update(plan_script, set(), result.new_branches)
            out = result.output or result.exception or "None"
            print(f"  -> new_branches={result.new_branches}, "
                  f"cumulative={runner.get_cumulative_coverage()}")
            print(f"  -> output: {out[:100]}")
            print()

    print(f"COV_PLANNED FINAL: {runner.get_cumulative_coverage()} branches\n")


if __name__ == "__main__":
    from datasets import load_dataset
    ds = load_dataset("kjain14/testgenevallite")
    test = ds["test"]
    examples = [ex for ex in test
                if ex["repo"] == "django/django" and ex["version"] == "4.0"]
    examples.sort(key=lambda ex: ex["baseline_covs"]["first"])

    # Use formsets.py — the clearest corridor example
    ex = examples[1]  # django/forms/formsets.py
    print(f"\nDEMO FILE: {ex['code_file']}\n")

    print("\n" + "=" * 70)
    print("PART 1: RANDOM (3 steps)")
    print("=" * 70 + "\n")
    demo_random(ex, steps=3)

    print("\n" + "=" * 70)
    print("PART 2: COV_PLANNED (2 plans = 6 tests)")
    print("=" * 70 + "\n")
    demo_cov_planned(ex, steps=2)
