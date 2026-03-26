"""Test coverage-map-guided exploration vs baselines.

Compares:
  random: standard gen, random pick (baseline)
  greedy: standard gen, LLM picks best
  cov_greedy: coverage map shown, target gaps (CoverUp-style)
  cov_planned: coverage map + trajectory planning (Sun et al.)
"""

import random as _random
import re
import time
import json
import logging

import config
from curiosity_explorer.llm import generate_with_model, batch_generate, get_cost, reset_cost
from curiosity_explorer.runner.docker_coverage import DockerCoverageRunner
from curiosity_explorer.explorer.coverage_exploration import (
    CoverageMap, generate_coverage_greedy, generate_coverage_planned,
    select_by_coverage_qvalue, _parse_script,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

STRATEGIES = ["random", "greedy", "cov_greedy", "cov_planned"]
BUDGET = 8
K = 3


def gen_standard(ctx, hist, K):
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
    responses = batch_generate([prompt] * K, temperature=0.9, max_tokens=400)
    return [_parse_script(r) for r in responses if _parse_script(r)]


def select_greedy(scripts, ctx, hist):
    sl = "\n".join(f"  {i+1}. {s.strip()[:100]}" for i, s in enumerate(scripts))
    prompt = (f"{ctx}\nWhich test covers the most NEW code?\n"
              f"{sl}\nRespond with ONLY the number.")
    resp = generate_with_model(config.MODEL, prompt, 0.3, 20)
    for n in re.findall(r'\d+', resp):
        idx = int(n) - 1
        if 0 <= idx < len(scripts):
            return scripts[idx]
    return scripts[0]


def run_strategy(ex, strategy):
    _random.seed(42)
    module = ex["code_file"].replace("/", ".").replace(".py", "")
    code = ex["code_src"]
    version = ex.get("version", "4.0")
    ctx = f"Module: {module}\n```python\n{code[:2000]}\n```"

    runner = DockerCoverageRunner(
        image=f"aorwall/swe-bench-django_django-testbed:{version}",
        source_module=module,
        setup_code="import django; django.setup()",
        working_dir="/opt/django__django",
        env={"DJANGO_SETTINGS_MODULE": "tests.test_sqlite"},
    )

    hist = []
    cov_map = CoverageMap()

    for step in range(BUDGET):
        if strategy == "cov_greedy":
            scripts = generate_coverage_greedy(
                code, module, hist, cov_map, K=K)
        elif strategy == "cov_planned":
            scripts = generate_coverage_planned(
                code, module, hist, cov_map, K=K, plan_length=3)
        else:
            scripts = gen_standard(ctx, hist, K)

        if not scripts:
            continue

        if strategy == "random":
            selected = _random.choice(scripts)
        elif strategy == "greedy":
            selected = select_greedy(scripts, ctx, hist)
        elif strategy == "cov_greedy":
            selected = _random.choice(scripts)  # coverage-aware gen, random select
        elif strategy == "cov_planned":
            # Execute the plan sequentially (scripts are ordered)
            for plan_script in scripts:
                result = runner.run_test(plan_script)
                hist.append((plan_script, result))
                cov_map.update(plan_script, set(), result.new_branches)
                print(f"        plan step: +{result.new_branches}, "
                      f"cum={runner.get_cumulative_coverage()}", flush=True)
            continue  # already executed all scripts in the plan

        result = runner.run_test(selected)
        hist.append((selected, result))
        cov_map.update(selected, set(), result.new_branches)

        print(f"      Step {step+1}: new={result.new_branches}, "
              f"cum={runner.get_cumulative_coverage()}", flush=True)

    return runner.get_cumulative_coverage()


def main():
    reset_cost()

    from datasets import load_dataset
    ds = load_dataset("kjain14/testgenevallite")
    test = ds["test"]
    examples = [ex for ex in test
                if ex["repo"] == "django/django" and ex["version"] == "4.0"]
    examples.sort(key=lambda ex: ex["baseline_covs"]["first"])
    examples = examples[:5]

    print("=" * 70, flush=True)
    print("Coverage Map Exploration Test", flush=True)
    print(f"Budget={BUDGET}, K={K}, Files={len(examples)}", flush=True)
    print(f"Strategies: {STRATEGIES}", flush=True)
    print("=" * 70, flush=True)

    t = generate_with_model(config.MODEL, "Say ok", 0.3, 10)
    print(f"Connectivity: {'OK' if t else 'FAILED'}", flush=True)
    if not t:
        return

    start = time.time()
    all_results = []

    for i, ex in enumerate(examples):
        print(f"\n[{i+1}/{len(examples)}] {ex['code_file']}", flush=True)
        file_results = {"file": ex["code_file"], "strategies": {}}

        for strategy in STRATEGIES:
            print(f"  {strategy}:", flush=True)
            final = run_strategy(ex, strategy)
            file_results["strategies"][strategy] = final
            print(f"    final={final}", flush=True)

        all_results.append(file_results)

    elapsed = time.time() - start
    cost = get_cost()

    print(f"\n{'=' * 70}", flush=True)
    print(f"{'File':<40}", end="", flush=True)
    for s in STRATEGIES:
        print(f" {s:>12}", end="")
    print(flush=True)
    print("-" * 90, flush=True)
    for r in all_results:
        print(f"{r['file']:<40}", end="")
        for s in STRATEGIES:
            print(f" {r['strategies'].get(s, 0):>12}", end="")
        print(flush=True)

    import statistics
    print("-" * 90, flush=True)
    for s in STRATEGIES:
        vals = [r["strategies"].get(s, 0) for r in all_results]
        m = statistics.mean(vals)
        print(f"  {s:<15} mean={m:.1f}", flush=True)

    print(f"\nCost: ${cost['total_cost_usd']:.4f} | "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "coverage_map_test.json", "w") as f:
        json.dump({"results": all_results, "cost": cost}, f, indent=2)
    print("Saved", flush=True)


if __name__ == "__main__":
    main()
