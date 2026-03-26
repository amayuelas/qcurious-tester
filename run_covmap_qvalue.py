"""Fair test of coverage Q-value plan selection.

Compares (all with equalized execution budget):
  random:         standard gen, random pick
  greedy:         standard gen, LLM picks best
  cov_greedy:     coverage-aware gen, random pick
  cov_planned:    1 plan generated, execute all (original)
  cov_qvalue:     K plans generated, Q-value selects best, execute it

All strategies get exactly EXEC_BUDGET test executions.
"""

import argparse
import random as _random
import re
import time
import json
import logging

import config
from curiosity_explorer.llm import generate_with_model, batch_generate, get_cost, reset_cost
from curiosity_explorer.runner.docker_coverage import DockerCoverageRunner
from curiosity_explorer.explorer.coverage_exploration import (
    CoverageMap, generate_coverage_greedy, generate_coverage_qvalue,
    generate_coverage_planned, _parse_script,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

STRATEGIES = ["random", "greedy", "cov_greedy", "cov_planned", "cov_qvalue"]
EXEC_BUDGET = 24
K = 3
PLAN_LENGTH = 3


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
    executions = 0

    while executions < EXEC_BUDGET:
        # --- Generation ---
        if strategy == "cov_greedy":
            scripts = generate_coverage_greedy(code, module, hist, cov_map, K=K)
        elif strategy == "cov_planned":
            scripts = generate_coverage_planned(code, module, hist, cov_map,
                                                 K=1, plan_length=PLAN_LENGTH)
        elif strategy == "cov_qvalue":
            scripts = generate_coverage_qvalue(code, module, hist, cov_map,
                                                K=K, plan_length=PLAN_LENGTH,
                                                gamma=0.5)
        else:
            scripts = gen_standard(ctx, hist, K)

        if not scripts:
            executions += 1
            continue

        # --- Selection & Execution ---
        if strategy in ("cov_planned", "cov_qvalue"):
            # Execute the plan sequentially, respecting budget
            for plan_script in scripts:
                if executions >= EXEC_BUDGET:
                    break
                result = runner.run_test(plan_script)
                hist.append((plan_script, result))
                cov_map.update(plan_script, set(), result.new_branches)
                executions += 1
                print(f"        exec {executions}/{EXEC_BUDGET}: "
                      f"+{result.new_branches}, cum={runner.get_cumulative_coverage()}",
                      flush=True)
        else:
            if strategy == "random":
                selected = _random.choice(scripts)
            elif strategy == "greedy":
                selected = select_greedy(scripts, ctx, hist)
            elif strategy == "cov_greedy":
                selected = _random.choice(scripts)
            else:
                selected = scripts[0]

            result = runner.run_test(selected)
            hist.append((selected, result))
            cov_map.update(selected, set(), result.new_branches)
            executions += 1
            print(f"      exec {executions}/{EXEC_BUDGET}: "
                  f"+{result.new_branches}, cum={runner.get_cumulative_coverage()}",
                  flush=True)

    return runner.get_cumulative_coverage()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-files", type=int, default=5)
    parser.add_argument("--strategies", nargs="+", default=None)
    args = parser.parse_args()

    reset_cost()
    strategies = args.strategies or STRATEGIES

    from datasets import load_dataset
    ds = load_dataset("kjain14/testgenevallite")
    test = ds["test"]
    examples = [ex for ex in test
                if ex["repo"] == "django/django" and ex["version"] == "4.0"]
    examples.sort(key=lambda ex: ex["baseline_covs"]["first"])
    examples = examples[:args.n_files]

    print("=" * 70, flush=True)
    print("Q-VALUE PLAN SELECTION — Fair Comparison", flush=True)
    print(f"Exec budget={EXEC_BUDGET}, K={K}, plan_length={PLAN_LENGTH}", flush=True)
    print(f"Files={len(examples)}, Strategies: {strategies}", flush=True)
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

        for strategy in strategies:
            print(f"  {strategy}:", flush=True)
            final = run_strategy(ex, strategy)
            file_results["strategies"][strategy] = final
            print(f"    final={final}", flush=True)

        all_results.append(file_results)

    elapsed = time.time() - start
    cost = get_cost()

    print(f"\n{'=' * 70}", flush=True)
    print(f"Q-VALUE PLAN SELECTION (all strategies: {EXEC_BUDGET} executions each)",
          flush=True)
    print(f"{'File':<40}", end="", flush=True)
    for s in strategies:
        print(f" {s:>14}", end="")
    print(flush=True)
    print("-" * (40 + 15 * len(strategies)), flush=True)
    for r in all_results:
        print(f"{r['file']:<40}", end="")
        for s in strategies:
            print(f" {r['strategies'].get(s, 0):>14}", end="")
        print(flush=True)

    import statistics
    print("-" * (40 + 15 * len(strategies)), flush=True)
    for s in strategies:
        vals = [r["strategies"].get(s, 0) for r in all_results]
        m = statistics.mean(vals)
        print(f"  {s:<20} mean={m:.1f}", flush=True)

    print(f"\nCost: ${cost['total_cost_usd']:.4f} | "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / "covmap_qvalue_results.json", "w") as f:
        json.dump({"exec_budget": EXEC_BUDGET, "K": K,
                    "plan_length": PLAN_LENGTH, "gamma": 0.5,
                    "results": all_results, "cost": cost,
                    "elapsed": round(elapsed, 1)}, f, indent=2)
    print("Saved", flush=True)


if __name__ == "__main__":
    main()
