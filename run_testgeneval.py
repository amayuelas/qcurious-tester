"""TestGenEval Lite runner.

Runs exploration strategies on all 160 files across 11 repos
in the TestGenEval Lite benchmark (SWE-bench Docker containers).

Usage:
    # All repos, 3 strategies, 1 seed
    python run_testgeneval.py

    # Django only
    python run_testgeneval.py --repos django/django

    # Quick test
    python run_testgeneval.py --max-examples 3 --exec-budget 6

    # Specific strategies
    python run_testgeneval.py --strategies random cov_qvalue
"""

import argparse
import random as _random
import re
import time
import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from curiosity_explorer.llm import generate_with_model, batch_generate, get_cost, reset_cost
from curiosity_explorer.runner.docker_coverage import DockerCoverageRunner
from curiosity_explorer.explorer.coverage_exploration import (
    CoverageMap, generate_coverage_greedy, generate_coverage_qvalue,
    generate_divhints_random, generate_plans_for_exec_selection, _parse_script,
)
from curiosity_explorer.benchmarks.testgeneval_config import (
    load_testgeneval_examples, get_repo_config,
)
from curiosity_explorer.explorer.covbayes import (
    module_functions, covered_quals, plan_expected_gain, predict_plan_functions,
)
from run_repo_explore_bench import (
    METHOD, gen_k_plans, target_hints, _score_plans_by_functions,
    uncovered_functions, _rescore_with_posterior,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

# The full method (targeting + function-grounded scorer) and CovBayes reuse
# RepoExploreBench's implementations; only the coverage->function mapping
# differs, since TestGenEval measures a whole package (see covered_in_target).
NEW_METHOD = METHOD  # frozen method, defined in run_repo_explore_bench
ALL_STRATEGIES = ["random", "greedy", "cov_greedy", "cov_qvalue"]
EXEC_BUDGET = 24
K = 3
PLAN_LENGTH = 3
GAMMA = 0.5


def parse_args():
    p = argparse.ArgumentParser(description="TestGenEval Lite runner")
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--code-files", nargs="+", default=None,
                   help="run only these code_file paths (used to repair "
                        "specific files, e.g. after an environment failure)")
    p.add_argument("--repos", nargs="+", default=None,
                   help="Filter by repo (e.g., django/django sympy/sympy)")
    p.add_argument("--strategies", nargs="+", default=ALL_STRATEGIES)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--exec-budget", type=int, default=EXEC_BUDGET)
    p.add_argument("--K", type=int, default=K)
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--parallel", type=int, default=4,
                   help="Number of targets to run in parallel")
    p.add_argument("--output", default="testgeneval_results.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def select_greedy(scripts, module, code, hist):
    """LLM picks the script most likely to cover new code."""
    code_ctx = f"```python\n{code[:2000]}\n```" if code else ""
    ctx = f"Module: {module}\n{code_ctx}"
    sl = "\n".join(f"  {i+1}. {s.strip()[:100]}" for i, s in enumerate(scripts))
    prompt = (f"{ctx}\nWhich test covers the most NEW code?\n"
              f"{sl}\nRespond with ONLY the number.")
    resp = generate_with_model(config.MODEL, prompt, 0.3, 20)
    for n in re.findall(r'\d+', resp):
        idx = int(n) - 1
        if 0 <= idx < len(scripts):
            return scripts[idx]
    return scripts[0]


def gen_standard(module, code, hist, K, prompt_note=""):
    """Standard generation — show source code and history."""
    code_ctx = f"```python\n{code[:2500]}\n```" if code else ""
    ctx = f"Module: {module}\n{code_ctx}"

    h = ""
    if hist:
        h = "\nPrevious:\n"
        for s, r in hist[-3:]:
            out = (r.output or r.exception or "None")[:50]
            h += f"  {s.strip()[:80]} -> {out}\n"

    note = f"\nIMPORTANT: {prompt_note}\n" if prompt_note else "\n"
    prompt = (f"{ctx}\n{h}\nWrite a test script (5-15 lines). "
              f"Import from the module and print results.{note}"
              f"Respond with ONLY executable Python code.\n\n```python\n")
    responses = batch_generate([prompt] * K, temperature=0.9, max_tokens=500)
    scripts = [_parse_script(r) for r in responses if _parse_script(r)]

    # Remove repo-specific setup if LLM adds it despite instructions
    cleaned = []
    for s in scripts:
        lines = s.split("\n")
        lines = [l for l in lines if "django.setup()" not in l
                 and l.strip() != "import django"
                 and "settings.configure" not in l]
        cleaned_script = "\n".join(lines).strip()
        if cleaned_script and len(cleaned_script) >= 10:
            cleaned.append(cleaned_script)
    return cleaned


def gen_cov_greedy(module, code, hist, cov_map, K, prompt_note=""):
    """Coverage-aware generation with source + coverage map."""
    if code:
        return generate_coverage_greedy(code, module, hist, cov_map, K=K)
    # Fallback for repos without embedded source
    return gen_standard(module, code, hist, K, prompt_note)


def gen_cov_qvalue(module, code, hist, cov_map, K, gamma, prompt_note=""):
    """Multi-plan generation with Q-value selection."""
    if code:
        return generate_coverage_qvalue(code, module, hist, cov_map,
                                         K=K, plan_length=PLAN_LENGTH, gamma=gamma)
    return gen_standard(module, code, hist, K, prompt_note)


def gen_cov_qvalue_plans(module, code, hist, cov_map, K, prompt_note=""):
    """K diverse multi-step plans, no selection (shared by cov_bayes)."""
    if code:
        return gen_k_plans(module, code, hist, cov_map, K, PLAN_LENGTH,
                           prompt_note=prompt_note)
    scripts = gen_standard(module, code, hist, K, prompt_note)
    return [[s] for s in scripts] if scripts else []


def gen_exec_plans(module, code, hist, cov_map, K, prompt_note=""):
    """K diverse plans for execution-based selection (cov_qvalue_exec)."""
    if code:
        return generate_plans_for_exec_selection(code, module, hist, cov_map,
                                                 K=K, plan_length=PLAN_LENGTH)
    scripts = gen_standard(module, code, hist, K, prompt_note)
    return [[s] for s in scripts] if scripts else []


def gen_divhints_random(module, code, hist, cov_map, K, prompt_note=""):
    """Experiment 1: cov_qvalue generation pipeline, random plan selection."""
    if code:
        return generate_divhints_random(code, module, hist, cov_map,
                                        K=K, plan_length=PLAN_LENGTH)
    return gen_standard(module, code, hist, K, prompt_note)


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

def run_strategy(example, strategy, seed, exec_budget, K, gamma):
    """Run one strategy on one example. Returns {final, curve}."""
    _random.seed(seed)

    module = example["module"]
    code = example["code_src"]
    prompt_note = example.get("prompt_note", "")

    # Build Docker runner with correct Python binary
    python_bin = example.get("python_bin", "python")
    pre_command = ""
    if example.get("pre_install"):
        pre_command = example["pre_install"]

    # Use the parent package of the target module for --source
    # e.g., sympy.physics.units.util → sympy.physics.units (not just "sympy")
    # This is broad enough to catch indirect imports but narrow enough to not
    # make coverage.py serialize 1400+ files
    parts = module.split(".")
    if len(parts) >= 3:
        source_package = ".".join(parts[:-1])  # parent package
    elif len(parts) == 2:
        source_package = parts[0]  # top-level package
    else:
        source_package = module
    target_file = example.get("code_file", None)

    runner = DockerCoverageRunner(
        image=example["image"],
        source_module=source_package,
        setup_code=example["setup_code"],
        working_dir=example["working_dir"],
        env=example["env"],
        python_bin=python_bin,
        pre_command=pre_command,
        target_file=target_file,
    )

    hist = []
    cov_map = CoverageMap()
    executions = 0
    branch_curve = []
    line_curve = []

    def covered_in_target():
        """Covered lines restricted to the target file.

        RepoExploreBench tracks one module, but here --source is a whole
        package, so cumulative_lines spans many files; covered_quals must only
        see the target file's lines or every qualname looks covered.
        """
        if not target_file:
            return runner.cumulative_lines
        return {(f, ln) for (f, ln) in runner.cumulative_lines
                if target_file in f}

    def execute(script):
        nonlocal executions
        result = runner.run_test(script)
        hist.append((script, result))
        cov_map.update(script, set(), result.new_branches)
        executions += 1
        branch_curve.append(runner.get_cumulative_coverage())
        line_curve.append(runner.get_cumulative_lines())
        return result

    strategy = {"covqvalue2": NEW_METHOD, "cov_qvalue_v2": NEW_METHOD}.get(
        strategy, strategy)
    funcs = module_functions(code) if code and strategy in (
        NEW_METHOD, "cov_bayes") else []
    cb_post = {f[0]: [1.0, 1.0] for f in funcs}
    fn_post = {f[0]: [1.0, 1.0] for f in funcs}  # Bayesian valuation posterior
    # Per scored round: the untested functions before and after the committed
    # plan, and the scorer's EXECUTES/ENABLES predictions for it, so the
    # accuracy of the LLM's predictions can be measured offline.
    pred_log = []

    while executions < exec_budget:
        # --- Full method: plans targeted at still-uncovered functions, scored
        #     by the function-grounded Q-value (see run_repo_explore_bench). ---
        if strategy == NEW_METHOD and funcs:
            # remaining unexecuted code per function, restricted to the
            # target file (coverage spans the whole package here)
            # touched/untouched map (the frozen default; see run_repo_explore_bench)
            uncovered = [list(f) for f in
                         uncovered_functions(runner, funcs, file_filter=target_file,
                                             touch_rule=True)]
            left = {f[0] for f in uncovered}
            covered_fn = {f[0] for f in funcs if f[0] not in left}
            targets = target_hints(code, funcs, covered_in_target(), K,
                                   uncovered=uncovered)
            plans = gen_k_plans(module, code, hist, cov_map, K, PLAN_LENGTH,
                                targets=targets, prompt_note=prompt_note)
            if not plans:
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                line_curve.append(runner.get_cumulative_lines())
                continue
            if len(plans) > 1 and uncovered:
                scores = _score_plans_by_functions(
                    plans, module, uncovered, gamma, with_future=True,
                    tight=True, covered=covered_fn)
                scores = _rescore_with_posterior(scores, uncovered, fn_post, gamma)
                best_q = max(scores[i]["q"] for i in range(len(plans)))
                sel = _random.choice([i for i in range(len(plans))
                                      if scores[i]["q"] == best_q])
            else:
                sel, scores = 0, None  # unscored round: no prediction to update on
            committed = [execute(step) for step in plans[sel]
                         if executions < exec_budget]
            if scores is None:
                continue
            # conjugate update of the reachability posterior
            left = {f[0] for f in uncovered_functions(runner, funcs,
                                                      file_filter=target_file,
                                                      touch_rule=True)}
            pred_log.append({
                "uncovered_before": sorted(f[0] for f in uncovered),
                "uncovered_after": sorted(left),
                "executes": sorted(scores[sel].get("executes") or ()),
                "enables": sorted(scores[sel].get("enables") or ()),
            })
            pred = set(scores[sel].get("executes") or ())
            pred |= set(scores[sel].get("enables") or ())
            for f in pred:
                if f in fn_post:
                    fn_post[f][0 if f not in left else 1] += 1.0
            continue

        # --- CovBayes: closed-form Bayesian selection over a per-function
        #     reachability posterior (Exp 11). ---
        if strategy == "cov_bayes" and funcs:
            covered_fn = covered_quals(covered_in_target(), funcs)
            uncovered = [f[0] for f in funcs if f[0] not in covered_fn]
            plans = gen_cov_qvalue_plans(module, code, hist, cov_map, K,
                                         prompt_note)
            if not plans:
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                line_curve.append(runner.get_cumulative_lines())
                continue
            if uncovered:
                best, best_score, best_preds = plans[0], -1.0, {}
                for plan in plans:
                    qf = predict_plan_functions(module, code, plan,
                                                set(uncovered), config.MODEL)
                    sc = plan_expected_gain(cb_post, uncovered, qf)
                    if sc > best_score:
                        best, best_score, best_preds = plan, sc, qf
            else:
                best, best_preds = _random.choice(plans), {}
            for step in best:
                if executions >= exec_budget:
                    break
                execute(step)
            new_cov = covered_quals(covered_in_target(), funcs)
            for q in cb_post:
                if q in new_cov and q not in covered_fn:
                    cb_post[q][0] += 1.0
                elif q in best_preds and q not in new_cov:
                    cb_post[q][1] += 1.0
            continue

        # --- Execution-based Q-value selection (ported from
        #     run_repo_explore_bench.py): execute step 1 of each of the K
        #     plans, observe real coverage, then finish the plan whose step 1
        #     found the most new branches. All executions count against the
        #     budget. ---
        if strategy == "cov_qvalue_exec":
            plans = gen_exec_plans(module, code, hist, cov_map, K, prompt_note)
            if not plans:
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                line_curve.append(runner.get_cumulative_lines())
                continue

            step1_results = []
            for plan in plans:
                if executions >= exec_budget:
                    break
                result = execute(plan[0])
                step1_results.append((plan, result.new_branches))

            if step1_results:
                best_plan, _ = max(step1_results, key=lambda x: x[1])
                for plan_script in best_plan[1:]:
                    if executions >= exec_budget:
                        break
                    execute(plan_script)
            continue

        # --- Generation ---
        if strategy == "cov_greedy":
            scripts = gen_cov_greedy(module, code, hist, cov_map, K, prompt_note)
        elif strategy == "cov_qvalue":
            scripts = gen_cov_qvalue(module, code, hist, cov_map, K, gamma, prompt_note)
        elif strategy == "divhints_random":
            scripts = gen_divhints_random(module, code, hist, cov_map, K, prompt_note)
        else:
            scripts = gen_standard(module, code, hist, K, prompt_note)

        if not scripts:
            executions += 1
            branch_curve.append(runner.get_cumulative_coverage())
            line_curve.append(runner.get_cumulative_lines())
            continue

        # --- Execution ---
        if strategy in ("cov_qvalue", "divhints_random"):
            for plan_script in scripts:
                if executions >= exec_budget:
                    break
                result = runner.run_test(plan_script)
                hist.append((plan_script, result))
                cov_map.update(plan_script, set(), result.new_branches)
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                line_curve.append(runner.get_cumulative_lines())
        else:
            if strategy == "greedy":
                selected = select_greedy(scripts, module, code, hist)
            else:
                selected = _random.choice(scripts)
            result = runner.run_test(selected)
            hist.append((selected, result))
            cov_map.update(selected, set(), result.new_branches)
            executions += 1
            branch_curve.append(runner.get_cumulative_coverage())
            line_curve.append(runner.get_cumulative_lines())

    stats = runner.get_stats()
    runner.cleanup()  # exec mode: remove the long-lived container now

    # Serialize execution trace
    trace = []
    for script, result in hist:
        trace.append({
            "script": script,
            "output": result.output,
            "exception": result.exception,
            "new_branches": result.new_branches,
            "new_lines": result.new_lines,
            "passed": result.passed,
        })

    return {
        "final": stats["branches"],
        "final_lines": stats["lines"],
        "pass_rate": stats["pass_rate"],
        "pass_count": stats["pass_count"],
        "fail_count": stats["fail_count"],
        "branch_curve": branch_curve,
        "line_curve": line_curve,
        "trace": trace,
        **({"pred_log": pred_log} if pred_log else {}),
    }


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def analyze_results(all_results, strategies):
    """Compute summary statistics and paired tests."""
    from scipy import stats as sp_stats

    analysis = {}

    for s in strategies:
        vals = [r["strategies"][s]["final"] for r in all_results
                if s in r["strategies"]]
        analysis[s] = {
            "mean": statistics.mean(vals) if vals else 0,
            "std": statistics.stdev(vals) if len(vals) > 1 else 0,
            "se": statistics.stdev(vals) / len(vals)**0.5 if len(vals) > 1 else 0,
            "n": len(vals),
        }

    if "random" in strategies:
        analysis["paired_vs_random"] = {}
        for s in strategies:
            if s == "random":
                continue
            deltas = []
            for r in all_results:
                if "random" in r["strategies"] and s in r["strategies"]:
                    d = r["strategies"][s]["final"] - r["strategies"]["random"]["final"]
                    deltas.append(d)
            if len(deltas) < 2:
                continue
            md = statistics.mean(deltas)
            sd = statistics.stdev(deltas)
            se = sd / len(deltas)**0.5
            wins = sum(1 for d in deltas if d > 0)
            losses = sum(1 for d in deltas if d < 0)
            ties = sum(1 for d in deltas if d == 0)

            if sd > 0:
                t_stat, p_val = sp_stats.ttest_1samp(deltas, 0)
                cohens_d = md / sd
            else:
                t_stat, p_val, cohens_d = 0, 1.0, 0

            analysis["paired_vs_random"][s] = {
                "mean_delta": md, "se": se,
                "wins": wins, "losses": losses, "ties": ties,
                "t_stat": t_stat, "p_value": p_val,
                "cohens_d": cohens_d, "n": len(deltas),
            }

    # Pairwise paired comparisons
    for a, b, key in [
        ("cov_qvalue", "cov_greedy", "qvalue_vs_greedy"),
        ("cov_qvalue", "divhints_random", "qvalue_vs_divhints_random"),
    ]:
        if a not in strategies or b not in strategies:
            continue
        deltas = []
        for r in all_results:
            if a in r["strategies"] and b in r["strategies"]:
                deltas.append(r["strategies"][a]["final"] -
                              r["strategies"][b]["final"])
        if len(deltas) >= 2 and statistics.stdev(deltas) > 0:
            sd = statistics.stdev(deltas)
            md = statistics.mean(deltas)
            t_stat, p_val = sp_stats.ttest_1samp(deltas, 0)
            analysis[key] = {
                "mean_delta": md,
                "se": sd / len(deltas)**0.5,
                "wins": sum(1 for d in deltas if d > 0),
                "losses": sum(1 for d in deltas if d < 0),
                "ties": sum(1 for d in deltas if d == 0),
                "t_stat": t_stat, "p_value": p_val,
                "cohens_d": md / sd,
                "n": len(deltas),
            }

    return analysis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    reset_cost()

    examples = load_testgeneval_examples(repos=args.repos,
                                          max_examples=args.max_examples)
    if args.code_files:
        wanted = set(args.code_files)
        examples = [e for e in examples if e["code_file"] in wanted]
        print(f"  Filtered to {len(examples)} of the requested "
              f"{len(wanted)} files", flush=True)
    strategies = args.strategies
    seeds = args.seeds

    total_runs = len(examples) * len(strategies) * len(seeds)
    repos_in_run = sorted(set(ex["repo"] for ex in examples))

    print("=" * 70, flush=True)
    print("TestGenEval Lite", flush=True)
    print(f"  Examples: {len(examples)} across {len(repos_in_run)} repos", flush=True)
    for repo in repos_in_run:
        count = sum(1 for ex in examples if ex["repo"] == repo)
        print(f"    {repo}: {count}", flush=True)
    print(f"  Strategies: {strategies}", flush=True)
    print(f"  Seeds: {seeds}", flush=True)
    print(f"  Exec budget: {args.exec_budget}", flush=True)
    print(f"  Total runs: {total_runs}", flush=True)
    print("=" * 70, flush=True)

    t = generate_with_model(config.MODEL, "Say ok", 0.3, 100)
    print(f"  Connectivity: {'OK' if t else 'FAILED'}", flush=True)
    if not t:
        return

    start = time.time()
    completed = [0]

    def run_one_example(i, ex, seed):
        """Run all strategies on one example. Returns result dict."""
        run_result = {
            "module": ex["module"],
            "repo": ex["repo"],
            "version": ex["version"],
            "code_file": ex["code_file"],
            "seed": seed,
            "strategies": {},
        }
        for strategy in strategies:
            result = run_strategy(ex, strategy, seed,
                                   args.exec_budget, args.K, args.gamma)
            run_result["strategies"][strategy] = result
        completed[0] += 1
        finals = {s: run_result["strategies"][s]["final"] for s in strategies}
        print(f"  [{completed[0]}/{len(examples)*len(seeds)}] "
              f"{ex['code_file']} ({ex['repo']} v{ex['version']}): {finals}",
              flush=True)
        return run_result

    jobs = [(i, ex, seed)
            for i, ex in enumerate(examples)
            for seed in seeds]

    print(f"\nRunning {len(jobs)} targets with {args.parallel} workers...",
          flush=True)

    all_results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(run_one_example, i, ex, s): (i, ex, s)
                   for i, ex, s in jobs}
        for future in as_completed(futures):
            try:
                all_results.append(future.result())
            except Exception as e:
                i, ex, s = futures[future]
                print(f"  ERROR on {ex['code_file']}: {e}", flush=True)

    elapsed = time.time() - start
    cost = get_cost()

    # --- Summary ---
    print(f"\n{'=' * 70}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)

    print(f"\n{'Module':<40} {'repo':<15} {'seed':>4}", end="", flush=True)
    for s in strategies:
        print(f" {s[:11]:>12}", end="")
    print(flush=True)
    print("-" * (60 + 13 * len(strategies)), flush=True)

    for r in all_results:
        print(f"{r['module'][:39]:<40} {r['repo'][:14]:<15} {r['seed']:>4}", end="")
        for s in strategies:
            v = r["strategies"].get(s, {}).get("final", 0)
            print(f" {v:>12}", end="")
        print(flush=True)

    # --- Statistics ---
    print(f"\n{'=' * 70}", flush=True)
    print("STATISTICS", flush=True)
    print(f"{'=' * 70}", flush=True)

    analysis = analyze_results(all_results, strategies)

    print(f"\nPer-strategy means (n={analysis[strategies[0]]['n']}):", flush=True)
    for s in strategies:
        a = analysis[s]
        lines = [r["strategies"][s].get("final_lines", 0) for r in all_results
                 if s in r["strategies"]]
        pass_rates = [r["strategies"][s].get("pass_rate", 0) for r in all_results
                      if s in r["strategies"]]
        mean_lines = statistics.mean(lines) if lines else 0
        mean_pr = statistics.mean(pass_rates) if pass_rates else 0
        print(f"  {s:<20} branches={a['mean']:>6.1f} ± {a['se']:.1f}  "
              f"lines={mean_lines:.1f}  pass_rate={mean_pr:.0%}", flush=True)

    if "paired_vs_random" in analysis:
        print(f"\nPaired vs random:", flush=True)
        for s, a in analysis["paired_vs_random"].items():
            sig = "***" if a["p_value"] < 0.001 else ("**" if a["p_value"] < 0.01
                    else ("*" if a["p_value"] < 0.05 else ""))
            print(f"  {s:<20} Δ={a['mean_delta']:>+6.1f} ± {a['se']:.1f}  "
                  f"W={a['wins']} L={a['losses']} T={a['ties']}  "
                  f"p={a['p_value']:.4f} d={a['cohens_d']:.2f} {sig}", flush=True)

    for key, label in [("qvalue_vs_greedy", "cov_qvalue vs cov_greedy"),
                       ("qvalue_vs_divhints_random",
                        "cov_qvalue vs divhints_random (Exp 1)")]:
        if key in analysis:
            a = analysis[key]
            sig = "***" if a["p_value"] < 0.001 else ("**" if a["p_value"] < 0.01
                    else ("*" if a["p_value"] < 0.05 else ""))
            print(f"\n{label}: Δ={a['mean_delta']:>+.1f} ± {a['se']:.1f}  "
                  f"W={a['wins']} L={a['losses']} T={a['ties']}  "
                  f"p={a['p_value']:.4f} d={a['cohens_d']:.2f} {sig}", flush=True)

    # Per-repo breakdown
    print(f"\nPer-repo means:", flush=True)
    for repo in repos_in_run:
        repo_results = [r for r in all_results if r["repo"] == repo]
        print(f"\n  {repo} ({len(repo_results)} runs):", flush=True)
        for s in strategies:
            vals = [r["strategies"][s]["final"] for r in repo_results
                    if s in r["strategies"]]
            if vals:
                print(f"    {s:<20} mean={statistics.mean(vals):.1f}", flush=True)

    print(f"\nCost: ${cost['total_cost_usd']:.4f} | "
          f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    # --- Save ---
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outpath = config.RESULTS_DIR / args.output
    with open(outpath, "w") as f:
        json.dump({
            "benchmark": "TestGenEval Lite",
            "config": {
                "strategies": strategies, "seeds": seeds,
                "exec_budget": args.exec_budget, "K": args.K,
                "gamma": args.gamma,
            },
            "results": all_results,
            "analysis": {k: v for k, v in analysis.items()
                          if k != "paired_vs_random"},
            "paired_vs_random": analysis.get("paired_vs_random", {}),
            "cost": cost,
            "elapsed": round(elapsed, 1),
        }, f, indent=2, default=str)
    print(f"Saved to {outpath}", flush=True)


if __name__ == "__main__":
    main()
