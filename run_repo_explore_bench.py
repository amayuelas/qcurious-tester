"""RepoExploreBench runner.

Evaluates exploration strategies on 100 real-world Python modules
across 9 repos, all running in the curiositybench Docker image.

Usage:
    # Full benchmark (100 files × 1 seed × 3 strategies)
    python run_repo_explore_bench.py

    # Quick smoke test
    python run_repo_explore_bench.py --max-targets 3 --seeds 42 --exec-budget 6

    # Single repo
    python run_repo_explore_bench.py --repos click

    # Only key comparison
    python run_repo_explore_bench.py --strategies random cov_qvalue

    # Multiple seeds for tighter CIs
    python run_repo_explore_bench.py --seeds 42 123 456
"""

import argparse
import random as _random
import re
import subprocess
import time
import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from curiosity_explorer.llm import generate_with_model, batch_generate, get_cost, reset_cost
from curiosity_explorer.runner.docker_coverage import DockerCoverageRunner
from curiosity_explorer.explorer.coverage_exploration import (
    CoverageMap, _parse_script, _parse_plan,
    generate_plans_for_exec_selection,
)
from curiosity_explorer.explorer.covbayes import (
    module_functions, covered_quals, plan_expected_gain,
    predict_plan_functions,
)
from curiosity_explorer.benchmarks.repo_explore_bench import (
    load_benchmark, get_benchmark_info, DEFAULT_SEEDS, DOCKER_IMAGE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

ALL_STRATEGIES = ["random", "greedy", "cov_greedy", "cov_qvalue"]
EXEC_BUDGET = 24
K = 3
PLAN_LENGTH = 3
GAMMA = 0.5
# Summary-content ablation (Exp 7): set from --map-mode in main(), read by
# run_strategy() when constructing each CoverageMap. Module-global so it reaches
# the ThreadPool workers without threading through run_strategy's signature.
MAP_MODE = "full"
# Optional override applied ONLY to the Q-value scorer's summary (None = use
# MAP_MODE). Holds generation fixed while ablating the scorer's summary, to
# separate a field's selection role from its generation role.
SCORE_MAP_MODE = None


def parse_args():
    p = argparse.ArgumentParser(description="RepoExploreBench runner")
    p.add_argument("--max-targets", type=int, default=None)
    p.add_argument("--repos", nargs="+", default=None)
    p.add_argument("--strategies", nargs="+", default=ALL_STRATEGIES)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--exec-budget", type=int, default=EXEC_BUDGET)
    p.add_argument("--K", type=int, default=K)
    p.add_argument("--gamma", type=float, default=GAMMA)
    p.add_argument("--parallel", type=int, default=4,
                   help="Number of targets to run in parallel")
    p.add_argument("--per-strategy-cost", action="store_true",
                   help="Record per-strategy tokens/calls (Exp 6). Requires "
                        "--parallel 1 (uses the global cost counter).")
    p.add_argument("--output", default="repo_explore_bench_results.json")
    p.add_argument("--map-mode", default="full",
                   choices=["full", "stats", "exemplars", "none"],
                   help="Summary-content ablation (Exp 7): which fields the "
                        "coverage-state summary exposes in BOTH generation and "
                        "Q-value scoring. full=count+rate+exemplars (default); "
                        "stats=count+rate only; exemplars=most-informative tests "
                        "only; none=source+history only.")
    p.add_argument("--score-map-mode", default=None,
                   choices=["full", "stats", "exemplars", "none"],
                   help="Exp 7 refinement: override the summary mode for the "
                        "Q-value SCORER only (generation stays at --map-mode). "
                        "Default None = same as --map-mode. Use to hold "
                        "generation fixed and ablate just the selection signal.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Source code fetching
# ---------------------------------------------------------------------------

def fetch_source(module_name):
    """Fetch module source code from Docker image."""
    cmd = (f"docker run --rm {DOCKER_IMAGE} python3 -c "
           f"\"import inspect, {module_name}; print(inspect.getsource({module_name}))\"")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception as e:
        log.warning(f"Failed to fetch source for {module_name}: {e}")
    return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def gen_standard(module, source, hist, K):
    """Standard generation — show module info and recent history."""
    code_ctx = f"```python\n{source[:2500]}\n```" if source else ""
    ctx = f"Module: {module}\n{code_ctx}"

    h = ""
    if hist:
        h = "\nPrevious:\n"
        for s, r in hist[-3:]:
            out = (r.output or r.exception or "None")[:50]
            h += f"  {s.strip()[:80]} -> {out}\n"

    prompt = (f"{ctx}\n{h}\nWrite a test script (5-15 lines). "
              f"Import from the module and print results.\n"
              f"Respond with ONLY executable Python code.\n\n```python\n")
    responses = batch_generate([prompt] * K, temperature=0.9, max_tokens=500)
    return [_parse_script(r) for r in responses if _parse_script(r)]


def select_greedy(scripts, module, source, hist):
    """LLM picks the script most likely to cover new code."""
    code_ctx = f"```python\n{source[:2000]}\n```" if source else ""
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


def gen_cov_greedy(module, source, hist, cov_map, K):
    """Coverage-aware generation — show source + coverage map, target gaps."""
    code_ctx = f"```python\n{source[:2500]}\n```" if source else ""
    cov_summary = cov_map.coverage_summary()
    history_str = ""
    if hist:
        for tc, res in hist[-3:]:
            out = (res.output or res.exception or "None")[:50]
            history_str += f"  {tc.strip()[:80]} -> {out} (new branches: {res.new_branches})\n"

    prompt = f"""Module: {module}
{code_ctx}

{cov_summary}

Previous tests:
{history_str}

Write a test script (5-15 lines) that covers branches NOT YET discovered.
Look at the code and target specific uncovered code paths.
Import from the module and print results.
Respond with ONLY executable Python code.

```python
"""
    responses = batch_generate([prompt] * K, temperature=0.9, max_tokens=500)
    return [_parse_script(r) for r in responses if _parse_script(r)]


def gen_k_plans(module, source, hist, cov_map, K, plan_length, use_diversity=True):
    """Generate K multi-step plans (generation only, no selection).

    Shared by cov_qvalue, divhints_random, and cov_greedy_multistep so the
    conditions use an identical generation pipeline. With use_diversity=True
    each plan gets a distinct diversity hint (CovQValue's generator); with
    use_diversity=False every plan gets the same neutral instruction, so the
    diversity hint is the ONLY difference from the diverse pipeline. Returns a
    list of plans, each a list of plan_length scripts.
    """
    code_ctx = f"```python\n{source[:2500]}\n```" if source else ""
    cov_summary = cov_map.coverage_summary()
    history_str = ""
    if hist:
        for tc, res in hist[-3:]:
            out = (res.output or res.exception or "None")[:50]
            history_str += f"  {tc.strip()[:80]} -> {out} (new branches: {res.new_branches})\n"

    diversity_hints = [
        "Focus on the MAIN functionality — constructors, primary methods.",
        "Focus on ERROR HANDLING — invalid inputs, edge cases, exceptions.",
        "Focus on INTERACTIONS — create objects, pass them to each other, chain calls.",
        "Focus on CONFIGURATION — different parameter combinations, options, flags.",
        "Focus on RARELY-USED features — optional arguments, deprecated paths.",
    ]
    neutral_hint = "Target the most promising uncovered code paths."

    prompts = []
    for i in range(K):
        hint = diversity_hints[i % len(diversity_hints)] if use_diversity else neutral_hint
        prompt = f"""Module: {module}
{code_ctx}

{cov_summary}

Previous tests:
{history_str}

PLAN a sequence of {plan_length} test scripts that TOGETHER will reach
UNCOVERED code paths. {hint}

Think about what setup is needed:
Step 1: What basic setup/import is needed to reach deeper code?
Step 2: Building on step 1's result, what exercises the next layer?
Step 3: Now target the deepest uncovered branches.

For each step, write a separate test script (5-15 lines each).
Import from the module and print results.

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

    # Generate K plans
    responses = batch_generate(prompts, temperature=0.9, max_tokens=1500)
    return [_parse_plan(r) for r in responses if _parse_plan(r)]


def gen_cov_qvalue(module, source, hist, cov_map, K, plan_length, gamma):
    """Generate K plans, score by Q-value, return the best plan."""
    plans = gen_k_plans(module, source, hist, cov_map, K, plan_length)

    if not plans:
        return gen_cov_greedy(module, source, hist, cov_map, K)
    if len(plans) == 1:
        return plans[0]

    # Score each plan by Q-value
    return _score_and_select(plans, module, source, cov_map, gamma)


def gen_divhints_random(module, source, hist, cov_map, K, plan_length):
    """Experiment 1: identical generation to cov_qvalue, but pick a plan
    uniformly at random instead of by Q-value. Isolates the contribution of
    the Q-value selection mechanism from the diversity-hinted generation."""
    plans = gen_k_plans(module, source, hist, cov_map, K, plan_length)

    if not plans:
        return gen_cov_greedy(module, source, hist, cov_map, K)
    return _random.choice(plans)


def gen_cov_greedy_multistep(module, source, hist, cov_map, K, plan_length):
    """Experiment 3: multi-step plan generation WITHOUT diversity hints,
    selected at random. Identical to divhints_random except the diversity
    hints are replaced by a neutral instruction — so it isolates the
    contribution of multi-step planning from the diversity hints (and from the
    Q-value selection). Bridges single-step cov_greedy and divhints_random."""
    plans = gen_k_plans(module, source, hist, cov_map, K, plan_length,
                        use_diversity=False)

    if not plans:
        return gen_cov_greedy(module, source, hist, cov_map, K)
    return _random.choice(plans)


def _score_plans(plans, module, source, cov_map, gamma):
    """Score each plan by LLM-estimated Q-value.

    Returns dict idx -> {"immediate": ĝ, "future": v̂, "q": ĝ + γ·v̂}. Shared by
    cov_qvalue (selection) and cov_qvalue_calib (calibration logging) so both
    use the identical scoring prompt — the calibration must describe the real
    scorer, not a copy that can drift.
    """
    code_ctx = f"```python\n{source[:2000]}\n```" if source else ""
    # Scorer-only summary override (Exp 7): None falls back to map_mode.
    cov_summary = cov_map.coverage_summary(cov_map.score_map_mode)

    def score_plan(idx):
        plan = plans[idx]
        plan_str = ""
        for i, s in enumerate(plan):
            plan_str += f"\nStep {i+1}:\n```python\n{s[:200]}\n```\n"

        prompt = f"""Module: {module}
{code_ctx}

{cov_summary}

Consider this TEST PLAN (a sequence of {len(plan)} scripts):
{plan_str}

Evaluate by answering TWO questions with just numbers:
1. IMMEDIATE GAIN: Total NEW branches this plan discovers? (0-50)
2. FUTURE VALUE: Additional branches reachable AFTER this plan? (0-50)

Format: immediate, future
Example: 15, 25"""

        # Score with the active model (config.MODEL), not a hardcoded gemini —
        # so single-model runs (e.g. a local vLLM gemma) score with the same
        # model that generates, fixing the cross-model fairness caveat.
        # max_tokens must clear a thinking-model's reasoning budget: at 50 the
        # gemini-3-flash-preview scorer was truncated mid-thought and returned
        # an empty string, silently parsing to 0 (the Exp-1 scorer bug).
        resp = generate_with_model(config.MODEL, prompt, 0.3, 256)
        nums = re.findall(r'\d+', resp)
        imm = int(nums[0]) if len(nums) >= 1 else 0
        fut = int(nums[1]) if len(nums) >= 2 else 0
        q = imm + gamma * fut
        log.info(f"Plan {idx}: ḡ={imm}, γE[v]={gamma*fut:.1f}, Q={q:.1f}")
        return {"immediate": imm, "future": fut, "q": q}

    scores = {}
    with ThreadPoolExecutor(max_workers=len(plans)) as ex:
        futures = {ex.submit(score_plan, i): i for i, _ in enumerate(plans)}
        for f in futures:
            idx = futures[f]
            try:
                scores[idx] = f.result()
            except Exception:
                scores[idx] = {"immediate": 0, "future": 0, "q": 0}
    return scores


def _score_and_select(plans, module, source, cov_map, gamma):
    """Score plans by Q-value and return the best."""
    scores = _score_plans(plans, module, source, cov_map, gamma)
    best = max(scores, key=lambda i: scores[i]["q"])
    log.info(f"Selected plan {best} with Q={scores[best]['q']:.1f}")
    return plans[best]


def _score_plans_ranking(plans, module, source, cov_map):
    """Score plans by LLM RANKING in one call (vs absolute scoring in _score_plans).

    Motivated by LLM-as-judge literature: comparative ranking is significantly more
    reliable than absolute numerical scoring on a 0–50 scale. One call per round
    instead of K (also helps cost normalization in Exp 6).
    """
    K = len(plans)
    if K == 1:
        return {0: {"q": 1, "rank": 1}}

    code_ctx = f"```python\n{source[:2000]}\n```" if source else ""
    cov_summary = cov_map.coverage_summary(cov_map.score_map_mode)  # Exp 7 scorer override
    labels = [chr(ord('A') + i) for i in range(K)]
    label_list = ", ".join(labels)

    plan_blocks = []
    for label, plan in zip(labels, plans):
        ps = ""
        for i, s in enumerate(plan):
            ps += f"\nStep {i+1}:\n```python\n{s[:200]}\n```\n"
        plan_blocks.append(f"PLAN {label}:{ps}")
    plans_section = "\n".join(plan_blocks)

    prompt = f"""Module: {module}
{code_ctx}

{cov_summary}

Consider these {K} test plans (each a sequence of {len(plans[0])} scripts):

{plans_section}

Which plan is MOST likely to discover the most new branches — counting BOTH
immediate discovery AND additional branches its setup unlocks for future tests?

Rank ALL {K} plans from BEST to WORST. Use the labels: {label_list}.
Output ONLY the ranking on a single line, separated by spaces, best first."""

    resp = generate_with_model(config.MODEL, prompt, 0.3, 256)

    # Parse: find labels in order of appearance
    label_set = set(labels)
    found = []
    for m in re.finditer(r'\b([A-Z])\b', resp):
        c = m.group(1)
        if c in label_set and c not in found:
            found.append(c)
        if len(found) == K:
            break
    # Fill any missing labels at the end (parse partial -> assume the rest are worst)
    if len(found) < K:
        for lbl in labels:
            if lbl not in found:
                found.append(lbl)

    scores = {}
    for rank, lbl in enumerate(found):
        idx = labels.index(lbl)
        scores[idx] = {"q": K - rank, "rank": rank + 1}
    log.info(f"Ranking: {' '.join(found)} -> plan {labels.index(found[0])}")
    return scores


def _score_and_select_ranking(plans, module, source, cov_map):
    scores = _score_plans_ranking(plans, module, source, cov_map)
    best = max(scores, key=lambda i: scores[i]["q"])
    return plans[best]


def gen_cov_qvalue_rank(module, source, hist, cov_map, K, plan_length):
    """Generate K plans, score by LLM ranking (1 call), return the best plan."""
    plans = gen_k_plans(module, source, hist, cov_map, K, plan_length)
    if not plans:
        return gen_cov_greedy(module, source, hist, cov_map, K)
    if len(plans) == 1:
        return plans[0]
    return _score_and_select_ranking(plans, module, source, cov_map)


# ---------------------------------------------------------------------------
# Strategy runner
# ---------------------------------------------------------------------------

def run_strategy(target, strategy, seed, exec_budget, K, gamma, source):
    """Run one strategy on one target. Returns {final, curve}."""
    _random.seed(seed)
    module = target["module"]

    runner = DockerCoverageRunner(
        image=target["docker_image"],
        source_module=module,
        setup_code=target["setup_code"],
        working_dir=target["working_dir"],
        env=target["env"],
    )

    hist = []
    cov_map = CoverageMap()
    cov_map.map_mode = MAP_MODE  # Exp 7: summary-content ablation
    cov_map.score_map_mode = SCORE_MAP_MODE  # Exp 7: scorer-only override
    executions = 0
    branch_curve = []
    line_curve = []
    calib_log = []  # per-round (predicted Q, realized gain) records for Exp 2
    exp9_log = []   # per-round (predicted vs actually-covered functions) for Exp 9

    # CovBayes (Exp 11): per-function reachability posterior Beta(alpha, beta).
    cb_funcs = module_functions(source) if strategy in (
        "cov_bayes", "cov_bayes_calib") else []
    cb_quals = [f[0] for f in cb_funcs]
    cb_post = {q: [1.0, 1.0] for q in cb_quals}

    while executions < exec_budget:
        # --- Generation ---
        if strategy == "cov_greedy":
            scripts = gen_cov_greedy(module, source, hist, cov_map, K)
        elif strategy == "cov_qvalue":
            scripts = gen_cov_qvalue(module, source, hist, cov_map,
                                      K, PLAN_LENGTH, gamma)
        elif strategy == "cov_qvalue_rank":
            scripts = gen_cov_qvalue_rank(module, source, hist, cov_map,
                                           K, PLAN_LENGTH)
        elif strategy == "divhints_random":
            scripts = gen_divhints_random(module, source, hist, cov_map,
                                          K, PLAN_LENGTH)
        elif strategy == "cov_greedy_multistep":
            scripts = gen_cov_greedy_multistep(module, source, hist, cov_map,
                                               K, PLAN_LENGTH)
        elif strategy in ("cov_qvalue_exec", "divhints_oracle",
                          "cov_qvalue_calib", "cov_bayes", "cov_bayes_calib"):
            scripts = None  # handled below
        else:
            scripts = gen_standard(module, source, hist, K)

        # --- Oracle selection (Exp 1 ceiling): trial-run all K plans,
        #     commit the one with the highest realized branch gain. The
        #     K-1 discarded trials are free lookahead; only the committed
        #     plan's steps count against the budget. ---
        if strategy == "divhints_oracle":
            plans = gen_k_plans(module, source, hist, cov_map, K, PLAN_LENGTH)
            if not plans:
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                line_curve.append(runner.get_cumulative_lines())
                continue

            snap = runner.snapshot()
            best_plan, best_gain = None, -1
            for plan in plans:
                runner.restore(snap)
                for step in plan:
                    runner.run_test(step)
                gain = runner.get_cumulative_coverage() - len(snap["branches"])
                if gain > best_gain:
                    best_gain, best_plan = gain, plan

            # Commit the winner: restore pre-round state, re-run within budget
            runner.restore(snap)
            for step in best_plan:
                if executions >= exec_budget:
                    break
                result = runner.run_test(step)
                hist.append((step, result))
                cov_map.update(step, set(), result.new_branches)
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                line_curve.append(runner.get_cumulative_lines())
            continue

        # --- Q-value calibration logging (Exp 2): identical to cov_qvalue
        #     (commits argmax-Q plan, counts only its steps against budget),
        #     but additionally scores ALL K plans and trial-runs ALL K from a
        #     snapshot to record each candidate's realized branch gain. Yields
        #     paired (predicted ĝ/v̂/Q, realized gain) per candidate per round
        #     so the offline analysis can measure scorer calibration and
        #     selection accuracy. Trial runs are free lookahead (rolled back). ---
        if strategy == "cov_qvalue_calib":
            plans = gen_k_plans(module, source, hist, cov_map, K, PLAN_LENGTH)
            if not plans:
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                line_curve.append(runner.get_cumulative_lines())
                continue

            scores = _score_plans(plans, module, source, cov_map, gamma)

            snap = runner.snapshot()
            base_branches = len(snap["branches"])
            candidates = []
            for idx, plan in enumerate(plans):
                runner.restore(snap)
                for step in plan:
                    runner.run_test(step)
                realized = runner.get_cumulative_coverage() - base_branches
                candidates.append({
                    "predicted_immediate": scores[idx]["immediate"],
                    "predicted_future": scores[idx]["future"],
                    "predicted_q": scores[idx]["q"],
                    "realized_gain": realized,
                    "n_steps": len(plan),
                })

            selected_idx = max(range(len(plans)),
                               key=lambda i: scores[i]["q"])

            # Commit argmax-Q (matches cov_qvalue), counting steps vs budget
            runner.restore(snap)
            committed_gain = 0
            for step in plans[selected_idx]:
                if executions >= exec_budget:
                    break
                result = runner.run_test(step)
                hist.append((step, result))
                cov_map.update(step, set(), result.new_branches)
                committed_gain += result.new_branches
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                line_curve.append(runner.get_cumulative_lines())

            calib_log.append({
                "round": len(calib_log),
                "base_branches": base_branches,
                "selected_idx": selected_idx,
                "committed_gain": committed_gain,
                "candidates": candidates,
            })
            continue

        # --- CovBayes (Exp 11): closed-form-IG selection over a per-function
        #     Beta posterior. Same generation as cov_qvalue; the LLM only
        #     predicts which functions a plan exercises, the math does the rest. ---
        if strategy == "cov_bayes":
            plans = gen_k_plans(module, source, hist, cov_map, K, PLAN_LENGTH)
            if not plans:
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                line_curve.append(runner.get_cumulative_lines())
                continue

            covered = covered_quals(runner.cumulative_lines, cb_funcs)
            uncovered = [q for q in cb_quals if q not in covered]

            if not uncovered or not cb_quals:
                # nothing to model (all covered, or source unparsed) -> random pick
                best_plan, best_preds = _random.choice(plans), {}
            else:
                best_ig, best_plan, best_preds = -1.0, plans[0], {}
                for plan in plans:
                    qf = predict_plan_functions(module, source, plan,
                                                set(uncovered), config.MODEL)
                    score = plan_expected_gain(cb_post, uncovered, qf)
                    log.info(f"CovBayes plan score={score:.2f} "
                             f"(predicts {len(qf)} uncovered fns)")
                    if score > best_ig:
                        best_ig, best_plan, best_preds = score, plan, qf

            for step in best_plan:
                if executions >= exec_budget:
                    break
                result = runner.run_test(step)
                hist.append((step, result))
                cov_map.update(step, set(), result.new_branches)
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                line_curve.append(runner.get_cumulative_lines())

            # conjugate update: newly covered fns -> alpha+1; predicted but
            # still-uncovered -> beta+1.
            new_cov = covered_quals(runner.cumulative_lines, cb_funcs)
            for q in cb_quals:
                if q in new_cov and q not in covered:
                    cb_post[q][0] += 1.0
                elif q in best_preds and q not in new_cov:
                    cb_post[q][1] += 1.0
            # Exp 9: log the committed plan's per-function predictions vs the
            # functions it actually newly covered, with the coverage-map size,
            # to test whether prediction quality improves as exploration proceeds.
            exp9_log.append({
                "round": len(exp9_log),
                "n_covered_before": len(covered),
                "n_uncovered_before": len(uncovered),
                "predicted": {f: round(q, 3) for f, q in best_preds.items()},
                "newly_covered": sorted(new_cov - covered),
            })
            continue

        # --- CovBayes calibration (isolates SELECTION from generation variance):
        #     score all K by CovBayes AND trial-run all K from a snapshot to get
        #     each plan's realized gain, log the pairs, commit argmax-CovBayes.
        #     Lets us measure whether argmax-CovBayes picks the best-of-K plan
        #     above 1/K chance — the clean test the cross-run comparison can't give. ---
        if strategy == "cov_bayes_calib":
            plans = gen_k_plans(module, source, hist, cov_map, K, PLAN_LENGTH)
            if not plans:
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                line_curve.append(runner.get_cumulative_lines())
                continue
            covered = covered_quals(runner.cumulative_lines, cb_funcs)
            uncovered = [q for q in cb_quals if q not in covered]

            snap = runner.snapshot()
            base = len(snap["branches"])
            cands = []
            preds_per_plan = []
            for plan in plans:
                qf = (predict_plan_functions(module, source, plan, set(uncovered),
                                             config.MODEL) if uncovered else {})
                cb_score = plan_expected_gain(cb_post, uncovered, qf) if uncovered else 0.0
                preds_per_plan.append(qf)
                runner.restore(snap)
                for step in plan:
                    runner.run_test(step)
                realized = runner.get_cumulative_coverage() - base
                cands.append({"cb_score": cb_score, "realized_gain": realized,
                              "n_pred": len(qf), "n_steps": len(plan)})

            sel = max(range(len(plans)), key=lambda i: cands[i]["cb_score"])
            runner.restore(snap)
            for step in plans[sel]:
                if executions >= exec_budget:
                    break
                result = runner.run_test(step)
                hist.append((step, result))
                cov_map.update(step, set(), result.new_branches)
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                line_curve.append(runner.get_cumulative_lines())

            new_cov = covered_quals(runner.cumulative_lines, cb_funcs)
            for q in cb_quals:
                if q in new_cov and q not in covered:
                    cb_post[q][0] += 1.0
                elif q in preds_per_plan[sel] and q not in new_cov:
                    cb_post[q][1] += 1.0
            calib_log.append({"round": len(calib_log), "selected_idx": sel,
                              "candidates": cands})
            continue

        # --- Execution-based Q-value selection ---
        if strategy == "cov_qvalue_exec":
            plans = generate_plans_for_exec_selection(
                source, module, hist, cov_map, K=K, plan_length=PLAN_LENGTH)

            if not plans or executions >= exec_budget:
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                line_curve.append(runner.get_cumulative_lines())
                continue

            # Execute step 1 of each plan — observe actual coverage
            step1_results = []
            for plan in plans:
                if executions >= exec_budget:
                    break
                result = runner.run_test(plan[0])
                hist.append((plan[0], result))
                cov_map.update(plan[0], set(), result.new_branches)
                executions += 1
                branch_curve.append(runner.get_cumulative_coverage())
                line_curve.append(runner.get_cumulative_lines())
                step1_results.append((plan, result.new_branches))

            # Select the plan whose step 1 discovered the most branches
            if step1_results:
                best_plan, _ = max(step1_results, key=lambda x: x[1])

                # Execute remaining steps of the winning plan
                for plan_script in best_plan[1:]:
                    if executions >= exec_budget:
                        break
                    result = runner.run_test(plan_script)
                    hist.append((plan_script, result))
                    cov_map.update(plan_script, set(), result.new_branches)
                    executions += 1
                    branch_curve.append(runner.get_cumulative_coverage())
                    line_curve.append(runner.get_cumulative_lines())

            continue

        if not scripts:
            executions += 1
            branch_curve.append(runner.get_cumulative_coverage())
            line_curve.append(runner.get_cumulative_lines())
            continue

        # --- Standard execution ---
        if strategy in ("cov_qvalue", "cov_qvalue_rank", "divhints_random",
                        "cov_greedy_multistep"):
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
            if strategy == "random":
                selected = _random.choice(scripts)
            elif strategy == "greedy":
                selected = select_greedy(scripts, module, source, hist)
            elif strategy == "cov_greedy":
                selected = _random.choice(scripts)
            else:
                selected = scripts[0]

            result = runner.run_test(selected)
            hist.append((selected, result))
            cov_map.update(selected, set(), result.new_branches)
            executions += 1
            branch_curve.append(runner.get_cumulative_coverage())
            line_curve.append(runner.get_cumulative_lines())

    stats = runner.get_stats()

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

    result = {
        "final": stats["branches"],
        "final_lines": stats["lines"],
        "pass_rate": stats["pass_rate"],
        "pass_count": stats["pass_count"],
        "fail_count": stats["fail_count"],
        "branch_curve": branch_curve,
        "line_curve": line_curve,
        "trace": trace,
    }
    if calib_log:
        result["calib"] = calib_log
    if exp9_log:
        result["exp9"] = exp9_log
    return result


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

    # Paired comparisons vs random
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
        ("divhints_oracle", "divhints_random", "oracle_vs_divhints_random"),
        ("divhints_oracle", "cov_qvalue", "oracle_vs_qvalue"),
        ("divhints_random", "cov_greedy_multistep", "diversity_increment"),
        ("cov_greedy_multistep", "cov_greedy", "multistep_increment"),
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
    global MAP_MODE, SCORE_MAP_MODE
    args = parse_args()
    MAP_MODE = args.map_mode
    SCORE_MAP_MODE = args.score_map_mode
    reset_cost()

    targets = load_benchmark(repos=args.repos, max_targets=args.max_targets)
    bench_info = get_benchmark_info()
    strategies = args.strategies
    seeds = args.seeds

    total_runs = len(targets) * len(strategies) * len(seeds)
    print("=" * 70, flush=True)
    print(f"RepoExploreBench v{bench_info['version']}", flush=True)
    print(f"  Targets: {len(targets)} files across "
          f"{len(set(t['repo'] for t in targets))} repos", flush=True)
    print(f"  Strategies: {strategies}", flush=True)
    print(f"  Seeds: {seeds}", flush=True)
    print(f"  Exec budget: {args.exec_budget} per run", flush=True)
    print(f"  Map mode: {args.map_mode}"
          + (f" (scorer: {args.score_map_mode})" if args.score_map_mode else ""),
          flush=True)
    print(f"  Total runs: {total_runs}", flush=True)
    print("=" * 70, flush=True)

    # Connectivity check
    t = generate_with_model(config.MODEL, "Say ok", 0.3, 100)
    print(f"  Connectivity: {'OK' if t else 'FAILED'}", flush=True)
    if not t:
        return

    # Pre-fetch source code for all targets
    print(f"\n  Fetching source code...", flush=True)
    source_cache = {}
    for target in targets:
        mod = target["module"]
        if mod not in source_cache:
            source_cache[mod] = fetch_source(mod)
            status = f"{len(source_cache[mod])} chars" if source_cache[mod] else "FAILED"
            print(f"    {mod}: {status}", flush=True)
    print(f"  Fetched {sum(1 for v in source_cache.values() if v)}/{len(source_cache)} modules",
          flush=True)

    start = time.time()
    completed = [0]  # mutable counter for thread safety

    def run_one_target(i, target, seed):
        """Run all strategies on one target. Returns result dict."""
        source = source_cache.get(target["module"])
        run_result = {
            "module": target["module"],
            "repo": target["repo"],
            "seed": seed,
            "strategies": {},
        }
        for strategy in strategies:
            # Per-strategy token/call accounting (Exp 6). Uses the global cost
            # counter, so it's only correct with --parallel 1.
            if args.per_strategy_cost:
                reset_cost()
            result = run_strategy(target, strategy, seed,
                                   args.exec_budget, args.K, args.gamma,
                                   source)
            if args.per_strategy_cost:
                c = get_cost()
                result["cost"] = {k: c[k] for k in
                                  ("api_calls", "input_tokens", "output_tokens",
                                   "total_tokens", "total_cost_usd")}
            run_result["strategies"][strategy] = result
        completed[0] += 1
        finals = {s: run_result["strategies"][s]["final"] for s in strategies}
        print(f"  [{completed[0]}/{len(targets)*len(seeds)}] "
              f"{target['module']} seed={seed}: {finals}", flush=True)
        return run_result

    # Build list of (index, target, seed) jobs
    jobs = [(i, target, seed)
            for i, target in enumerate(targets)
            for seed in seeds]

    print(f"\nRunning {len(jobs)} targets with {args.parallel} workers...",
          flush=True)

    all_results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(run_one_target, i, t, s): (i, t, s)
                   for i, t, s in jobs}
        for future in as_completed(futures):
            try:
                all_results.append(future.result())
            except Exception as e:
                i, t, s = futures[future]
                print(f"  ERROR on {t['module']} seed={s}: {e}", flush=True)

    elapsed = time.time() - start
    cost = get_cost()

    # --- Summary table ---
    print(f"\n{'=' * 70}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'=' * 70}", flush=True)

    print(f"\n{'Module':<35} {'seed':>4}", end="", flush=True)
    for s in strategies:
        print(f" {s[:11]:>12}", end="")
    print(flush=True)
    print("-" * (40 + 13 * len(strategies)), flush=True)

    for r in all_results:
        print(f"{r['module'][:34]:<35} {r['seed']:>4}", end="")
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
        # Line coverage and pass rate
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
                        "cov_qvalue vs divhints_random (Exp 1)"),
                       ("oracle_vs_divhints_random",
                        "divhints_oracle vs divhints_random (Exp 1 ceiling)"),
                       ("oracle_vs_qvalue",
                        "divhints_oracle vs cov_qvalue (Exp 1 headroom)"),
                       ("diversity_increment",
                        "divhints_random vs cov_greedy_multistep (+diversity, Exp 3)"),
                       ("multistep_increment",
                        "cov_greedy_multistep vs cov_greedy (+multistep, Exp 3)")]:
        if key in analysis:
            a = analysis[key]
            sig = "***" if a["p_value"] < 0.001 else ("**" if a["p_value"] < 0.01
                    else ("*" if a["p_value"] < 0.05 else ""))
            extra = (f"  W={a['wins']} L={a['losses']} T={a['ties']}  "
                     f"d={a['cohens_d']:.2f}" if "wins" in a else "")
            print(f"\n{label}: "
                  f"Δ={a['mean_delta']:>+.1f} ± {a['se']:.1f}{extra}  "
                  f"p={a['p_value']:.4f} {sig}", flush=True)

    # --- Per-repo breakdown ---
    print(f"\nPer-repo means:", flush=True)
    repos = sorted(set(r["repo"] for r in all_results))
    for repo in repos:
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
            "benchmark": bench_info,
            "config": {
                "strategies": strategies, "seeds": seeds,
                "exec_budget": args.exec_budget, "K": args.K,
                "gamma": args.gamma, "map_mode": args.map_mode,
                "score_map_mode": args.score_map_mode,
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
