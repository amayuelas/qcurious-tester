"""Exp 10 — LLM condition: run the real pipeline on synthetic corridor modules.

Generates depth-d corridor modules (see scripts/active/gen_synth_corridor.py),
bakes them into a Docker image so they're importable + coverage-measurable, then
runs the ACTUAL strategies (random/greedy/cov_greedy/divhints_random/cov_qvalue
via run_repo_explore_bench.run_strategy) on them and compares branch coverage to
each module's total-branch ceiling (measured with an exhaustive "god test").

Tests whether the method navigates corridors: deep terminal branches require a
multi-step setup sequence, so multi-step strategies should reach them while
single-step ones plateau — and (per Exp 1/2) cov_qvalue should ≈ divhints_random.

Usage:
    python run_synth_corridor.py --build            # (re)build the image first
    python run_synth_corridor.py --depths 2 4 6 8 --seeds 42 123 456
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts", "active"))
from gen_synth_corridor import gen_corridor_source  # noqa: E402

import config  # noqa: E402
from curiosity_explorer.runner.docker_coverage import DockerCoverageRunner  # noqa: E402
from run_repo_explore_bench import run_strategy  # noqa: E402

IMAGE = "curiositybench-synth:latest"
BUILD_DIR = os.path.join(os.path.dirname(__file__), "build", "synth")
ALL_DEPTHS = [2, 4, 6, 8]
STRATEGIES = ["random", "greedy", "cov_greedy", "divhints_random", "cov_qvalue"]


def module_name(d):
    return f"synthbench.corridor_d{d}"


def build_image(depths, k, m):
    """Write the synthbench package + Dockerfile and build the synth image."""
    pkg = os.path.join(BUILD_DIR, "synthbench")
    os.makedirs(pkg, exist_ok=True)
    open(os.path.join(pkg, "__init__.py"), "w").close()
    for d in depths:
        with open(os.path.join(pkg, f"corridor_d{d}.py"), "w") as f:
            f.write(gen_corridor_source(d, k, m))
    # site-packages path is fixed in the curiosity image (py3.11)
    dockerfile = (
        "FROM curiositybench:latest\n"
        "COPY synthbench /usr/local/lib/python3.11/site-packages/synthbench\n"
        "WORKDIR /opt\n")
    with open(os.path.join(BUILD_DIR, "Dockerfile"), "w") as f:
        f.write(dockerfile)
    print(f"Building {IMAGE} with depths {depths} (k={k}, m={m})...", flush=True)
    r = subprocess.run(["docker", "build", "-t", IMAGE, "."],
                       cwd=BUILD_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:]); sys.exit(1)
    print("  built.", flush=True)


def god_test(d, k, m):
    """Exhaustive test covering every reachable branch → total-branch ceiling."""
    mod = module_name(d)
    L = [f"import {mod} as M"]
    for i in range(m):  # both distractor branches
        L += [f"M.dist{i}(1)", f"M.dist{i}(0)"]
    L.append("c = M.Corridor(); c.terminal(0)")          # locked_terminal
    for i in range(1, d):                                # locked{i} branches
        L.append(f"M.Corridor().s{i}('x')")
    L.append("c = M.Corridor()")
    L.append("c.s0('bad'); c.s0('k0')")                  # no0, ok0
    for i in range(1, d):
        L.append(f"c.s{i}('bad'); c.s{i}('k{i}')")       # wrong{i}, ok{i}
    for j in range(k):
        L.append(f"c.terminal({j})")                     # t0..t{k-1}
    L.append("c.terminal(99999)")                        # tdefault
    L.append("print('god ok')")
    return "\n".join(L)


def ceiling(d, k, m):
    runner = DockerCoverageRunner(image=IMAGE, source_module=module_name(d),
                                  setup_code="", working_dir="/opt", env={})
    res = runner.run_test(god_test(d, k, m))
    return res.cumulative_branches


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true", help="(re)build the synth image")
    p.add_argument("--depths", nargs="+", type=int, default=ALL_DEPTHS)
    p.add_argument("--k", type=int, default=20, help="terminal branches")
    p.add_argument("--m", type=int, default=6, help="distractors")
    p.add_argument("--strategies", nargs="+", default=STRATEGIES)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--exec-budget", type=int, default=24)
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--parallel", type=int, default=8)
    p.add_argument("--output", default="repo_explore_bench/exp10_llm_synth.json")
    return p.parse_args()


def main():
    args = parse_args()
    if args.build:
        build_image(args.depths, args.k, args.m)

    print(f"\nComputing per-depth ceilings (total reachable branches)...", flush=True)
    ceil = {d: ceiling(d, args.k, args.m) for d in args.depths}
    for d in args.depths:
        print(f"  depth {d}: {ceil[d]} total branches", flush=True)

    sources = {d: gen_corridor_source(d, args.k, args.m) for d in args.depths}
    jobs = [(d, s, seed) for d in args.depths for s in args.strategies
            for seed in args.seeds]
    print(f"\nRunning {len(jobs)} (depth×strategy×seed) jobs on {config.MODEL} "
          f"(parallel {args.parallel})...", flush=True)

    results = {}  # (d, strat) -> [finals]

    def run_one(d, strat, seed):
        target = {"module": module_name(d), "repo": "synth",
                  "docker_image": IMAGE, "setup_code": "", "working_dir": "/opt",
                  "env": {}}
        r = run_strategy(target, strat, seed, args.exec_budget, args.K,
                         args.gamma, sources[d])
        return d, strat, r["final"]

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futs = [ex.submit(run_one, *j) for j in jobs]
        for f in as_completed(futs):
            d, strat, final = f.result()
            results.setdefault((d, strat), []).append(final)

    # ---- Report: mean branches + % of ceiling, per depth × strategy ----
    print(f"\n{'='*70}\nExp 10 LLM condition — branches (mean over seeds) / % of ceiling")
    print(f"{'='*70}")
    hdr = f"  {'depth':>5} {'ceil':>5}  " + "".join(f"{s[:11]:>13}" for s in args.strategies)
    print(hdr)
    out = {"config": vars(args), "ceiling": ceil, "by_depth": {}}
    for d in args.depths:
        row = f"  {d:>5} {ceil[d]:>5}  "
        out["by_depth"][d] = {"ceiling": ceil[d], "strategies": {}}
        for s in args.strategies:
            vals = results.get((d, s), [])
            mv = statistics.mean(vals) if vals else 0
            pct = 100 * mv / ceil[d] if ceil[d] else 0
            row += f"{mv:>6.1f}/{pct:>4.0f}% "
            out["by_depth"][d]["strategies"][s] = {"mean": mv, "pct": pct,
                                                   "vals": vals}
        print(row)

    outpath = config.RESULTS_DIR / args.output
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
