"""Run TestGenEval Lite repo-by-repo to bound Docker disk usage.

The 140-example headline subset needs ~40 SWE-bench images (~39 GB compressed,
far more unpacked). Holding them all at once filled the disk in earlier runs,
and an image pulled implicitly by `docker run` counts against the 60 s test
timeout, so every test on that target fails. Instead, for each repo:

  1. pull that repo's images up front (with a free-disk guard),
  2. run every requested model on that repo concurrently,
  3. remove the images this script pulled.

Per-repo outputs go to results/testgeneval/by_repo/<key>/<repo>.json and are
skipped if already present, so the driver is resumable. At the end they are
merged into results/testgeneval/full_run_<key>.json.

Usage:
  python scripts/active/run_tge_by_repo.py \
      --models gemini38=gemini-3.8-flash glm53=zai-glm-5-3 gemma4=gemma-4-31B-it
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from curiosity_explorer.benchmarks.testgeneval_config import (  # noqa: E402
    load_testgeneval_examples,
)

# Paper headline subset: TestGenEval Lite minus scikit-learn (140 examples).
HEADLINE_REPOS = [
    "django/django", "sympy/sympy", "pytest-dev/pytest",
    "matplotlib/matplotlib", "astropy/astropy", "pydata/xarray",
    "sphinx-doc/sphinx", "mwaskom/seaborn", "pylint-dev/pylint",
    "pallets/flask",
]
STRATEGIES = ["random", "greedy", "cov_greedy", "cov_qvalue", "cov_qvalue_exec"]
OUT_DIR = config.RESULTS_DIR / "testgeneval"


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def free_gb(path="/var/lib/docker"):
    p = path if os.path.exists(path) else "/"
    return shutil.disk_usage(p).free / 1024**3


def image_present(image):
    return subprocess.run(["docker", "image", "inspect", image],
                          capture_output=True).returncode == 0


def pull(image, retries=3):
    for attempt in range(retries):
        r = subprocess.run(["docker", "pull", "-q", image],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return True
        log(f"  pull failed ({attempt+1}/{retries}) {image}: {r.stderr[-200:]}")
        time.sleep(10 * (attempt + 1))
    return False


def merge(key, repos):
    """Merge per-repo outputs for one model into full_run_<key>.json."""
    from run_testgeneval import analyze_results

    parts = []
    for repo in repos:
        p = OUT_DIR / "by_repo" / key / f"{repo.replace('/', '__')}.json"
        if p.exists():
            parts.append(json.load(open(p)))
        else:
            log(f"  merge {key}: missing {repo}")
    if not parts:
        return
    results = [r for d in parts for r in d["results"]]
    strategies = parts[0]["config"]["strategies"]
    analysis = analyze_results(results, strategies)

    per_model, totals = {}, {"api_calls": 0, "input_tokens": 0,
                             "output_tokens": 0, "total_cost_usd": 0.0}
    for d in parts:
        c = d["cost"]
        for k in ("api_calls", "input_tokens", "output_tokens"):
            totals[k] += c.get(k, 0)
        totals["total_cost_usd"] += c.get("total_cost_usd", 0.0)
        for m, u in c.get("per_model", {}).items():
            agg = per_model.setdefault(m, {"api_calls": 0, "input_tokens": 0,
                                           "output_tokens": 0, "cost_usd": 0.0})
            for k in agg:
                agg[k] += u.get(k, 0)
    cost = {"model": parts[0]["cost"].get("model"), **totals,
            "total_tokens": totals["input_tokens"] + totals["output_tokens"],
            "per_model": per_model}

    out = OUT_DIR / f"full_run_{key}.json"
    with open(out, "w") as f:
        json.dump({
            "benchmark": "TestGenEval Lite",
            "config": {**parts[0]["config"], "repos": repos},
            "results": results,
            "analysis": {k: v for k, v in analysis.items()
                         if k != "paired_vs_random"},
            "paired_vs_random": analysis.get("paired_vs_random", {}),
            "cost": cost,
            "elapsed": round(sum(d["elapsed"] for d in parts), 1),
        }, f, indent=2, default=str)
    log(f"merged {len(results)} results -> {out} (${cost['total_cost_usd']:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="key=model_id pairs, e.g. gemini38=gemini-3.8-flash")
    ap.add_argument("--repos", nargs="+", default=HEADLINE_REPOS)
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--exec-budget", type=int, default=24)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--min-free-gb", type=float, default=30,
                    help="refuse to pull if less than this would remain free")
    ap.add_argument("--keep-images", action="store_true")
    args = ap.parse_args()

    models = dict(m.split("=", 1) for m in args.models)
    examples = load_testgeneval_examples(repos=args.repos)

    for repo in args.repos:
        slug = repo.replace("/", "__")
        todo = {k: m for k, m in models.items()
                if not (OUT_DIR / "by_repo" / k / f"{slug}.json").exists()}
        if not todo:
            log(f"{repo}: all models done, skipping")
            continue

        images = sorted({e["image"] for e in examples if e["repo"] == repo})
        pulled = []
        log(f"{repo}: {len(images)} images, models {list(todo)}, "
            f"{free_gb():.0f} GB free")
        ok = True
        for img in images:
            if image_present(img):
                continue
            if free_gb() < args.min_free_gb:
                log(f"  ABORT: only {free_gb():.0f} GB free before pulling {img}")
                ok = False
                break
            if pull(img):
                pulled.append(img)
            else:
                ok = False
                break
        if not ok:
            log(f"{repo}: image setup failed — skipping repo")
            continue

        procs = []
        for key, model in todo.items():
            out_rel = f"testgeneval/by_repo/{key}/{slug}.json"
            (OUT_DIR / "by_repo" / key).mkdir(parents=True, exist_ok=True)
            logf = open(OUT_DIR / "by_repo" / key / f"{slug}.log", "w")
            cmd = [sys.executable, "run_testgeneval.py", "--repos", repo,
                   "--strategies", *args.strategies, "--seeds", "42",
                   "--exec-budget", str(args.exec_budget), "--K", str(args.K),
                   "--gamma", str(args.gamma), "--parallel", str(args.parallel),
                   "--output", out_rel]
            procs.append((key, subprocess.Popen(
                cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                env={**os.environ, "MODEL": model}), logf))
        for key, p, logf in procs:
            rc = p.wait()
            logf.close()
            log(f"  {repo} / {key}: exit {rc}")

        if pulled and not args.keep_images:
            subprocess.run(["docker", "rmi", *pulled], capture_output=True)
            log(f"  removed {len(pulled)} images, {free_gb():.0f} GB free")

    for key in models:
        merge(key, args.repos)


if __name__ == "__main__":
    main()
