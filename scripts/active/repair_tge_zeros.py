"""Re-run TestGenEval files that scored zero for every strategy, and patch them in.

An all-zero file is almost always an environment failure rather than a result:
the SWE-bench testbeds ship no coverage module, so if its one-time install in
the container fails, every test in that file reports "No module named coverage"
and the file records 0 for all strategies. (The runner now installs coverage at
container start, verifies it and raises otherwise; this script repairs runs
made before that.)

For each model it finds the affected files, re-runs only those, and replaces
those entries in the per-repo result files, leaving everything else untouched.

Usage:
    MODEL=gemini-3.8-flash python scripts/active/repair_tge_zeros.py \
        --model-key gemini38 --model gemini-3.8-flash
    python scripts/active/repair_tge_zeros.py --model-key gemini38 --list-only
"""

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BY_REPO = ROOT / "results/testgeneval/by_repo"

os.environ.setdefault("DOCKER_NETWORK", "host")


def zero_files(model_key):
    """{repo_slug: [code_file, ...]} for files that are 0 on every strategy."""
    out = defaultdict(list)
    for path in sorted((BY_REPO / model_key).glob("*.json")):
        data = json.load(open(path))
        strategies = data["config"]["strategies"]
        for r in data["results"]:
            if all(r["strategies"][s]["final"] == 0 for s in strategies
                   if s in r["strategies"]):
                out[path.stem].append(r["code_file"])
    return out


def repair_repo(model_key, model, repo_slug, files, args):
    repo = repo_slug.replace("__", "/", 1)
    out_rel = f"testgeneval/repair/{model_key}__{repo_slug}.json"
    (ROOT / "results/testgeneval/repair").mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "run_testgeneval.py", "--repos", repo,
           "--code-files", *files,
           "--strategies", *args.strategies,
           "--seeds", "42", "--exec-budget", str(args.exec_budget),
           "--K", str(args.K), "--gamma", str(args.gamma),
           "--parallel", str(min(args.parallel, len(files))),
           "--output", out_rel]
    log = ROOT / f"results/testgeneval/repair/{model_key}__{repo_slug}.log"
    print(f"  re-running {len(files)} file(s) of {repo} ...", flush=True)
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                            env={**os.environ, "MODEL": model}).returncode
    if rc != 0:
        print(f"    FAILED (exit {rc}); see {log}")
        return 0
    fixed = json.load(open(ROOT / "results" / out_rel))
    # key on (file, version): TestGenEval holds the same path at several repo
    # versions (django/db/migrations/serializer.py appears 5 times), so keying
    # on the path alone can patch an entry with another version's results.
    new = {(r["code_file"], str(r.get("version"))): r for r in fixed["results"]}

    target = BY_REPO / model_key / f"{repo_slug}.json"
    data = json.load(open(target))
    patched = 0
    for i, r in enumerate(data["results"]):
        # --code-files selects by path, so the rerun also covers other
        # versions of the same file that were never zero; only replace
        # entries that were all-zero (the ones this script is repairing).
        was_zero = all(v["final"] == 0 for v in r["strategies"].values())
        repl = new.get((r["code_file"], str(r.get("version"))))
        if was_zero and repl and any(repl["strategies"][s]["final"] > 0
                                     for s in repl["strategies"]):
            data["results"][i] = repl
            patched += 1
    json.dump(data, open(target, "w"), indent=2, default=str)
    print(f"    patched {patched}/{len(files)} file(s) into {target.name}")
    return patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, help="e.g. gemini38")
    ap.add_argument("--model", help="model id for MODEL= (required unless --list-only)")
    ap.add_argument("--strategies", nargs="+",
                    default=["random", "greedy", "cov_greedy", "covqvalue2"])
    ap.add_argument("--exec-budget", type=int, default=24)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    zeros = zero_files(args.model_key)
    total = sum(len(v) for v in zeros.values())
    print(f"{args.model_key}: {total} all-zero file(s) across {len(zeros)} repo(s)")
    for repo, files in zeros.items():
        print(f"  {repo}: {len(files)}")
    if args.list_only or not total:
        return
    if not args.model:
        sys.exit("--model is required to repair")

    fixed = sum(repair_repo(args.model_key, args.model, repo, files, args)
                for repo, files in zeros.items())
    print(f"repaired {fixed}/{total} file(s)")


if __name__ == "__main__":
    main()
