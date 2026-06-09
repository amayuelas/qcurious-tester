"""Exp 10 add-on: run the external baselines (Pynguin, CoverUp) on the synthetic
corridor modules, for a controlled head-to-head with our strategies.

Builds a combined image (curiositybench-baselines + the synthbench package) and
reuses run_external_baselines' adapters, measuring branch coverage as % of each
module's total-branch ceiling (same metric as the LLM strategies in Exp 10).
"""

import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import config  # noqa: E402
import run_external_baselines as REB  # noqa: E402
from run_synth_corridor import ceiling, module_name, BUILD_DIR  # noqa: E402

COMBINED_IMAGE = "curiositybench-synth-baselines:latest"
DEPTHS = [2, 4, 6, 8]
K, M = 20, 6


def build_combined():
    """FROM curiositybench-baselines + COPY the synthbench package (already
    generated under build/synth/synthbench by run_synth_corridor --build)."""
    pkg = os.path.join(BUILD_DIR, "synthbench")
    assert os.path.isdir(pkg), f"missing {pkg}; run run_synth_corridor.py --build first"
    dockerfile = os.path.join(BUILD_DIR, "Dockerfile.baselines")
    with open(dockerfile, "w") as f:
        f.write("FROM curiositybench-baselines:latest\n"
                "COPY synthbench /usr/local/lib/python3.11/site-packages/synthbench\n"
                "ENV PYNGUIN_DANGER_AWARE=1\nWORKDIR /opt\n")
    print(f"Building {COMBINED_IMAGE}...", flush=True)
    r = subprocess.run(["docker", "build", "-f", "Dockerfile.baselines",
                        "-t", COMBINED_IMAGE, "."],
                       cwd=BUILD_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-1500:]); sys.exit(1)
    print("  built.", flush=True)


def main():
    build_combined()
    REB.BASELINE_IMAGE = COMBINED_IMAGE   # point the adapters at the combined image

    out = {"image": COMBINED_IMAGE, "by_depth": {}}
    print(f"\n{'depth':>5} {'ceil':>5} {'pynguin':>16} {'coverup':>16}")
    for d in DEPTHS:
        ceil = ceiling(d, K, M)   # measured on the synth image (identical modules)
        target = {"module": module_name(d), "repo": "synth",
                  "docker_image": COMBINED_IMAGE, "setup_code": "",
                  "working_dir": "/opt", "env": {}}
        pyn = REB.run_pynguin_target(target, search_time=60, docker_timeout=300)
        cov = REB.run_coverup_target(target, model="gemma-4-31B-it",
                                     docker_timeout=900)
        pp = 100 * pyn["final"] / ceil if ceil else 0
        cp = 100 * cov["final"] / ceil if ceil else 0
        print(f"{d:>5} {ceil:>5} {pyn['final']:>7}/{pp:>4.0f}% "
              f"{cov['final']:>7}/{cp:>4.0f}%", flush=True)
        out["by_depth"][d] = {"ceiling": ceil,
                              "pynguin": {"final": pyn["final"], "pct": pp},
                              "coverup": {"final": cov["final"], "pct": cp}}

    outpath = config.RESULTS_DIR / "repo_explore_bench/exp10_synth_baselines.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
