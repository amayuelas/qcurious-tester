"""TestGenEval results as file-coverage %, against the human-written suite.

Pynguin and CoverUp cannot run on TestGenEval: both require Python >= 3.10,
while the SWE-bench testbeds pin 3.9.19 for every repo but flask (checked in
the images). TestGenEval, however, ships each file's `baseline_cov` — the
coverage achieved by the project's OWN test suite — which is a stronger
reference point than a classical generator anyway, and is the metric the
benchmark is built around.

To compare on that scale we express our suites as statement coverage of the
target file:

    pct = covered statement lines in the file / executable statements in it

The numerator is what the runner already records (`final_lines`; the Docker
runner filters coverage to the target file). The denominator comes from
coverage.py's own parser applied to the file's source, so no container and no
re-run is needed.

Usage:
    python scripts/active/analyze_tge_vs_human.py \
        --inputs results/testgeneval/full_run_*.json
"""

import argparse
import glob
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from curiosity_explorer.benchmarks.testgeneval_config import (  # noqa: E402
    load_testgeneval_examples,
)

# coverage.py lives in the project venv; this script runs under the interpreter
# that has `datasets`. Count statements there and hand the numbers back.
_COUNTER = """
import sys, json
from coverage.parser import PythonParser
out = {}
for key, src in json.load(sys.stdin).items():
    try:
        p = PythonParser(text=src); p.parse_source()
        out[key] = len(p.statements)
    except Exception:
        out[key] = 0
print(json.dumps(out))
"""


def statement_counts(examples, venv_python=None):
    """{code_file: executable statements} via coverage.py's parser."""
    srcs = {e["code_file"]: e["code_src"] for e in examples if e.get("code_src")}
    py = venv_python or str(ROOT / ".venv/bin/python")
    r = subprocess.run([py, "-c", _COUNTER], input=json.dumps(srcs),
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"statement counting failed: {r.stderr[-300:]}")
    return json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+",
                    default=["results/testgeneval/full_run_*.json"])
    ap.add_argument("--venv-python", default=None)
    args = ap.parse_args()

    examples = load_testgeneval_examples()
    human = {e["code_file"]: e.get("baseline_cov") for e in examples}
    stmts = statement_counts(examples, args.venv_python)

    paths = [p for pat in args.inputs for p in sorted(glob.glob(pat))]
    if not paths:
        sys.exit(f"no result files matched {args.inputs}")

    for path in paths:
        data = json.load(open(path))
        strategies = data["config"]["strategies"]
        rows, skipped = [], 0
        for r in data["results"]:
            cf = r.get("code_file")
            n = stmts.get(cf, 0)
            if not n or human.get(cf) is None:
                skipped += 1
                continue
            rows.append({
                "human": human[cf],
                **{s: 100.0 * r["strategies"][s].get("final_lines", 0) / n
                   for s in strategies if s in r["strategies"]},
            })
        if not rows:
            print(f"{path}: no comparable rows"); continue

        print(f"\n{Path(path).name}  (n={len(rows)} files"
              f"{f', {skipped} skipped' if skipped else ''})")
        hm = statistics.mean(x["human"] for x in rows)
        print(f"  {'human test suite':<22}{hm:6.1f}% of file statements")
        for s in strategies:
            vals = [x[s] for x in rows if s in x]
            if not vals:
                continue
            beat = sum(1 for x in rows if s in x and x[s] >= x["human"])
            print(f"  {s:<22}{statistics.mean(vals):6.1f}%"
                  f"   reaches the human suite on {beat}/{len(vals)} files"
                  f" ({beat/len(vals):.0%})")


if __name__ == "__main__":
    main()
