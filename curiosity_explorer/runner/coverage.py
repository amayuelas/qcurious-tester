"""Coverage measurement with per-branch tracking."""

import json
import os
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import Any

from .sandbox import execute_in_sandbox


@dataclass
class CoverageResult:
    output: Any = None
    exception: str = None
    branches_hit: int = 0
    new_branches: int = 0
    branches_detail: set = field(default_factory=set)     # set of (from, to) arcs
    new_branches_detail: set = field(default_factory=set)  # arcs not hit before
    lines_hit: int = 0
    total_lines: int = 0
    line_coverage: float = 0.0


class CoverageRunner:
    """Runs a function with coverage tracking and per-branch accounting."""

    def __init__(self, func_name: str, source_code: str):
        self.func_name = func_name
        self.source_code = source_code
        self.cumulative_branches: set[tuple[int, int]] = set()
        self.test_history: list[tuple[str, CoverageResult]] = []
        self.temp_dir = tempfile.mkdtemp()
        self.func_file = os.path.join(self.temp_dir, f"{func_name}.py")
        self._cleaned_up = False

        with open(self.func_file, "w") as f:
            f.write(source_code)

    def run_test(self, test_input_code: str) -> CoverageResult:
        """Execute the function with given input and measure coverage."""
        runner_code = textwrap.dedent(f'''\
            import json, sys, coverage
            sys.path.insert(0, "{self.temp_dir}")

            cov = coverage.Coverage(branch=True, source=["{self.temp_dir}"])
            cov.start()

            from {self.func_name} import {self.func_name}
            try:
                result = {test_input_code}
                print(json.dumps({{"output": repr(result), "exception": None}}))
            except Exception as e:
                print(json.dumps({{"output": None, "exception": f"{{type(e).__name__}}: {{e}}"}}))

            cov.stop()
            cov.save()

            analysis = cov.analysis2("{self.func_file}")
            executed_lines = analysis[1]
            missing_lines = analysis[3]
            total_lines = len(executed_lines) + len(missing_lines)

            branch_data = cov.get_data()
            arcs = branch_data.arcs("{self.func_file}") or []

            print("---COVERAGE---")
            print(json.dumps({{
                "lines_hit": sorted(executed_lines),
                "total_lines": total_lines,
                "arcs": [list(a) for a in arcs],
            }}))
        ''')

        runner_file = os.path.join(self.temp_dir, "_runner.py")
        with open(runner_file, "w") as f:
            f.write(runner_code)

        result = execute_in_sandbox(runner_file, cwd=self.temp_dir)

        if result["timed_out"]:
            return CoverageResult(exception="TimeoutError")

        stdout = result["stdout"].strip()
        if "---COVERAGE---" not in stdout:
            return CoverageResult(
                exception=f"Coverage extraction failed: {result['stderr'][:200]}"
            )

        parts = stdout.split("---COVERAGE---")
        try:
            exec_result = json.loads(parts[0].strip())
            cov_data = json.loads(parts[1].strip())
        except (json.JSONDecodeError, IndexError):
            return CoverageResult(exception="Parse error")

        arcs = set(tuple(a) for a in cov_data.get("arcs", []))
        lines = set(cov_data.get("lines_hit", []))
        new_branches = arcs - self.cumulative_branches
        self.cumulative_branches |= arcs

        total_lines = cov_data.get("total_lines", 1)
        cov_result = CoverageResult(
            output=exec_result.get("output"),
            exception=exec_result.get("exception"),
            branches_hit=len(arcs),
            new_branches=len(new_branches),
            branches_detail=arcs,
            new_branches_detail=new_branches,
            lines_hit=len(lines),
            total_lines=total_lines,
            line_coverage=len(lines) / max(total_lines, 1),
        )

        self.test_history.append((test_input_code, cov_result))
        return cov_result

    def get_cumulative_coverage(self) -> int:
        """Return cumulative branch (arc) count."""
        return len(self.cumulative_branches)

    def get_cumulative_branches(self) -> set[tuple[int, int]]:
        """Return the actual set of covered arcs."""
        return set(self.cumulative_branches)

    def reset(self):
        self.cumulative_branches = set()
        self.test_history = []

    def cleanup(self):
        """Remove temporary directory."""
        if not self._cleaned_up:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self._cleaned_up = True

    def __del__(self):
        self.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()
