"""Run tests inside Docker containers and measure coverage.

For TestGenEval: runs generated test scripts inside SWE-bench Docker
containers with coverage.py tracking branch coverage on the target file.
"""

import json
import logging
import subprocess
import tempfile
import os

log = logging.getLogger(__name__)


class DockerCoverageRunner:
    """Run test scripts in a Docker container, tracking cumulative coverage."""

    def __init__(self, image: str, source_module: str, setup_code: str = "",
                 working_dir: str = "/opt/django__django",
                 env: dict = None, python_bin: str = "python",
                 pre_command: str = "", target_file: str = None):
        """
        Args:
            image: Docker image name (e.g. 'aorwall/swe-bench-django_django-testbed:4.0')
            source_module: Module to track coverage on (e.g. 'django.forms.boundfield').
                For broad tracking, use top-level package (e.g. 'sympy') and
                set target_file to filter results.
            setup_code: Python code to run before each test (e.g. 'import django; django.setup()')
            working_dir: Working directory inside container
            env: Environment variables to set
            python_bin: Path to Python binary (for conda envs, e.g.
                '/home/swe-bench/miniconda3/envs/sympy__sympy__1.13/bin/python')
            pre_command: Shell command to run before Python (e.g. 'pip install -e .')
            target_file: If set, only count branches from files matching this
                substring (e.g. 'sympy/physics/units/util.py'). Used when
                --source is a broad package but we want file-level coverage.
        """
        self.image = image
        self.source_module = source_module
        self.setup_code = setup_code
        self.working_dir = working_dir
        self.env = env or {}
        self.python_bin = python_bin
        self.pre_command = pre_command
        self.target_file = target_file
        self.cumulative_branches = set()
        self.cumulative_lines = set()
        self._coverage_data_dir = tempfile.mkdtemp(prefix="docker_cov_")
        # Make world-writable so non-root Docker users (e.g. swe-bench) can write
        os.chmod(self._coverage_data_dir, 0o777)
        self._test_count = 0
        self._pass_count = 0
        self._fail_count = 0

    def run_test(self, test_script: str, timeout: int = 30):
        """Run a test script inside the container and measure coverage.

        Args:
            test_script: Python code to execute as a test
            timeout: Maximum seconds for execution

        Returns:
            DockerTestResult with output, exception, new_branches, coverage info
        """
        self._test_count += 1

        # Build full script with setup
        full_script = f"{self.setup_code}\n{test_script}"

        # Write script to temp file
        script_path = os.path.join(self._coverage_data_dir,
                                   f"test_{self._test_count}.py")
        with open(script_path, "w") as f:
            f.write(full_script)

        # Build docker command
        env_args = []
        for k, v in self.env.items():
            env_args.extend(["-e", f"{k}={v}"])

        # Run with coverage, write JSON inside container then cat to stdout
        py = self.python_bin
        pre = f"{self.pre_command} && " if self.pre_command else ""
        ensure_coverage = f"{py} -m pip install coverage -q 2>/dev/null; "

        # Strategy: run coverage, generate JSON inside container, print a
        # separator then cat the JSON to stdout. We parse it from the output.
        separator = "===COVERAGE_JSON_START==="
        cmd = [
            "docker", "run", "--rm",
            "--entrypoint", "bash",
            "-v", f"{script_path}:/tmp/test_script.py:ro",
            *env_args,
            self.image,
            "-c",
            f"cd {self.working_dir} && {pre}{ensure_coverage}"
            f"{py} -m coverage run --rcfile=/dev/null --branch "
            f"--source={self.source_module} "
            f"/tmp/test_script.py 2>&1; "
            f"echo '{separator}'; "
            f"{py} -m coverage json --rcfile=/dev/null -o /tmp/cov.json 2>/dev/null && "
            f"cat /tmp/cov.json 2>/dev/null"
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            raw_stdout = result.stdout.strip()
            stderr = result.stderr.strip()
        except subprocess.TimeoutExpired:
            self._fail_count += 1
            return DockerTestResult(
                output=None, exception="TimeoutError", new_branches=0,
                cumulative_branches=len(self.cumulative_branches),
                cumulative_lines=len(self.cumulative_lines),
            )
        except Exception as e:
            self._fail_count += 1
            return DockerTestResult(
                output=None, exception=str(e), new_branches=0,
                cumulative_branches=len(self.cumulative_branches),
                cumulative_lines=len(self.cumulative_lines),
            )

        # Split output: test output before separator, coverage JSON after
        separator = "===COVERAGE_JSON_START==="
        if separator in raw_stdout:
            parts = raw_stdout.split(separator, 1)
            output_text = parts[0].strip()
            after_sep = parts[1].strip()
            # Find the JSON object start (skip "Wrote JSON report..." line)
            json_start = after_sep.find("{")
            cov_json_str = after_sep[json_start:] if json_start >= 0 else ""
        else:
            output_text = raw_stdout
            cov_json_str = ""

        # Parse coverage JSON from stdout
        exception_text = None
        new_branches = set()
        new_lines = set()

        if cov_json_str:
            try:
                cov_data = json.loads(cov_json_str)
                for file_path, file_data in cov_data.get("files", {}).items():
                    # Filter to target file if specified
                    if self.target_file and self.target_file not in file_path:
                        continue
                    # Track branches (coverage.py 7.x)
                    exec_branches = file_data.get("executed_branches", [])
                    if exec_branches:
                        for arc in exec_branches:
                            branch = (file_path, tuple(arc))
                            if branch not in self.cumulative_branches:
                                new_branches.add(branch)
                                self.cumulative_branches.add(branch)
                    # Track lines (always available)
                    for line in file_data.get("executed_lines", []):
                        line_key = (file_path, line)
                        if line_key not in self.cumulative_lines:
                            new_lines.add(line_key)
                            self.cumulative_lines.add(line_key)
                    # Fallback: if no branches, use lines as branch proxy
                    if not exec_branches:
                        for line in file_data.get("executed_lines", []):
                            branch = (file_path, line)
                            if branch not in self.cumulative_branches:
                                new_branches.add(branch)
                                self.cumulative_branches.add(branch)
            except (json.JSONDecodeError, KeyError) as e:
                log.debug(f"Coverage parse error: {e}")
        else:
            log.debug("No coverage JSON in output")

        # Determine pass/fail
        # A test "passes" if it produces output without crashing
        has_output = bool(output_text and output_text.strip())
        has_error = (result.returncode != 0) or ("Traceback" in (output_text or ""))
        passed = has_output and not has_error

        if passed:
            self._pass_count += 1
        else:
            self._fail_count += 1

        if result.returncode != 0 and not output_text:
            exception_text = stderr[:200] if stderr else f"exit code {result.returncode}"

        return DockerTestResult(
            output=output_text[:500] if output_text else None,
            exception=exception_text,
            new_branches=len(new_branches),
            new_lines=len(new_lines),
            cumulative_branches=len(self.cumulative_branches),
            cumulative_lines=len(self.cumulative_lines),
            passed=passed,
        )

    def get_cumulative_coverage(self):
        return len(self.cumulative_branches)

    def get_cumulative_lines(self):
        return len(self.cumulative_lines)

    def get_pass_rate(self):
        total = self._pass_count + self._fail_count
        return self._pass_count / total if total > 0 else 0.0

    def get_stats(self):
        """Return all tracked metrics."""
        total = self._pass_count + self._fail_count
        return {
            "branches": len(self.cumulative_branches),
            "lines": len(self.cumulative_lines),
            "pass_count": self._pass_count,
            "fail_count": self._fail_count,
            "pass_rate": self._pass_count / total if total > 0 else 0.0,
        }

    def reset(self):
        self.cumulative_branches = set()
        self.cumulative_lines = set()
        self._test_count = 0
        self._pass_count = 0
        self._fail_count = 0

    def cleanup(self):
        """Remove temporary coverage data directory."""
        import shutil
        shutil.rmtree(self._coverage_data_dir, ignore_errors=True)

    def __del__(self):
        self.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cleanup()


class DockerTestResult:
    """Result of running a test in Docker."""
    def __init__(self, output, exception, new_branches, cumulative_branches,
                 new_lines=0, cumulative_lines=0, passed=False):
        self.output = output
        self.exception = exception
        self.new_branches = new_branches
        self.new_lines = new_lines
        self.cumulative = cumulative_branches  # backward compat
        self.cumulative_branches = cumulative_branches
        self.cumulative_lines = cumulative_lines
        self.passed = passed
