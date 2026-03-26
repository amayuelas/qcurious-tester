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
        self._coverage_data_dir = tempfile.mkdtemp(prefix="docker_cov_")
        # Make world-writable so non-root Docker users (e.g. swe-bench) can write
        os.chmod(self._coverage_data_dir, 0o777)
        self._test_count = 0

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

        # Coverage JSON output file (mounted volume)
        cov_json_path = os.path.join(self._coverage_data_dir,
                                     f"cov_{self._test_count}.json")

        # Run with coverage, write JSON to mounted volume
        py = self.python_bin
        pre = f"{self.pre_command} && " if self.pre_command else ""
        # pip install coverage if not available (some testbeds don't have it)
        ensure_coverage = f"{py} -m pip install coverage -q 2>/dev/null; "

        cmd = [
            "docker", "run", "--rm",
            "--entrypoint", "bash",
            "-v", f"{script_path}:/tmp/test_script.py:ro",
            "-v", f"{self._coverage_data_dir}:/tmp/covdata",
            *env_args,
            self.image,
            "-c",
            f"cd {self.working_dir} && {pre}{ensure_coverage}"
            f"{py} -m coverage run --branch "
            f"--source={self.source_module} "
            f"/tmp/test_script.py 2>&1; "
            f"{py} -m coverage json -o /tmp/covdata/cov_{self._test_count}.json 2>/dev/null"
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            output_text = result.stdout.strip()
            stderr = result.stderr.strip()
        except subprocess.TimeoutExpired:
            return DockerTestResult(
                output=None, exception="TimeoutError", new_branches=0,
                cumulative=len(self.cumulative_branches)
            )
        except Exception as e:
            return DockerTestResult(
                output=None, exception=str(e), new_branches=0,
                cumulative=len(self.cumulative_branches)
            )

        # Parse coverage JSON
        exception_text = None
        new_branches = set()

        if os.path.exists(cov_json_path):
            try:
                with open(cov_json_path) as f:
                    cov_data = json.load(f)
                for file_path, file_data in cov_data.get("files", {}).items():
                    # Filter to target file if specified
                    if self.target_file and self.target_file not in file_path:
                        continue
                    # Prefer executed_branches (coverage.py 7.x)
                    exec_branches = file_data.get("executed_branches", [])
                    if exec_branches:
                        for arc in exec_branches:
                            branch = (file_path, tuple(arc))
                            if branch not in self.cumulative_branches:
                                new_branches.add(branch)
                                self.cumulative_branches.add(branch)
                    else:
                        # Fallback: use executed_lines as branch proxy
                        # (older coverage.py without executed_branches)
                        for line in file_data.get("executed_lines", []):
                            branch = (file_path, line)
                            if branch not in self.cumulative_branches:
                                new_branches.add(branch)
                                self.cumulative_branches.add(branch)
            except (json.JSONDecodeError, KeyError) as e:
                log.debug(f"Coverage parse error: {e}")
        else:
            log.debug(f"Coverage JSON not found at {cov_json_path}")

        if result.returncode != 0 and not output_text:
            exception_text = stderr[:200] if stderr else f"exit code {result.returncode}"

        return DockerTestResult(
            output=output_text[:500] if output_text else None,
            exception=exception_text,
            new_branches=len(new_branches),
            cumulative=len(self.cumulative_branches),
        )

    def get_cumulative_coverage(self):
        return len(self.cumulative_branches)

    def reset(self):
        self.cumulative_branches = set()
        self._test_count = 0

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
    def __init__(self, output, exception, new_branches, cumulative):
        self.output = output
        self.exception = exception
        self.new_branches = new_branches
        self.cumulative = cumulative
