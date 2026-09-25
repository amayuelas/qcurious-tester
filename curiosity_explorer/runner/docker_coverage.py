"""Run tests inside Docker containers and measure coverage.

For TestGenEval: runs generated test scripts inside SWE-bench Docker
containers with coverage.py tracking branch coverage on the target file.
"""

import atexit
import json
import logging
import os
import signal
import subprocess
import tempfile
import threading
import time
import uuid

log = logging.getLogger(__name__)

# Resource caps for each test container. LLM-generated tests can loop forever
# or allocate unboundedly; without caps, a few of them in parallel exhaust host
# RAM and the OOM killer takes down co-located services (e.g. the vLLM server).
# Swap equals memory, which disables swap for the container.
DOCKER_MEMORY = os.environ.get("DOCKER_MEMORY", "4g")
DOCKER_CPUS = os.environ.get("DOCKER_CPUS", "2")
DOCKER_PIDS = os.environ.get("DOCKER_PIDS", "512")

# "run": a fresh `docker run --rm` container per test (original behaviour).
# Default "exec": validated identical to "run" on 216 replayed tests across all
# 9 RepoExploreBench repos (scripts/active/validate_docker_exec_mode.py), 10x faster.
# "exec": one long-lived container per runner, each test via `docker exec`.
# Container start costs ~5 s on a busy daemon versus ~1-2 s for the test
# itself, so exec mode is several times faster. Tests of one runner then share
# the container's filesystem; timeouts still get a fresh container.
DOCKER_MODE = os.environ.get("DOCKER_MODE", "exec")

# Network mode for test containers. Neither default touches docker0: every
# bridge-networked container consumes one veth slot, a host-wide limit (1023)
# that other workloads on a shared machine can exhaust, after which
# `docker run` fails with "exchange full".
#   "none" — no interface at all. Fine when the image already has coverage
#            (our curiositybench image does).
#   "host" — shares the host stack, so `pip install coverage` works. Needed
#            for the SWE-bench testbeds, which ship no coverage (and pythons
#            as old as 3.8).
# TestGenEval runners set DOCKER_NETWORK=host; RepoExploreBench keeps "none".
DOCKER_NETWORK = os.environ.get("DOCKER_NETWORK", "none")

# Every container this process starts carries this label, so leftovers can be
# killed at exit (or by hand: docker ps -q --filter label=qcurious.owner=<pid>).
_OWNER_LABEL = f"qcurious.owner={os.getpid()}"

# Set on shutdown: run_test refuses to start new containers.
_shutting_down = threading.Event()


def _kill_container(name: str):
    subprocess.run(["docker", "kill", name], capture_output=True, timeout=30)


def _kill_own_containers():
    """Stop launching, kill in-flight `docker run` clients, then our containers.

    Order matters: a `docker run` client that outlives this process still
    creates its container, so the clients die first. The label sweep runs twice
    to catch a create request the daemon had already accepted.
    """
    _shutting_down.set()
    try:
        subprocess.run(["pkill", "-KILL", "-P", str(os.getpid()), "-x", "docker"],
                       capture_output=True, timeout=30)
        for attempt in range(2):
            ids = subprocess.run(
                ["docker", "ps", "-q", "--filter", f"label={_OWNER_LABEL}"],
                capture_output=True, text=True, timeout=30).stdout.split()
            if ids:
                subprocess.run(["docker", "kill", *ids], capture_output=True,
                               timeout=60)
            if attempt == 0:
                time.sleep(2)
    except Exception as e:
        log.warning(f"Container cleanup failed: {e}")


atexit.register(_kill_own_containers)


def _on_sigterm(signum, frame):
    # Kill our containers, then exit at once. Raising SystemExit is not enough:
    # the interpreter waits for ThreadPoolExecutor workers at shutdown, and they
    # keep starting new containers until their jobs drain.
    _kill_own_containers()
    os._exit(128 + signum)


if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except ValueError:  # not in main thread
        pass


class DockerCoverageRunner:
    """Run test scripts in a Docker container, tracking cumulative coverage."""

    def __init__(self, image: str, source_module: str, setup_code: str = "",
                 working_dir: str = "/opt/django__django",
                 env: dict = None, python_bin: str = "python",
                 pre_command: str = "", target_file: str = None,
                 mode: str = None):
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
            mode: "run" or "exec" (default: DOCKER_MODE env var); see DOCKER_MODE.
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
        # Static extent of the code under measurement, as coverage.py reports
        # it (executed + missing): every statement line and every possible arc,
        # plus per-function regions. Lets callers ask what REMAINS unexecuted
        # inside each function, not just whether it was touched.
        self.all_lines = set()          # (file, line)
        self.all_branches = set()       # (file, (from, to))
        self.func_regions = {}          # (file, qualname) -> {"lines": set, "branches": set}
        self._coverage_data_dir = tempfile.mkdtemp(prefix="docker_cov_")
        # Make world-writable so non-root Docker users (e.g. swe-bench) can write
        os.chmod(self._coverage_data_dir, 0o777)
        self._test_count = 0
        self._pass_count = 0
        self._fail_count = 0
        self.mode = mode or DOCKER_MODE
        if self.mode not in ("run", "exec"):
            raise ValueError(f"unknown docker mode {self.mode!r}")
        self._container = None  # exec mode: name of the long-lived container

    def _resource_args(self):
        return ["--label", _OWNER_LABEL, "--network", DOCKER_NETWORK,
                "--memory", DOCKER_MEMORY, "--memory-swap", DOCKER_MEMORY,
                "--cpus", DOCKER_CPUS, "--pids-limit", DOCKER_PIDS]

    def _env_args(self):
        env_args = []
        for k, v in self.env.items():
            env_args.extend(["-e", f"{k}={v}"])
        return env_args

    def _ensure_coverage_installed(self, name):
        """Install coverage once per container, verify it, or fail loudly.

        Doing this per test was both wasteful and fragile: one failed install
        (a network hiccup under load) left every later test in that container
        reporting "No module named coverage", i.e. a whole file silently
        scoring 0. SWE-bench testbeds ship no coverage at all.
        """
        py = self.python_bin
        check = ["docker", "exec", name, "bash", "-c",
                 f"{py} -c 'import coverage' 2>/dev/null && echo PRESENT"]
        r = subprocess.run(check, capture_output=True, text=True, timeout=120)
        if "PRESENT" in r.stdout:
            return
        for attempt in range(3):
            subprocess.run(
                ["docker", "exec", name, "bash", "-c",
                 f"{py} -m pip install -q coverage 2>&1 | tail -2"],
                capture_output=True, text=True, timeout=600)
            r = subprocess.run(check, capture_output=True, text=True, timeout=120)
            if "PRESENT" in r.stdout:
                log.info(f"installed coverage in {self.image} "
                         f"(attempt {attempt + 1})")
                return
            time.sleep(5 * (attempt + 1))
        raise RuntimeError(
            f"{self.image}: could not install coverage after 3 attempts "
            f"(DOCKER_NETWORK={DOCKER_NETWORK}, python={py}) — every test "
            f"would score 0")

    def _ensure_container(self):
        """exec mode: start the long-lived container if it isn't running."""
        if self._container:
            return self._container
        name = f"qcurious-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        cmd = ["docker", "run", "-d", "--init", "--name", name,
               *self._resource_args(),
               "-v", f"{self._coverage_data_dir}:/qc:ro",
               *self._env_args(),
               "--entrypoint", "sleep", self.image, "infinity"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"could not start container: {r.stderr[-300:]}")
        self._container = name
        self._ensure_coverage_installed(name)
        return name

    def _drop_container(self):
        """exec mode: remove the long-lived container (next test gets a new one)."""
        if self._container:
            subprocess.run(["docker", "rm", "-f", self._container],
                           capture_output=True, timeout=60)
            self._container = None

    def run_test(self, test_script: str, timeout: int = 60):
        """Run a test script inside the container and measure coverage.

        Args:
            test_script: Python code to execute as a test
            timeout: Maximum seconds for execution

        Returns:
            DockerTestResult with output, exception, new_branches, coverage info
        """
        if _shutting_down.is_set():
            raise RuntimeError("DockerCoverageRunner is shutting down")
        self._test_count += 1

        # Build full script with setup
        full_script = f"{self.setup_code}\n{test_script}"

        # Write script to temp file
        script_path = os.path.join(self._coverage_data_dir,
                                   f"test_{self._test_count}.py")
        with open(script_path, "w") as f:
            f.write(full_script)

        # Run with coverage, write JSON inside container then cat to stdout
        py = self.python_bin
        pre = f"{self.pre_command} && " if self.pre_command else ""
        # Install coverage only if missing: an unconditional `pip install`
        # contacts the package index every run (~25 s), dwarfing the test itself.
        ensure_coverage = (f"{py} -c 'import coverage' 2>/dev/null || "
                           f"{py} -m pip install coverage -q 2>/dev/null; ")

        # Strategy: run coverage, generate JSON inside container, print a
        # separator then cat the JSON to stdout. We parse it from the output.
        separator = "===COVERAGE_JSON_START==="
        if self.mode == "exec":
            script_in = f"/qc/{os.path.basename(script_path)}"
            # Clear the previous test's coverage data so a failed run can't
            # report stale results.
            reset = "rm -f .coverage /tmp/cov.json; "
        else:
            script_in = "/tmp/test_script.py"
            reset = ""
        body = (f"cd {self.working_dir} && {reset}{pre}{ensure_coverage}"
                f"{py} -m coverage run --rcfile=/dev/null --branch "
                f"--source={self.source_module} "
                f"{script_in} 2>&1; "
                f"echo '{separator}'; "
                f"{py} -m coverage json --rcfile=/dev/null -o /tmp/cov.json 2>/dev/null && "
                f"cat /tmp/cov.json 2>/dev/null")

        if self.mode == "exec":
            try:
                container_name = self._ensure_container()
            except Exception as e:
                self._fail_count += 1
                return DockerTestResult(
                    output=None, exception=str(e)[:200], new_branches=0,
                    cumulative_branches=len(self.cumulative_branches),
                    cumulative_lines=len(self.cumulative_lines),
                )
            cmd = ["docker", "exec", container_name, "bash", "-c", body]
        else:
            container_name = f"qcurious-{os.getpid()}-{uuid.uuid4().hex[:12]}"
            cmd = ["docker", "run", "--rm", "--name", container_name,
                   *self._resource_args(),
                   "--entrypoint", "bash",
                   "-v", f"{script_path}:{script_in}:ro",
                   *self._env_args(),
                   self.image, "-c", body]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            if (self.mode == "exec" and result.returncode != 0
                    and ("is not running" in result.stderr
                         or "No such container" in result.stderr)):
                # container died (e.g. killed externally): one retry on a new one
                self._container = None
                cmd[2] = self._ensure_container()
                container_name = cmd[2]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout
                )
            raw_stdout = result.stdout.strip()
            stderr = result.stderr.strip()
        except subprocess.TimeoutExpired:
            # Killing the docker CLI client does not stop the container (nor,
            # in exec mode, the test process inside it); kill it explicitly or
            # it keeps running and holding memory.
            _kill_container(container_name)
            if self.mode == "exec":
                self._drop_container()
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

        # A missing coverage module makes every test score 0 — loudly abort
        # instead of filling the results with zeros (this cost a full
        # TestGenEval run once, when --network none blocked its pip install).
        if "No module named coverage" in raw_stdout:
            raise RuntimeError(
                f"{self.image}: coverage is not installed and could not be "
                f"installed (DOCKER_NETWORK={DOCKER_NETWORK}); "
                f"use DOCKER_NETWORK=host for images without coverage")

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
                    # Static extent + per-function regions (coverage.py >= 7.5)
                    self.all_lines.update(
                        (file_path, ln) for ln in file_data.get("executed_lines", [])
                        + file_data.get("missing_lines", []))
                    self.all_branches.update(
                        (file_path, tuple(a)) for a in file_data.get("executed_branches", [])
                        + file_data.get("missing_branches", []))
                    for qual, fd in file_data.get("functions", {}).items():
                        if not qual:
                            continue
                        reg = self.func_regions.setdefault(
                            (file_path, qual), {"lines": set(), "branches": set()})
                        reg["lines"].update(fd.get("executed_lines", [])
                                            + fd.get("missing_lines", []))
                        reg["branches"].update(
                            tuple(a) for a in fd.get("executed_branches", [])
                            + fd.get("missing_branches", []))
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

    def remaining_by_function(self, file_filter: str = None):
        """{qualname: (missing_lines, missing_branches)} still unexecuted.

        Uses coverage.py's own per-function regions, so a function counts as
        remaining while any of its statements or arcs has not run — a locked
        early-return does not mark the whole function done. Empty until the
        first test has produced a coverage report.
        """
        out = {}
        for (fp, qual), reg in self.func_regions.items():
            if file_filter and file_filter not in fp:
                continue
            ml = sum(1 for ln in reg["lines"] if (fp, ln) not in self.cumulative_lines)
            mb = sum(1 for a in reg["branches"] if (fp, a) not in self.cumulative_branches)
            prev = out.get(qual, (0, 0))
            out[qual] = (prev[0] + ml, prev[1] + mb)
        return out

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

    def snapshot(self):
        """Capture mutable coverage state so it can be rolled back.

        Used for counterfactual best-of-K (oracle) selection: trial-run each
        candidate plan from the same state, then restore and commit the best.
        """
        return {
            "branches": set(self.cumulative_branches),
            "lines": set(self.cumulative_lines),
            "test_count": self._test_count,
            "pass_count": self._pass_count,
            "fail_count": self._fail_count,
        }

    def restore(self, snap):
        """Restore state previously captured with snapshot()."""
        self.cumulative_branches = set(snap["branches"])
        self.cumulative_lines = set(snap["lines"])
        self._test_count = snap["test_count"]
        self._pass_count = snap["pass_count"]
        self._fail_count = snap["fail_count"]

    def reset(self):
        self.cumulative_branches = set()
        self.cumulative_lines = set()
        self._test_count = 0
        self._pass_count = 0
        self._fail_count = 0

    def cleanup(self):
        """Remove the exec-mode container and the temporary script directory."""
        import shutil
        try:
            self._drop_container()
        except Exception as e:
            log.warning(f"Container removal failed: {e}")
        shutil.rmtree(self._coverage_data_dir, ignore_errors=True)

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass  # interpreter shutdown: the atexit label sweep covers it

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
