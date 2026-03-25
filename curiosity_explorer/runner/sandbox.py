"""Safe execution sandbox with resource limits."""

import os
import resource
import subprocess
import sys

DEFAULT_TIMEOUT = 10  # seconds
DEFAULT_MEMORY_MB = 256


def _set_limits(memory_mb):
    """Create a preexec_fn that sets memory limits."""
    def _inner():
        mem_bytes = memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        except ValueError:
            pass  # some systems don't support this
    return _inner


def execute_in_sandbox(script_path: str, timeout: int = DEFAULT_TIMEOUT,
                       memory_mb: int = DEFAULT_MEMORY_MB,
                       cwd: str = None) -> dict:
    """Execute a Python script in a subprocess with resource limits.

    Returns dict with keys: stdout, stderr, returncode, timed_out
    """
    try:
        proc = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True,
            timeout=timeout,
            cwd=cwd,
            preexec_fn=_set_limits(memory_mb),
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "TimeoutError",
            "returncode": -1,
            "timed_out": True,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "timed_out": False,
        }
