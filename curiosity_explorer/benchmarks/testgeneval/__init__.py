"""TestGenEval benchmark loader stub (Phase 2).

TestGenEval requires Docker and SWE-Bench-style containerized execution.
This stub will be implemented when Phase 2 begins.

Upstream repos:
  - https://github.com/facebookresearch/testgeneval
  - HuggingFace: kjain14/testgenevallite (lightweight variant)

Setup notes:
  - Requires Docker for isolated execution (SWE-Bench harness)
  - Each task includes a full repo context, not just a single function
  - Evaluation uses pytest pass/fail + coverage delta
"""


def load_testgeneval_functions(**kwargs):
    """Load TestGenEval functions. Not yet implemented (Phase 2)."""
    raise NotImplementedError(
        "TestGenEval loader requires Docker and SWE-Bench setup. "
        "See https://github.com/facebookresearch/testgeneval for details. "
        "Will be implemented in Phase 2."
    )
