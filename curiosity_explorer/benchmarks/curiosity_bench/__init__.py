"""CuriosityBench: Benchmark for evaluating curiosity-guided code exploration.

50 functions extracted from popular Python libraries, each forked with
modified internal logic. The LLM has trained knowledge of the original
code but the modifications make its predictions partially wrong —
forcing genuine learning through interaction.

Design:
- Functions from 22 well-known repos (requests, click, email, ast, etc.)
- Each function is standalone (no self references, minimal deps)
- Each has 2-3 internal modifications (changed thresholds, swapped conditions)
- Run in black-box mode: LLM sees only function signature + history
- Coverage measured by CoverageRunner (branch-level)

Usage:
    from curiosity_explorer.benchmarks.curiosity_bench import load_benchmark

    programs = load_benchmark()          # all 50
    programs = load_benchmark(n=10)      # first 10
    programs = load_benchmark(repo="click")  # just click functions

    for key, prog in programs.items():
        # prog["func_name"], prog["source"], prog["metadata"]
        runner = CoverageRunner(prog["func_name"], prog["source"])
        result = runner.run_test(f'{prog["func_name"]}(...)')
"""

from .registry import load_benchmark, FUNCTION_REGISTRY

__all__ = ["load_benchmark", "FUNCTION_REGISTRY"]
