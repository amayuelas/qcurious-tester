"""Benchmark loaders.

Dispatches to per-benchmark modules and provides a common data contract
(BenchmarkFunction) so run_experiment.py doesn't need to know internals.
"""

from dataclasses import dataclass, field


@dataclass
class BenchmarkFunction:
    """Common representation of a benchmark function for the experiment pipeline."""

    func_name: str
    source: str
    task_id: str = ""
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def to_program_dict(self) -> dict:
        """Convert to the dict format expected by run_experiment.run_experiment()."""
        return {
            "func_name": self.func_name,
            "source": self.source,
            "description": self.description or self.task_id,
        }


def load_benchmark(name: str, **kwargs) -> dict:
    """Load a benchmark by name, returning a dict of {key: program_dict}.

    Args:
        name: One of "toy", "cruxeval", "ult", "testgeneval".
        **kwargs: Passed through to the underlying loader.

    Returns:
        dict mapping program keys to dicts with "func_name", "source", "description".
    """
    if name == "toy":
        from .toy_programs import TOY_PROGRAMS
        return TOY_PROGRAMS

    elif name == "cruxeval":
        from .cruxeval import load_cruxeval_functions
        return load_cruxeval_functions(**kwargs)

    elif name == "ult":
        from .ult import load_ult_functions
        return load_ult_functions(**kwargs)

    elif name == "testgeneval":
        from .testgeneval import load_testgeneval_functions
        return load_testgeneval_functions(**kwargs)

    else:
        raise ValueError(f"Unknown benchmark: {name!r}. "
                         f"Choose from: toy, cruxeval, ult, testgeneval")
