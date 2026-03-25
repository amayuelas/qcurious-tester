"""Parse and format execution traces for LLM context."""


def format_test_history(test_history, max_recent=10):
    """Format recent test history for LLM prompt context."""
    if not test_history:
        return ""
    recent = test_history[-max_recent:]
    lines = ["Previous tests and results:"]
    for i, (test_code, result) in enumerate(recent):
        lines.append(f"  Test {i+1}: {test_code}")
        lines.append(f"    → Output: {result.output}, Exception: {result.exception}")
        lines.append(f"    → New branches discovered: {result.new_branches}")
    return "\n".join(lines)


def format_coverage_summary(runner):
    """Format current coverage state for LLM context."""
    branches = runner.get_cumulative_branches()
    return (f"Cumulative branches covered: {len(branches)}\n"
            f"Arcs: {sorted(branches)}")


def extract_function_signature(source_code):
    """Extract function signature and docstring from source."""
    lines = source_code.strip().split("\n")
    sig_lines = []
    for line in lines:
        sig_lines.append(line)
        if ('"""' in line or "'''" in line) and len(sig_lines) > 1:
            break
    return "\n".join(sig_lines)
