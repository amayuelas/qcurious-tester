"""Generate synthetic corridor modules (real Python) for Exp 10's LLM condition.

Each module has a `Corridor` class whose DEEP branches (a k-way terminal) are
gated behind a depth-d setup sequence: you must call s0('k0'), s1('k1'), …,
s{d-1}('k{d-1}') in order to raise `stage` to d, and only then does `terminal(x)`
expose its k branches. Plus m standalone `dist*` distractor functions with easy,
independent branches.

A single test (random/greedy) can hit distractors + shallow/locked branches but
NOT the terminal's k branches — those require a multi-step plan that walks the
corridor first. So branch coverage of these modules directly measures whether a
strategy navigates corridors.
"""


def gen_corridor_source(depth: int, k: int, m: int) -> str:
    L = [f'"""Synthetic corridor: depth={depth}, terminal_k={k}, distractors={m}."""',
         "", "", "class Corridor:", "    def __init__(self):",
         "        self.stage = 0", ""]
    for i in range(depth):
        L.append(f"    def s{i}(self, key):")
        if i == 0:
            L += [f"        if key == 'k{i}':",
                  f"            self.stage = {i + 1}",
                  f"            return 'ok{i}'",
                  f"        return 'no{i}'", ""]
        else:
            L += [f"        if self.stage >= {i}:",
                  f"            if key == 'k{i}':",
                  f"                self.stage = {i + 1}",
                  f"                return 'ok{i}'",
                  f"            return 'wrong{i}'",
                  f"        return 'locked{i}'", ""]
    L.append("    def terminal(self, x):")
    L.append(f"        if self.stage >= {depth}:")
    for j in range(k):
        L.append(f"            {'if' if j == 0 else 'elif'} x == {j}:")
        L.append(f"                return 't{j}'")
    L += ["            return 'tdefault'", "        return 'locked_terminal'", ""]
    for d in range(m):
        L += [f"", f"def dist{d}(x):", f"    if x:", f"        return 'd{d}a'",
              f"    return 'd{d}b'"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    # quick local self-check: source is valid and the corridor gates correctly
    src = gen_corridor_source(depth=3, k=5, m=4)
    print(src)
    ns = {}
    exec(compile(src, "corridor_d3.py", "exec"), ns)
    c = ns["Corridor"]()
    # without setup, terminal is locked
    assert c.terminal(0) == "locked_terminal"
    # walk the corridor
    assert c.s0("k0") == "ok0"
    assert c.s1("k1") == "ok1"
    assert c.s2("k2") == "ok2"
    assert c.terminal(0) == "t0"
    assert c.terminal(3) == "t3"
    assert ns["dist0"](1) == "d0a" and ns["dist0"](0) == "d0b"
    print("\nself-check OK: terminal gated behind 3-step setup; distractors independent")
