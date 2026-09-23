"""Generate hidden-key corridor modules: corridors that require exploration.

The Exp-10 corridor (gen_synth_corridor.py) hardcodes the stage keys in the
source, and the LLM reads the source — so one test script walks the whole
corridor and every coverage-aware method saturates it (~100% at all depths).
It therefore cannot discriminate between values of gamma.

Here the corridor has to be *discovered by running it*:

  * entry (`s0()`) needs no key, and each correct step returns the token the
    NEXT step needs — so a plan written before execution cannot contain the
    tokens for later stages;
  * a process may advance the corridor AT MOST ONCE, so a single script cannot
    chain `t = c.s0(); c.s1(t); ...` to the end;
  * the stage persists in a state file, so progress accumulates across test
    executions the way knowledge accumulates in the test history.

A depth-d corridor therefore needs at least d separate test executions before
`terminal()`'s k branches become reachable: the setup steps carry almost no
immediate gain and pay off only later, which is exactly the regime the
Q-value future term (gamma * v) is supposed to handle.

Requires DOCKER_MODE=exec (the default): in "run" mode every test gets a fresh
container, so the stage file never persists and the corridor is unreachable.
"""

import hashlib


def token(salt: str, i: int) -> str:
    return hashlib.sha256(f"{salt}:{i}".encode()).hexdigest()[:8]


def gen_hidden_corridor_source(depth: int, k: int, m: int, salt: str) -> str:
    """Real Python module for a depth-d hidden-key corridor."""
    toks = [token(salt, i) for i in range(depth + 1)]
    L = [f'"""Hidden-key corridor: depth={depth}, terminal_k={k}, distractors={m}.',
         "",
         "Call s0() to enter. Each step returns the key the next step needs.",
         'A process may advance the corridor only once."""',
         "",
         "import os",
         "",
         f'_STATE = "/tmp/.corridor_d{depth}_stage"',
         "_ADVANCED = False",
         "",
         "",
         "def _stage():",
         "    try:",
         "        with open(_STATE) as f:",
         "            return int(f.read().strip() or 0)",
         "    except (OSError, ValueError):",
         "        return 0",
         "",
         "",
         "def _advance(to):",
         "    global _ADVANCED",
         "    with open(_STATE, 'w') as f:",
         "        f.write(str(to))",
         "    _ADVANCED = True",
         "",
         "",
         "class Corridor:",
         "    def __init__(self):",
         "        self.stage = _stage()",
         ""]
    for i in range(depth):
        args = "self" if i == 0 else "self, key"
        L.append(f"    def s{i}({args}):")
        L.append(f"        if self.stage != {i}:")
        L.append(f"            return 'locked{i}'")
        if i > 0:
            L.append(f"        if key != {toks[i]!r}:")
            L.append(f"            return 'wrong{i}'")
        L.append("        if _ADVANCED:")
        L.append(f"            return 'cooldown{i}'")
        L.append(f"        _advance({i + 1})")
        L.append(f"        self.stage = {i + 1}")
        L.append(f"        return 'ok{i}:' + {toks[i + 1]!r}")
        L.append("")
    L.append("    def terminal(self, x):")
    L.append(f"        if self.stage >= {depth}:")
    for j in range(k):
        L.append(f"            {'if' if j == 0 else 'elif'} x == {j}:")
        L.append(f"                return 't{j}'")
    L += ["            return 'tdefault'", "        return 'locked_terminal'", ""]
    for d in range(m):
        L += ["", f"def dist{d}(x):", "    if x:", f"        return 'd{d}a'",
              f"    return 'd{d}b'"]
    return "\n".join(L) + "\n"


def god_scripts(module: str, depth: int, k: int, m: int, salt: str):
    """Scripts that together cover every branch (one advance per script).

    Returns a list of test scripts to run in order through ONE runner, which
    is how the total-branch ceiling is measured for these modules.
    """
    toks = [token(salt, i) for i in range(depth + 1)]
    out = []
    first = [f"import {module} as M"]
    for i in range(m):
        first += [f"M.dist{i}(1)", f"M.dist{i}(0)"]
    first += ["c = M.Corridor()", "c.terminal(0)"]          # locked_terminal
    for i in range(1, depth):                               # locked{i}
        first.append(f"M.Corridor().s{i}('x')")
    for i in range(1, depth):                               # wrong{i}
        first.append(f"M.Corridor().s{i}({toks[i]!r} + 'x')")
    first += ["print(M.Corridor().s0())"]                   # ok0 (+ advance)
    first += ["print(M.Corridor().s0())"]                   # locked0 next time
    out.append("\n".join(first))
    for i in range(1, depth):                               # one advance each
        s = [f"import {module} as M", "c = M.Corridor()",
             f"print(c.s{i}({toks[i]!r}))"]
        if i + 1 < depth:                                   # cooldown{i+1}
            s.append(f"print(M.Corridor().s{i + 1}({toks[i + 1]!r}))")
        out.append("\n".join(s))
    last = [f"import {module} as M", "c = M.Corridor()"]
    last += [f"c.terminal({j})" for j in range(k)]
    last += ["c.terminal(99999)", "print('god ok', c.stage)"]
    out.append("\n".join(last))
    return out


if __name__ == "__main__":
    # Self-check without Docker: each script runs in its own process, exactly
    # like separate test executions against one container.
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    depth, k, m, salt = 4, 6, 2, "selfcheck"
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "hc.py").write_text(gen_hidden_corridor_source(depth, k, m, salt))

        def run(code):
            return subprocess.run([sys.executable, "-c", code], cwd=tmp,
                                  capture_output=True, text=True).stdout.strip()

        # 1. a single script cannot chain to the end: the 2nd advance cools down
        chain = ("import hc\nc = hc.Corridor()\nr = c.s0()\n"
                 "print(r)\nprint(hc.Corridor().s1(r.split(':')[1]))\n"
                 "print(hc.Corridor().terminal(0))")
        out = run(chain).splitlines()
        assert out[0].startswith("ok0:"), out
        assert out[1] == "cooldown1", out
        assert out[2] == "locked_terminal", out

        # 2. separate processes do walk it, one stage at a time
        tok = out[0].split(":")[1]
        for i in range(1, depth):
            r = run(f"import hc\nprint(hc.Corridor().s{i}({tok!r}))")
            assert r.startswith(f"ok{i}:"), (i, r)
            tok = r.split(":")[1]
        assert run("import hc\nprint(hc.Corridor().terminal(3))") == "t3"
        print(f"self-check OK: depth {depth} needs {depth} executions; "
              "one script cannot shortcut it")
