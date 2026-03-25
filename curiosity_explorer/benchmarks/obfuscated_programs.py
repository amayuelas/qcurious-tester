"""Obfuscated corridor programs.

Takes corridor programs and renames everything so the LLM can't use
prior knowledge. Forces genuine exploration through interaction.

The LLM sees:
  - Meaningless function/variable names
  - No docstrings or comments
  - Opaque validation logic
  - Must discover valid input format through trial and error
"""


def func_alpha(x):
    if not isinstance(x, dict):
        return {"e": 1}
    if "k1" not in x:
        return {"e": 2}
    if "k2" not in x:
        return {"e": 3}
    if not isinstance(x["k1"], list) or len(x["k1"]) == 0:
        return {"e": 4}
    if not isinstance(x["k2"], int):
        return {"e": 5}
    if x["k2"] < 0:
        return {"e": 6}

    # Deep logic: behavior depends on runtime values
    t = 0
    for item in x["k1"]:
        if not isinstance(item, dict):
            return {"e": 7}
        a = item.get("a", 0)
        b = item.get("b", 1)
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return {"e": 8}
        t += a * b

    if t > 500:
        if x["k2"] > 50:
            if len(x["k1"]) > 3:
                return {"r": "X1", "v": t, "n": len(x["k1"])}
            return {"r": "X2", "v": t}
        else:
            if "k3" in x and x["k3"] is True:
                return {"r": "X3", "v": t, "f": True}
            return {"r": "X4", "v": t}
    elif t > 100:
        if x["k2"] % 2 == 0:
            return {"r": "Y1", "v": t}
        else:
            return {"r": "Y2", "v": t}
    elif t > 0:
        return {"r": "Z1", "v": t}
    elif t == 0:
        return {"r": "Z2"}
    else:
        return {"r": "Z3", "v": t}


def func_beta(m, p, h, b=None, q=None):
    if not isinstance(m, str):
        return {"e": 10}
    m = m.upper()
    if m not in ("A", "B", "C", "D", "E"):
        return {"e": 11, "m": m}

    if not isinstance(p, str) or not p.startswith("/"):
        return {"e": 12}
    if not isinstance(h, dict):
        return {"e": 13}

    # Auth gate
    t = h.get("t", "")
    if not t:
        k = h.get("k", "")
        if not k or len(k) < 8:
            return {"e": 14}
        u = hash(k) % 1000
        at = "k"
    elif t.startswith("V:"):
        v = t[2:]
        if len(v) < 5:
            return {"e": 15}
        u = hash(v) % 1000
        at = "v"
    else:
        return {"e": 16}

    # Rate limit
    if u % 100 == 0:
        return {"e": 17, "w": 30}

    q = q or {}

    # Route matching
    import re
    r1 = re.match(r"^/s/(\d+)/(\w+)(?:/(\d+))?$", p)
    if r1:
        s = int(r1.group(1))
        c = r1.group(2)
        i = r1.group(3)

        if s not in (1, 2, 3):
            return {"e": 18, "s": s}

        if c == "items":
            if m == "A":
                if i:
                    return {"s": 200, "d": {"id": int(i), "at": at}}
                else:
                    lim = int(q.get("l", 10))
                    return {"s": 200, "d": [], "pg": {"l": lim}}
            elif m == "B":
                if i:
                    return {"e": 19}
                ct = h.get("ct", "")
                if "json" in ct:
                    if isinstance(b, dict):
                        if "n" not in b:
                            return {"e": 20}
                        return {"s": 201, "d": {"id": u, **b}}
                    return {"e": 21}
                return {"e": 22}
            elif m == "D":
                if not i:
                    return {"e": 23}
                return {"s": 204}
            elif m == "C":
                if not i:
                    return {"e": 24}
                return {"s": 200, "u": int(i), "m": m}

        elif c == "data":
            if m == "A":
                f = q.get("f", "id")
                o = q.get("o", "asc")
                if f not in ("id", "n", "v", "ts"):
                    return {"e": 25}
                if o not in ("asc", "desc"):
                    return {"e": 26}
                return {"s": 200, "d": [], "f": f, "o": o}

        elif c == "status":
            return {"s": 200, "d": {"ok": True, "ver": s}}

        return {"e": 27, "c": c}

    # Static
    r2 = re.match(r"^/f/(.+)$", p)
    if r2:
        if m != "A":
            return {"e": 28}
        fn = r2.group(1)
        if ".." in fn:
            return {"e": 29}
        ext = fn.rsplit(".", 1)[-1] if "." in fn else ""
        ct = {"h": "t/h", "c": "t/c", "j": "a/j", "p": "i/p"}.get(ext, "a/o")
        return {"s": 200, "ct": ct, "fn": fn}

    if p == "/":
        return {"s": 200, "d": {"msg": "ok"}}

    return {"e": 30, "p": p}


def func_gamma(expr, v=None, fn=None):
    if not isinstance(expr, str):
        return {"e": 40}
    expr = expr.strip()
    if not expr:
        return {"e": 41}

    d = 0
    for ch in expr:
        if ch == "(": d += 1
        elif ch == ")": d -= 1
        if d < 0:
            return {"e": 42}
    if d != 0:
        return {"e": 43}

    if any(w in expr for w in [";", "__", "import"]):
        return {"e": 44}

    import math
    v = dict(v or {})
    v.setdefault("p", math.pi)
    v.setdefault("e", math.e)

    ops = {
        "sin": math.sin, "cos": math.cos, "sqrt": math.sqrt,
        "abs": abs, "log": math.log, "round": round,
    }
    ops.update(fn or {})

    # Tokenize
    toks = []
    i = 0
    while i < len(expr):
        if expr[i].isspace():
            i += 1
        elif expr[i] in "+-*/%().,":
            if expr[i] == "*" and i+1 < len(expr) and expr[i+1] == "*":
                toks.append(("O", "**")); i += 2
            elif expr[i] == "/" and i+1 < len(expr) and expr[i+1] == "/":
                toks.append(("O", "//")); i += 2
            else:
                toks.append(("O", expr[i])); i += 1
        elif expr[i].isdigit() or (expr[i] == "." and i+1 < len(expr) and expr[i+1].isdigit()):
            j = i
            while i < len(expr) and (expr[i].isdigit() or expr[i] == "."):
                i += 1
            toks.append(("N", expr[j:i]))
        elif expr[i].isalpha() or expr[i] == "_":
            j = i
            while i < len(expr) and (expr[i].isalnum() or expr[i] == "_"):
                i += 1
            toks.append(("W", expr[j:i]))
        else:
            return {"e": 45, "ch": expr[i]}

    # Parse (recursive descent)
    pos = [0]
    def pk():
        return toks[pos[0]] if pos[0] < len(toks) else None
    def nx(tt=None, tv=None):
        t = pk()
        if t and (tt is None or t[0] == tt) and (tv is None or t[1] == tv):
            pos[0] += 1; return t
        return None

    def p_add():
        l = p_mul()
        if isinstance(l, dict): return l
        while True:
            t = pk()
            if t and t[0] == "O" and t[1] in "+-":
                nx(); r = p_mul()
                if isinstance(r, dict): return r
                l = l + r if t[1] == "+" else l - r
            else: break
        return l

    def p_mul():
        l = p_pow()
        if isinstance(l, dict): return l
        while True:
            t = pk()
            if t and t[0] == "O" and t[1] in ("*", "/", "//", "%"):
                nx(); r = p_pow()
                if isinstance(r, dict): return r
                if t[1] == "*": l = l * r
                elif t[1] == "/":
                    if r == 0: return {"e": 46}
                    l = l / r
                elif t[1] == "//":
                    if r == 0: return {"e": 46}
                    l = l // r
                elif t[1] == "%":
                    if r == 0: return {"e": 47}
                    l = l % r
            else: break
        return l

    def p_pow():
        b = p_una()
        if isinstance(b, dict): return b
        if pk() and pk()[0] == "O" and pk()[1] == "**":
            nx(); exp = p_pow()
            if isinstance(exp, dict): return exp
            try: return b ** exp
            except: return {"e": 48}
        return b

    def p_una():
        t = pk()
        if t and t[0] == "O" and t[1] in "+-":
            nx(); val = p_una()
            if isinstance(val, dict): return val
            return val if t[1] == "+" else -val
        return p_pri()

    def p_pri():
        t = pk()
        if t is None: return {"e": 49}
        if t[0] == "N":
            nx(); return float(t[1]) if "." in t[1] else int(t[1])
        if t[0] == "W":
            nx(); nm = t[1]
            if pk() and pk()[1] == "(":
                nx()
                if nm not in ops: return {"e": 50, "fn": nm}
                args = []
                if not (pk() and pk()[1] == ")"):
                    a = p_add()
                    if isinstance(a, dict): return a
                    args.append(a)
                    while pk() and pk()[1] == ",":
                        nx(); a = p_add()
                        if isinstance(a, dict): return a
                        args.append(a)
                if not nx("O", ")"): return {"e": 51}
                try: return ops[nm](*args)
                except Exception as ex: return {"e": 52, "d": str(ex)}
            if nm in v: return v[nm]
            return {"e": 53, "nm": nm}
        if t[0] == "O" and t[1] == "(":
            nx(); val = p_add()
            if isinstance(val, dict): return val
            if not nx("O", ")"): return {"e": 54}
            return val
        return {"e": 55, "t": t}

    result = p_add()
    if isinstance(result, dict): return result
    if pos[0] < len(toks):
        return {"e": 56, "rem": [t[1] for t in toks[pos[0]:]]}
    return {"r": result}


def func_delta(tasks, res, con=None):
    if not isinstance(tasks, list):
        return {"e": 60}
    if not isinstance(res, list):
        return {"e": 61}
    if len(tasks) == 0:
        return {"e": 62}
    if len(res) == 0:
        return {"e": 63}

    tids = set()
    for t in tasks:
        if not isinstance(t, dict):
            return {"e": 64}
        for f in ("id", "d", "p", "rt"):
            if f not in t:
                return {"e": 65, "t": t.get("id"), "f": f}
        if t["id"] in tids:
            return {"e": 66, "id": t["id"]}
        tids.add(t["id"])
        if not isinstance(t["d"], int) or t["d"] <= 0:
            return {"e": 67, "t": t["id"]}

    for t in tasks:
        for dep in t.get("deps", []):
            if dep not in tids:
                return {"e": 68, "t": t["id"], "dep": dep}
            if dep == t["id"]:
                return {"e": 69, "t": t["id"]}

    vis = set(); path = set()
    def cyc(tid):
        if tid in path: return True
        if tid in vis: return False
        vis.add(tid); path.add(tid)
        tk = next(t for t in tasks if t["id"] == tid)
        for dep in tk.get("deps", []):
            if cyc(dep): return True
        path.remove(tid); return False
    for t in tasks:
        if cyc(t["id"]): return {"e": 70, "t": t["id"]}

    rm = {}
    for r in res:
        if not isinstance(r, dict) or "id" not in r or "tp" not in r:
            return {"e": 71}
        rm[r["id"]] = r

    atypes = {r["tp"] for r in res}
    for t in tasks:
        if t["rt"] not in atypes:
            return {"e": 72, "t": t["id"], "rt": t["rt"]}

    con = con or {}
    done = {}; sched = []
    rtl = {r["id"]: r.get("af", 0) for r in res}
    np = set()
    for pair in con.get("np", []):
        if len(pair) == 2:
            np.add((pair[0], pair[1])); np.add((pair[1], pair[0]))

    rem = list(tasks)
    mx = con.get("mx", float("inf"))

    while rem:
        ready = [t for t in rem if all(d in done for d in t.get("deps", []))]
        if not ready:
            return {"e": 73, "rem": [t["id"] for t in rem]}
        ready.sort(key=lambda t: -t["p"])
        task = ready[0]; rem.remove(task)

        es = 0
        for dep in task.get("deps", []):
            es = max(es, done[dep])
        for s in sched:
            if (task["id"], s["tid"]) in np:
                es = max(es, s["et"])

        br = None; bs = float("inf")
        for r in res:
            if r["tp"] != task["rt"]: continue
            st = max(es, rtl[r["id"]])
            if st < bs: bs = st; br = r

        if br is None:
            return {"e": 74, "t": task["id"]}

        et = bs + task["d"]
        if et > mx:
            return {"e": 75, "t": task["id"], "et": et, "mx": mx}

        rtl[br["id"]] = et; done[task["id"]] = et
        sched.append({"tid": task["id"], "rid": br["id"], "st": bs, "et": et, "d": task["d"]})

    tt = max(s["et"] for s in sched) if sched else 0
    util = {}
    for r in res:
        busy = sum(s["d"] for s in sched if s["rid"] == r["id"])
        util[r["id"]] = round(busy / tt, 2) if tt > 0 else 0

    return {"s": "ok", "sched": sched, "tt": tt, "util": util, "n": len(sched)}


OBFUSCATED_PROGRAMS = {
    "alpha": {
        "func_name": "func_alpha",
        "description": "Unknown function with dict input, list field, nested processing",
        "expected_branches": 40,
        "corridor_depth": 6,
    },
    "beta": {
        "func_name": "func_beta",
        "description": "Unknown function with string+dict inputs, auth, routing",
        "expected_branches": 80,
        "corridor_depth": 5,
    },
    "gamma": {
        "func_name": "func_gamma",
        "description": "Unknown function with string input, parsing, evaluation",
        "expected_branches": 70,
        "corridor_depth": 3,
    },
    "delta": {
        "func_name": "func_delta",
        "description": "Unknown function with list+list inputs, validation, scheduling",
        "expected_branches": 60,
        "corridor_depth": 5,
    },
}


def load_obfuscated_programs():
    """Load obfuscated programs with source code."""
    import inspect
    programs = {}
    source_map = {
        "func_alpha": func_alpha,
        "func_beta": func_beta,
        "func_gamma": func_gamma,
        "func_delta": func_delta,
    }
    for key, prog in OBFUSCATED_PROGRAMS.items():
        func = source_map[prog["func_name"]]
        source = inspect.getsource(func)
        import textwrap
        source = textwrap.dedent(source)
        programs[key] = {
            "func_name": prog["func_name"],
            "source": source,
            "metadata": {
                "cyclomatic_complexity": prog["expected_branches"],
                "corridor_depth": prog["corridor_depth"],
            },
        }
    return programs
