"""Non-trivial programs with genuine corridor structure.

Each has:
- Multiple validation gates (corridor) that must be passed
- Deep branching logic (clique) that requires understanding the domain
- Enough branches that 15 steps won't saturate coverage
"""


def validate_and_process_csv_row(row, schema, strict=False):
    """Process a CSV row against a schema with type coercion and validation.

    Args:
        row: dict mapping column names to string values
        schema: dict mapping column names to dicts with keys:
            type: 'int', 'float', 'str', 'bool', 'date'
            required: bool
            min/max: optional numeric bounds
            choices: optional list of valid values
            pattern: optional regex pattern for str type
        strict: if True, reject unknown columns
    """
    import re
    from datetime import datetime

    # Gate 1: type checks
    if not isinstance(row, dict):
        return {"error": "row_not_dict"}
    if not isinstance(schema, dict):
        return {"error": "schema_not_dict"}
    if len(schema) == 0:
        return {"error": "empty_schema"}

    # Gate 2: strict mode - reject unknown columns
    if strict:
        unknown = set(row.keys()) - set(schema.keys())
        if unknown:
            return {"error": "unknown_columns", "columns": sorted(unknown)}

    # Gate 3: required field check
    for col, spec in schema.items():
        if spec.get("required", False) and col not in row:
            return {"error": "missing_required", "column": col}

    # Deep logic: type coercion and validation per column
    result = {}
    warnings = []

    for col, spec in schema.items():
        if col not in row:
            result[col] = None
            continue

        raw = row[col]
        col_type = spec.get("type", "str")

        # Type coercion
        try:
            if col_type == "int":
                if isinstance(raw, str) and raw.strip() == "":
                    if spec.get("required"):
                        return {"error": "empty_required_int", "column": col}
                    result[col] = None
                    continue
                value = int(float(raw))
            elif col_type == "float":
                if isinstance(raw, str) and raw.strip() == "":
                    if spec.get("required"):
                        return {"error": "empty_required_float", "column": col}
                    result[col] = None
                    continue
                value = float(raw)
            elif col_type == "bool":
                if isinstance(raw, str):
                    if raw.lower() in ("true", "1", "yes", "y"):
                        value = True
                    elif raw.lower() in ("false", "0", "no", "n"):
                        value = False
                    else:
                        return {"error": "invalid_bool", "column": col, "value": raw}
                else:
                    value = bool(raw)
            elif col_type == "date":
                if isinstance(raw, str):
                    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y%m%d"):
                        try:
                            value = datetime.strptime(raw, fmt).date().isoformat()
                            break
                        except ValueError:
                            continue
                    else:
                        return {"error": "invalid_date", "column": col, "value": raw}
                else:
                    return {"error": "date_not_string", "column": col}
            else:
                value = str(raw)
        except (ValueError, TypeError) as e:
            return {"error": "coercion_failed", "column": col, "detail": str(e)}

        # Bounds checking
        if col_type in ("int", "float"):
            if "min" in spec and value < spec["min"]:
                if strict:
                    return {"error": "below_min", "column": col, "value": value,
                            "min": spec["min"]}
                warnings.append(f"{col}: {value} below min {spec['min']}")
                value = spec["min"]
            if "max" in spec and value > spec["max"]:
                if strict:
                    return {"error": "above_max", "column": col, "value": value,
                            "max": spec["max"]}
                warnings.append(f"{col}: {value} above max {spec['max']}")
                value = spec["max"]

        # Choices validation
        if "choices" in spec and value not in spec["choices"]:
            if strict:
                return {"error": "invalid_choice", "column": col, "value": value,
                        "choices": spec["choices"]}
            warnings.append(f"{col}: {value} not in choices")

        # Pattern validation for strings
        if col_type == "str" and "pattern" in spec:
            if not re.match(spec["pattern"], str(value)):
                if strict:
                    return {"error": "pattern_mismatch", "column": col,
                            "value": value, "pattern": spec["pattern"]}
                warnings.append(f"{col}: pattern mismatch")

        result[col] = value

    output = {"status": "ok", "data": result}
    if warnings:
        output["warnings"] = warnings
    return output


def route_http_request(method, path, headers, body=None, query_params=None):
    """Route an HTTP request through middleware and handlers.

    Simulates a web framework's request routing with:
    - Method validation
    - Authentication via headers
    - Rate limiting
    - Path matching with parameters
    - Content-type negotiation
    - Request body parsing and validation

    Args:
        method: HTTP method string ('GET', 'POST', etc.)
        path: URL path string (e.g., '/api/v1/users/123')
        headers: dict of HTTP headers
        body: request body (string or dict)
        query_params: dict of query parameters
    """
    import json as _json
    import re

    # Gate 1: method validation
    if not isinstance(method, str):
        return {"status": 400, "error": "method_not_string"}
    method = method.upper()
    if method not in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
        return {"status": 405, "error": "method_not_allowed", "method": method}

    # Gate 2: path validation
    if not isinstance(path, str) or not path.startswith("/"):
        return {"status": 400, "error": "invalid_path"}

    # Gate 3: headers must be dict
    if not isinstance(headers, dict):
        return {"status": 400, "error": "headers_not_dict"}

    # Gate 4: authentication
    auth = headers.get("Authorization", headers.get("authorization", ""))
    if not auth:
        # Check for API key
        api_key = headers.get("X-API-Key", headers.get("x-api-key", ""))
        if not api_key:
            return {"status": 401, "error": "unauthorized", "detail": "no credentials"}
        if len(api_key) < 16:
            return {"status": 401, "error": "invalid_api_key"}
        auth_method = "api_key"
        user_id = hash(api_key) % 10000
    elif auth.startswith("Bearer "):
        token = auth[7:]
        if len(token) < 10:
            return {"status": 401, "error": "invalid_token"}
        auth_method = "bearer"
        user_id = hash(token) % 10000
    elif auth.startswith("Basic "):
        auth_method = "basic"
        user_id = hash(auth) % 10000
    else:
        return {"status": 401, "error": "unsupported_auth_scheme"}

    # Gate 5: rate limiting (simulated)
    if user_id % 100 == 0:
        return {"status": 429, "error": "rate_limited", "retry_after": 60}

    # Deep logic: route matching
    query_params = query_params or {}

    # API routes
    api_match = re.match(r"^/api/v(\d+)/(\w+)(?:/(\d+))?(?:/(\w+))?$", path)
    if api_match:
        version = int(api_match.group(1))
        resource = api_match.group(2)
        resource_id = api_match.group(3)
        action = api_match.group(4)

        if version not in (1, 2):
            return {"status": 400, "error": "unsupported_api_version",
                    "version": version}

        # Resource-specific handling
        if resource == "users":
            if method == "GET":
                if resource_id:
                    return {"status": 200, "data": {"id": int(resource_id),
                            "auth": auth_method}}
                else:
                    limit = int(query_params.get("limit", 10))
                    offset = int(query_params.get("offset", 0))
                    return {"status": 200, "data": [],
                            "pagination": {"limit": limit, "offset": offset}}
            elif method == "POST":
                if resource_id:
                    return {"status": 405, "error": "cannot_post_to_specific_user"}
                # Parse body
                content_type = headers.get("Content-Type",
                                          headers.get("content-type", ""))
                if "json" in content_type:
                    if isinstance(body, str):
                        try:
                            data = _json.loads(body)
                        except _json.JSONDecodeError:
                            return {"status": 400, "error": "invalid_json"}
                    elif isinstance(body, dict):
                        data = body
                    else:
                        return {"status": 400, "error": "invalid_body_type"}

                    if "name" not in data:
                        return {"status": 422, "error": "missing_field",
                                "field": "name"}
                    return {"status": 201, "data": {"id": user_id, **data}}
                else:
                    return {"status": 415, "error": "unsupported_media_type"}
            elif method == "DELETE":
                if not resource_id:
                    return {"status": 405, "error": "cannot_delete_collection"}
                if action == "soft":
                    return {"status": 200, "deleted": "soft",
                            "id": int(resource_id)}
                return {"status": 204}
            elif method == "PUT" or method == "PATCH":
                if not resource_id:
                    return {"status": 405, "error": "cannot_update_collection"}
                return {"status": 200, "updated": int(resource_id),
                        "method": method}

        elif resource == "items":
            if method == "GET":
                sort = query_params.get("sort", "id")
                order = query_params.get("order", "asc")
                if sort not in ("id", "name", "price", "created"):
                    return {"status": 400, "error": "invalid_sort_field"}
                if order not in ("asc", "desc"):
                    return {"status": 400, "error": "invalid_sort_order"}
                return {"status": 200, "data": [], "sort": sort, "order": order}
            elif method == "POST":
                return {"status": 201, "data": {"id": user_id}}

        elif resource == "health":
            return {"status": 200, "data": {"status": "healthy",
                    "version": version}}

        return {"status": 404, "error": "unknown_resource", "resource": resource}

    # Static file routes
    static_match = re.match(r"^/static/(.+)$", path)
    if static_match:
        if method != "GET":
            return {"status": 405, "error": "static_files_get_only"}
        filename = static_match.group(1)
        if ".." in filename:
            return {"status": 403, "error": "path_traversal_blocked"}
        ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
        content_types = {"html": "text/html", "css": "text/css",
                        "js": "application/javascript", "json": "application/json",
                        "png": "image/png", "jpg": "image/jpeg"}
        ct = content_types.get(ext, "application/octet-stream")
        return {"status": 200, "content_type": ct, "file": filename}

    # Root
    if path == "/":
        return {"status": 200, "data": {"message": "welcome"}}

    return {"status": 404, "error": "not_found", "path": path}


def parse_and_evaluate_expression(expr, variables=None, functions=None):
    """Parse and evaluate a mathematical expression with variables and functions.

    Supports: +, -, *, /, //, %, **, parentheses, variables, function calls.

    Args:
        expr: string expression like "2 * x + sin(pi/4)"
        variables: dict mapping names to numeric values
        functions: dict mapping names to callables
    """
    import math
    import re

    # Gate 1: type check
    if not isinstance(expr, str):
        return {"error": "expr_not_string"}
    expr = expr.strip()
    if not expr:
        return {"error": "empty_expression"}

    # Gate 2: balanced parentheses
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth < 0:
            return {"error": "unbalanced_parens", "detail": "extra closing paren"}
    if depth != 0:
        return {"error": "unbalanced_parens", "detail": "unclosed paren"}

    # Gate 3: no dangerous characters
    if any(c in expr for c in [";", "import", "__", "exec", "eval", "open"]):
        return {"error": "forbidden_content"}

    # Setup environment
    variables = dict(variables or {})
    variables.setdefault("pi", math.pi)
    variables.setdefault("e", math.e)

    builtins = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "sqrt": math.sqrt, "abs": abs, "log": math.log,
        "log2": math.log2, "log10": math.log10,
        "floor": math.floor, "ceil": math.ceil,
        "min": min, "max": max,
        "round": round,
    }
    all_funcs = {**builtins, **(functions or {})}

    # Tokenizer
    tokens = []
    i = 0
    while i < len(expr):
        if expr[i].isspace():
            i += 1
        elif expr[i] in "+-*/%().,":
            if expr[i] == "*" and i + 1 < len(expr) and expr[i+1] == "*":
                tokens.append(("OP", "**"))
                i += 2
            elif expr[i] == "/" and i + 1 < len(expr) and expr[i+1] == "/":
                tokens.append(("OP", "//"))
                i += 2
            else:
                tokens.append(("OP", expr[i]))
                i += 1
        elif expr[i].isdigit() or (expr[i] == "." and i + 1 < len(expr) and expr[i+1].isdigit()):
            j = i
            has_dot = expr[i] == "."
            i += 1
            while i < len(expr) and (expr[i].isdigit() or (expr[i] == "." and not has_dot)):
                if expr[i] == ".":
                    has_dot = True
                i += 1
            # Scientific notation
            if i < len(expr) and expr[i] in "eE":
                i += 1
                if i < len(expr) and expr[i] in "+-":
                    i += 1
                while i < len(expr) and expr[i].isdigit():
                    i += 1
            tokens.append(("NUM", expr[j:i]))
        elif expr[i].isalpha() or expr[i] == "_":
            j = i
            while i < len(expr) and (expr[i].isalnum() or expr[i] == "_"):
                i += 1
            name = expr[j:i]
            tokens.append(("NAME", name))
        else:
            return {"error": "unexpected_char", "char": expr[i], "position": i}

    # Recursive descent parser
    pos = [0]

    def peek():
        if pos[0] < len(tokens):
            return tokens[pos[0]]
        return None

    def consume(expected_type=None, expected_val=None):
        t = peek()
        if t is None:
            return None
        if expected_type and t[0] != expected_type:
            return None
        if expected_val and t[1] != expected_val:
            return None
        pos[0] += 1
        return t

    def parse_expr():
        return parse_additive()

    def parse_additive():
        left = parse_multiplicative()
        if isinstance(left, dict) and "error" in left:
            return left
        while True:
            t = peek()
            if t and t[0] == "OP" and t[1] in ("+", "-"):
                consume()
                right = parse_multiplicative()
                if isinstance(right, dict) and "error" in right:
                    return right
                if t[1] == "+":
                    left = left + right
                else:
                    left = left - right
            else:
                break
        return left

    def parse_multiplicative():
        left = parse_power()
        if isinstance(left, dict) and "error" in left:
            return left
        while True:
            t = peek()
            if t and t[0] == "OP" and t[1] in ("*", "/", "//", "%"):
                consume()
                right = parse_power()
                if isinstance(right, dict) and "error" in right:
                    return right
                if t[1] == "*":
                    left = left * right
                elif t[1] == "/":
                    if right == 0:
                        return {"error": "division_by_zero"}
                    left = left / right
                elif t[1] == "//":
                    if right == 0:
                        return {"error": "division_by_zero"}
                    left = left // right
                elif t[1] == "%":
                    if right == 0:
                        return {"error": "modulo_by_zero"}
                    left = left % right
            else:
                break
        return left

    def parse_power():
        base = parse_unary()
        if isinstance(base, dict) and "error" in base:
            return base
        t = peek()
        if t and t[0] == "OP" and t[1] == "**":
            consume()
            exp = parse_power()  # right-associative
            if isinstance(exp, dict) and "error" in exp:
                return exp
            try:
                return base ** exp
            except (OverflowError, ValueError) as e:
                return {"error": "math_error", "detail": str(e)}
        return base

    def parse_unary():
        t = peek()
        if t and t[0] == "OP" and t[1] in ("+", "-"):
            consume()
            val = parse_unary()
            if isinstance(val, dict) and "error" in val:
                return val
            return val if t[1] == "+" else -val
        return parse_primary()

    def parse_primary():
        t = peek()
        if t is None:
            return {"error": "unexpected_end"}

        if t[0] == "NUM":
            consume()
            return float(t[1]) if "." in t[1] or "e" in t[1].lower() else int(t[1])

        if t[0] == "NAME":
            consume()
            name = t[1]
            # Function call?
            if peek() and peek()[0] == "OP" and peek()[1] == "(":
                consume()
                if name not in all_funcs:
                    return {"error": "unknown_function", "name": name}
                args = []
                if not (peek() and peek()[0] == "OP" and peek()[1] == ")"):
                    arg = parse_expr()
                    if isinstance(arg, dict) and "error" in arg:
                        return arg
                    args.append(arg)
                    while peek() and peek()[0] == "OP" and peek()[1] == ",":
                        consume()
                        arg = parse_expr()
                        if isinstance(arg, dict) and "error" in arg:
                            return arg
                        args.append(arg)
                if not consume("OP", ")"):
                    return {"error": "expected_closing_paren"}
                try:
                    return all_funcs[name](*args)
                except Exception as e:
                    return {"error": "function_error", "name": name,
                            "detail": str(e)}
            # Variable
            if name in variables:
                return variables[name]
            return {"error": "unknown_variable", "name": name}

        if t[0] == "OP" and t[1] == "(":
            consume()
            val = parse_expr()
            if isinstance(val, dict) and "error" in val:
                return val
            if not consume("OP", ")"):
                return {"error": "expected_closing_paren"}
            return val

        return {"error": "unexpected_token", "token": t}

    result = parse_expr()
    if isinstance(result, dict) and "error" in result:
        return result

    if pos[0] < len(tokens):
        return {"error": "trailing_tokens",
                "remaining": [t[1] for t in tokens[pos[0]:]]}

    return {"result": result}


def schedule_tasks(tasks, resources, constraints=None):
    """Schedule tasks onto resources with constraints.

    Args:
        tasks: list of dicts with keys:
            id: unique task identifier
            duration: int (time units)
            priority: int (higher = more urgent)
            dependencies: list of task ids that must complete first
            resource_type: string (what kind of resource needed)
        resources: list of dicts with keys:
            id: unique resource identifier
            type: string
            capacity: int (how many tasks simultaneously)
            available_from: int (time unit when available)
        constraints: optional dict with keys:
            max_time: int (deadline)
            no_parallel: list of task id pairs that can't run simultaneously
    """
    # Gate 1: type validation
    if not isinstance(tasks, list):
        return {"error": "tasks_not_list"}
    if not isinstance(resources, list):
        return {"error": "resources_not_list"}
    if len(tasks) == 0:
        return {"error": "no_tasks"}
    if len(resources) == 0:
        return {"error": "no_resources"}

    # Gate 2: task validation
    task_ids = set()
    for t in tasks:
        if not isinstance(t, dict):
            return {"error": "task_not_dict"}
        for key in ("id", "duration", "priority", "resource_type"):
            if key not in t:
                return {"error": "task_missing_field", "task": t.get("id"),
                        "field": key}
        if t["id"] in task_ids:
            return {"error": "duplicate_task_id", "id": t["id"]}
        task_ids.add(t["id"])
        if not isinstance(t["duration"], int) or t["duration"] <= 0:
            return {"error": "invalid_duration", "task": t["id"]}

    # Gate 3: dependency validation
    for t in tasks:
        for dep in t.get("dependencies", []):
            if dep not in task_ids:
                return {"error": "unknown_dependency", "task": t["id"],
                        "dependency": dep}
            if dep == t["id"]:
                return {"error": "self_dependency", "task": t["id"]}

    # Gate 4: cycle detection
    visited = set()
    path = set()

    def has_cycle(tid):
        if tid in path:
            return True
        if tid in visited:
            return False
        visited.add(tid)
        path.add(tid)
        task = next(t for t in tasks if t["id"] == tid)
        for dep in task.get("dependencies", []):
            if has_cycle(dep):
                return True
        path.remove(tid)
        return False

    for t in tasks:
        if has_cycle(t["id"]):
            return {"error": "dependency_cycle", "task": t["id"]}

    # Gate 5: resource validation
    resource_map = {}
    for r in resources:
        if not isinstance(r, dict):
            return {"error": "resource_not_dict"}
        if "id" not in r or "type" not in r:
            return {"error": "resource_missing_field"}
        resource_map[r["id"]] = r

    # Check all resource types are available
    available_types = {r["type"] for r in resources}
    for t in tasks:
        if t["resource_type"] not in available_types:
            return {"error": "no_resource_for_type", "task": t["id"],
                    "type": t["resource_type"]}

    constraints = constraints or {}

    # Deep logic: scheduling algorithm
    # Topological sort by priority within dependency order
    completed = {}  # task_id -> end_time
    schedule = []
    resource_timeline = {r["id"]: r.get("available_from", 0) for r in resources}
    no_parallel = set()
    for pair in constraints.get("no_parallel", []):
        if len(pair) == 2:
            no_parallel.add((pair[0], pair[1]))
            no_parallel.add((pair[1], pair[0]))

    # Sort tasks: dependencies first, then by priority (descending)
    remaining = list(tasks)
    max_time = constraints.get("max_time", float("inf"))

    while remaining:
        # Find ready tasks (all dependencies completed)
        ready = [t for t in remaining
                 if all(d in completed for d in t.get("dependencies", []))]

        if not ready:
            return {"error": "deadlock", "remaining": [t["id"] for t in remaining]}

        # Sort ready tasks by priority
        ready.sort(key=lambda t: -t["priority"])

        # Schedule the highest priority ready task
        task = ready[0]
        remaining.remove(task)

        # Find earliest available resource of the right type
        earliest_start = 0
        # Must start after all dependencies complete
        for dep in task.get("dependencies", []):
            earliest_start = max(earliest_start, completed[dep])

        # Check no-parallel constraints
        for scheduled in schedule:
            if (task["id"], scheduled["task_id"]) in no_parallel:
                earliest_start = max(earliest_start, scheduled["end_time"])

        # Find best resource
        best_resource = None
        best_start = float("inf")

        for r in resources:
            if r["type"] != task["resource_type"]:
                continue
            start = max(earliest_start, resource_timeline[r["id"]])
            if start < best_start:
                best_start = start
                best_resource = r

        if best_resource is None:
            return {"error": "no_available_resource", "task": task["id"]}

        end_time = best_start + task["duration"]

        if end_time > max_time:
            return {"error": "deadline_exceeded", "task": task["id"],
                    "end_time": end_time, "deadline": max_time}

        resource_timeline[best_resource["id"]] = end_time
        completed[task["id"]] = end_time

        schedule.append({
            "task_id": task["id"],
            "resource_id": best_resource["id"],
            "start_time": best_start,
            "end_time": end_time,
            "duration": task["duration"],
        })

    total_time = max(s["end_time"] for s in schedule) if schedule else 0
    utilization = {}
    for r in resources:
        busy = sum(s["duration"] for s in schedule if s["resource_id"] == r["id"])
        utilization[r["id"]] = round(busy / total_time, 2) if total_time > 0 else 0

    return {
        "status": "scheduled",
        "schedule": schedule,
        "total_time": total_time,
        "utilization": utilization,
        "tasks_scheduled": len(schedule),
    }


# Registry for loading
CORRIDOR_PROGRAMS = {
    "csv_processor": {
        "func_name": "validate_and_process_csv_row",
        "source": None,  # loaded from this module
        "expected_branches": 80,
        "corridor_depth": 3,
        "description": "CSV row validation + type coercion + bounds checking",
    },
    "http_router": {
        "func_name": "route_http_request",
        "source": None,
        "expected_branches": 120,
        "corridor_depth": 5,
        "description": "HTTP routing with auth, rate limiting, path matching, body parsing",
    },
    "expr_parser": {
        "func_name": "parse_and_evaluate_expression",
        "source": None,
        "expected_branches": 100,
        "corridor_depth": 3,
        "description": "Math expression parser with tokenizer + recursive descent",
    },
    "task_scheduler": {
        "func_name": "schedule_tasks",
        "source": None,
        "expected_branches": 90,
        "corridor_depth": 5,
        "description": "Task scheduling with dependency resolution + resource allocation",
    },
}


def load_corridor_programs():
    """Load corridor programs with source code from this module."""
    import inspect
    import textwrap

    source_map = {
        "validate_and_process_csv_row": validate_and_process_csv_row,
        "route_http_request": route_http_request,
        "parse_and_evaluate_expression": parse_and_evaluate_expression,
        "schedule_tasks": schedule_tasks,
    }

    programs = {}
    for key, prog in CORRIDOR_PROGRAMS.items():
        func = source_map[prog["func_name"]]
        source = inspect.getsource(func)
        # Dedent to remove module-level indentation
        source = textwrap.dedent(source)
        programs[key] = {
            "func_name": prog["func_name"],
            "source": source,
            "metadata": {
                "cyclomatic_complexity": prog["expected_branches"],
                "corridor_depth": prog["corridor_depth"],
            },
            "description": prog["description"],
        }
    return programs
