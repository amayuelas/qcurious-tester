"""Generate corridor-structured functions with controlled parameters.

Each generated function has:
- A configurable number of validation gates (corridor depth)
- Deep branching logic after the gates
- Known expected branch count
- Enough complexity to not saturate in 15 steps

Parameters:
- n_gates: number of sequential validation checks
- n_deep_branches: number of branches in the deep logic
- input_complexity: how complex the valid input needs to be
"""

import random
import textwrap


def generate_corridor_function(seed=0, n_gates=3, n_deep_branches=8,
                               domain="data"):
    """Generate a corridor-structured function.

    Returns dict with func_name, source, metadata.
    """
    rng = random.Random(seed)

    # Pick a domain template
    generators = {
        "data": _gen_data_processor,
        "config": _gen_config_validator,
        "event": _gen_event_handler,
        "transform": _gen_data_transformer,
        "query": _gen_query_processor,
    }

    gen_fn = generators.get(domain, _gen_data_processor)
    return gen_fn(rng, n_gates, n_deep_branches, seed)


def _gen_data_processor(rng, n_gates, n_deep, seed):
    """Generate a data processing function with validation gates."""
    func_name = f"process_data_{seed}"

    # Build gate checks
    field_types = ["str", "int", "float", "bool", "list"]
    required_fields = []
    gate_lines = []

    # Gate: type check
    gate_lines.append("    if not isinstance(data, dict):")
    gate_lines.append('        return {"error": "not_dict"}')

    for i in range(n_gates - 1):
        field = f"field_{chr(97 + i)}"
        ftype = rng.choice(field_types)
        required_fields.append((field, ftype))
        gate_lines.append(f'    if "{field}" not in data:')
        gate_lines.append(f'        return {{"error": "missing_{field}"}}')

        if ftype == "int":
            gate_lines.append(f'    if not isinstance(data["{field}"], int):')
            gate_lines.append(f'        return {{"error": "{field}_not_int"}}')
            gate_lines.append(f'    if data["{field}"] < 0:')
            gate_lines.append(f'        return {{"error": "{field}_negative"}}')
        elif ftype == "str":
            gate_lines.append(f'    if not isinstance(data["{field}"], str):')
            gate_lines.append(f'        return {{"error": "{field}_not_str"}}')
            gate_lines.append(f'    if len(data["{field}"]) == 0:')
            gate_lines.append(f'        return {{"error": "{field}_empty"}}')
        elif ftype == "float":
            gate_lines.append(f'    if not isinstance(data["{field}"], (int, float)):')
            gate_lines.append(f'        return {{"error": "{field}_not_numeric"}}')
        elif ftype == "list":
            gate_lines.append(f'    if not isinstance(data["{field}"], list):')
            gate_lines.append(f'        return {{"error": "{field}_not_list"}}')
            gate_lines.append(f'    if len(data["{field}"]) == 0:')
            gate_lines.append(f'        return {{"error": "{field}_empty_list"}}')
        elif ftype == "bool":
            gate_lines.append(f'    if not isinstance(data["{field}"], bool):')
            gate_lines.append(f'        return {{"error": "{field}_not_bool"}}')

    # Build deep logic branches
    deep_lines = []
    deep_lines.append("    # Deep processing logic")
    deep_lines.append("    result = {}")

    # Use the validated fields to create branching logic
    for i in range(n_deep):
        if not required_fields:
            # Fallback: branch on arbitrary conditions
            threshold = rng.randint(1, 100)
            deep_lines.append(f"    if hash(str(data)) % 100 < {threshold}:")
            deep_lines.append(f'        result["branch_{i}"] = "path_a_{i}"')
            deep_lines.append(f"    else:")
            deep_lines.append(f'        result["branch_{i}"] = "path_b_{i}"')
        else:
            field, ftype = rng.choice(required_fields)
            if ftype == "int":
                threshold = rng.randint(1, 50)
                deep_lines.append(f'    if data["{field}"] > {threshold}:')
                if rng.random() < 0.5:
                    t2 = rng.randint(threshold + 1, 100)
                    deep_lines.append(f'        if data["{field}"] > {t2}:')
                    deep_lines.append(f'            result["cat_{i}"] = "very_high"')
                    deep_lines.append(f"        else:")
                    deep_lines.append(f'            result["cat_{i}"] = "high"')
                else:
                    deep_lines.append(f'        result["cat_{i}"] = "above_{threshold}"')
                deep_lines.append(f"    else:")
                deep_lines.append(f'        result["cat_{i}"] = "below_{threshold}"')
            elif ftype == "str":
                patterns = ["alpha", "digit", "upper", "lower", "mixed"]
                pat = rng.choice(patterns)
                deep_lines.append(f'    if data["{field}"].is{pat}():')
                deep_lines.append(f'        result["fmt_{i}"] = "{pat}"')
                deep_lines.append(f"    else:")
                if rng.random() < 0.4:
                    deep_lines.append(f'        if len(data["{field}"]) > 10:')
                    deep_lines.append(f'            result["fmt_{i}"] = "long_other"')
                    deep_lines.append(f"        else:")
                    deep_lines.append(f'            result["fmt_{i}"] = "short_other"')
                else:
                    deep_lines.append(f'        result["fmt_{i}"] = "other"')
            elif ftype == "float":
                threshold = rng.uniform(0, 100)
                deep_lines.append(f'    if data["{field}"] > {threshold:.1f}:')
                deep_lines.append(f'        result["range_{i}"] = "high"')
                deep_lines.append(f"    elif data[\"{field}\"] > {threshold/2:.1f}:")
                deep_lines.append(f'        result["range_{i}"] = "medium"')
                deep_lines.append(f"    else:")
                deep_lines.append(f'        result["range_{i}"] = "low"')
            elif ftype == "list":
                length = rng.randint(1, 5)
                deep_lines.append(f'    if len(data["{field}"]) > {length}:')
                deep_lines.append(f'        result["size_{i}"] = "large"')
                if rng.random() < 0.5:
                    deep_lines.append(f'        if all(isinstance(x, int) for x in data["{field}"]):')
                    deep_lines.append(f'            result["type_{i}"] = "int_list"')
                    deep_lines.append(f"        else:")
                    deep_lines.append(f'            result["type_{i}"] = "mixed_list"')
                deep_lines.append(f"    else:")
                deep_lines.append(f'        result["size_{i}"] = "small"')
            elif ftype == "bool":
                deep_lines.append(f'    if data["{field}"]:')
                deep_lines.append(f'        result["flag_{i}"] = "enabled"')
                deep_lines.append(f"    else:")
                deep_lines.append(f'        result["flag_{i}"] = "disabled"')

    deep_lines.append('    result["status"] = "processed"')
    deep_lines.append("    return result")

    # Assemble
    fields_doc = ", ".join(f"{f}: {t}" for f, t in required_fields)
    source = f'def {func_name}(data):\n'
    source += f'    """Process data with {n_gates} validation gates and {n_deep} deep branches.\n'
    source += f'    Required fields: {fields_doc}\n'
    source += f'    """\n'
    source += "\n".join(gate_lines) + "\n"
    source += "\n".join(deep_lines) + "\n"

    return {
        "func_name": func_name,
        "source": source,
        "metadata": {
            "n_gates": n_gates,
            "n_deep_branches": n_deep,
            "domain": "data",
            "seed": seed,
        },
    }


def _gen_config_validator(rng, n_gates, n_deep, seed):
    """Generate a config validation function."""
    func_name = f"validate_config_{seed}"

    sections = ["database", "cache", "logging", "auth", "storage",
                "network", "scheduler", "metrics", "alerts", "plugins"]
    rng.shuffle(sections)
    used_sections = sections[:max(2, n_gates - 1)]

    gate_lines = []
    gate_lines.append("    if not isinstance(config, dict):")
    gate_lines.append('        return {"error": "config_not_dict"}')

    for sec in used_sections:
        gate_lines.append(f'    if "{sec}" not in config:')
        gate_lines.append(f'        return {{"error": "missing_section_{sec}"}}')
        gate_lines.append(f'    if not isinstance(config["{sec}"], dict):')
        gate_lines.append(f'        return {{"error": "{sec}_not_dict"}}')

    deep_lines = []
    deep_lines.append("    warnings = []")
    deep_lines.append("    validated = {}")

    for i, sec in enumerate(used_sections):
        keys = [f"host", f"port", f"timeout", f"enabled", f"max_retries",
                f"level", f"format", f"path"]
        rng.shuffle(keys)
        for j, key in enumerate(keys[:max(1, n_deep // len(used_sections))]):
            deep_lines.append(f'    if "{key}" in config["{sec}"]:')
            deep_lines.append(f'        val = config["{sec}"]["{key}"]')
            if key == "port":
                deep_lines.append(f"        if isinstance(val, int) and 1 <= val <= 65535:")
                deep_lines.append(f'            validated["{sec}_{key}"] = val')
                deep_lines.append(f"        else:")
                deep_lines.append(f'            warnings.append("{sec}.{key}: invalid port")')
            elif key == "timeout":
                deep_lines.append(f"        if isinstance(val, (int, float)) and val > 0:")
                deep_lines.append(f'            validated["{sec}_{key}"] = val')
                deep_lines.append(f"        elif val == 0:")
                deep_lines.append(f'            validated["{sec}_{key}"] = 30  # default')
                deep_lines.append(f"        else:")
                deep_lines.append(f'            warnings.append("{sec}.{key}: invalid")')
            elif key == "enabled":
                deep_lines.append(f"        if isinstance(val, bool):")
                deep_lines.append(f'            validated["{sec}_{key}"] = val')
                deep_lines.append(f"        else:")
                deep_lines.append(f'            validated["{sec}_{key}"] = bool(val)')
            else:
                deep_lines.append(f'        validated["{sec}_{key}"] = val')
            deep_lines.append(f"    else:")
            deep_lines.append(f'        validated["{sec}_{key}"] = None')

    deep_lines.append('    return {"status": "valid", "config": validated, "warnings": warnings}')

    source = f'def {func_name}(config):\n'
    source += f'    """Validate config with {n_gates} gates, {len(used_sections)} sections.\n'
    source += f'    Required sections: {", ".join(used_sections)}\n'
    source += f'    """\n'
    source += "\n".join(gate_lines) + "\n"
    source += "\n".join(deep_lines) + "\n"

    return {
        "func_name": func_name,
        "source": source,
        "metadata": {"n_gates": n_gates, "n_deep_branches": n_deep,
                     "domain": "config", "seed": seed},
    }


def _gen_event_handler(rng, n_gates, n_deep, seed):
    """Generate an event handling function."""
    func_name = f"handle_event_{seed}"

    event_types = ["click", "submit", "hover", "scroll", "keypress",
                   "load", "error", "timeout", "resize", "focus"]
    rng.shuffle(event_types)

    gate_lines = []
    gate_lines.append("    if not isinstance(event, dict):")
    gate_lines.append('        return {"error": "event_not_dict"}')
    gate_lines.append('    if "type" not in event:')
    gate_lines.append('        return {"error": "missing_event_type"}')
    gate_lines.append('    if "timestamp" not in event:')
    gate_lines.append('        return {"error": "missing_timestamp"}')

    for i in range(n_gates - 3):
        field = rng.choice(["source", "target", "data", "user_id", "session"])
        gate_lines.append(f'    if "{field}" not in event:')
        gate_lines.append(f'        return {{"error": "missing_{field}"}}')

    deep_lines = []
    deep_lines.append('    etype = event["type"]')
    deep_lines.append("    actions = []")

    for i, et in enumerate(event_types[:n_deep]):
        deep_lines.append(f'    {"if" if i == 0 else "elif"} etype == "{et}":')
        if rng.random() < 0.5:
            deep_lines.append(f'        if event.get("priority", 0) > 5:')
            deep_lines.append(f'            actions.append("{et}_urgent")')
            deep_lines.append(f"        else:")
            deep_lines.append(f'            actions.append("{et}_normal")')
        else:
            deep_lines.append(f'        actions.append("{et}_processed")')
            if rng.random() < 0.3:
                deep_lines.append(f'        if event.get("data") and len(str(event["data"])) > 100:')
                deep_lines.append(f'            actions.append("{et}_large_payload")')

    deep_lines.append("    else:")
    deep_lines.append('        actions.append("unknown_event")')
    deep_lines.append('    return {"status": "handled", "actions": actions, "type": event["type"]}')

    source = f'def {func_name}(event):\n'
    source += f'    """Handle events with {n_gates} validation gates.\n'
    source += f'    Handles: {", ".join(event_types[:n_deep])}\n'
    source += f'    """\n'
    source += "\n".join(gate_lines) + "\n"
    source += "\n".join(deep_lines) + "\n"

    return {
        "func_name": func_name,
        "source": source,
        "metadata": {"n_gates": n_gates, "n_deep_branches": n_deep,
                     "domain": "event", "seed": seed},
    }


def _gen_data_transformer(rng, n_gates, n_deep, seed):
    """Generate a data transformation function."""
    return _gen_data_processor(rng, n_gates, n_deep, seed + 1000)


def _gen_query_processor(rng, n_gates, n_deep, seed):
    """Generate a query processing function."""
    return _gen_event_handler(rng, n_gates, n_deep, seed + 2000)


def generate_batch(n=20, seed=42, gate_range=(3, 7), branch_range=(6, 15)):
    """Generate a batch of corridor functions with varying parameters."""
    rng = random.Random(seed)
    domains = ["data", "config", "event", "data", "config"]

    programs = {}
    for i in range(n):
        n_gates = rng.randint(*gate_range)
        n_deep = rng.randint(*branch_range)
        domain = rng.choice(domains)
        func_seed = seed * 100 + i

        prog = generate_corridor_function(
            seed=func_seed, n_gates=n_gates,
            n_deep_branches=n_deep, domain=domain,
        )
        key = f"gen_{i:03d}"
        programs[key] = prog

    return programs
