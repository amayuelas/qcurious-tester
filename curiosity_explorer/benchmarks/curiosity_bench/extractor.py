"""Extract and fork functions from popular Python libraries.

Extracts standalone functions, applies modifications to create
"strong-but-wrong prior" test cases, and bundles with metadata.
"""

import inspect
import importlib
import textwrap
import re
import random


def extract_function(module_name, func_name):
    """Extract a function's source code from an installed module.

    Returns dict with source, lines, ifs count, imports needed.
    """
    mod = importlib.import_module(module_name)
    obj = getattr(mod, func_name)
    source = textwrap.dedent(inspect.getsource(obj))

    # Detect imports needed by scanning for module references
    imports = _detect_imports(source)

    return {
        "module": module_name,
        "func_name": func_name,
        "source": source,
        "lines": len(source.splitlines()),
        "ifs": source.count("if "),
        "imports": imports,
    }


def fork_function(extracted, modifications, new_name=None):
    """Apply modifications to an extracted function.

    Args:
        extracted: dict from extract_function()
        modifications: list of (old_str, new_str) replacements
        new_name: rename the function (default: add '_fork' suffix)

    Returns forked source string with imports prepended.
    """
    source = extracted["source"]

    # Apply modifications
    for old, new in modifications:
        source = source.replace(old, new)

    # Rename function
    old_name = extracted["func_name"]
    new_name = new_name or f"{old_name}_fork"
    source = source.replace(f"def {old_name}(", f"def {new_name}(", 1)

    # Prepend imports
    import_lines = "\n".join(extracted["imports"])
    full_source = f"{import_lines}\n\n{source}" if import_lines else source

    return {
        "func_name": new_name,
        "source": full_source,
        "original_module": extracted["module"],
        "original_func": extracted["func_name"],
        "modifications": modifications,
        "lines": len(full_source.splitlines()),
    }


def _detect_imports(source):
    """Detect which imports the function needs to work standalone."""
    imports = set()

    # Common stdlib modules referenced in function bodies
    module_patterns = {
        "re.": "import re",
        "os.": "import os",
        "sys.": "import sys",
        "io.": "import io",
        "math.": "import math",
        "json.": "import json",
        "base64.": "import base64",
        "struct.": "import struct",
        "hashlib.": "import hashlib",
        "warnings.": "import warnings",
        "calendar.": "import calendar",
        "datetime.": "import datetime",
        "collections.": "from collections import Counter, OrderedDict, defaultdict",
        "functools.": "import functools",
        "itertools.": "import itertools",
        "codecs.": "import codecs",
        "string.": "import string",
        "binascii.": "import binascii",
        "locale.": "import locale",
        "copy.": "import copy",
        "time.": "import time",
        "errno.": "import errno",
        "stat.": "import stat",
        "shutil.": "import shutil",
        "tempfile.": "import tempfile",
        "pathlib.": "import pathlib",
        "zipfile.": "import zipfile",
        "tarfile.": "import tarfile",
        "gzip.": "import gzip",
        "socket.": "import socket",
        "ssl.": "import ssl",
        "tokenize.": "import tokenize",
    }

    for pattern, imp in module_patterns.items():
        if pattern in source:
            imports.add(imp)

    # Check for builtins that might need importing in sandbox
    if "ValueError" in source or "TypeError" in source:
        pass  # builtins, always available

    return sorted(imports)


# ---------------------------------------------------------------------------
# Modification strategies
# ---------------------------------------------------------------------------

def swap_threshold(source, old_val, new_val):
    """Swap a numeric threshold."""
    return (str(old_val), str(new_val))


def swap_string(source, old_str, new_str):
    """Swap a string constant."""
    return (repr(old_str), repr(new_str))


def swap_condition_order(line1, line2):
    """Swap two elif branches (returns two replacements)."""
    return [(line1, "___TEMP___"), ("___TEMP___", line2), (line2, line1)]


def invert_condition(old_cond, new_cond):
    """Invert a boolean condition."""
    return (old_cond, new_cond)
