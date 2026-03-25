"""Full-module forked benchmarks for CuriosityBench.

Extracts entire stdlib modules (300-2000 lines) and applies targeted
modifications. The LLM has partial observability — modules are too large
to hold entirely in context, forcing genuine exploration.

Each module has 3-5 modifications that change internal behavior while
preserving the API surface.
"""

import inspect
import importlib
import logging

log = logging.getLogger(__name__)


# Each entry: module to extract, list of (old, new) string replacements,
# description of what changed
FULL_MODULE_FORKS = [
    {
        "module": "shlex",
        "mods": [
            ("self.commenters = '#'", "self.commenters = '@'"),
            ("self.escape = '\\\\'", "self.escape = '^'"),
            ("self.quotes = '\\'\"\\'", "self.quotes = '`\"'"),
        ],
        "description": "Shell lexer: @ for comments, ^ for escape, ` for quotes",
    },
    {
        "module": "http.cookies",
        "mods": [
            ('_LegalKeyChars = r"\\w\\d!#%&\'~_`><@{}\\|\\^\\-\\+\\*\\/"',
             '_LegalKeyChars = r"\\w\\d!#%&\'~_`><@{}\\^\\-\\+\\*\\/"'),
            ("_Translator = {", "# Modified translator\n_Translator = {"),
            ('_semispacejoin = "; ".join', '_semispacejoin = " | ".join'),
        ],
        "description": "Cookies: | separator instead of ;",
    },
    {
        "module": "configparser",
        "mods": [
            ("'=' : '='", "'=' : '->'"),
            ("':' : ':'", "'->' : '->'"),
            ("comment_prefixes=('#', ';')", "comment_prefixes=('//', ';;')"),
            ("inline_comment_prefixes=None", "inline_comment_prefixes=('%%',)"),
        ],
        "description": "Config parser: -> delimiter, // comments, %% inline comments",
    },
    {
        "module": "email.header",
        "mods": [
            ("ecre = re.compile(r'=\\?'", "ecre = re.compile(r'=!'"),
            ("'=?'", "'=!'"),
            ("'?='", "'!='"),
        ],
        "description": "Email header: =! encoding marker instead of =?",
    },
    {
        "module": "csv",
        "mods": [
            ("self.preferred = [',', '\\t', ';', ' ', ':']",
             "self.preferred = ['|', '^', '~', '\\t', ',']"),
        ],
        "description": "CSV sniffer: | ^ ~ as preferred delimiters",
    },
    {
        "module": "textwrap",
        "mods": [
            ("width=70", "width=60"),
            ("initial_indent=''", "initial_indent='  '"),
            ("break_long_words=True", "break_long_words=False"),
            ("tabsize=8", "tabsize=4"),
        ],
        "description": "Text wrapping: 60 width, 2-space indent, no break, tab=4",
    },
    {
        "module": "json.decoder",
        "mods": [
            ("NaN", "Null"),
            ("Infinity", "Inf"),
        ],
        "description": "JSON decoder: Null instead of NaN, Inf instead of Infinity",
    },
    {
        "module": "html.parser",
        "mods": [
            ("'script'", "'code'"),
            ("'style'", "'css'"),
            ("CDATA_CONTENT_ELEMENTS = ('script', 'style')",
             "CDATA_CONTENT_ELEMENTS = ('code', 'css')"),
        ],
        "description": "HTML parser: code/css instead of script/style as CDATA elements",
    },
    {
        "module": "pprint",
        "mods": [
            ("indent=1", "indent=2"),
            ("width=80", "width=60"),
            ("depth=None", "depth=5"),
            ("compact=False", "compact=True"),
        ],
        "description": "Pretty printer: 2-indent, 60-width, depth=5, compact",
    },
]


def _extract_and_fork(entry):
    """Extract a full module and apply modifications."""
    try:
        mod = importlib.import_module(entry["module"])
        source = inspect.getsource(mod)

        # Apply modifications
        for old, new in entry["mods"]:
            if old in source:
                source = source.replace(old, new)
            else:
                log.debug(f"Mod not found in {entry['module']}: {old[:50]}")

        return {
            "source": source,
            "lines": len(source.splitlines()),
            "module": entry["module"],
            "description": entry["description"],
            "mods_applied": len(entry["mods"]),
        }
    except Exception as e:
        log.warning(f"Failed to extract {entry['module']}: {e}")
        return None


def load_full_module_benchmark():
    """Load full-module forked benchmarks.

    Returns dict mapping key -> {source, metadata, main_entry}.
    """
    programs = {}

    for entry in FULL_MODULE_FORKS:
        result = _extract_and_fork(entry)
        if not result:
            continue

        mod_name = entry["module"].replace(".", "_")
        key = f"fm_{mod_name}"

        # Determine what the LLM should test
        # Extract public class/function names for hints
        mod = importlib.import_module(entry["module"])
        public_names = [n for n in dir(mod)
                        if not n.startswith("_")
                        and (inspect.isclass(getattr(mod, n))
                             or inspect.isfunction(getattr(mod, n)))]

        programs[key] = {
            "func_name": public_names[0] if public_names else mod_name,
            "source": result["source"],
            "metadata": {
                "original_module": entry["module"],
                "description": entry["description"],
                "lines": result["lines"],
                "mods_applied": result["mods_applied"],
                "public_names": public_names[:10],
                "type": "full_module",
            },
        }

    return programs
