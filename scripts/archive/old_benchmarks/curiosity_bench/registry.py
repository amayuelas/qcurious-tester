"""Registry of 50 forked functions for CuriosityBench.

Each entry specifies:
- The source module and function to extract
- Modifications to apply (threshold changes, string swaps, condition inversions)
- Metadata for analysis (repo, category, expected complexity)
"""

import logging
from .extractor import extract_function, fork_function

log = logging.getLogger(__name__)

# Each entry: (module, func_name, modifications, category, description)
# Modifications: list of (old_string, new_string) replacements
FUNCTION_REGISTRY = [
    # =====================================================================
    # email (5) — date/header parsing corridors
    # =====================================================================
    {
        "module": "email.utils",
        "func": "_parsedate_tz",
        "mods": [
            ("tm_year = tm_year + 2000", "tm_year = tm_year + 1900"),  # Y2K handling
            ("if len(s) == 2", "if len(s) <= 2"),
        ],
        "category": "email",
        "description": "Date parsing with modified year handling",
    },
    {
        "module": "email.header",
        "func": "decode_header",
        "mods": [
            ("=?", "=!"),  # Change encoded-word marker
        ],
        "category": "email",
        "description": "Header decoding with changed marker",
    },
    {
        "module": "email._header_value_parser",
        "func": "get_parameter",
        "mods": [
            ("*0", "*1"),  # Change continuation index
        ],
        "category": "email",
        "description": "MIME parameter parsing with shifted continuation",
    },
    {
        "module": "email._header_value_parser",
        "func": "parse_mime_version",
        "mods": [
            ("1", "2"),  # Change expected major version
        ],
        "category": "email",
        "description": "MIME version with changed major version",
    },
    {
        "module": "email._header_value_parser",
        "func": "_fold_as_ew",
        "mods": [
            ("75", "60"),  # Change line length limit
        ],
        "category": "email",
        "description": "Header folding with shorter line limit",
    },

    # =====================================================================
    # urllib/http (6) — URL and cookie parsing
    # =====================================================================
    {
        "module": "urllib.parse",
        "func": "urljoin",
        "mods": [
            ("://", "::"),  # Change scheme separator
        ],
        "category": "url",
        "description": "URL joining with modified scheme separator",
    },
    {
        "module": "urllib.parse",
        "func": "parse_qsl",
        "mods": [
            ("'&'", "'|'"),  # Change default separator
            ("max_num_fields", "max_fields"),  # Rename parameter
        ],
        "category": "url",
        "description": "Query string parsing with | separator",
    },
    {
        "module": "urllib.parse",
        "func": "urlencode",
        "mods": [
            ("'&'", "';'"),  # Change join separator
        ],
        "category": "url",
        "description": "URL encoding with ; separator",
    },
    {
        "module": "http.cookiejar",
        "func": "parse_ns_headers",
        "mods": [
            ("';'", "','"),  # Change cookie attribute separator
        ],
        "category": "http",
        "description": "Cookie header parsing with , separator",
    },
    {
        "module": "http.cookiejar",
        "func": "_str2time",
        "mods": [
            ("2000", "2010"),  # Change Y2K cutoff
        ],
        "category": "http",
        "description": "Date parsing with shifted Y2K cutoff",
    },
    {
        "module": "http.cookiejar",
        "func": "lwp_cookie_str",
        "mods": [
            ("'\"'", "\"'\""),  # Swap quote characters
        ],
        "category": "http",
        "description": "Cookie string with swapped quotes",
    },

    # =====================================================================
    # click (5) — CLI processing
    # =====================================================================
    {
        "module": "click.core",
        "func": "style",
        "mods": [
            ("\\033[", "\\033("),  # Change ANSI escape prefix
            ("reset = True", "reset = False"),
        ],
        "category": "cli",
        "description": "ANSI styling with modified escape sequence",
    },
    {
        "module": "click.core",
        "func": "echo",
        "mods": [
            ("'utf-8'", "'ascii'"),  # Change default encoding
        ],
        "category": "cli",
        "description": "Echo with ascii default encoding",
    },
    {
        "module": "click.core",
        "func": "prompt",
        "mods": [
            ("': '", "' > '"),  # Change prompt suffix
            ("'Abort!'", "'Cancelled!'"),
        ],
        "category": "cli",
        "description": "Prompt with > suffix and Cancelled message",
    },
    {
        "module": "click.types",
        "func": "convert_type",
        "mods": [
            ("bool", "int"),  # Swap type mapping
        ],
        "category": "cli",
        "description": "Type conversion with bool→int swap",
    },
    {
        "module": "click.types",
        "func": "open_stream",
        "mods": [
            ("'-'", "'@'"),  # Change stdin/stdout marker
        ],
        "category": "cli",
        "description": "Stream opening with @ as stdin marker",
    },

    # =====================================================================
    # json (2) — parsing
    # =====================================================================
    {
        "module": "json.decoder",
        "func": "JSONObject",
        "mods": [
            ("':'", "'='"),  # Change key-value separator
        ],
        "category": "json",
        "description": "JSON object parsing with = separator",
    },
    {
        "module": "json.encoder",
        "func": "_make_iterencode",
        "mods": [
            ("', '", "'; '"),  # Change item separator
            ("': '", "' = '"),  # Change key-value separator
        ],
        "category": "json",
        "description": "JSON encoding with ; and = separators",
    },

    # =====================================================================
    # ast (3) — code evaluation
    # =====================================================================
    {
        "module": "ast",
        "func": "literal_eval",
        "mods": [
            ("'__builtins__'", "'__safe__'"),
        ],
        "category": "ast",
        "description": "Literal eval with renamed builtins key",
    },
    {
        "module": "ast",
        "func": "dump",
        "mods": [
            ("indent", "padding"),  # Rename parameter
        ],
        "category": "ast",
        "description": "AST dump with renamed indent parameter",
    },
    {
        "module": "ast",
        "func": "_simple_enum",
        "mods": [
            ("'_generate_next_value_'", "'_next_value_'"),
        ],
        "category": "ast",
        "description": "Simple enum with renamed generator",
    },

    # =====================================================================
    # requests (4) — HTTP utilities
    # =====================================================================
    {
        "module": "requests.utils",
        "func": "parse_url",
        "mods": [
            ("'http'", "'https'"),  # Change default scheme
        ],
        "category": "requests",
        "description": "URL parsing defaulting to https",
    },
    {
        "module": "requests.utils",
        "func": "should_bypass_proxies",
        "mods": [
            ("'no_proxy'", "'bypass_proxy'"),  # Change env var name
            ("'NO_PROXY'", "'BYPASS_PROXY'"),
        ],
        "category": "requests",
        "description": "Proxy bypass with renamed env var",
    },
    {
        "module": "requests.utils",
        "func": "guess_json_utf",
        "mods": [
            ("'utf-32'", "'utf-16'"),  # Swap encoding detection
        ],
        "category": "requests",
        "description": "JSON encoding detection with swapped UTF",
    },
    {
        "module": "requests.utils",
        "func": "super_len",
        "mods": [
            ("'len'", "'size'"),  # Change attribute name to check
        ],
        "category": "requests",
        "description": "Content length with size attribute",
    },

    # =====================================================================
    # shutil (5) — file operations
    # =====================================================================
    {
        "module": "shutil",
        "func": "which",
        "mods": [
            ("'PATH'", "'BINPATH'"),  # Change PATH env var name
            ("os.defpath", "'/usr/bin:/bin'"),
        ],
        "category": "shutil",
        "description": "which() using BINPATH env var",
    },
    {
        "module": "shutil",
        "func": "_make_tarball",
        "mods": [
            ("'.gz'", "'.bz2'"),  # Change default compression
            ("'gz'", "'bz2'"),
        ],
        "category": "shutil",
        "description": "Tar creation with bz2 default",
    },
    {
        "module": "shutil",
        "func": "_make_zipfile",
        "mods": [
            ("'.zip'", "'.pk'"),  # Change extension
        ],
        "category": "shutil",
        "description": "Zip creation with .pk extension",
    },
    {
        "module": "shutil",
        "func": "copystat",
        "mods": [
            ("follow_symlinks=True", "follow_symlinks=False"),
        ],
        "category": "shutil",
        "description": "Copystat not following symlinks by default",
    },
    {
        "module": "shutil",
        "func": "rmtree",
        "mods": [
            ("onerror", "on_fail"),  # Rename parameter
        ],
        "category": "shutil",
        "description": "rmtree with renamed error callback",
    },

    # =====================================================================
    # xml (3) — serialization
    # =====================================================================
    {
        "module": "xml.etree.ElementTree",
        "func": "_namespaces",
        "mods": [
            ("'xmlns:'", "'ns:'"),  # Change namespace prefix
        ],
        "category": "xml",
        "description": "XML namespaces with ns: prefix",
    },
    {
        "module": "xml.etree.ElementTree",
        "func": "_serialize_html",
        "mods": [
            ("'/>'", "' />'"),  # Change self-closing tag format
        ],
        "category": "xml",
        "description": "HTML serialization with space before />",
    },
    {
        "module": "xml.etree.ElementTree",
        "func": "_serialize_xml",
        "mods": [
            ("'<?xml'", "'<?XML'"),  # Change processing instruction case
        ],
        "category": "xml",
        "description": "XML serialization with uppercase PI",
    },

    # =====================================================================
    # tokenize/locale/fnmatch (3)
    # =====================================================================
    {
        "module": "tokenize",
        "func": "detect_encoding",
        "mods": [
            ("'utf-8'", "'latin-1'"),  # Change default encoding
            ("'coding'", "'charset'"),
        ],
        "category": "tokenize",
        "description": "Encoding detection defaulting to latin-1",
    },
    {
        "module": "locale",
        "func": "normalize",
        "mods": [
            ("'UTF-8'", "'UTF8'"),  # Change canonical name
        ],
        "category": "locale",
        "description": "Locale normalization with UTF8 variant",
    },
    {
        "module": "fnmatch",
        "func": "translate",
        "mods": [
            ("'*'", "'%'"),  # Change glob wildcard
            ("'?'", "'_'"),
        ],
        "category": "fnmatch",
        "description": "Glob translation with SQL-style wildcards",
    },

    # =====================================================================
    # rich/pydantic (4)
    # =====================================================================
    {
        "module": "rich.markup",
        "func": "render",
        "mods": [
            ("'[/'", "'[~'"),  # Change closing tag marker
        ],
        "category": "rich",
        "description": "Rich markup with ~ as closing marker",
    },
    {
        "module": "pydantic.fields",
        "func": "inspect_annotation",
        "mods": [
            ("'Field'", "'Param'"),  # Rename field reference
        ],
        "category": "pydantic",
        "description": "Annotation inspection with Param name",
    },
    {
        "module": "pydantic.main",
        "func": "create_model",
        "mods": [
            ("'__module__'", "'__origin__'"),  # Change metadata key
        ],
        "category": "pydantic",
        "description": "Model creation with renamed module key",
    },
    {
        "module": "pydantic.main",
        "func": "getattr_migration",
        "mods": [
            ("'__pydantic'", "'__schema'"),  # Change attribute prefix
        ],
        "category": "pydantic",
        "description": "Attribute migration with schema prefix",
    },

    # =====================================================================
    # logging/ssl/base64/tarfile/copy/gzip/glob/locale/socket (10)
    # =====================================================================
    {
        "module": "logging",
        "func": "basicConfig",
        "mods": [
            ("'%(levelname)s'", "'%(level)s'"),  # Change format key
            ("'WARNING'", "'NOTICE'"),
        ],
        "category": "logging",
        "description": "Logging config with renamed level format",
    },
    {
        "module": "ssl",
        "func": "_create_stdlib_context",
        "mods": [
            ("PROTOCOL_TLS_CLIENT", "PROTOCOL_TLS"),
        ],
        "category": "ssl",
        "description": "SSL context with different protocol",
    },
    {
        "module": "base64",
        "func": "a85decode",
        "mods": [
            ("'<~'", "'(~'"),  # Change prefix marker
            ("'~>'", "'~)'"),  # Change suffix marker
        ],
        "category": "base64",
        "description": "Ascii85 decoding with changed markers",
    },
    {
        "module": "tarfile",
        "func": "_get_filtered_attrs",
        "mods": [
            ("100000", "50000"),  # Change size threshold
        ],
        "category": "tarfile",
        "description": "Tar filtering with lower size limit",
    },
    {
        "module": "copy",
        "func": "_reconstruct",
        "mods": [
            ("'__reduce_ex__'", "'__reduce_v2__'"),
        ],
        "category": "copy",
        "description": "Object reconstruction with renamed reduce",
    },
    {
        "module": "copy",
        "func": "deepcopy",
        "mods": [
            ("'__deepcopy__'", "'__clone__'"),
        ],
        "category": "copy",
        "description": "Deep copy looking for __clone__ method",
    },
    {
        "module": "gzip",
        "func": "_read_gzip_header",
        "mods": [
            ("b'\\037\\213'", "b'\\037\\214'"),  # Change magic bytes
        ],
        "category": "gzip",
        "description": "Gzip header with modified magic bytes",
    },
    {
        "module": "glob",
        "func": "_iglob",
        "mods": [
            ("'**'", "'***'"),  # Change recursive glob marker
        ],
        "category": "glob",
        "description": "Glob with *** as recursive marker",
    },
    {
        "module": "locale",
        "func": "currency",
        "mods": [
            ("grouping=False", "grouping=True"),
        ],
        "category": "locale",
        "description": "Currency formatting with grouping default",
    },
    {
        "module": "socket",
        "func": "create_server",
        "mods": [
            ("SOL_SOCKET", "IPPROTO_TCP"),
        ],
        "category": "socket",
        "description": "Server creation with TCP protocol level",
    },
]


def _build_function(entry):
    """Extract and fork a single function."""
    try:
        extracted = extract_function(entry["module"], entry["func"])
        forked = fork_function(extracted, entry["mods"])
        return {
            "func_name": forked["func_name"],
            "source": forked["source"],
            "metadata": {
                "original_module": entry["module"],
                "original_func": entry["func"],
                "category": entry["category"],
                "description": entry["description"],
                "modifications": entry["mods"],
                "lines": forked["lines"],
            },
        }
    except Exception as e:
        log.warning(f"Failed to extract {entry['module']}.{entry['func']}: {e}")
        return None


def load_benchmark(n=None, repo=None, category=None):
    """Load CuriosityBench functions.

    Args:
        n: Max number of functions (None = all)
        repo: Filter by original repo name (e.g., "click", "requests")
        category: Filter by category (e.g., "email", "url", "cli")

    Returns:
        dict mapping key -> {func_name, source, metadata}
    """
    entries = FUNCTION_REGISTRY

    if category:
        entries = [e for e in entries if e["category"] == category]
    elif repo:
        entries = [e for e in entries if repo in e["module"]]

    if n:
        entries = entries[:n]

    programs = {}
    for i, entry in enumerate(entries):
        result = _build_function(entry)
        if result:
            key = f"cb_{i:03d}_{entry['category']}"
            programs[key] = result

    return programs
