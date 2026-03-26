"""RepoExploreBench: Benchmark for evaluating exploration strategies on real repos.

~100 real-world Python source files across 9 repositories, curated for
corridor structure (import chains, class setup, configuration requirements).

All targets run inside the curiositybench Docker image.

Selection criteria (principled):
  1. Minimum 200 source lines (enough branches to create meaningful exploration)
  2. ~10 modules per repo for balanced cross-project diversity
  3. Larger modules preferred (more paths to discover, more room for curiosity)
  4. Functional diversity within each repo (avoid picking 10 similar utils)
  5. Each module should expose classes/functions to test, not just constants
  6. Skip test-only modules (except testing utilities like click.testing)
  7. Skip private modules EXCEPT httpx, which uses _ prefix by convention
  8. Skip vendored code and legacy/deprecated compatibility layers

Repos (all pre-installed in curiositybench:latest):
  click, requests, flask, rich, jinja2, httpx, pydantic, werkzeug, starlette

Statistical design:
  - ~100 files across 9 repos (balanced diversity)
  - 3 seeds per file for reproducibility (default: 42, 123, 456)
  - Paired comparisons within (file, seed) for significance testing
"""

import logging

log = logging.getLogger(__name__)

DOCKER_IMAGE = "curiositybench:latest"

DEFAULT_SEEDS = [42, 123, 456]

# ---------------------------------------------------------------------------
# ~100 curated targets -- real-world modules with corridor structure
#
# Discovered via pkgutil.walk_packages inside the Docker image.
# All line counts verified against curiositybench:latest.
# ---------------------------------------------------------------------------

TARGETS = [
    # == click (10) -- CLI framework ==========================================
    # Corridors: command setup -> option parsing -> type coercion -> completion
    {"module": "click.core", "repo": "click", "lines": 3042,
     "description": "Command/Group/Option classes with complex dispatch"},
    {"module": "click.types", "repo": "click", "lines": 1089,
     "description": "Type system with conversion and validation corridors"},
    {"module": "click.termui", "repo": "click", "lines": 784,
     "description": "Terminal UI helpers: prompts, progress bars, pagers"},
    {"module": "click.utils", "repo": "click", "lines": 624,
     "description": "Utility functions: file handling, lazy loading"},
    {"module": "click.shell_completion", "repo": "click", "lines": 596,
     "description": "Shell completion for bash/zsh/fish"},
    {"module": "click.decorators", "repo": "click", "lines": 561,
     "description": "Decorator-based command construction"},
    {"module": "click.parser", "repo": "click", "lines": 529,
     "description": "Option/argument parser (optparse-like)"},
    {"module": "click.testing", "repo": "click", "lines": 479,
     "description": "CLI test runner with isolated filesystem"},
    {"module": "click.formatting", "repo": "click", "lines": 301,
     "description": "Help text formatting with wrapping"},
    {"module": "click.exceptions", "repo": "click", "lines": 288,
     "description": "Exception hierarchy with formatting"},

    # == requests (6) -- HTTP client ==========================================
    # Corridors: session setup -> auth -> adapters -> connection pool
    {"module": "requests.utils", "repo": "requests", "lines": 1094,
     "description": "URL parsing, encoding detection, proxy config"},
    {"module": "requests.models", "repo": "requests", "lines": 1034,
     "description": "Request/Response/PreparedRequest models"},
    {"module": "requests.sessions", "repo": "requests", "lines": 833,
     "description": "Session with auth, cookies, adapters"},
    {"module": "requests.cookies", "repo": "requests", "lines": 561,
     "description": "Cookie jar with domain/path matching"},
    {"module": "requests.adapters", "repo": "requests", "lines": 538,
     "description": "HTTP adapter with connection pooling"},
    {"module": "requests.auth", "repo": "requests", "lines": 315,
     "description": "Auth handlers (Basic, Digest, Token)"},

    # == flask (10) -- Web framework ==========================================
    # Corridors: app creation -> config -> routing -> request context -> response
    {"module": "flask.app", "repo": "flask", "lines": 1478,
     "description": "Flask app with routing, config, error handling"},
    {"module": "flask.cli", "repo": "flask", "lines": 1068,
     "description": "CLI integration with click commands"},
    {"module": "flask.helpers", "repo": "flask", "lines": 623,
     "description": "URL generation, flashing, send_file"},
    {"module": "flask.ctx", "repo": "flask", "lines": 440,
     "description": "Application and request context management"},
    {"module": "flask.sessions", "repo": "flask", "lines": 367,
     "description": "Session interface and cookie-based sessions"},
    {"module": "flask.config", "repo": "flask", "lines": 347,
     "description": "Config with env loading and defaults"},
    {"module": "flask.json.tag", "repo": "flask", "lines": 314,
     "description": "Tagged JSON serialization for sessions"},
    {"module": "flask.testing", "repo": "flask", "lines": 295,
     "description": "Test client with session handling"},
    {"module": "flask.templating", "repo": "flask", "lines": 221,
     "description": "Jinja2 template rendering integration"},
    {"module": "flask.json.provider", "repo": "flask", "lines": 216,
     "description": "JSON provider with custom encoders"},

    # == rich (12) -- Terminal rendering ======================================
    # Corridors: console setup -> style parsing -> segment building -> rendering
    {"module": "rich.console", "repo": "rich", "lines": 2633,
     "description": "Console with style, markup, table rendering"},
    {"module": "rich.progress", "repo": "rich", "lines": 1699,
     "description": "Progress bars with multiple tracks and columns"},
    {"module": "rich.text", "repo": "rich", "lines": 1357,
     "description": "Styled text with spans and highlights"},
    {"module": "rich.table", "repo": "rich", "lines": 1000,
     "description": "Table with columns, rows, formatting"},
    {"module": "rich.pretty", "repo": "rich", "lines": 995,
     "description": "Pretty-printing with rich formatting"},
    {"module": "rich.syntax", "repo": "rich", "lines": 958,
     "description": "Syntax highlighting with Pygments integration"},
    {"module": "rich.markdown", "repo": "rich", "lines": 800,
     "description": "Markdown rendering to terminal"},
    {"module": "rich.style", "repo": "rich", "lines": 796,
     "description": "Style parsing and combination"},
    {"module": "rich.traceback", "repo": "rich", "lines": 753,
     "description": "Rich traceback formatting"},
    {"module": "rich.segment", "repo": "rich", "lines": 738,
     "description": "Segment-based rendering pipeline"},
    {"module": "rich.color", "repo": "rich", "lines": 621,
     "description": "Color parsing (named, hex, RGB, system)"},
    {"module": "rich.layout", "repo": "rich", "lines": 442,
     "description": "Terminal layout with split panels"},

    # == jinja2 (12) -- Template engine =======================================
    # Corridors: env setup -> lexer -> parser -> AST -> compiler -> runtime
    {"module": "jinja2.compiler", "repo": "jinja2", "lines": 1956,
     "description": "Template AST to Python code compiler"},
    {"module": "jinja2.filters", "repo": "jinja2", "lines": 1854,
     "description": "Built-in template filters (50+)"},
    {"module": "jinja2.environment", "repo": "jinja2", "lines": 1667,
     "description": "Environment with loader, globals, filters"},
    {"module": "jinja2.nodes", "repo": "jinja2", "lines": 1204,
     "description": "Template AST node definitions"},
    {"module": "jinja2.runtime", "repo": "jinja2", "lines": 1051,
     "description": "Template runtime (context, loops, macros)"},
    {"module": "jinja2.parser", "repo": "jinja2", "lines": 1034,
     "description": "Template source to AST parser"},
    {"module": "jinja2.ext", "repo": "jinja2", "lines": 869,
     "description": "Extension system (i18n, do, loopcontrols)"},
    {"module": "jinja2.lexer", "repo": "jinja2", "lines": 866,
     "description": "Template lexer/tokenizer"},
    {"module": "jinja2.utils", "repo": "jinja2", "lines": 755,
     "description": "Utility functions and LRU cache"},
    {"module": "jinja2.loaders", "repo": "jinja2", "lines": 661,
     "description": "Template loaders (filesystem, package, dict)"},
    {"module": "jinja2.sandbox", "repo": "jinja2", "lines": 428,
     "description": "Sandboxed execution environment"},
    {"module": "jinja2.bccache", "repo": "jinja2", "lines": 406,
     "description": "Bytecode cache (file, memcached)"},

    # == httpx (10) -- Async HTTP client ======================================
    # Corridors: client config -> auth -> transport -> URL parsing -> response
    # Note: httpx uses _ prefix by convention for all its modules
    {"module": "httpx._client", "repo": "httpx", "lines": 2052,
     "description": "Sync and async HTTP client implementations"},
    {"module": "httpx._models", "repo": "httpx", "lines": 1209,
     "description": "Request/Response models with headers and content"},
    {"module": "httpx._urls", "repo": "httpx", "lines": 646,
     "description": "URL class with query parameter handling"},
    {"module": "httpx._main", "repo": "httpx", "lines": 509,
     "description": "Main entry point and convenience functions"},
    {"module": "httpx._urlparse", "repo": "httpx", "lines": 502,
     "description": "URL parsing with IDNA and percent encoding"},
    {"module": "httpx._api", "repo": "httpx", "lines": 467,
     "description": "Top-level API functions (get, post, etc.)"},
    {"module": "httpx._utils", "repo": "httpx", "lines": 440,
     "description": "Utility functions for URL and header handling"},
    {"module": "httpx._transports.default", "repo": "httpx", "lines": 385,
     "description": "Default HTTP transport with connection pooling"},
    {"module": "httpx._config", "repo": "httpx", "lines": 370,
     "description": "SSL, timeout, proxy, and limits configuration"},
    {"module": "httpx._auth", "repo": "httpx", "lines": 345,
     "description": "Authentication flows (Basic, Digest, custom)"},

    # == pydantic (10) -- Data validation =====================================
    # Corridors: model definition -> schema generation -> validation -> serialization
    {"module": "pydantic.types", "repo": "pydantic", "lines": 2877,
     "description": "Constrained types (PositiveInt, EmailStr, etc.)"},
    {"module": "pydantic.json_schema", "repo": "pydantic", "lines": 2425,
     "description": "JSON Schema generation from models"},
    {"module": "pydantic.main", "repo": "pydantic", "lines": 1500,
     "description": "BaseModel with validation and serialization"},
    {"module": "pydantic.fields", "repo": "pydantic", "lines": 1154,
     "description": "Field definitions with metadata and defaults"},
    {"module": "pydantic.config", "repo": "pydantic", "lines": 912,
     "description": "Model configuration options"},
    {"module": "pydantic.networks", "repo": "pydantic", "lines": 708,
     "description": "Network types (URLs, emails, IPs)"},
    {"module": "pydantic.functional_validators", "repo": "pydantic", "lines": 695,
     "description": "Decorator-based validators (before, after, wrap)"},
    {"module": "pydantic.color", "repo": "pydantic", "lines": 603,
     "description": "Color type with parsing and conversion"},
    {"module": "pydantic.type_adapter", "repo": "pydantic", "lines": 458,
     "description": "TypeAdapter for non-BaseModel validation"},
    {"module": "pydantic.functional_serializers", "repo": "pydantic", "lines": 395,
     "description": "Custom serialization decorators"},

    # == werkzeug (12) -- WSGI toolkit ========================================
    # Corridors: routing setup -> request parsing -> response building -> serving
    {"module": "werkzeug.http", "repo": "werkzeug", "lines": 1372,
     "description": "HTTP header parsing and generation"},
    {"module": "werkzeug.serving", "repo": "werkzeug", "lines": 1109,
     "description": "Development WSGI server with reloader"},
    {"module": "werkzeug.datastructures.structures", "repo": "werkzeug", "lines": 1006,
     "description": "MultiDict, OrderedMultiDict, ImmutableDict"},
    {"module": "werkzeug.routing.map", "repo": "werkzeug", "lines": 946,
     "description": "URL routing map with adapter binding"},
    {"module": "werkzeug.routing.rules", "repo": "werkzeug", "lines": 909,
     "description": "URL rule definitions with converters"},
    {"module": "werkzeug.exceptions", "repo": "werkzeug", "lines": 879,
     "description": "HTTP exception classes (404, 500, etc.)"},
    {"module": "werkzeug.wrappers.response", "repo": "werkzeug", "lines": 835,
     "description": "Response wrapper with headers and cookies"},
    {"module": "werkzeug.sansio.response", "repo": "werkzeug", "lines": 751,
     "description": "Sans-IO response implementation"},
    {"module": "werkzeug.utils", "repo": "werkzeug", "lines": 690,
     "description": "Utility functions: redirect, secure filename"},
    {"module": "werkzeug.wrappers.request", "repo": "werkzeug", "lines": 650,
     "description": "Request wrapper with form and file parsing"},
    {"module": "werkzeug.local", "repo": "werkzeug", "lines": 643,
     "description": "Context locals and proxy objects"},
    {"module": "werkzeug.wsgi", "repo": "werkzeug", "lines": 595,
     "description": "WSGI helpers: ClosingIterator, responder"},

    # == starlette (11) -- ASGI framework =====================================
    # Corridors: app -> routing -> middleware -> request/response -> websockets
    {"module": "starlette.routing", "repo": "starlette", "lines": 932,
     "description": "Route matching with path parameters"},
    {"module": "starlette.testclient", "repo": "starlette", "lines": 806,
     "description": "Test client for ASGI apps"},
    {"module": "starlette.datastructures", "repo": "starlette", "lines": 715,
     "description": "URL, Headers, QueryParams, State"},
    {"module": "starlette.responses", "repo": "starlette", "lines": 354,
     "description": "Response types (JSON, HTML, Streaming)"},
    {"module": "starlette.requests", "repo": "starlette", "lines": 324,
     "description": "ASGI request with body parsing"},
    {"module": "starlette.formparsers", "repo": "starlette", "lines": 276,
     "description": "Multipart and form data parsing"},
    {"module": "starlette.applications", "repo": "starlette", "lines": 266,
     "description": "Starlette application with middleware"},
    {"module": "starlette.middleware.errors", "repo": "starlette", "lines": 261,
     "description": "Error handling middleware"},
    {"module": "starlette.staticfiles", "repo": "starlette", "lines": 243,
     "description": "Static file serving"},
    {"module": "starlette.templating", "repo": "starlette", "lines": 231,
     "description": "Jinja2 template integration"},
    {"module": "starlette.middleware.base", "repo": "starlette", "lines": 217,
     "description": "Base middleware class"},
]


def load_benchmark(repos=None, max_targets=None):
    """Load RepoExploreBench targets.

    Args:
        repos: Filter by repo name(s). E.g., ["click", "flask"]
        max_targets: Limit total targets

    Returns:
        List of target dicts with: module, repo, description, lines,
        docker_image, setup_code, working_dir, env
    """
    targets = []
    for spec in TARGETS:
        targets.append({
            "module": spec["module"],
            "repo": spec["repo"],
            "code_file": None,
            "code_src": None,
            "version": None,
            "description": spec["description"],
            "lines": spec["lines"],
            "docker_image": DOCKER_IMAGE,
            "setup_code": "",
            "working_dir": "/opt",
            "env": {},
            "source": "repo_explore_bench",
        })

    if repos:
        targets = [t for t in targets if t["repo"] in repos]
    if max_targets:
        targets = targets[:max_targets]

    log.info(f"RepoExploreBench: {len(targets)} targets across "
             f"{len(set(t['repo'] for t in targets))} repos")
    return targets


def get_benchmark_info():
    """Return benchmark metadata for reporting."""
    from collections import Counter
    repo_counts = Counter(t["repo"] for t in TARGETS)
    return {
        "name": "RepoExploreBench",
        "version": "2.0",
        "total_files": len(TARGETS),
        "repos": dict(repo_counts),
        "seeds": DEFAULT_SEEDS,
        "description": (
            f"{len(TARGETS)} real-world Python source files across "
            f"{len(repo_counts)} repositories. Each file requires navigating "
            f"import chains, class setup, and configuration corridors."
        ),
    }
