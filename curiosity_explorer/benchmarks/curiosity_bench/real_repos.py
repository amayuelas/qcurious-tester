"""Real-repo benchmark for CuriosityBench.

Uses pip-installed popular Python repos in a Docker container.
Each target is a real source file with natural corridor structure
(import chains, class setup, configuration requirements).

The LLM generates test scripts that exercise the target file.
Coverage is measured with coverage.py inside the container.
"""

# Target files: module path → metadata
# Selected for: corridor structure, sufficient complexity, LLM partial knowledge
REAL_REPO_TARGETS = [
    # click — CLI framework (deep corridors: command setup → option parsing → type coercion)
    {"module": "click.core", "repo": "click", "lines": 3042, "description": "Command/Group/Option classes with complex dispatch"},
    {"module": "click.types", "repo": "click", "lines": 1089, "description": "Type system with conversion and validation corridors"},
    {"module": "click.testing", "repo": "click", "lines": 479, "description": "CLI test runner with isolated filesystem"},
    {"module": "click.decorators", "repo": "click", "lines": 561, "description": "Decorator-based command construction"},
    {"module": "click.formatting", "repo": "click", "lines": 301, "description": "Help text formatting with wrapping"},

    # requests — HTTP client (corridors: session setup → auth → adapters → connection pool)
    {"module": "requests.sessions", "repo": "requests", "lines": 833, "description": "Session with auth, cookies, adapters"},
    {"module": "requests.models", "repo": "requests", "lines": 1034, "description": "Request/Response/PreparedRequest models"},
    {"module": "requests.auth", "repo": "requests", "lines": 315, "description": "Auth handlers (Basic, Digest, Token)"},
    {"module": "requests.adapters", "repo": "requests", "lines": 538, "description": "HTTP adapter with connection pooling"},
    {"module": "requests.cookies", "repo": "requests", "lines": 561, "description": "Cookie jar with domain/path matching"},
    {"module": "requests.utils", "repo": "requests", "lines": 1094, "description": "URL parsing, encoding detection, proxy config"},

    # flask — Web framework (corridors: app creation → config → routing → request context)
    {"module": "flask.app", "repo": "flask", "lines": 1478, "description": "Flask app with routing, config, error handling"},
    {"module": "flask.blueprints", "repo": "flask", "lines": 0, "description": "Blueprint registration and URL rules"},
    {"module": "flask.config", "repo": "flask", "lines": 347, "description": "Config with env loading and defaults"},
    {"module": "flask.testing", "repo": "flask", "lines": 295, "description": "Test client with session handling"},

    # marshmallow — Serialization (corridors: schema definition → field validation → deserialization)
    {"module": "marshmallow.schema", "repo": "marshmallow", "lines": 1230, "description": "Schema with load/dump, validation, nesting"},
    {"module": "marshmallow.fields", "repo": "marshmallow", "lines": 2114, "description": "38 field types with validation chains"},
    {"module": "marshmallow.validate", "repo": "marshmallow", "lines": 681, "description": "Validators (Range, Length, OneOf, Regexp)"},

    # rich — Terminal UI (corridors: console setup → style parsing → rendering pipeline)
    {"module": "rich.console", "repo": "rich", "lines": 2633, "description": "Console with style, markup, table rendering"},
    {"module": "rich.text", "repo": "rich", "lines": 1357, "description": "Styled text with spans and highlights"},
    {"module": "rich.table", "repo": "rich", "lines": 1000, "description": "Table with columns, rows, formatting"},
    {"module": "rich.style", "repo": "rich", "lines": 796, "description": "Style parsing and combination"},
    {"module": "rich.markup", "repo": "rich", "lines": 251, "description": "Markup tag parsing"},

    # httpx — Async HTTP (corridors: client config → transport → auth → response)
    {"module": "httpx._client", "repo": "httpx", "lines": 2052, "description": "Sync/Async client with middleware pipeline"},
    {"module": "httpx._urls", "repo": "httpx", "lines": 646, "description": "URL parsing and manipulation"},
    {"module": "httpx._auth", "repo": "httpx", "lines": 345, "description": "Auth flow with challenge/response"},
    {"module": "httpx._config", "repo": "httpx", "lines": 370, "description": "SSL, timeout, proxy configuration"},

    # jinja2 — Template engine (corridors: env setup → lexer → parser → compiler → render)
    {"module": "jinja2.environment", "repo": "jinja2", "lines": 1667, "description": "Environment with loader, globals, filters"},
    {"module": "jinja2.lexer", "repo": "jinja2", "lines": 866, "description": "Template lexer/tokenizer"},

    # pydantic — Data validation (corridors: model definition → schema generation → validation)
    {"module": "pydantic.main", "repo": "pydantic", "lines": 1500, "description": "BaseModel with validation, serialization"},
    {"module": "pydantic.fields", "repo": "pydantic", "lines": 1154, "description": "Field definitions with metadata"},

    # werkzeug — WSGI utilities (corridors: routing → request parsing → response building)
    {"module": "werkzeug.routing", "repo": "werkzeug", "lines": 0, "description": "URL routing with converters and rules"},
    {"module": "werkzeug.datastructures", "repo": "werkzeug", "lines": 0, "description": "MultiDict, Headers, ImmutableDict"},

    # starlette — ASGI framework
    {"module": "starlette.routing", "repo": "starlette", "lines": 932, "description": "Route matching with path parameters"},
    {"module": "starlette.testclient", "repo": "starlette", "lines": 806, "description": "Test client for ASGI apps"},
    {"module": "starlette.responses", "repo": "starlette", "lines": 354, "description": "Response types (JSON, HTML, Streaming)"},
]

DOCKER_IMAGE = "curiositybench:latest"


def get_targets(n=None, repo=None):
    """Get target modules for benchmarking.

    Args:
        n: Max number of targets
        repo: Filter by repo name

    Returns list of target dicts.
    """
    targets = REAL_REPO_TARGETS
    if repo:
        targets = [t for t in targets if t["repo"] == repo]
    if n:
        targets = targets[:n]
    return targets
