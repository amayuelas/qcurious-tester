"""File-level benchmark modules for CuriosityBench.

Each module contains multiple related classes/functions that require
setup to test — mimicking real-world file-level code like Django.
Forked from well-known stdlib modules with modified internals.

The LLM must:
1. Import the right classes
2. Create objects with correct parameters
3. Call methods in the right order
4. Discover modified behavior through interaction
"""


# =========================================================================
# Module 1: Mini Cookie Jar (forked from http.cookies)
#
# A simplified cookie parsing/formatting module with modified rules:
# - Cookie separator is '|' instead of ';'
# - Attribute prefix is '$' instead of nothing
# - Max-Age format changed
# =========================================================================

MODULE_COOKIES = '''
import re
import time

class CookieError(Exception):
    pass


class Cookie:
    """A single cookie with name, value, and attributes."""

    # MODIFIED: attribute names differ from standard
    VALID_ATTRS = {
        "path", "domain", "max-age", "secure", "httponly",
        "samesite", "priority", "expires"
    }

    def __init__(self, name, value=""):
        if not name or not isinstance(name, str):
            raise CookieError("Cookie name must be a non-empty string")
        # MODIFIED: stricter name validation
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_.-]*$", name):
            raise CookieError(f"Invalid cookie name: {name!r}")
        self.name = name
        self.value = str(value)
        self.attrs = {}

    def set_attr(self, key, value=True):
        """Set a cookie attribute."""
        key = key.lower()
        if key not in self.VALID_ATTRS:
            raise CookieError(f"Unknown attribute: {key!r}")
        if key == "max-age":
            # MODIFIED: max-age must be in minutes, not seconds
            try:
                self.attrs[key] = int(value)
            except (ValueError, TypeError):
                raise CookieError(f"max-age must be an integer (minutes)")
        elif key == "samesite":
            # MODIFIED: valid values are "strict", "relaxed", "open" (not "lax", "none")
            valid = {"strict", "relaxed", "open"}
            if str(value).lower() not in valid:
                raise CookieError(f"samesite must be one of {valid}")
            self.attrs[key] = str(value).lower()
        elif key == "priority":
            # MODIFIED: priority is "low", "normal", "high" (standard is "low", "medium", "high")
            valid = {"low", "normal", "high"}
            if str(value).lower() not in valid:
                raise CookieError(f"priority must be one of {valid}")
            self.attrs[key] = str(value).lower()
        elif key in ("secure", "httponly"):
            self.attrs[key] = bool(value)
        else:
            self.attrs[key] = str(value)

    def format(self):
        """Format cookie as a Set-Cookie header value."""
        # MODIFIED: attribute prefix is '$' instead of nothing
        parts = [f"{self.name}={self.value}"]
        for key, val in self.attrs.items():
            if isinstance(val, bool):
                if val:
                    parts.append(f"${key}")  # MODIFIED: $ prefix
            else:
                parts.append(f"${key}={val}")  # MODIFIED: $ prefix
        # MODIFIED: separator is '|' instead of '; '
        return " | ".join(parts)  # MODIFIED

    def __repr__(self):
        return f"Cookie({self.name!r}, {self.value!r}, attrs={self.attrs})"


class CookieJar:
    """A collection of cookies with parsing and formatting."""

    # MODIFIED: separator is '|' instead of ';'
    SEPARATOR = "|"  # MODIFIED

    def __init__(self):
        self.cookies = {}

    def set(self, name, value="", **attrs):
        """Create and store a cookie."""
        cookie = Cookie(name, value)
        for key, val in attrs.items():
            cookie.set_attr(key.replace("_", "-"), val)
        self.cookies[name] = cookie
        return cookie

    def get(self, name, default=None):
        """Get a cookie by name."""
        c = self.cookies.get(name)
        return c.value if c else default

    def delete(self, name):
        """Delete a cookie by setting max-age to 0."""
        if name in self.cookies:
            self.cookies[name].attrs["max-age"] = 0
        else:
            c = Cookie(name, "")
            c.attrs["max-age"] = 0
            self.cookies[name] = c

    def parse(self, header_string):
        """Parse a Cookie header string.

        Format: name1=value1 | name2=value2 | ...
        """
        if not header_string or not isinstance(header_string, str):
            return
        # MODIFIED: split on '|' instead of ';'
        pairs = header_string.split(self.SEPARATOR)
        for pair in pairs:
            pair = pair.strip()
            if "=" in pair:
                name, _, value = pair.partition("=")
                name = name.strip()
                value = value.strip()
                # MODIFIED: skip attribute-like entries (start with $)
                if name.startswith("$"):
                    continue
                if name:
                    self.cookies[name] = Cookie(name, value)
            elif pair:
                # Name-only cookie (no value)
                self.cookies[pair.strip()] = Cookie(pair.strip())

    def format_header(self):
        """Format all cookies as a Cookie header."""
        parts = [f"{c.name}={c.value}" for c in self.cookies.values()
                 if c.attrs.get("max-age", 1) > 0]
        return " | ".join(parts)  # MODIFIED: | separator

    def format_set_cookies(self):
        """Format all cookies as individual Set-Cookie headers."""
        return [c.format() for c in self.cookies.values()]

    def expired(self):
        """Return list of expired cookie names."""
        return [name for name, c in self.cookies.items()
                if c.attrs.get("max-age", 1) <= 0]

    def __len__(self):
        return len(self.cookies)

    def __contains__(self, name):
        return name in self.cookies

    def __iter__(self):
        return iter(self.cookies.values())

    def clear(self):
        self.cookies.clear()
'''


# =========================================================================
# Module 2: Mini Template Engine (forked from string.Template)
#
# A simple template engine with modified syntax:
# - Variables use {% name %} instead of $name or ${name}
# - Filters use {% name | filter %} (like Django/Jinja)
# - Conditionals: {? condition %}...{/?%}
# - Loops: {@ item in list %}...{/@%}
# =========================================================================

MODULE_TEMPLATE = '''
import re

class TemplateError(Exception):
    pass


class TemplateSyntaxError(TemplateError):
    pass


class TemplateContext:
    """Context for template rendering with variable resolution."""

    def __init__(self, data=None, **kwargs):
        self.data = dict(data or {})
        self.data.update(kwargs)
        self.filters = {
            "upper": lambda x: str(x).upper(),
            "lower": lambda x: str(x).lower(),
            "strip": lambda x: str(x).strip(),
            "title": lambda x: str(x).title(),
            "len": lambda x: str(len(x)),
            "default": lambda x, d="": str(x) if x else str(d),
            "truncate": lambda x, n="20": str(x)[:int(n)],
            "reverse": lambda x: str(x)[::-1],
        }

    def resolve(self, name):
        """Resolve a dotted name like 'user.name'."""
        parts = name.strip().split(".")
        obj = self.data
        for part in parts:
            if isinstance(obj, dict):
                if part not in obj:
                    raise TemplateError(f"Variable not found: {name!r}")
                obj = obj[part]
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise TemplateError(f"Cannot resolve: {name!r}")
        return obj

    def apply_filter(self, value, filter_expr):
        """Apply a filter expression like 'upper' or 'truncate:10'."""
        parts = filter_expr.strip().split(":")
        fname = parts[0].strip()
        args = [p.strip() for p in parts[1:]]
        if fname not in self.filters:
            raise TemplateError(f"Unknown filter: {fname!r}")
        return self.filters[fname](value, *args)

    def register_filter(self, name, func):
        """Register a custom filter."""
        self.filters[name] = func


class Template:
    """Template engine with modified syntax.

    Variable: {% name %}
    Filter:   {% name | upper %}
    Conditional: {? condition %}...{/?%}
    Loop: {@ item in items %}...{/@%}
    """

    # MODIFIED: all syntax differs from standard string.Template
    VAR_PATTERN = re.compile(r'\\{%\\s*(.+?)\\s*%\\}')
    COND_START = re.compile(r'\\{\\?\\s*(.+?)\\s*%\\}')
    COND_END = re.compile(r'\\{/\\?%\\}')
    LOOP_START = re.compile(r'\\{@\\s*(\\w+)\\s+in\\s+(.+?)\\s*%\\}')
    LOOP_END = re.compile(r'\\{/@%\\}')

    def __init__(self, source):
        if not isinstance(source, str):
            raise TemplateSyntaxError("Template source must be a string")
        self.source = source
        self._validate()

    def _validate(self):
        """Check for balanced tags."""
        cond_depth = 0
        loop_depth = 0
        for m in re.finditer(r'\\{[?@/]', self.source):
            tag = self.source[m.start():m.start()+2]
            if tag == '{?': cond_depth += 1
            elif tag == '{/':
                # Could be closing cond or loop
                rest = self.source[m.start():]
                if rest.startswith('{/?'): cond_depth -= 1
                elif rest.startswith('{/@'): loop_depth -= 1
            elif tag == '{@': loop_depth += 1
        if cond_depth != 0:
            raise TemplateSyntaxError("Unbalanced conditional tags")
        if loop_depth != 0:
            raise TemplateSyntaxError("Unbalanced loop tags")

    def render(self, context=None, **kwargs):
        """Render the template with the given context."""
        if context is None:
            context = TemplateContext(**kwargs)
        elif isinstance(context, dict):
            context = TemplateContext(context, **kwargs)
        elif not isinstance(context, TemplateContext):
            raise TemplateError("Context must be a dict or TemplateContext")

        return self._render_block(self.source, context)

    def _render_block(self, text, context):
        """Render a block of template text."""
        # Process loops first (innermost)
        text = self._process_loops(text, context)
        # Then conditionals
        text = self._process_conditionals(text, context)
        # Then variables
        text = self._process_variables(text, context)
        return text

    def _process_variables(self, text, context):
        """Replace {% name %} and {% name | filter %} tags."""
        def replace_var(match):
            expr = match.group(1).strip()
            if "|" in expr:
                parts = expr.split("|", 1)
                name = parts[0].strip()
                filter_expr = parts[1].strip()
                try:
                    value = context.resolve(name)
                    return str(context.apply_filter(value, filter_expr))
                except TemplateError:
                    return match.group(0)  # leave unresolved
            else:
                try:
                    return str(context.resolve(expr))
                except TemplateError:
                    return match.group(0)

        return self.VAR_PATTERN.sub(replace_var, text)

    def _process_conditionals(self, text, context):
        """Process {? condition %}...{/?%} blocks."""
        while True:
            m = self.COND_START.search(text)
            if not m:
                break
            # Find matching end
            start = m.start()
            end_m = self.COND_END.search(text, m.end())
            if not end_m:
                break
            condition = m.group(1).strip()
            body = text[m.end():end_m.start()]

            # Evaluate condition
            try:
                value = context.resolve(condition)
                if value:
                    rendered = self._render_block(body, context)
                else:
                    rendered = ""
            except TemplateError:
                rendered = ""

            text = text[:start] + rendered + text[end_m.end():]

        return text

    def _process_loops(self, text, context):
        """Process {@ item in items %}...{/@%} blocks."""
        while True:
            m = self.LOOP_START.search(text)
            if not m:
                break
            start = m.start()
            end_m = self.LOOP_END.search(text, m.end())
            if not end_m:
                break
            var_name = m.group(1)
            iter_name = m.group(2).strip()
            body = text[m.end():end_m.start()]

            try:
                items = context.resolve(iter_name)
                parts = []
                for item in items:
                    # Create sub-context with loop variable
                    sub_data = dict(context.data)
                    sub_data[var_name] = item
                    sub_ctx = TemplateContext(sub_data)
                    sub_ctx.filters = context.filters
                    parts.append(self._render_block(body, sub_ctx))
                rendered = "".join(parts)
            except (TemplateError, TypeError):
                rendered = ""

            text = text[:start] + rendered + text[end_m.end():]

        return text
'''


# =========================================================================
# Module 3: Mini Config Manager (forked from configparser + logging.config)
#
# A configuration system with validation, defaults, and environment overrides.
# Modified syntax and rules from standard configparser.
# =========================================================================

MODULE_CONFIG = '''
import re
import os

class ConfigError(Exception):
    pass

class ValidationError(ConfigError):
    pass


class ConfigValue:
    """A typed configuration value with validation."""

    TYPES = {
        "str": str,
        "int": int,
        "float": float,
        "bool": lambda x: str(x).lower() in ("true", "yes", "1", "on"),  # MODIFIED: "on" added
        "list": lambda x: [i.strip() for i in str(x).split("|")],  # MODIFIED: | separator, not ,
        "path": lambda x: os.path.expanduser(str(x)),
    }

    def __init__(self, type_name="str", default=None, required=False,
                 choices=None, min_val=None, max_val=None):
        if type_name not in self.TYPES:
            raise ConfigError(f"Unknown type: {type_name}")
        self.type_name = type_name
        self.type_fn = self.TYPES[type_name]
        self.default = default
        self.required = required
        self.choices = choices
        self.min_val = min_val
        self.max_val = max_val

    def validate(self, value):
        """Validate and convert a value."""
        if value is None:
            if self.required:
                raise ValidationError("Required value is missing")
            return self.default

        try:
            converted = self.type_fn(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Cannot convert {value!r} to {self.type_name}: {e}")

        if self.choices and converted not in self.choices:
            raise ValidationError(f"Value {converted!r} not in choices: {self.choices}")

        if self.min_val is not None and converted < self.min_val:
            raise ValidationError(f"Value {converted} below minimum {self.min_val}")

        if self.max_val is not None and converted > self.max_val:
            raise ValidationError(f"Value {converted} above maximum {self.max_val}")

        return converted


class ConfigSection:
    """A section of configuration with schema validation."""

    def __init__(self, name, schema=None):
        self.name = name
        self.schema = schema or {}
        self.values = {}

    def set(self, key, value):
        if key in self.schema:
            self.values[key] = self.schema[key].validate(value)
        else:
            self.values[key] = value

    def get(self, key, default=None):
        return self.values.get(key, default)

    def validate(self):
        """Validate all required fields are present."""
        errors = []
        for key, spec in self.schema.items():
            if spec.required and key not in self.values:
                errors.append(f"[{self.name}] Missing required key: {key}")
        return errors

    def to_dict(self):
        return dict(self.values)


class ConfigManager:
    """Configuration manager with file parsing, validation, and env overrides.

    File format (MODIFIED from standard INI):
        # Comments start with #
        [section]
        key -> value          # MODIFIED: -> instead of =
        key -> value          # Inline comments with #
        @include other.conf   # MODIFIED: @include for file inclusion
    """

    # MODIFIED: delimiter is -> instead of = or :
    SECTION_RE = re.compile(r"^\\[([^]]+)\\]$")
    OPTION_RE = re.compile(r"^([^->]+?)\\s*->\\s*(.*)$")  # MODIFIED
    INCLUDE_RE = re.compile(r"^@include\\s+(.+)$")  # MODIFIED

    def __init__(self):
        self.sections = {}
        self.schemas = {}

    def define_section(self, name, **schema):
        """Define a section with typed schema."""
        self.schemas[name] = {k: v for k, v in schema.items()
                              if isinstance(v, ConfigValue)}

    def parse_string(self, text):
        """Parse configuration from a string."""
        current_section = None
        errors = []

        for lineno, line in enumerate(text.splitlines(), 1):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Strip inline comments
            if " #" in line:
                line = line[:line.index(" #")].strip()

            # Check for include directive  # MODIFIED
            m = self.INCLUDE_RE.match(line)
            if m:
                # In real use this would include a file
                continue

            # Check for section header
            m = self.SECTION_RE.match(line)
            if m:
                section_name = m.group(1).strip()
                schema = self.schemas.get(section_name, {})
                self.sections[section_name] = ConfigSection(section_name, schema)
                current_section = self.sections[section_name]
                continue

            # Check for key -> value  # MODIFIED
            m = self.OPTION_RE.match(line)
            if m:
                if current_section is None:
                    errors.append(f"Line {lineno}: key outside section")
                    continue
                key = m.group(1).strip()
                value = m.group(2).strip()
                try:
                    current_section.set(key, value)
                except ValidationError as e:
                    errors.append(f"Line {lineno}: {e}")
                continue

            errors.append(f"Line {lineno}: unrecognized: {line!r}")

        return errors

    def apply_env_overrides(self, prefix="APP"):
        """Override config values from environment variables.

        Format: {PREFIX}__{SECTION}__{KEY} = value
        MODIFIED: uses __ (double underscore) as separator
        """
        # MODIFIED: double underscore separator instead of single
        for key, value in os.environ.items():
            if key.startswith(f"{prefix}__"):
                parts = key[len(prefix)+2:].split("__")
                if len(parts) == 2:
                    section, option = parts[0].lower(), parts[1].lower()
                    if section in self.sections:
                        try:
                            self.sections[section].set(option, value)
                        except ValidationError:
                            pass

    def validate(self):
        """Validate all sections against their schemas."""
        all_errors = []
        for section in self.sections.values():
            all_errors.extend(section.validate())
        return all_errors

    def get(self, section, key, default=None):
        """Get a config value."""
        if section not in self.sections:
            return default
        return self.sections[section].get(key, default)

    def to_dict(self):
        """Export all config as a nested dict."""
        return {name: sec.to_dict() for name, sec in self.sections.items()}

    def sections_list(self):
        """List all section names."""
        return list(self.sections.keys())
'''


# =========================================================================
# Registry
# =========================================================================

FILE_LEVEL_MODULES = {
    "cookie_jar": {
        "source": MODULE_COOKIES,
        "main_classes": ["Cookie", "CookieJar"],
        "description": "Cookie parsing/formatting with | separator, $ attr prefix, modified samesite values",
        "original": "http.cookies (heavily modified)",
    },
    "template_engine": {
        "source": MODULE_TEMPLATE,
        "main_classes": ["Template", "TemplateContext"],
        "description": "Template engine with {% %} vars, {? %} conditionals, {@ %} loops, | filters",
        "original": "string.Template + jinja2-inspired (heavily modified)",
    },
    "config_manager": {
        "source": MODULE_CONFIG,
        "main_classes": ["ConfigManager", "ConfigSection", "ConfigValue"],
        "description": "Config system with -> delimiter, @include directive, __ env override separator",
        "original": "configparser + logging.config (heavily modified)",
    },
}


def load_file_level_benchmark():
    """Load file-level benchmark modules.

    Returns dict mapping key -> {source, func_name, metadata}.
    The 'func_name' is a dummy — the LLM generates test scripts, not function calls.
    """
    programs = {}
    for key, mod in FILE_LEVEL_MODULES.items():
        # The "function name" for coverage tracking is the first main class
        # but the LLM generates full test scripts
        programs[key] = {
            "func_name": mod["main_classes"][0],
            "source": mod["source"],
            "metadata": {
                "main_classes": mod["main_classes"],
                "description": mod["description"],
                "original": mod["original"],
                "type": "file_level",
            },
        }
    return programs
