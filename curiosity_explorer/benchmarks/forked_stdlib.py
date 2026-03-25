"""Forked stdlib functions with modified logic.

Each function is extracted from Python's stdlib and modified so the LLM's
trained knowledge is partially wrong. The LLM recognizes the pattern
(shlex tokenizer, config parser, etc.) but the specific rules have changed.

Modifications are marked with # MODIFIED comments.
"""

import re
import io


# =========================================================================
# 1. Forked shlex.read_token — state machine tokenizer
#
# MODIFICATIONS:
# - '#' is no longer a comment char; '@' is
# - Single quotes don't work for quoting; backticks do
# - Escape char changed from '\' to '^'
# - Punctuation chars changed
# - Wordchars expanded to include '.'
# =========================================================================

def tokenize_modified(s):
    """Tokenize a string using modified shell-like rules.

    Returns a list of tokens. Raises ValueError on unterminated quotes.

    Differences from standard shlex:
    - Comment character is '@' (not '#')
    - Quoting uses backticks ` and double quotes " (not single quotes)
    - Escape character is '^' (not backslash)
    - Word characters include '.' and '-'
    - Punctuation is '|', '&', '>'
    """
    # MODIFIED: different special characters
    commenters = '@'          # MODIFIED: was '#'
    wordchars = ('abcdfeghijklmnopqrstuvwxyz'
                 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-')  # MODIFIED: added .-
    whitespace = ' \t\n\r'
    quotes = '`"'             # MODIFIED: backtick instead of single quote
    escape = '^'              # MODIFIED: was '\\'
    punctuation_chars = '|&>'  # MODIFIED: was ';'

    tokens = []
    token = ''
    state = ' '  # start in whitespace state
    quoted = False
    escape_state = ' '

    stream = io.StringIO(s)

    while True:
        nextchar = stream.read(1)
        if not nextchar:
            break

        if state is None:
            # past end of input
            break
        elif state == ' ':
            # whitespace state
            if not nextchar:
                state = None
                break
            elif nextchar in whitespace:
                if token:
                    tokens.append(token)
                    token = ''
            elif nextchar in commenters:
                # MODIFIED: '@' starts a comment (rest of line ignored)
                stream.readline()
                break
            elif nextchar in wordchars:
                token = nextchar
                state = 'a'
            elif nextchar in punctuation_chars:
                tokens.append(nextchar)
            elif nextchar in quotes:
                state = nextchar
                quoted = True
                token = ''
            elif nextchar == escape:
                # MODIFIED: '^' escapes next char
                escape_state = 'a'
                state = escape
            else:
                token = nextchar
                tokens.append(token)
                token = ''
        elif state in quotes:
            # in quoted string
            quoted = True
            if not nextchar:
                raise ValueError("unterminated quote")
            if nextchar == state:
                # end of quote
                state = 'a'
            elif nextchar == escape and state == '"':
                # MODIFIED: '^' escapes inside double quotes only
                escape_state = state
                state = escape
            else:
                token += nextchar
        elif state == escape:
            if not nextchar:
                raise ValueError("escape at end of input")
            token += nextchar
            state = escape_state
        elif state == 'a':
            # in a word
            if not nextchar:
                state = None
            elif nextchar in whitespace:
                state = ' '
                if token or quoted:
                    tokens.append(token)
                    token = ''
                    quoted = False
            elif nextchar in commenters:
                stream.readline()
                if token or quoted:
                    tokens.append(token)
                    token = ''
                break
            elif nextchar in wordchars:
                token += nextchar
            elif nextchar in punctuation_chars:
                if token or quoted:
                    tokens.append(token)
                    token = ''
                    quoted = False
                tokens.append(nextchar)
                state = ' '
            elif nextchar in quotes:
                state = nextchar
            elif nextchar == escape:
                escape_state = 'a'
                state = escape
            else:
                token += nextchar

    if token or quoted:
        tokens.append(token)

    return tokens


# =========================================================================
# 2. Forked configparser._read — INI-like config parser
#
# MODIFICATIONS:
# - Section delimiters changed from [] to <>
# - Comment prefix changed from '#'/';' to '//'
# - Key-value delimiter is '->' instead of '=' or ':'
# - Continuation lines use '+' prefix instead of whitespace indentation
# - Inline comments use '%%' instead of '#' or ';'
# =========================================================================

def parse_config_modified(text):
    """Parse a modified INI-like config format.

    Format:
        <section_name>
        key -> value
        +continuation of value
        // this is a comment
        key2 -> value2 %% inline comment

    Returns dict of {section: {key: value}}.
    Raises ValueError on errors.
    """
    # MODIFIED: different patterns
    SECTCRE = re.compile(r'\<(?P<header>.+)\>')           # MODIFIED: <> not []
    OPTCRE = re.compile(r'(?P<option>[^->]+?)\s*->\s*(?P<value>.*)$')  # MODIFIED: -> not = or :
    COMMENT_PREFIX = '//'                                   # MODIFIED: was # or ;
    INLINE_COMMENT = '%%'                                   # MODIFIED: was # or ;
    CONTINUATION = '+'                                      # MODIFIED: was whitespace

    result = {}
    cursect = None
    curkey = None
    curval = []
    errors = []
    lineno = 0

    for lineno, line in enumerate(text.splitlines(), 1):
        # Strip inline comments
        if INLINE_COMMENT in line:
            idx = line.index(INLINE_COMMENT)
            # Only strip if not inside a value
            if cursect is not None and curkey is not None and line.startswith(CONTINUATION):
                pass  # don't strip from continuation lines
            else:
                line = line[:idx]

        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped:
            # Save accumulated multiline value
            if curkey is not None and curval:
                cursect[curkey] = '\n'.join(curval)
                curkey = None
                curval = []
            continue

        if stripped.startswith(COMMENT_PREFIX):
            continue

        # Check for continuation  # MODIFIED: '+' prefix, not indentation
        if stripped.startswith(CONTINUATION) and curkey is not None:
            curval.append(stripped[1:].strip())
            continue

        # Save previous key's value
        if curkey is not None and curval:
            cursect[curkey] = '\n'.join(curval)
            curkey = None
            curval = []

        # Check for section header  # MODIFIED: <> not []
        mo = SECTCRE.match(stripped)
        if mo:
            sectname = mo.group('header').strip()
            if sectname in result:
                errors.append(f"line {lineno}: duplicate section '{sectname}'")
            else:
                result[sectname] = {}
                cursect = result[sectname]
            continue

        # Check for key-value pair  # MODIFIED: -> delimiter
        mo = OPTCRE.match(stripped)
        if mo:
            if cursect is None:
                errors.append(f"line {lineno}: key outside section")
                continue
            curkey = mo.group('option').strip()
            value = mo.group('value').strip()
            if curkey in cursect:
                errors.append(f"line {lineno}: duplicate key '{curkey}'")
            curval = [value] if value else []
            continue

        # Unrecognized line
        errors.append(f"line {lineno}: parsing error: {stripped!r}")

    # Save last key
    if curkey is not None and curval:
        cursect[curkey] = '\n'.join(curval)

    if errors:
        return {"errors": errors, "partial": result}

    return {"config": result}


# =========================================================================
# 3. Forked html.parser — tag parser with modified rules
#
# MODIFICATIONS:
# - Tags use {} instead of <>
# - Attributes use : instead of =
# - Comments are {-- --} instead of <!-- -->
# - Self-closing tags end with /} instead of />
# - Entity references use @ instead of &
# =========================================================================

def parse_markup_modified(text):
    """Parse a modified markup format using {} for tags.

    Format:
        {tag attr:value attr2:value2}content{/tag}
        {-- this is a comment --}
        {img src:url /}
        Use @amp; for special characters.

    Returns list of events: ('start', tag, attrs), ('end', tag),
    ('data', text), ('comment', text), ('entity', name).
    """
    events = []
    i = 0
    n = len(text)

    while i < n:
        # Look for tag start or entity
        if text[i] == '{':
            # Check for comment  # MODIFIED: {-- --} not <!-- -->
            if text[i:i+3] == '{--':
                j = text.find('--}', i + 3)
                if j < 0:
                    events.append(('error', 'unterminated comment'))
                    break
                events.append(('comment', text[i+3:j].strip()))
                i = j + 3
                continue

            # Check for end tag  # MODIFIED: {/tag} not </tag>
            if i + 1 < n and text[i+1] == '/':
                j = text.find('}', i + 2)
                if j < 0:
                    events.append(('error', 'unterminated end tag'))
                    break
                tag = text[i+2:j].strip()
                events.append(('end', tag))
                i = j + 1
                continue

            # Start tag or self-closing
            j = text.find('}', i + 1)
            if j < 0:
                events.append(('error', 'unterminated start tag'))
                break

            tag_content = text[i+1:j]

            # Check self-closing  # MODIFIED: /} not />
            self_closing = False
            if tag_content.endswith('/'):
                self_closing = True
                tag_content = tag_content[:-1].strip()

            # Parse tag name and attributes
            parts = tag_content.split(None, 1)
            if not parts:
                events.append(('error', 'empty tag'))
                i = j + 1
                continue

            tag = parts[0]
            attrs = {}
            if len(parts) > 1:
                # Parse attributes  # MODIFIED: attr:value not attr=value
                attr_str = parts[1]
                for attr_match in re.finditer(
                    r'(\w[\w-]*)\s*:\s*(?:"([^"]*)"|\'([^\']*)\'|(\S+))',
                    attr_str
                ):
                    key = attr_match.group(1)
                    val = (attr_match.group(2) or attr_match.group(3)
                           or attr_match.group(4) or '')
                    attrs[key] = val

            events.append(('start', tag, attrs, self_closing))
            if self_closing:
                events.append(('end', tag))
            i = j + 1

        elif text[i] == '@':
            # Entity reference  # MODIFIED: @ not &
            j = text.find(';', i + 1)
            if j < 0 or j - i > 20:
                events.append(('data', text[i]))
                i += 1
            else:
                entity = text[i+1:j]
                if entity.startswith('#'):
                    # Numeric entity
                    try:
                        if entity[1] == 'x':
                            char = chr(int(entity[2:], 16))
                        else:
                            char = chr(int(entity[1:]))
                        events.append(('data', char))
                    except (ValueError, OverflowError):
                        events.append(('entity_error', entity))
                else:
                    # Named entity
                    entity_map = {
                        'amp': '@', 'lt': '{', 'gt': '}',  # MODIFIED: different chars
                        'quot': '"', 'apos': "'",
                        'nbsp': '\xa0', 'copy': '\xa9',
                    }
                    if entity in entity_map:
                        events.append(('data', entity_map[entity]))
                    else:
                        events.append(('entity_error', entity))
                i = j + 1
        else:
            # Regular text
            j = i
            while j < n and text[j] not in ('{', '@'):
                j += 1
            if j > i:
                events.append(('data', text[i:j]))
            i = j

    return events


# =========================================================================
# 4. Forked csv.Sniffer._guess_delimiter — with modified preferences
#
# MODIFICATIONS:
# - Preferred delimiters changed: '|', '^', '~', '\t', ',' (not ',', '\t', ';', ' ', ':')
# - Consistency threshold starts at 0.8 (not 1.0)
# - Minimum frequency threshold is 3 (not 0)
# - Returns a dict with metadata instead of just the delimiter
# =========================================================================

def guess_delimiter_modified(data, candidates=None):
    """Guess the delimiter of a CSV-like text.

    Returns dict with 'delimiter', 'confidence', 'frequency' keys.
    Returns {'delimiter': None} if no delimiter found.

    Preferred delimiters: |, ^, ~, tab, comma (in that order).
    """
    if not isinstance(data, str) or not data.strip():
        return {'delimiter': None, 'error': 'empty_input'}

    if candidates is None:
        candidates = [',', '\t', '|', '^', '~', ';', ':']  # MODIFIED: added ^, ~

    # MODIFIED: preferred order is different from standard csv module
    preferred = ['|', '^', '~', '\t', ',']  # MODIFIED: was [',', '\t', ';', ' ', ':']

    lines = data.split('\n')
    lines = [l for l in lines if l.strip()]

    if len(lines) < 2:
        return {'delimiter': None, 'error': 'need_multiple_lines'}

    # Build frequency table
    freqs = {}
    for line in lines:
        for char in candidates:
            cnt = line.count(char)
            if cnt > 0:
                freqs.setdefault(char, []).append(cnt)

    if not freqs:
        return {'delimiter': None, 'error': 'no_candidates_found'}

    # Find consistent delimiters
    # MODIFIED: threshold starts at 0.8 (not 1.0)
    threshold = 0.8  # MODIFIED: was 1.0
    consistent = {}

    for char, counts in freqs.items():
        if len(counts) < len(lines) * threshold:
            continue
        # MODIFIED: minimum frequency of 3
        if max(counts) < 3:  # MODIFIED: was 0 (any frequency)
            continue
        # Check consistency: mode should cover most lines
        from collections import Counter
        mode_count = Counter(counts).most_common(1)[0][1]
        consistency = mode_count / len(counts)
        if consistency >= threshold:
            total_freq = sum(counts)
            consistent[char] = {
                'frequency': total_freq,
                'consistency': round(consistency, 3),
                'per_line': Counter(counts).most_common(1)[0][0],
            }

    if not consistent:
        return {'delimiter': None, 'error': 'no_consistent_delimiter'}

    if len(consistent) == 1:
        delim = list(consistent.keys())[0]
        return {'delimiter': delim, 'confidence': 'high', **consistent[delim]}

    # Multiple candidates: use preference order  # MODIFIED: different preference
    for p in preferred:
        if p in consistent:
            return {'delimiter': p, 'confidence': 'preferred', **consistent[p]}

    # Fallback: highest frequency
    best = max(consistent.items(), key=lambda x: x[1]['frequency'])
    return {'delimiter': best[0], 'confidence': 'frequency', **best[1]}


# =========================================================================
# Registry
# =========================================================================

FORKED_PROGRAMS = {
    "shlex_modified": {
        "func_name": "tokenize_modified",
        "description": "Shell tokenizer with changed special chars (@=comment, `=quote, ^=escape)",
        "original": "shlex.shlex.read_token",
    },
    "configparser_modified": {
        "func_name": "parse_config_modified",
        "description": "INI parser with changed syntax (<> sections, -> delimiter, // comments)",
        "original": "configparser.ConfigParser._read",
    },
    "htmlparser_modified": {
        "func_name": "parse_markup_modified",
        "description": "Markup parser with {} tags, : attrs, @entities",
        "original": "html.parser.HTMLParser.goahead",
    },
    "csv_modified": {
        "func_name": "guess_delimiter_modified",
        "description": "CSV delimiter guesser with changed preferences and thresholds",
        "original": "csv.Sniffer._guess_delimiter",
    },
}


_IMPORT_HEADER = """\
import re
import io
from collections import Counter
"""


def load_forked_programs():
    """Load forked programs with source code for use with CoverageRunner."""
    import inspect
    import textwrap

    source_map = {
        "tokenize_modified": tokenize_modified,
        "parse_config_modified": parse_config_modified,
        "parse_markup_modified": parse_markup_modified,
        "guess_delimiter_modified": guess_delimiter_modified,
    }

    programs = {}
    for key, prog in FORKED_PROGRAMS.items():
        func = source_map[prog["func_name"]]
        source = textwrap.dedent(inspect.getsource(func))
        # Prepend imports so the function works in the sandbox
        source = _IMPORT_HEADER + source
        programs[key] = {
            "func_name": prog["func_name"],
            "source": source,
            "metadata": {
                "original": prog["original"],
                "description": prog["description"],
            },
        }
    return programs
