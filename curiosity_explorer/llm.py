"""LLM client with response caching, batch support, multi-model, and cost tracking."""

import hashlib
import logging
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import config

# Transient API failures (rate limits, server hiccups, timeouts) that are worth
# retrying with backoff rather than silently returning "" — an empty response
# parses to a zero score and can corrupt Q-value selection. Matched as substrings
# of the exception text (provider-agnostic across the OpenAI-compatible clients).
_RETRYABLE = ("429", "rate", "limit", "quota", "resource_exhausted", "overloaded",
              "timeout", "timed out", "temporarily", "503", "502", "500",
              "unavailable", "connection")
_MAX_RETRIES = 8      # attempts after the first try
_BACKOFF_CAP = 120.0  # seconds; rate limits can need minute-scale waits
# Errors that will not fix themselves: an exhausted account budget, a revoked
# key. Retrying is pointless and, worse, the caller keeps going with empty
# responses — a GLM run once "completed" 34/93 targets on nothing but "" after
# the Mistral budget ran out. Abort the run instead.
_FATAL = ("budget_exhausted", "budget exhausted", "insufficient_quota",
          "invalid_api_key", "account is not active", "402")
_FATAL_STREAK = 20    # consecutive empty responses that abort the run
_BACKOFF_BASE = 2.0   # seconds; exponential with full jitter, capped at 60s

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai")
    sys.exit(1)

log = logging.getLogger(__name__)


def _make_client():
    """Create the OpenAI-compatible client for the configured model."""
    return _client_for_model(config.MODEL)


def _wire_name(model: str) -> str:
    """The id the provider expects (our ids carry a routing prefix)."""
    return model[len("openrouter/"):] if model.startswith("openrouter/") else model


def _client_for_model(model: str) -> OpenAI:
    """Create an OpenAI-compatible client for a specific model."""
    if model.startswith("openrouter/"):
        return OpenAI(base_url=config.OPENROUTER_API_BASE,
                      api_key=config.OPENROUTER_API_KEY)
    elif model.startswith("accounts/fireworks/"):
        return OpenAI(base_url=config.FIREWORKS_API_BASE,
                      api_key=config.FIREWORKS_API_KEY)
    elif model.startswith("mistral") or "glm" in model.lower():
        # Mistral's API also serves Z.ai GLM models (e.g. "zai-glm-5-3").
        return OpenAI(base_url=config.MISTRAL_API_BASE,
                      api_key=config.MISTRAL_API_KEY)
    elif model.startswith("gpt"):
        return OpenAI(api_key=config.OPENAI_API_KEY)
    elif model.startswith("google/") or "gemma" in model.lower():
        return OpenAI(base_url=config.VLLM_API_BASE,
                      api_key=config.VLLM_API_KEY)
    else:
        return OpenAI(base_url=config.GEMINI_API_BASE,
                      api_key=config.GEMINI_API_KEY)


client: OpenAI = _make_client()

# Per-model client cache for ensemble calls
_model_clients: dict[str, OpenAI] = {}


def _get_client(model: str) -> OpenAI:
    """Get or create a cached client for a model."""
    if model == config.MODEL:
        return client
    if model not in _model_clients:
        _model_clients[model] = _client_for_model(model)
    return _model_clients[model]


def reconfigure():
    """Reinitialize the client after config.MODEL changes."""
    global client
    client = _make_client()

# Per-provider concurrency caps. Our own fan-out multiplies fast: each worker
# issues K generation calls plus K scoring calls in parallel, so N workers can
# mean ~6N simultaneous requests. Mistral answers that with 429 storms (the
# rebuttal abandoned mistral-large after 4,340 of them, correlated with two
# host crashes), and retry inflation then dominates the run. A semaphore caps
# in-flight requests per provider regardless of how many workers exist.
_PROVIDER_LIMITS = {
    "mistral": int(os.environ.get("MISTRAL_MAX_CONCURRENCY", "12")),
}
class AdaptiveLimiter:
    """In-flight cap that shrinks on rate limits and recovers on success.

    A fixed cap has to be guessed: too high and the provider answers with 429s
    (retry inflation, no extra goodput, since the real limit is tokens/minute);
    too low and we leave throughput on the table. This is AIMD, as in TCP
    congestion control: multiplicative decrease when the provider pushes back,
    additive increase after a run of clean responses.
    """

    def __init__(self, limit, min_limit=2, max_limit=None):
        self.limit = limit
        self.min_limit = min_limit
        self.max_limit = max_limit or max(limit * 2, limit + 8)
        self._in_flight = 0
        self._ok_streak = 0
        self._last_cut = 0.0
        self._cv = threading.Condition()

    def acquire(self):
        with self._cv:
            while self._in_flight >= self.limit:
                self._cv.wait(timeout=5)
            self._in_flight += 1

    def release(self):
        with self._cv:
            self._in_flight -= 1
            self._cv.notify()

    def on_rate_limited(self):
        with self._cv:
            now = time.time()
            # one cut per cooldown: a burst of 429s is one signal, not many
            if now - self._last_cut < 10.0:
                return
            new = max(self.min_limit, int(self.limit * 0.7))
            if new < self.limit:
                log.warning(f"rate limited: concurrency {self.limit} -> {new}")
                self.limit = new
            self._last_cut = now
            self._ok_streak = 0

    def on_success(self):
        with self._cv:
            self._ok_streak += 1
            if self._ok_streak >= 50 and self.limit < self.max_limit:
                self.limit += 1
                self._ok_streak = 0
                self._cv.notify()


_provider_sems: dict[str, AdaptiveLimiter] = {}
_sem_lock = threading.Lock()


def _provider_of(model: str) -> str:
    if model.startswith("openrouter/"):
        return "openrouter"
    if model.startswith("mistral") or "glm" in model.lower():
        return "mistral"
    if model.startswith("gpt"):
        return "openai"
    if model.startswith("accounts/fireworks/"):
        return "fireworks"
    if model.startswith("google/") or "gemma" in model.lower():
        return "vllm"
    return "gemini"


def _semaphore_for(model: str):
    """Concurrency gate for this model's provider, or None if uncapped."""
    prov = _provider_of(model)
    limit = _PROVIDER_LIMITS.get(prov)
    if not limit:
        return None
    with _sem_lock:
        if prov not in _provider_sems:
            _provider_sems[prov] = AdaptiveLimiter(limit)
    return _provider_sems[prov]


class FatalAPIError(RuntimeError):
    """Raised when the provider reports a condition retrying cannot fix."""


_empty_streak = 0

# Response cache: key -> response string
_cache: dict[str, str] = {}
_cache_hits = 0
_cache_misses = 0

# Token / cost accounting — tracks per-model usage
_total_input_tokens = 0
_total_output_tokens = 0
_total_api_calls = 0
_per_model_usage: dict[str, dict] = {}


def _extract_text(msg) -> str:
    """Return the visible answer text of a chat completion message.

    Mistral's API returns reasoning models' (e.g. GLM) content as a list of
    chunks — {"type": "thinking", ...} followed by {"type": "text", "text": ...};
    only the text chunks are the answer.
    """
    content = msg.content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, dict):
                ctype, ctext = chunk.get("type"), chunk.get("text")
            else:
                ctype, ctext = getattr(chunk, "type", None), getattr(chunk, "text", None)
            if ctype == "text" and ctext:
                parts.append(ctext)
        content = "".join(parts)
    return content or getattr(msg, "reasoning_content", None) or ""


def _request_params(model: str, max_tokens: int) -> dict:
    """Token-limit and per-model extra params for a chat completion call."""
    params = dict(config.MODEL_EXTRA_PARAMS.get(model)
                  or config.MODEL_EXTRA_PARAMS.get(_wire_name(model), {}))
    if "reasoning_effort" in params:
        max_tokens += config.THINKING_TOKEN_ALLOWANCE
    # OpenAI gpt-5+ models require max_completion_tokens
    if model.startswith("gpt"):
        params["max_completion_tokens"] = max_tokens
    else:
        params["max_tokens"] = max_tokens
    return params


def _cache_key(prompt: str, temperature: float, max_tokens: int,
               model: str = "") -> str:
    h = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    return f"{model or config.MODEL}:{h}:{temperature}:{max_tokens}"


def _track_usage(model: str, input_toks: int, output_toks: int):
    """Track token usage globally and per-model."""
    global _total_input_tokens, _total_output_tokens, _total_api_calls
    _total_input_tokens += input_toks
    _total_output_tokens += output_toks
    _total_api_calls += 1

    if model not in _per_model_usage:
        _per_model_usage[model] = {"input_tokens": 0, "output_tokens": 0,
                                   "api_calls": 0}
    _per_model_usage[model]["input_tokens"] += input_toks
    _per_model_usage[model]["output_tokens"] += output_toks
    _per_model_usage[model]["api_calls"] += 1


def llm_generate(prompt: str, temperature: float = 0.7, max_tokens: int = 256,
                 use_cache: bool = True) -> str:
    """Call the LLM API (default model) with optional caching."""
    return generate_with_model(config.MODEL, prompt, temperature, max_tokens,
                               use_cache)


def generate_with_model(model: str, prompt: str, temperature: float = 0.7,
                        max_tokens: int = 256,
                        use_cache: bool = True) -> str:
    """Call a specific model's API with optional caching."""
    global _cache_hits, _cache_misses

    if use_cache and temperature == 0:
        key = _cache_key(prompt, temperature, max_tokens, model)
        if key in _cache:
            _cache_hits += 1
            return _cache[key]

    tok_param = _request_params(model, max_tokens)

    result = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            cli = _get_client(model)
            sem = _semaphore_for(model)
            if sem:
                sem.acquire()
            try:
                response = cli.chat.completions.create(
                    model=_wire_name(model),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    **tok_param,
                )
            finally:
                if sem:
                    sem.release()
            if sem:
                sem.on_success()
            msg = response.choices[0].message
            result = _extract_text(msg).strip()

            # Track token usage
            input_toks = (response.usage.prompt_tokens or 0) if response.usage else 0
            output_toks = (response.usage.completion_tokens or 0) if response.usage else 0
            _track_usage(model, input_toks, output_toks)
            break

        except Exception as e:
            if any(t in str(e).lower() for t in _FATAL):
                raise FatalAPIError(f"{model}: {e}") from e
            transient = any(t in str(e).lower() for t in _RETRYABLE)
            sem = _semaphore_for(model)
            if sem and ("429" in str(e) or "rate" in str(e).lower()):
                sem.on_rate_limited()
            if transient and attempt < _MAX_RETRIES:
                # Exponential backoff with full jitter, capped at 60s.
                delay = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_CAP)
                delay = random.uniform(0, delay)
                log.warning(f"Transient API error ({model}), retry "
                            f"{attempt+1}/{_MAX_RETRIES} in {delay:.1f}s: {e}")
                time.sleep(delay)
                continue
            # Exhausted retries on a transient error: never hand back "" —
            # it parses as a zero score and silently corrupts results (2 such
            # calls slipped through a GLM rate-limit storm before this guard).
            if transient:
                raise FatalAPIError(
                    f"{model}: still failing after {_MAX_RETRIES} retries "
                    f"(reduce concurrency): {e}") from e
            log.warning(f"API error ({model}): {e}")
            return ""

    global _empty_streak
    if result:
        _empty_streak = 0
    else:
        _empty_streak += 1
        if _empty_streak >= _FATAL_STREAK:
            raise FatalAPIError(
                f"{model}: {_empty_streak} consecutive empty responses — "
                f"aborting rather than filling results with blanks")

    if result is None:
        return ""

    if use_cache and temperature == 0:
        _cache[key] = result
        _cache_misses += 1

    return result


def batch_generate(prompts: list[str], temperature: float = 0.7,
                   max_tokens: int = 256, max_workers: int = 16) -> list[str]:
    """Generate responses for multiple prompts in parallel (default model)."""
    return batch_generate_with_model(config.MODEL, prompts, temperature,
                                     max_tokens, max_workers)


def batch_generate_with_model(model: str, prompts: list[str],
                              temperature: float = 0.7, max_tokens: int = 256,
                              max_workers: int = 16) -> list[str]:
    """Generate responses for multiple prompts in parallel using a specific model."""
    results = [""] * len(prompts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(generate_with_model, model, p, temperature,
                            max_tokens, False): i
            for i, p in enumerate(prompts)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                log.warning(f"Batch error for prompt {idx}: {e}")

    return results


def generate_with_logprobs(model: str, prompt: str, temperature: float = 0.3,
                           max_tokens: int = 100,
                           top_logprobs: int = 5) -> dict | None:
    """Generate a response with token-level logprobs.

    Returns dict with keys: text, token_logprobs (list of dicts with
    token, logprob, top_logprobs).
    Returns None on failure.
    """
    try:
        cli = _get_client(model)
        response = cli.chat.completions.create(
            model=_wire_name(model),
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_logprobs,
        )

        msg = response.choices[0].message
        text = _extract_text(msg).strip()

        # Track usage
        input_toks = (response.usage.prompt_tokens or 0) if response.usage else 0
        output_toks = (response.usage.completion_tokens or 0) if response.usage else 0
        _track_usage(model, input_toks, output_toks)

        # Extract logprobs, filtering out special/control tokens
        token_data = []
        logprobs_content = response.choices[0].logprobs
        if logprobs_content and logprobs_content.content:
            for token_info in logprobs_content.content:
                # Skip special tokens (reasoning model control tokens)
                if token_info.token.startswith("<|") and token_info.token.endswith("|>"):
                    continue
                entry = {
                    "token": token_info.token,
                    "logprob": token_info.logprob,
                    "top_logprobs": {
                        t.token: t.logprob
                        for t in (token_info.top_logprobs or [])
                        if not (t.token.startswith("<|") and t.token.endswith("|>"))
                    },
                }
                token_data.append(entry)

        return {"text": text, "token_logprobs": token_data}

    except Exception as e:
        log.warning(f"Logprobs API error ({model}): {e}")
        return None


def get_cost() -> dict:
    """Return current token usage and estimated cost (all models combined)."""
    total_cost = 0.0
    per_model_costs = {}
    for model, usage in _per_model_usage.items():
        pricing = config.MODEL_PRICING.get(model, {"input": 0, "output": 0})
        ic = usage["input_tokens"] / 1_000_000 * pricing["input"]
        oc = usage["output_tokens"] / 1_000_000 * pricing["output"]
        per_model_costs[model] = {
            "api_calls": usage["api_calls"],
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "cost_usd": round(ic + oc, 6),
        }
        total_cost += ic + oc

    return {
        "model": config.MODEL,
        "api_calls": _total_api_calls,
        "input_tokens": _total_input_tokens,
        "output_tokens": _total_output_tokens,
        "total_tokens": _total_input_tokens + _total_output_tokens,
        "total_cost_usd": round(total_cost, 6),
        "per_model": per_model_costs,
    }


def cache_stats() -> dict:
    return {"hits": _cache_hits, "misses": _cache_misses, "size": len(_cache)}


def clear_cache():
    global _cache, _cache_hits, _cache_misses
    _cache = {}
    _cache_hits = 0
    _cache_misses = 0


def reset_cost():
    global _total_input_tokens, _total_output_tokens, _total_api_calls
    global _per_model_usage
    _total_input_tokens = 0
    _total_output_tokens = 0
    _total_api_calls = 0
    _per_model_usage = {}
