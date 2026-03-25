"""LLM client with response caching, batch support, multi-model, and cost tracking."""

import hashlib
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import config

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai")
    sys.exit(1)

log = logging.getLogger(__name__)


def _make_client():
    """Create the OpenAI-compatible client for the configured model."""
    return _client_for_model(config.MODEL)


def _client_for_model(model: str) -> OpenAI:
    """Create an OpenAI-compatible client for a specific model."""
    if model.startswith("accounts/fireworks/"):
        return OpenAI(base_url=config.FIREWORKS_API_BASE,
                      api_key=config.FIREWORKS_API_KEY)
    elif model.startswith("mistral"):
        return OpenAI(base_url=config.MISTRAL_API_BASE,
                      api_key=config.MISTRAL_API_KEY)
    elif model.startswith("gpt"):
        return OpenAI(api_key=config.OPENAI_API_KEY)
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

# Response cache: key -> response string
_cache: dict[str, str] = {}
_cache_hits = 0
_cache_misses = 0

# Token / cost accounting — tracks per-model usage
_total_input_tokens = 0
_total_output_tokens = 0
_total_api_calls = 0
_per_model_usage: dict[str, dict] = {}


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

    try:
        cli = _get_client(model)
        # OpenAI gpt-5+ models require max_completion_tokens
        if model.startswith("gpt"):
            tok_param = {"max_completion_tokens": max_tokens}
        else:
            tok_param = {"max_tokens": max_tokens}
        response = cli.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            **tok_param,
        )
        msg = response.choices[0].message
        text = msg.content or getattr(msg, "reasoning_content", None) or ""
        result = text.strip()

        # Track token usage
        input_toks = (response.usage.prompt_tokens or 0) if response.usage else 0
        output_toks = (response.usage.completion_tokens or 0) if response.usage else 0
        _track_usage(model, input_toks, output_toks)

    except Exception as e:
        log.warning(f"API error ({model}): {e}")
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
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_logprobs,
        )

        msg = response.choices[0].message
        text = (msg.content or getattr(msg, "reasoning_content", None)
                or "").strip()

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
