"""Injected via PYTHONPATH for CoverUp runs (local vLLM or an API model).

CoverUp drives the LLM through litellm, which needs three nudges here:

1. Function calling. litellm gatekeeps it on a static model-name registry.
   Neither our vLLM served-model name nor a newer API model (gemini-3.8-flash,
   zai-glm-5-3, ...) is in it, so litellm refuses tool calls before anything
   reaches the provider ("model does not support function calling"). We
   register the name as tool-capable; the provider does the real parsing —
   for vLLM, started with `--enable-auto-tool-choice --tool-call-parser gemma4`.
   The name comes from COVERUP_REGISTER_MODEL (or the older COVERUP_VLLM_MODEL).

2. Reasoning effort. CoverUp exposes no setting for it, so a thinking model
   would use its default — thousands of extra output tokens per code segment,
   which turned one target into an hour and put the run over budget. We pin
   "low" for GLM, matching what our own runs use, and pass
   allowed_openai_params so litellm forwards the parameter instead of
   rejecting the call outright.

3. Content shape. Thinking models return `content` as a list of chunks
   ({"type": "thinking"...}, {"type": "text"...}); CoverUp expects a string
   and finds no test in a list.

CoverUp awaits `litellm.acreate`, so that is the function to wrap — patching
`litellm.completion` alone changes nothing.
"""
import os

try:
    import litellm

    name = (os.environ.get("COVERUP_REGISTER_MODEL")
            or os.environ.get("COVERUP_VLLM_MODEL", ""))
    if name:
        litellm.register_model({
            name: {"supports_function_calling": True},
            f"hosted_vllm/{name}": {"supports_function_calling": True},
            f"openai/{name}": {"supports_function_calling": True},
            f"mistral/{name}": {"supports_function_calling": True},
        })

    # "default" leaves the model's own reasoning setting alone (used to check
    # whether pinning it low changes CoverUp's output quality).
    _effort = os.environ.get("COVERUP_REASONING_EFFORT", "low")

    def _prepare(kw, args):
        """Set the thinking budget for thinking models (see note 2).

        COVERUP_REASONING_EFFORT is either a level the provider accepts
        ("low"/"high"/"max"), a NUMBER of thinking tokens (OpenRouter's
        reasoning.max_tokens, the only setting between "low" and "high"), or
        "default" to leave the model alone. Left alone, GLM thinks until it
        hits max_tokens — 6000 tokens on a one-line prompt, 30x the cost of
        "low" — which is what made a full CoverUp run unaffordable.
        """
        model = kw.get("model") or (args[0] if args else "")
        if _effort == "default" or "glm" not in str(model).lower():
            return kw
        if _effort.isdigit():
            body = dict(kw.get("extra_body") or {})
            body.setdefault("reasoning", {"max_tokens": int(_effort)})
            kw["extra_body"] = body
        elif "reasoning_effort" not in kw:
            kw["reasoning_effort"] = _effort
            allowed = list(kw.get("allowed_openai_params") or [])
            if "reasoning_effort" not in allowed:
                allowed.append("reasoning_effort")
            kw["allowed_openai_params"] = allowed
        return kw

    def _flatten_text(content):
        if not isinstance(content, list):
            return content
        parts = []
        for chunk in content:
            if isinstance(chunk, dict):
                ctype, ctext = chunk.get("type"), chunk.get("text")
            else:
                ctype, ctext = getattr(chunk, "type", None), getattr(chunk, "text", None)
            if ctype == "text" and ctext:
                parts.append(ctext)
        return "".join(parts)

    def _normalise(resp):
        """Flatten chunked content in place (see note 3)."""
        try:
            for choice in getattr(resp, "choices", []):
                msg = getattr(choice, "message", None)
                if msg is not None and isinstance(getattr(msg, "content", None), list):
                    msg.content = _flatten_text(msg.content)
        except Exception:
            pass
        return resp

    if hasattr(litellm, "acreate"):
        _orig_acreate = litellm.acreate

        async def _acreate(*a, **kw):
            return _normalise(await _orig_acreate(*a, **_prepare(kw, a)))

        litellm.acreate = _acreate

    _orig_completion = litellm.completion

    def _completion(*a, **kw):
        return _normalise(_orig_completion(*a, **_prepare(kw, a)))

    litellm.completion = _completion
except Exception:
    pass
