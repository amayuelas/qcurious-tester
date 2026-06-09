"""Injected via PYTHONPATH for CoverUp runs against a local vLLM model.

CoverUp drives the LLM through litellm, which gatekeeps function calling on a
static model-name registry. Our custom vLLM served-model name isn't in it, so
litellm refuses tool calls *before* anything reaches the server. This module
runs at interpreter startup (sitecustomize is auto-imported) and registers the
model as tool-capable, so litellm's check passes; vLLM — started with
`--enable-auto-tool-choice --tool-call-parser gemma4` — does the real tool-call
parsing. The model name is passed in via COVERUP_VLLM_MODEL.
"""
import os

try:
    import litellm

    name = os.environ.get("COVERUP_VLLM_MODEL", "")
    if name:
        litellm.register_model({
            name: {"supports_function_calling": True},
            f"hosted_vllm/{name}": {"supports_function_calling": True},
        })
except Exception:
    pass
