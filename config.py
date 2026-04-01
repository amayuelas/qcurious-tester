"""Central configuration — reads .env, exports settings."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# API settings
GEMINI_API_BASE = os.environ.get("API_BASE", "https://generativelanguage.googleapis.com/v1beta/openai")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MISTRAL_API_BASE = "https://api.mistral.ai/v1"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
FIREWORKS_API_BASE = "https://api.fireworks.ai/inference/v1"
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
LOGPROB_MODEL = "accounts/fireworks/models/gpt-oss-120b"
MODEL = os.environ.get("MODEL", "gemini-3-flash-preview")

# Default ensemble for multi-model disagreement (Estimator A)
ENSEMBLE_MODELS = [
    "gemini-3-flash-preview",
    "mistral-large-latest",
    "gpt-5.4-mini",
]

# Experiment parameters
BUDGET = int(os.environ.get("BUDGET", "15"))
K = int(os.environ.get("K", "5"))
S = int(os.environ.get("S", "6"))

# Paths
RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent / "data"

# Model pricing (USD per 1M tokens, standard paid tier, text)
# Source: https://ai.google.dev/gemini-api/docs/pricing (retrieved 2026-03-22)
# Output prices include thinking tokens where applicable.
# For tiered models (prompt length ≤200k vs >200k), we use the ≤200k price.
MODEL_PRICING = {
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
    "gemini-3-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    # Source: https://docs.mistral.ai/models/ (retrieved 2026-03-22)
    "mistral-large-latest": {"input": 0.50, "output": 1.50},
    "mistral-large-2512": {"input": 0.50, "output": 1.50},
    "mistral-small-latest": {"input": 0.15, "output": 0.60},
    "mistral-small-2603": {"input": 0.15, "output": 0.60},
    # Source: https://developers.openai.com/api/docs/pricing (retrieved 2026-03-23)
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "output": 1.25},
    "gpt-5.4-pro": {"input": 30.00, "output": 180.00},
    # Source: https://fireworks.ai/models/fireworks/gpt-oss-120b (retrieved 2026-03-23)
    "accounts/fireworks/models/gpt-oss-120b": {"input": 0.15, "output": 0.60},
}
