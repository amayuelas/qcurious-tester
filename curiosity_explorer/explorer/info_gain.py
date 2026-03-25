"""Information gain estimation — multiple readout methods.

Estimators:
  - output_entropy: Single-model sampling entropy (Phase 1, deprecated)
  - coverage_disagreement: Single-model binary disagreement (Phase 1, deprecated)
  - multi_model_disagreement: Cross-model prediction disagreement (Estimator A)
  - logprob_entropy: Token-level logprob entropy from single prediction (Estimator B)
  - verbalized_confidence: Model self-reported confidence (Estimator C)
  - hybrid: Weighted combination of A + C (Estimator D)
"""

import math
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from ..llm import batch_generate, generate_with_model, generate_with_logprobs
from ..runner.trace_parser import extract_function_signature

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared prompt building
# ---------------------------------------------------------------------------

def _build_context(func_name: str, source_code: str, test_history: list,
                   code_visible: bool = True) -> tuple[str, str]:
    """Build code section and history string for prompts."""
    if code_visible:
        code_section = f"```python\n{source_code}\n```"
    else:
        sig = extract_function_signature(source_code)
        code_section = f"```python\n{sig}\n```"

    history_str = ""
    if test_history:
        recent = test_history[-5:]
        history_str = "Known test results:\n"
        for test_code, result in recent:
            history_str += f"  {test_code} → {result.output or result.exception}\n"

    return code_section, history_str


# ---------------------------------------------------------------------------
# Estimator A: Multi-Model Disagreement
# ---------------------------------------------------------------------------

def estimate_multi_model_disagreement(
    func_name: str, source_code: str, test_history: list,
    candidate_test: str, models: list[str] = None,
    code_visible: bool = True,
) -> float:
    """Estimate info gain via cross-model prediction disagreement.

    Ask each model in the ensemble to predict the output of candidate_test.
    Compute Shannon entropy over the set of distinct predictions.
    High disagreement = models have different beliefs about program behavior.
    """
    models = models or config.ENSEMBLE_MODELS
    code_section, history_str = _build_context(func_name, source_code,
                                                test_history, code_visible)

    prompt = f"""Given this function:
{code_section}

{history_str}

What will be the output of: {candidate_test}

Respond with ONLY the expected output value (the return value or exception), nothing else."""

    # Query each model in parallel
    predictions = []
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(generate_with_model, m, prompt, 0.3, 100): m
            for m in models
        }
        for future in as_completed(futures):
            model = futures[future]
            try:
                pred = future.result().strip().lower()[:100]
                if pred:
                    predictions.append(pred)
            except Exception as e:
                log.warning(f"Ensemble prediction failed for {model}: {e}")

    if len(predictions) < 2:
        return 0.0

    return _entropy(predictions)


# ---------------------------------------------------------------------------
# Estimator B: Token-Level Logprob Entropy
# ---------------------------------------------------------------------------

def estimate_logprob_entropy(
    func_name: str, source_code: str, test_history: list,
    candidate_test: str, model: str = None,
    code_visible: bool = True,
) -> float:
    """Estimate info gain via mean per-token entropy from logprobs.

    Generate a single output prediction with logprobs enabled.
    Compute entropy of the top-k token distribution at each position.
    Average across all output tokens.

    High token entropy = model uncertain about output = informative test.
    Eliminates the zero-entropy floor problem (71% of candidates scoring 0
    with sampling) because even a deterministic greedy decode can have high
    token-level entropy in the underlying distribution.
    """
    model = model or config.LOGPROB_MODEL
    code_section, history_str = _build_context(func_name, source_code,
                                                test_history, code_visible)

    prompt = f"""Given this function:
{code_section}

{history_str}

What will be the output of: {candidate_test}

Respond with ONLY the expected output value (the return value or exception), nothing else."""

    result = generate_with_logprobs(model, prompt, temperature=0.3,
                                    max_tokens=100, top_logprobs=5)
    if not result or not result["token_logprobs"]:
        return 0.0

    # Compute per-token entropy from top-k logprob distributions
    token_entropies = []
    for token_info in result["token_logprobs"]:
        top_lps = token_info.get("top_logprobs", {})
        if not top_lps:
            continue
        # Convert logprobs to probabilities
        probs = [math.exp(lp) for lp in top_lps.values()]
        # Normalize (top-k may not sum to 1)
        total_p = sum(probs)
        if total_p <= 0:
            continue
        probs = [p / total_p for p in probs]
        # Shannon entropy
        ent = -sum(p * math.log2(p) for p in probs if p > 0)
        token_entropies.append(ent)

    if not token_entropies:
        return 0.0

    return sum(token_entropies) / len(token_entropies)


# ---------------------------------------------------------------------------
# Estimator C: Verbalized Confidence
# ---------------------------------------------------------------------------

def estimate_verbalized_confidence(
    func_name: str, source_code: str, test_history: list,
    candidate_test: str, code_visible: bool = True,
) -> float:
    """Estimate info gain via model's self-reported confidence.

    Ask the model to rate its confidence about output prediction (0-100).
    Lower confidence = higher information gain.
    Returns: information gain score in [0, 1].
    """
    code_section, history_str = _build_context(func_name, source_code,
                                                test_history, code_visible)

    prompt = f"""Given this function:
{code_section}

{history_str}

Consider this test call: {candidate_test}

Answer these two questions with ONLY two numbers separated by a comma, nothing else:
1. How confident are you (0-100) that you can predict the EXACT output of this call?
2. How confident are you (0-100) that this test will hit branches NOT covered by previous tests?

Format: confidence_output, confidence_new_branches
Example: 75, 30"""

    response = generate_with_model(config.MODEL, prompt, temperature=0.3,
                                   max_tokens=50)
    output_conf, branch_conf = _parse_confidence_scores(response)

    # Information gain proxy: low output confidence = high info gain
    # Also factor in branch confidence (high = believes it will explore)
    # Score = (1 - output_confidence/100) * 0.7 + (branch_confidence/100) * 0.3
    ig = (1 - output_conf / 100) * 0.7 + (branch_conf / 100) * 0.3
    return ig


def _parse_confidence_scores(response: str) -> tuple[float, float]:
    """Parse two confidence scores from LLM response."""
    # Try to find two numbers
    numbers = re.findall(r'\d+(?:\.\d+)?', response)
    if len(numbers) >= 2:
        output_conf = max(0, min(100, float(numbers[0])))
        branch_conf = max(0, min(100, float(numbers[1])))
        return output_conf, branch_conf
    elif len(numbers) == 1:
        return max(0, min(100, float(numbers[0]))), 50.0
    return 50.0, 50.0  # Default: maximally uncertain


# ---------------------------------------------------------------------------
# Estimator D: Hybrid (Multi-Model Disagreement + Verbalized Confidence)
# ---------------------------------------------------------------------------

def estimate_hybrid(
    func_name: str, source_code: str, test_history: list,
    candidate_test: str, models: list[str] = None,
    code_visible: bool = True, alpha: float = 0.6,
) -> float:
    """Hybrid estimator combining multi-model disagreement and verbalized confidence.

    score = alpha * disagreement + (1-alpha) * verbalized_ig
    Default alpha=0.6 weights disagreement more heavily.
    """
    disagreement = estimate_multi_model_disagreement(
        func_name, source_code, test_history, candidate_test,
        models=models, code_visible=code_visible,
    )
    verbalized_ig = estimate_verbalized_confidence(
        func_name, source_code, test_history, candidate_test,
        code_visible=code_visible,
    )

    # Normalize disagreement to [0, 1] range
    # Max entropy for N models = log2(N), normalize by that
    n_models = len(models or config.ENSEMBLE_MODELS)
    max_ent = math.log2(n_models) if n_models > 1 else 1.0
    norm_disagreement = min(disagreement / max_ent, 1.0)

    return alpha * norm_disagreement + (1 - alpha) * verbalized_ig


# ---------------------------------------------------------------------------
# Pre-compute all Phase 0b estimators for a candidate (for efficiency)
# ---------------------------------------------------------------------------

def score_candidate_phase0b(
    func_name: str, source_code: str, test_history: list,
    candidate_test: str, models: list[str] = None,
    code_visible: bool = True,
) -> dict:
    """Score a candidate with all Phase 0b estimators in one call.

    Returns dict with keys: multi_model_disagreement, verbalized_confidence,
    hybrid, plus raw sub-scores.
    """
    models = models or config.ENSEMBLE_MODELS
    code_section, history_str = _build_context(func_name, source_code,
                                                test_history, code_visible)

    # --- Multi-model predictions (Estimator A) ---
    predict_prompt = f"""Given this function:
{code_section}

{history_str}

What will be the output of: {candidate_test}

Respond with ONLY the expected output value (the return value or exception), nothing else."""

    predictions = {}
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(generate_with_model, m, predict_prompt, 0.3, 100): m
            for m in models
        }
        for future in as_completed(futures):
            m = futures[future]
            try:
                pred = future.result().strip().lower()[:100]
                if pred:
                    predictions[m] = pred
            except Exception as e:
                log.warning(f"Prediction failed for {m}: {e}")

    pred_list = list(predictions.values())
    disagreement = _entropy(pred_list) if len(pred_list) >= 2 else 0.0

    # --- Logprob entropy (Estimator B) + Verbalized confidence (C) in parallel ---
    logprob_model = config.LOGPROB_MODEL

    conf_prompt = f"""Given this function:
{code_section}

{history_str}

Consider this test call: {candidate_test}

Answer these two questions with ONLY two numbers separated by a comma, nothing else:
1. How confident are you (0-100) that you can predict the EXACT output of this call?
2. How confident are you (0-100) that this test will hit branches NOT covered by previous tests?

Format: confidence_output, confidence_new_branches
Example: 75, 30"""

    with ThreadPoolExecutor(max_workers=2) as executor:
        logprob_future = executor.submit(
            _compute_logprob_entropy, logprob_model, predict_prompt,
        )
        conf_future = executor.submit(
            generate_with_model, config.MODEL, conf_prompt, 0.3, 50,
        )

    try:
        logprob_ent = logprob_future.result()
    except Exception as e:
        log.warning(f"Logprob entropy failed: {e}")
        logprob_ent = 0.0

    try:
        conf_response = conf_future.result()
    except Exception:
        conf_response = ""

    output_conf, branch_conf = _parse_confidence_scores(conf_response)
    verbalized_ig = (1 - output_conf / 100) * 0.7 + (branch_conf / 100) * 0.3

    # --- Hybrid (Estimator D) ---
    n_models = len(models)
    max_ent = math.log2(n_models) if n_models > 1 else 1.0
    norm_disagreement = min(disagreement / max_ent, 1.0)
    hybrid = 0.6 * norm_disagreement + 0.4 * verbalized_ig

    return {
        "multi_model_disagreement": round(disagreement, 4),
        "logprob_entropy": round(logprob_ent, 4),
        "verbalized_confidence": round(verbalized_ig, 4),
        "hybrid": round(hybrid, 4),
        "output_confidence": round(output_conf, 1),
        "branch_confidence": round(branch_conf, 1),
        "model_predictions": predictions,
        "logprob_model": logprob_model,
        "n_models_responded": len(predictions),
    }


def _compute_logprob_entropy(model: str, prompt: str) -> float:
    """Helper: compute mean per-token entropy from logprobs."""
    result = generate_with_logprobs(model, prompt, temperature=0.3,
                                    max_tokens=100, top_logprobs=5)
    if not result or not result["token_logprobs"]:
        return 0.0

    token_entropies = []
    for token_info in result["token_logprobs"]:
        top_lps = token_info.get("top_logprobs", {})
        if not top_lps:
            continue
        probs = [math.exp(lp) for lp in top_lps.values()]
        total_p = sum(probs)
        if total_p <= 0:
            continue
        probs = [p / total_p for p in probs]
        ent = -sum(p * math.log2(p) for p in probs if p > 0)
        token_entropies.append(ent)

    if not token_entropies:
        return 0.0
    return sum(token_entropies) / len(token_entropies)


# ---------------------------------------------------------------------------
# Legacy estimators (Phase 1 — kept for backward compatibility)
# ---------------------------------------------------------------------------

def estimate_output_entropy(func_name: str, source_code: str,
                            test_history: list, candidate_test: str,
                            S: int = 8, code_visible: bool = True) -> float:
    """Estimate info gain by measuring LLM's uncertainty about program output.

    High entropy = LLM doesn't know what will happen = informative test.
    Maps to g(a|h) = I(O; Theta|h,a) from Sun et al.
    """
    code_section, history_str = _build_context(func_name, source_code,
                                                test_history, code_visible)

    prompt = f"""Given this function:
{code_section}

{history_str}

What will be the output of: {candidate_test}

Respond with ONLY the expected output value (the return value or exception), nothing else."""

    # Batch sample S predictions in parallel
    predictions = batch_generate([prompt] * S, temperature=0.9, max_tokens=100)
    predictions = [p.strip().lower()[:100] for p in predictions if p]

    return _entropy(predictions)


def estimate_coverage_disagreement(func_name: str, source_code: str,
                                   test_history: list, candidate_test: str,
                                   cumulative_arcs: int,
                                   S: int = 20, code_visible: bool = True) -> float:
    """Estimate info gain by measuring LLM's disagreement about whether
    a test will hit new branches."""
    code_section, history_str = _build_context(func_name, source_code,
                                                test_history, code_visible)
    if test_history:
        # Override history to include coverage info
        recent = test_history[-5:]
        history_str = "Previous tests and their coverage results:\n"
        for test_code, result in recent:
            history_str += (f"  {test_code} → "
                           f"new_branches={result.new_branches}, "
                           f"output={result.output or result.exception}\n")

    prompt = f"""Given this function:
{code_section}

{history_str}
Total branches covered so far: {cumulative_arcs}

Will the following test discover NEW branches not yet covered?
Test: {candidate_test}

Answer with ONLY one of: "yes", "no", or a number (how many new branches you expect). Nothing else."""

    predictions = batch_generate([prompt] * S, temperature=0.9, max_tokens=20)
    predictions = [p.strip().lower()[:50] for p in predictions if p]

    yes_count = sum(1 for p in predictions if _predicts_new_coverage(p))
    no_count = len(predictions) - yes_count

    total = yes_count + no_count
    if total == 0:
        return 0.0

    p_yes = yes_count / total
    return _binary_entropy(p_yes)


def _predicts_new_coverage(pred: str) -> bool:
    """Classify a prediction as expecting new coverage or not."""
    pred = pred.strip().lower()
    if pred in ("no", "no.", "0", "none", "unlikely", "no new branches",
                "no, it will not", "no new", "false"):
        return False
    if pred in ("yes", "yes.", "likely", "yes, it will", "true"):
        return True
    try:
        n = int(pred.split()[0])
        return n > 0
    except (ValueError, IndexError):
        pass
    if "yes" in pred:
        return True
    if "no" in pred:
        return False
    return True


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _binary_entropy(p: float) -> float:
    """Binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def _entropy(predictions: list[str]) -> float:
    """Compute Shannon entropy over a list of predictions."""
    if not predictions:
        return 0.0
    counts = Counter(predictions)
    total = len(predictions)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy
