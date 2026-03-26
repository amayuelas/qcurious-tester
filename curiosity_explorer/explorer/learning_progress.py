"""Learning progress exploration — Schmidhuber-grounded approach.

Instead of optimizing candidate SELECTION, optimize the WORLD MODEL UPDATE.
Make the history more informative so the LLM's generation improves faster.

Key idea: the LLM's in-context learning IS the Bayesian update.
We improve the update by:
1. Summarizing what was learned from each test (structured knowledge)
2. Highlighting surprising results (high prediction error)
3. Tracking known vs unknown territory to guide exploration

The generation prompt receives enriched history, then generates freely
(no constrained selection — random pick from diverse candidates).
"""

import logging
from ..llm import generate_with_model, batch_generate

log = logging.getLogger(__name__)


class LearningProgressTracker:
    """Track what the world model has learned and what remains unknown."""

    def __init__(self, model=None):
        import config
        self.model = model or config.MODEL
        self.knowledge = []      # structured learnings
        self.surprises = []      # tests where prediction was wrong
        self.predictions = {}    # step -> predicted output
        self.step = 0

    def predict(self, script, context_prompt):
        """Predict what a test script will output."""
        prompt = (f"{context_prompt}\n\n"
                  f"What will this script output?\n"
                  f"```python\n{script[:300]}\n```\n"
                  f"Respond with ONLY the expected output (1-2 lines).")
        pred = generate_with_model(self.model, prompt, temperature=0.3,
                                   max_tokens=100)
        self.predictions[self.step] = pred
        return pred

    def update(self, script, predicted, actual, context_prompt):
        """Update the world model after observing a result.

        This is the Bayesian update step — extract structured knowledge
        from the prediction error.
        """
        self.step += 1

        # Compute surprise
        pred_clean = (predicted or "").strip().lower()[:100]
        actual_clean = (actual or "").strip().lower()[:100]
        is_surprised = pred_clean != actual_clean

        if is_surprised:
            self.surprises.append({
                "step": self.step,
                "script": script[:100],
                "predicted": pred_clean[:50],
                "actual": actual_clean[:50],
            })

        # Extract structured learning via LLM
        learning = self._extract_learning(script, predicted, actual,
                                          is_surprised, context_prompt)
        if learning:
            self.knowledge.append(learning)

        # Keep knowledge concise (last 8 learnings)
        if len(self.knowledge) > 8:
            self.knowledge = self.knowledge[-8:]

    def _extract_learning(self, script, predicted, actual, is_surprised,
                          context_prompt):
        """Ask the LLM to extract a structured learning from the observation."""
        if is_surprised:
            prompt = (f"{context_prompt}\n\n"
                      f"You predicted this script would output:\n  {predicted[:100]}\n"
                      f"But it actually output:\n  {actual[:100]}\n\n"
                      f"In ONE sentence, what specific fact did you learn about "
                      f"this module's behavior? Focus on the unexpected part.")
        else:
            prompt = (f"{context_prompt}\n\n"
                      f"This script output:\n  {actual[:100]}\n"
                      f"(As you predicted.)\n\n"
                      f"In ONE sentence, what does this confirm about the module?")

        learning = generate_with_model(self.model, prompt, temperature=0.3,
                                       max_tokens=100)
        return learning.strip() if learning else None

    def build_enriched_history(self, test_history):
        """Build an enriched history string for the generation prompt.

        Instead of just showing (test → output), shows:
        - Structured knowledge extracted from each test
        - Which tests were surprising (high prediction error)
        - What territory remains unexplored
        """
        parts = []

        # Structured knowledge
        if self.knowledge:
            parts.append("WHAT YOU'VE LEARNED SO FAR:")
            for i, k in enumerate(self.knowledge, 1):
                parts.append(f"  {i}. {k}")
            parts.append("")

        # Recent test results (with surprise markers)
        if test_history:
            parts.append("RECENT TEST RESULTS:")
            for i, (script, result) in enumerate(test_history[-5:]):
                short = script.strip()[:100]
                out = (result.output or result.exception or "None")[:60]
                step_idx = len(test_history) - 5 + i + 1
                surprise = ""
                for s in self.surprises:
                    if s["step"] == step_idx:
                        surprise = " [SURPRISING — you predicted something different]"
                        break
                parts.append(f"  Test: {short}")
                parts.append(f"  → {out}{surprise}")
                parts.append("")

        # What's unknown
        if self.surprises:
            parts.append(f"AREAS OF UNCERTAINTY (you were wrong {len(self.surprises)} "
                         f"times out of {self.step} tests):")
            recent_surprises = self.surprises[-3:]
            for s in recent_surprises:
                parts.append(f"  - Step {s['step']}: expected '{s['predicted']}' "
                             f"got '{s['actual']}'")
            parts.append("")

        return "\n".join(parts)


def generate_with_learning_progress(func_name, source, test_history,
                                     tracker, K=5, code_visible=True):
    """Generate candidates using enriched history from learning progress.

    The key difference from standard generation: the prompt includes
    structured knowledge, surprise markers, and uncertainty areas —
    giving the LLM better information to learn from.
    """
    if code_visible:
        code_section = f"```python\n{source[:3000]}\n```"
    else:
        from ..runner.trace_parser import extract_function_signature
        code_section = f"```python\n{extract_function_signature(source)}\n```"

    enriched_history = tracker.build_enriched_history(test_history)

    prompt = f"""You are testing this Python module:

{code_section}

{enriched_history}

Based on what you've learned (and what surprised you), write a test script
(5-15 lines) that explores UNKNOWN behavior — something you haven't confirmed yet.

Focus on areas where you were wrong or uncertain.
Import what you need and print results to verify.

Respond with ONLY Python code.

```python
"""

    responses = batch_generate([prompt] * K, temperature=0.9, max_tokens=500)

    scripts = []
    for resp in responses:
        script = _parse_script(resp)
        if script and script not in scripts:
            scripts.append(script)
    return scripts


def _parse_script(response):
    if not response:
        return None
    code = response.strip()
    if code.startswith("```python"):
        code = code[9:]
    if code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()
    if not code or len(code) < 10:
        return None
    if "subprocess" in code or "os.system" in code:
        return None
    # Reject non-code responses (explanations, descriptions)
    first_line = code.split("\n")[0].strip()
    if not any(first_line.startswith(kw) for kw in
               ["import ", "from ", "#", "class ", "def ", "print(",
                "try:", "with ", "for ", "if ", "assert "]) \
       and "=" not in first_line and "(" not in first_line:
        return None
    return code
