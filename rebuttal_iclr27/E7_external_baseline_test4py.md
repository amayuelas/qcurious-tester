# E7. New external baseline: Test4Py

**Answers:** jDfu W2 and Q2 ("TestWeaver, Test4Py, and TestForge ... Where
feasible, could the authors compare against representative methods under
comparable budgets?").

Timebox: 1 working day to a first valid number on 3 targets. If not reached,
stop and answer in text only (see Fallback).

## Why Test4Py (and not the other two)

| Tool | Code | Fit to our setup | Cost |
|---|---|---|---|
| Test4Py | public (github.com/Test4DT/Test4Py), CLI `python -m test4dt.start --project_path ... --source_path ...` | Python modules from real projects, like RepoExploreBench; OpenAI-style endpoint with configurable base URL, so it can run on the local Gemma server | $0 on Gemma |
| TestForge | built on OpenHands; evaluated on TestGenEval | TestGenEval testbeds pin Python 3.6-3.9 (the constraint that already kept CoverUp/Pynguin off it); agent loop needs a strong tool-calling model | paper reports $0.63 per file with their model; an agent on paid APIs is outside our budget |
| TestWeaver | no released code confirmed yet (check the ICSE 2026 artifact page once more) | targets one uncovered line at a time via backward slices | n/a |

The choice is made on availability, fit and cost. We do not know how Test4Py
will score against us, and the result is reported whichever way it goes.

Note: the paper already cites TestForge (as greedy) and TELPA; Test4Py and
TestWeaver are not cited and must be added to related work either way.

## Setup checklist

1. Clone at a pinned commit; record the hash here.
2. New image `curiositybench-test4py` FROM `curiositybench:latest` (so the
   target packages and versions are identical), + `requirements.txt`, +
   `pytest coverage pylint pytest-json-report` in the target env
   (`USER_PYTHON_PATH`).
3. Embedding model `BAAI/bge-large-en-v1.5` downloaded once on the host and
   mounted read-only (`TRANSFORMER_PATH`); containers stay offline otherwise.
4. `.env`: `OPENAI_API_BASE=$VLLM_API_BASE`, model `gemma-4-31B-it`,
   `--network host` as for CoverUp. Reuse `scripts/coverup_inject/
   sitecustomize.py` if it calls through litellm/openai so calls and tokens
   are counted the same way.
5. Check the Python version Test4Py needs against the image before anything
   else; this is the most likely blocker.
6. It runs per project (`--project_path`, `--source_path`). Find how to
   restrict generation to one module; if there is no flag, run it on a copy of
   the package and keep only tests for the target module, and note that here.

## Protocol

- Targets: the same 30 stratified targets as E6 (`E6_targets.txt`), so E6 can
  also include the Test4Py suite at no extra design cost.
- Model: Gemma only. Gemini only if budget remains after E1/E2.
- Budgets: Test4Py has an iterative repair loop and no execution cap that we
  know of. Report (a) native budget, with its measured LLM calls, tokens and
  test executions; (b) if its loop has an iteration/round setting, a setting
  whose calls are closest to our 48 per target. Do not patch its algorithm.
- Measurement: identical to the other external baselines: run the generated
  tests with `coverage run --branch --source=<module>` and count covered
  branches of the target module.
- Wall-clock cap per target: 30 min (`--docker-timeout`); timeouts recorded
  as the coverage of whatever tests exist, and counted.

## Analysis

Table on the 30 targets: branches, calls, tokens, executions for Test4Py
(native, and matched if available), CoverUp, Pynguin, CovGreedy, `covqvalue2`
(all Gemma). Paired test vs `covqvalue2`. State clearly that this is a
30-target subset.

## What we say in each outcome

- Ours higher at comparable budget: add the row; note Test4Py addresses a
  different bottleneck (type-correct inputs) and the two could be combined.
- Test4Py higher at its native (larger) budget: report both the native and
  the per-call-normalised view, as we do for CoverUp in E2, and say so plainly.
- Test4Py higher at matched budget: report it. Its type inference would then
  be a strong generator to put under our targeting, and we say that.

## Fallback (setup fails within the timebox)

Text-only answer, with the concrete blocker stated, plus related-work
additions: TestWeaver (backward slice toward one target line, closest-test
retrieval, execution in-lining), Test4Py (call graph + behaviour-based type
inference for valid inputs, iterative repair), TestForge (agentic refinement
from execution and coverage reports on TestGenEval). All three improve HOW a
given target gets covered; ours decides WHICH code to go after next under a
fixed execution budget, so they are complementary generators for our loop.
