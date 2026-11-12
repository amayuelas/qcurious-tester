# E6. Fault detection by mutation analysis

**Answers:** 1zqK Q5 ("How is the fault-detection effectiveness of the
generated scripts assessed?").

Cost: $0 in API spend. Uses Gemma suites (ours and CovGreedy from the stored
traces; CoverUp regenerated on the local server; Pynguin needs no LLM).

## Question

Does higher coverage translate into detecting more injected faults, and how do
our print-style exploration scripts compare with assertion-bearing suites?

## Suites compared (Gemma, seed 42)

| Suite | Source | Form |
|---|---|---|
| `covqvalue2` | stored trace, 24 scripts | scripts that print |
| `cov_greedy` | stored trace, 24 scripts | scripts that print |
| CoverUp | regenerate, SAVE tests (currently in `/tmp/gen` of a `--rm` container; add a `-v` mount) | pytest with assertions |
| Pynguin | regenerate with assertion generation ON (existing runs used `--assertion-generation NONE`, which would make its mutation score unfairly low) and save tests | pytest with assertions |

## Targets and mutants

- 30 targets: stratified over the 9 repos (3-4 each), fixed list committed in
  `rebuttal_iclr27/E6_targets.txt` before any results are seen (random
  with seed 0 within each repo).
- Mutants: `scripts/active/gen_mutants.py` (NEW), an AST mutator with five
  standard operators: relational flip (`<` to `<=`, `==` to `!=`), boolean
  negation of an `if`/`while` condition, arithmetic operator swap, constant
  replacement (0/1, True/False, "" for strings), and `return None`. One
  mutation per mutant, inside function bodies of the target module only.
- 50 mutants per target, sampled uniformly over mutation sites with seed 0
  (1,500 mutants in total). Discard mutants that fail to import (count them).

## Oracle

Print-style scripts have no assertions, so they are used as regression tests:
record behaviour on the original, compare on the mutant.

- Script suites: a mutant is KILLED if for any script the triple (normalised
  stdout, exception type, exit status) differs from the original. Normalise
  memory addresses (`0x...`), timestamps and temp paths. Timeouts (3x the
  original run time, min 10 s) count as kills.
- Flakiness filter: run each script twice on the original; drop scripts whose
  triple differs between the two runs (report how many).
- pytest suites: KILLED if any test that passes on the original fails or
  errors on the mutant. Tests failing on the original are dropped first.

Stated limitation: the regression oracle and assertion oracles differ in
strength (ours detects any visible behaviour change in printed values;
assertions check only what was asserted). We report it and do not claim the
scores are oracle-equivalent.

## Execution

One long-lived container per target (`DOCKER_MODE=exec`). For each mutant:
copy the mutated module over the installed file, run the suite, restore the
original file. Per target: 50 mutants x (24 scripts + pytest suites), about
1-2 s per script, so about 30-40 min per target per container; 30 targets at
`--parallel 30` finish in under an hour.

## Metrics

Per suite, mean over targets (+ paired tests vs `covqvalue2`):

- **Mutation score**: killed / valid mutants.
- **Covered-mutant kill rate**: killed / mutants whose mutated line the suite
  executes. Separates reaching code from detecting the fault once reached.
- Share of mutants on lines the suite executes (coverage of mutation sites).
- Scatter: per-target branch coverage vs mutation score, all suites.

## Decision rule / reporting

Report all four suites whichever way it goes. Expected pattern, to be
confirmed or not: our mutation score tracks our coverage advantage, while the
covered-mutant kill rate is similar across suites. If assertion suites kill
clearly more of the mutants they cover, say so: the method optimises
exploration, and adding assertions over the explored behaviour is the natural
next step (the recorded outputs are exactly what such assertions would pin).
