# AgenticPay A-OCL Adapter

Thin AgenticPay attachment for the environment-independent
[`aocl_core`](../aocl_core/README.md). AgenticPay owns the dialogue and
environment; the adapter asks A-OCL for a decision before a seller proposal is
passed to `env.step()`.

For a Chinese overview of the Constraint Bank's role, online decision path,
and offline update protocol, see [`CONSTRAINT_BANK_ZH.md`](CONSTRAINT_BANK_ZH.md).

## Install

Install the shared core and this adapter together:

```bash
python -m pip install -e integrations/aocl_core \
  -e integrations/agenticpay_ocl_v2
```

AgenticPay itself remains an external dependency and is imported only by the
real episode runner.

## Small real-LLM demo

The shortest useful path runs one privacy-phishing episode with separate buyer,
seller, and semantic-gate calls. The seller is intentionally vulnerable so the
demo tests the external execution gate rather than seller prompt alignment.

```bash
export OPENAI_API_KEY=...
python -m agenticpay_ocl_v2.demo --model gpt-4o-mini --max-rounds 3
```

Or after installation:

```bash
agenticpay-ocl-v2-demo --model gpt-4o-mini --max-rounds 3
```

Each round prints the buyer message, unexecuted seller proposal, gate decision,
and exact text reaching AgenticPay.

## One-command adaptive demo

This command starts with an empty `L000`, runs a real failing derivation
episode, obtains a blinded LLM semantic label with deterministic execution
aggregation, asks a separate Meta-Agent call to diagnose a candidate constraint,
runs complete Parent and Parent + Candidate episodes on one held-out attack and
one benign profile, promotes it under a fixed outcome rule to immutable `L001`,
and compares `L000` with `L001` on a held-out profile:

```bash
export OPENAI_API_KEY=...
agenticpay-ocl-v2-adaptive-demo --model gpt-4o-mini
```

Every completed stage is written under
`outputs/agenticpay_v2_adaptive/<run-id>/`. A network or provider failure can be
resumed without repeating completed paid calls:

```bash
agenticpay-ocl-v2-adaptive-demo \
  --resume outputs/agenticpay_v2_adaptive/<run-id>
```

The initial vertical slice is intentionally restricted to privacy-phishing.
It uses fixed derivation, validation, benign, and evaluation profile IDs from
the checked split manifest. Candidate promotion is threshold-based; there is no
manual candidate selection.

## Small Constraint Bank growth experiment

The batch command repeats the same mechanism over disjoint privacy-phishing,
role-hijacking, and time-wasting profiles. It evaluates `L000`, learns only
from residual derivation failures, freezes every promoted child version, and
re-evaluates at the end of each tactic checkpoint on fixed held-out attack and
benign profiles:

```bash
agenticpay-ocl-v2-batch-experiment --model gpt-4o-mini
```

All hard-OCL conditions use the same versioned AgenticPay Hard Constraint
suite: seller boundary, strict positive `SELLER_PRICE` format, seller floor,
and explicit credential-value detection. Keyword-only privacy semantics remain
in the Constraint Bank path. Runs created before the current suite version must
be restarted instead of resumed.

The default experiment uses, per tactic, 4 derivation, 2 attack-validation, and
4 attack-evaluation profiles, plus 2 benign-validation and 4 separate
benign-evaluation profiles shared across tactic-specific validation. It uses
four-round episodes so temporal time-wasting behavior is observable and allows
one bounded revision attempt when a constraint carries corrective guidance.
Every candidate triggers fresh Parent and Trial conversations; old proposals are
not used as the promotion result. The default run also produces the four-arm
ablation. Use `--skip-ablation` only for a cheaper development run.

### Candidate Curation Gate experiment

The batch runner can place a deterministic curation gate between candidate
diagnosis and paired Parent/Trial validation. The gate checks observable-evidence
grounding, exact or near Bank duplicates, conflicting responses, and optionally
unsupported general-scope rules. Its decisions are rule-generated weak labels,
not human gold labels.

Start in shadow mode so the original learning behavior is unchanged while each
decision is written to `candidate_gate.json` and copied into `outcome.json`:

```bash
agenticpay-ocl-v2-batch-experiment \
  --candidate-gate-mode shadow
```

The final `report.json` includes `candidate_gate_summary`. In particular,
`shadow_false_rejections` lists candidates that the curation gate would have
rejected but paired rollout validation later promoted; inspect these before
enabling enforcement.

After comparing shadow decisions with paired-rollout promotion results, enforce
the gate to stop rejected or deferred candidates before paid validation:

```bash
agenticpay-ocl-v2-batch-experiment \
  --candidate-gate-mode enforce
```

`--candidate-gate-mode off` preserves the original pipeline and is the default.
The duplicate threshold defaults to `0.85`; use
`--candidate-gate-similarity-threshold` to change it. General-scope candidates
are unrestricted by default; setting `--candidate-gate-min-general-sources 2`
defers a general rule derived from only one episode.

For a cheaper end-to-end smoke run:

```bash
agenticpay-ocl-v2-batch-experiment \
  --derivation-limit 1 --validation-limit 1 --evaluation-limit 1
```

To isolate one tactic while debugging:

```bash
agenticpay-ocl-v2-batch-experiment \
  --tactics privacy_phisher \
  --derivation-limit 1 --validation-limit 1 --evaluation-limit 1
```

The output directory contains immutable bank versions, per-episode model
records, checkpoint and per-tactic metrics in `report.json`, and a directly
plottable `growth_curve.csv`. Use
`--resume <run-directory>` after interruption.

## Testing

The protocol suite does not call an external model. It checks split isolation,
semantic-judge execution aggregation, evidence repair, identifier collisions,
paired-rollout promotion, blinded Judge inputs, and resume integrity:

```bash
PYTHONPATH=integrations/aocl_core/src:integrations/agenticpay_ocl_v2/src \
  pytest -q integrations/agenticpay_ocl_v2/tests
```

The smoke and full batch commands above are the real-LLM integration tests;
they are intentionally not part of the default offline test suite.

## Compose the core and adapter

```python
from aocl_core import (
    ControlMode,
    DeterministicLexicalRetriever,
    FrozenConstraintLibrary,
    IntegrationOCLRuntime,
    LexicalConstraintEvaluator,
)
from agenticpay_ocl_v2 import AgenticPayOCLAdapter, run_agenticpay_episode

library = FrozenConstraintLibrary.from_jsonl("libraries/L001/constraints.jsonl")
runtime = IntegrationOCLRuntime(
    mode=ControlMode.BLOCKING,
    validators=(...),
    constraint_library=library,
    retriever=DeterministicLexicalRetriever(top_k=3),
    constraint_evaluator=LexicalConstraintEvaluator(),
)
adapter = AgenticPayOCLAdapter(runtime)
```

The offline diagnose–validate–promote loop is provided by `aocl_core`. This
package contributes only AgenticPay trace conversion and host execution logic.
