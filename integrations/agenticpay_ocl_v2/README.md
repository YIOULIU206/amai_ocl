# AgenticPay A-OCL Adapter

Thin AgenticPay attachment for the environment-independent
[`aocl_core`](../aocl_core/README.md). AgenticPay owns the dialogue and
environment; the adapter asks A-OCL for a decision before a seller proposal is
passed to `env.step()`.

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
validates it on one held-out attack and one benign episode, promotes it to
immutable `L001`, and compares `L000` with `L001` on a held-out profile:

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
