# AgenticPay A-OCL V2

Independent adaptive organizational control runtime and AgenticPay adapter.
The package separates an online action gate from an offline
diagnose–validate–promote learning pipeline. It does not import the legacy
repository package `aimai_ocl`.

See [`DESIGN.md`](DESIGN.md) for the architecture, observability rules, and
experiment protocol.

## Install

```bash
python -m pip install -e integrations/agenticpay_ocl_v2
```

Core imports do not require AgenticPay. The real episode runner imports
AgenticPay lazily when no test environment factory is supplied.

## Online runtime

```python
from agenticpay_ocl_v2 import (
    ControlMode,
    DeterministicLexicalRetriever,
    FrozenConstraintLibrary,
    IntegrationOCLRuntime,
    LexicalConstraintEvaluator,
)

library = FrozenConstraintLibrary.from_jsonl("libraries/L001/constraints.jsonl")
runtime = IntegrationOCLRuntime(
    mode=ControlMode.BLOCKING,
    validators=(...),
    constraint_library=library,
    retriever=DeterministicLexicalRetriever(top_k=3),
    constraint_evaluator=LexicalConstraintEvaluator(),
)
```

Attach it to the AgenticPay seller boundary:

```python
from agenticpay_ocl_v2.agenticpay_adapter import AgenticPayOCLAdapter
from agenticpay_ocl_v2.agenticpay_runner import run_agenticpay_episode

result = run_agenticpay_episode(
    episode_id="run-001",
    env_id="Task1_basic_price_negotiation-v0",
    buyer_agent=buyer,
    seller_agent=seller,
    ocl_adapter=AgenticPayOCLAdapter(runtime),
    reset_kwargs={...},
    env_kwargs={...},
)
```

The runtime never executes an action. The host runner executes only
`approve`/`warn` proposals, may regenerate a `revise` proposal within a fixed
budget, and sends `None` to AgenticPay for `block`/`escalate`.

## Offline adaptive loop

`AdaptiveLearningPipeline.process_failure(...)` performs one complete offline
step:

```text
derivation trace + independent label
  -> injected Meta-Agent diagnosis
  -> candidate constraint
  -> validation/benign replay
  -> fixed promotion policy
  -> immutable child library version
```

Only approved constraints are active online. A candidate cannot validate on
its own source episode, and promotion requires both positive and negative
replay cases.

The prompted diagnoser and semantic evaluator accept small injected interfaces
with `generate(prompt: str) -> str`; model-provider configuration remains owned
by the host.

## Data protocol

- [`data/split_manifest_v1.json`](data/split_manifest_v1.json) defines stable
  derivation, validation, evaluation, benign, and control IDs.
- [`data/benign_buyers.json`](data/benign_buyers.json) supplies ten independently
  authored benign controls.
- `datasets.load_profiles` assigns deterministic IDs to the legacy profile file
  when explicit IDs are absent and validates manifest disjointness.
- `trace_export.learning_trace_from_run` converts a completed AgenticPay run to
  an observable offline trace; `append_learning_trace` writes JSONL.

Evaluation profiles must not feed candidate generation, validation, threshold
selection, or library updates.
