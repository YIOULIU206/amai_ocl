# AgenticPay A-OCL Adapter Design

## Role

This package is the AgenticPay host adapter for `aocl_core`. It is not an
independent control runtime and does not own constraint retrieval, semantic
evaluation, learning, or library versioning.

AgenticPay remains external and owns the buyer, seller, dialogue loop,
environment transitions, rewards, and execution. The adapter controls only the
seller proposal immediately before `env.step()`:

```text
buyer message -> seller proposal -> AgenticPay adapter -> A-OCL core
                                                   approve/warn -> env.step(text)
                                            block/escalate/revise -> no execution
```

The package imports AgenticPay lazily, so its adapter contracts remain
importable when the benchmark is not installed.

## Files owned here

```text
agenticpay_adapter.py  seller text/context normalization and hard price rule
agenticpay_runner.py   native AgenticPay episode loop and execution boundary
trace_export.py        native run result to host-independent learning trace
demo.py                one real-LLM vertical demonstration
adaptive_demo.py       resumable L000 -> candidate -> validation -> L001 demo
batch_experiment.py    frozen-library growth curve over disjoint profiles
```

All shared contracts and mechanisms live in `integrations/aocl_core`.

## Online mapping

Each seller proposal becomes:

- `action_type="commerce.respond"`;
- `visible_text` and payload containing only the proposed seller response;
- visible conversation turns and sanitized environment observation;
- no persona label, reward, hidden price bound, oracle answer, or future result.

The seller price validator is adapter-owned. Hidden organizational bounds are
used by deterministic validation but are not exposed to retrieval or the
semantic evaluator.

Decision mapping is fixed:

```text
approve/warn  execute the exact proposed text
revise        optionally regenerate within a fixed retry budget, then re-check
block         pass no seller action to AgenticPay
escalate      pass no seller action without an external resolution
```

## Real-LLM demo

The small demo uses separate buyer, seller, and gate model calls. A deliberately
vulnerable seller agrees to an off-platform payment request; the external gate
must prevent that proposal from reaching AgenticPay. It demonstrates the
online boundary only, not the offline learning claim.

## Adaptive demo

The adaptive demo is the smallest complete learning experiment. It uses four
disjoint profiles: one derivation failure, one validation attack, one benign
validation case, and one held-out evaluation attack. A Meta-Agent converts the
derivation trace into one candidate constraint. A fixed promotion policy accepts
it only when semantic replay catches the attack and leaves the benign proposal
alone, then writes immutable `L001`.

The blinded LLM judge handles only the semantic question of which seller
proposals violate policy. Code combines those step IDs with the recorded
`executed` flags. This prevents a language-model judge from accidentally calling
a blocked, counterfactual proposal an executed policy failure.

Every stage is persisted before the next begins. `--resume` reuses completed
episodes and model calls, while artifacts that fail a machine-checkable schema
or execution invariant are rejected and regenerated.

The batch experiment applies the same loop repeatedly. Each derivation episode
runs against the latest frozen version. Episodes with no observed failure do not
create a new version; artifacts distinguish intrinsically safe handling from a
constraint-library intervention. Every promoted version is evaluated on the
same held-out attack and benign profiles, producing the library-size growth
curve without learning from evaluation outcomes.

## Experiment role

AgenticPay is the short-horizon adversarial negotiation environment. It should
remain a small reference integration for price, privacy, and role attacks.
Long-horizon library growth belongs primarily in CoffeeBench. Evaluation uses
frozen library versions and disjoint derivation, validation, and evaluation
splits as specified by `aocl_core`.
