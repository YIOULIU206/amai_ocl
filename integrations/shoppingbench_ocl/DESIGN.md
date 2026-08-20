# ShoppingBench A-OCL Adapter Plan

## Status

Planned, not implemented. ShoppingBench remains external. No benchmark code,
dataset, search index, or evaluation logic is vendored here.

## Intended boundary

ShoppingBench is primarily a search-and-recommendation environment rather than
an adversarial negotiation environment. The main controlled action is the final
recommendation:

```text
find_product / view_product_information / web_search
  -> observable evidence

recommend_product
  -> ProposedAction
  -> shared A-OCL decision
  -> native recommendation or blocked/revised result
```

`python_execute`, if enabled, needs a separate code-execution security policy;
it must not be treated as an ordinary shopping recommendation. `terminate` is
normally scheduling rather than an economic commitment.

## Host-owned checks

Candidate deterministic checks include total budget, required item count,
voucher eligibility, same-shop requirements, mandatory attributes, and product
identifier existence. Soft constraints may capture contextual errors such as
excluding shipping from the effective total, optimizing items independently
while violating a bundle constraint, or trusting title keywords over product
specifications.

## Research role

ShoppingBench is a secondary cross-task generalization study, not the primary
evidence for negotiation governance. The main risk is confounding governance
with ordinary recommendation improvement. Experiments must therefore separate:

- hard intent/transaction violations;
- learned defensive constraint activation;
- ordinary search or reasoning accuracy.

Implementation starts only after `aocl_core` and the CoffeeBench adapter have a
stable action/outcome contract. It should add only native mapping, validators,
and a runner; it must reuse the core library, retrieval, evaluator, audit, and
offline learning pipeline.
