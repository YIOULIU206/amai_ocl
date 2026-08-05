# A-OCL Core Design

## Purpose

A-OCL is a host-independent governance layer at the action execution boundary.
It is not an agent, dialogue loop, benchmark runner, or general-purpose RAG
system. A benchmark owns its agents, state transitions, rewards, and execution.
A-OCL observes a normalized proposal and returns a control decision.

## Two levels

Level 1 hard constraints are deterministic host-owned checks such as schema,
authority, inventory, cash, price, or disclosure rules. Level 2 soft
constraints are reviewed defensive heuristics distilled from failed episodes.
Hard checks provide the safety boundary; soft constraints improve semantic and
contextual handling over time.

## Stable host contract

```text
host proposal
  -> ProposedAction + ObservableContext + optional host pre-checks
  -> hard validators
  -> retrieve approved soft constraints
  -> evaluate constraint activation
  -> APPROVE / WARN / REVISE / BLOCK / ESCALATE
  -> host decides whether to execute
  -> host reports ObservedOutcome
```

The runtime never calls a host tool. Adapters must exclude oracle labels,
future outcomes, hidden benchmark answers, and reward signals from online
context.

## Frozen experience library

Learning is asynchronous and split-safe:

```text
derivation failure -> diagnosis -> candidate -> held-out replay
                   -> promotion -> new immutable library version
```

An active library is frozen for an evaluation run. Failed evaluation episodes
never feed back into that run. Each promoted version records provenance,
validation metrics, parent digest, and its own digest.

## Adapter responsibility

Each adapter owns native action mapping, visible-state mapping, host-specific
hard checks, execution/block semantics, and outcome mapping. It must not copy
the core library, retrieval, evaluator, learning, or versioning code.

Current adapters:

- AgenticPay: seller response before `env.step()`.
- CoffeeBench: focal `BusinessApp` operational tool before invocation.
- ShoppingBench: planned; final `recommend_product` is the primary controlled
  boundary while search and inspection tools are normally observable context.
