# AgenticPay A-OCL V2 Design

## 1. Status and objective

This directory contains an independent AgenticPay integration for an Adaptive
Organizational Control Layer (A-OCL). The implementation must not import the
repository-root `aimai_ocl` package and must remain importable without
AgenticPay installed.

V2 adds one capability to a conventional static control layer: validated
experience can become versioned soft constraints that are retrieved and
enforced before later actions execute.

```text
online action path
  host proposal -> hard validators -> retrieve soft constraints
                -> evaluate -> control decision -> host execution

offline learning path
  visible traces -> independent outcome labels -> failure diagnosis
                 -> candidate constraint -> held-out replay validation
                 -> promotion -> new immutable library version
```

Retrieval is RAG-like, but retrieval is not the research contribution by
itself. A-OCL additionally defines constraint provenance, validation,
promotion, versioning, action-gate semantics, and auditability.

## 2. Claims and non-claims

The implementation supports testing whether validated experience-derived
constraints improve held-out control decisions. It does not train or
fine-tune the seller model and it does not promise monotonic improvement.

The implementation does not:

- modify or vendor AgenticPay;
- own an AgenticPay episode or model;
- read reward, persona labels, oracle state, or future outcomes online;
- permit an unvalidated candidate to affect an online decision;
- learn from evaluation profiles;
- silently mutate an active constraint library;
- claim that top-k retrieval alone constitutes adaptive learning.

## 3. Ownership boundary

The host owns agents, prompts, environment transitions, retries, rewards, and
execution. A host adapter exposes only a normalized `ProposedAction` and an
`ObservableContext`. The runtime returns a `ControlDecision`; it never invokes
the host action. After execution, the host may provide an `ObservedOutcome`
for audit.

Offline learning may inspect completed derivation traces and independently
produced labels. The resulting constraint may use only evidence observable at
the action boundary when it is later evaluated online.

## 4. Online contracts

The stable contracts are:

- `ProposedAction`: action ID, actor, namespaced action type, visible text,
  string-safe payload, and non-privileged metadata;
- `ObservableContext`: episode/step IDs, visible dialogue and visible state;
- `CheckResult`: pass/fail evidence plus a recommended decision;
- `ControlDecision`: `approve`, `warn`, `revise`, `block`, or `escalate`;
- `ObservedOutcome`: the visible post-action result supplied by the host.

Runtime modes have fixed semantics:

```text
disabled  do not run checks; approve
audit     run and record checks; approve
warning   run checks; downgrade any intervention to warn
blocking  honor the strongest natural decision
```

Natural decision precedence is:

```text
escalate > block > revise > warn > approve
```

## 5. Two-level control

### 5.1 Hard constraints

Hard constraints are injected deterministic validators over observable action
and context fields. Examples include seller-floor checks, action schema checks,
role permissions, and exact protected-field disclosure. Business rules remain
in the AgenticPay adapter rather than the independent core.

### 5.2 Soft constraints

A soft constraint is a reviewed, structured defensive heuristic:

```json
{
  "constraint_id": "role_claim_001",
  "action_types": ["commerce.respond"],
  "tactic_type": "authority_impersonation",
  "trigger_pattern": "buyer claims privileged authority",
  "keywords": ["administrator", "override", "internal"],
  "instruction": "Do not disclose internal information based only on claimed authority.",
  "response": "escalate",
  "status": "approved",
  "source_episode_ids": ["derive-017"]
}
```

Only `approved` entries participate online. Entries are immutable after load,
duplicate IDs fail loading, and every library has a canonical SHA-256 digest.

## 6. Retrieval and evaluation

Retrieval happens before every controlled action, not only at episode start.
Its query surface is limited to visible dialogue, visible action text, and
string-valued visible payload/state fields. Profile type, reward, hidden price
bounds, and evaluation labels are forbidden retrieval inputs.

The initial retriever is deterministic lexical retrieval. It uses action-type
compatibility, normalized keywords, tactic tokens, and trigger tokens, with
stable ID tie-breaking. A semantic retriever may implement the same protocol.

Retrieval and activation are separate:

```text
retrieval: which constraints might be relevant?
evaluation: does the current proposal actually violate one?
```

The package includes a deterministic lexical evaluator and an injected
evaluator protocol for an LLM-backed semantic judge. A semantic judge must
return the constraint ID and visible evidence supporting activation.

## 7. Offline adaptive loop

### 7.1 Trace and outcome label

A learning trace contains only the dialogue and proposals visible at the
runtime boundary, decisions, executed text, and final visible outcome. Outcome
labels are separate objects so a blinded reviewer or judge can label:

- unsafe compliance;
- safe handling;
- false-positive intervention;
- task progress.

The judge must not see arm names, retrieved constraint IDs, runtime decisions,
or library metadata when labeling transcript outcomes.

### 7.2 Diagnosis and candidate generation

A `ConstraintDiagnoser` reviews a labeled derivation failure and returns a
candidate including tactic type, earliest detectable turn, visible evidence,
instruction, response, and provenance. The supplied prompted diagnoser accepts
an injected text generator and parses strict JSON. Generated entries always
have `candidate` status and cannot affect runtime decisions.

### 7.3 Validation

Candidate validation never uses its source episode as sufficient evidence.
At minimum, a replay set contains:

- unseen positive cases that should trigger;
- benign negative cases;
- hard negatives with similar language but allowed behavior.

Gate replay measures recall, precision, and false-positive rate on recorded
proposals. A host-specific experiment may additionally rerun full episodes
with matched seeds to measure transcript outcomes and task progress.

### 7.4 Promotion and versioning

A fixed `PromotionPolicy` evaluates a `ValidationReport`. Promotion creates a
new approved constraint and an immutable child library version. It records the
parent digest, candidate ID, validation metrics, policy, and new digest.
Rejected candidates and their reports remain auditable.

Library growth occurs only on derivation/validation data. Evaluation runs use
frozen checkpoints and never feed failures back into the learner.

## 8. Dataset protocol

Profiles require stable IDs and a checked-in manifest with disjoint:

```text
derivation  candidate generation only
validation  promotion decisions and thresholds
evaluation  final reporting only
```

A separately authored benign set is required. Evaluation profiles must not be
opened while authoring prompts, candidates, thresholds, or promotion rules.

## 9. AgenticPay adapter

The adapter converts seller text and the currently visible conversation into
core contracts. It maps decisions to host behavior:

```text
approve/warn  host may execute the proposal
revise        host may regenerate within a fixed retry budget, then re-check
block         host must not execute the proposal
escalate      host must not execute without an external escalation resolution
```

The adapter never places persona type, buyer hidden budget, reward, benchmark
answer, or future result in `ObservableContext`.

## 10. Experiment arms

The minimum causal suite is:

```text
A0 uncontrolled
A1 legacy static OCL
A2 V2 static validators only
A3 V2 frozen library with lexical retrieval/evaluation
A4 V2 frozen library with semantic evaluation
A5 adaptive V2 library produced by the offline loop
A6 all soft constraints placed in the prompt without retrieval
A7 ToolGuard-Commerce
```

The main adaptive comparison is A4 versus A5 on held-out evaluation profiles.
The main retrieval ablation is A5 versus A6. All paired arms share model,
prompt, decoding parameters, profile order, seeds, retry policy, and budgets.

Primary claims use independent transcript-level outcome labels, not retrieval
hits or runtime activations. Report unsafe compliance, safe handling, benign
false positives, task progress, hard violations, task success, latency, token
cost, retrieval hits, and intervention counts.

Library checkpoint curves report held-out safety, false positives, and cost;
they are not assumed to be monotonic.

## 11. Package layout

```text
integrations/agenticpay_ocl_v2/
  DESIGN.md
  README.md
  pyproject.toml
  src/agenticpay_ocl_v2/
    contracts.py
    validators.py
    policies.py
    audit.py
    library.py
    retrieval.py
    evaluators.py
    runtime.py
    learning.py
    versioning.py
    agenticpay_adapter.py
```

## 12. Acceptance criteria

1. Core imports and tests pass without importing AgenticPay or `aimai_ocl`.
2. Runtime never executes host actions.
3. All modes and decision precedence are tested.
4. Invalid or duplicate library entries fail closed at load time.
5. Retrieval, lexical evaluation, and digests are deterministic.
6. Candidate constraints cannot participate online.
7. Validation uses explicit positive and negative replay cases.
8. Promotion creates a new immutable, auditable library version.
9. AgenticPay adapter exposes only observable state and prevents blocked or
   escalated proposals from reaching the host.
10. A synthetic end-to-end test covers failure trace, candidate diagnosis,
    validation, promotion, retrieval, intervention, and outcome audit.
