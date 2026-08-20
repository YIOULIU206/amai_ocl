# A-OCL Core

Environment-independent online action gate and offline validated-constraint
learning pipeline. The core never imports a benchmark and never executes a
host action.

Host adapters normalize their native boundary into four contracts:

```text
ProposedAction + ObservableContext
                -> IntegrationOCLRuntime
                -> ControlDecision
host execution  -> ObservedOutcome
```

Install the core together with the adapter being used:

```bash
python -m pip install -e integrations/aocl_core \
  -e integrations/agenticpay_ocl_v2
```

Only approved entries from an immutable Adaptive Constraint Bank participate
online. The backward-compatible Python names are `FrozenConstraintLibrary` and
its alias `FrozenConstraintBank`. One learned constraint carries its trigger,
defensive principle, control response, and optional revision guidance; there is
no separate repair-skill type. Candidate generation and promotion remain
offline. The core supports both proposal-level replay checks and conservative
paired-rollout reports supplied by a host adapter; AgenticPay V2 uses complete
fresh episodes for formal promotion. Evaluation runs never mutate the active
bank.
