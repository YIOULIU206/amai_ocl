"""Runtime rule evaluation for the independent AgentSpec-Commerce baseline.

The runtime follows AgentSpec's declarative execution structure:

    trigger -> predicate -> enforcement

It does not use LangChain, ToolGuard, EGI control, deterministic repair,
price clamping, escalation, or replanning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from aimai_ocl.baselines.agentspec_commerce.policy import (
    ParsedRule,
    load_policy,
)
from aimai_ocl.baselines.agentspec_commerce.predicates import (
    PredicateContext,
    evaluate_predicate,
)
from aimai_ocl.schemas import RawAction, ViolationType

SELLER_ACTION_EVENT = "SellerAction"


class EnforcementDecision(str, Enum):
    """Supported AgentSpec-Commerce runtime outcomes."""

    CONTINUE = "continue"
    SELF_REFLECT = "llm_self_reflect"
    SKIP = "skip"
    STOP = "stop"


class AgentSpecRuntimeError(RuntimeError):
    """Base error for AgentSpec-Commerce runtime evaluation."""


class UnsupportedEnforcementError(AgentSpecRuntimeError):
    """Raised when a policy uses an unsupported enforcement."""


@dataclass(frozen=True)
class RuleEvaluation:
    """Evaluation record for one declarative AgentSpec rule."""

    rule_id: str
    event: str
    predicate_name: str
    predicate_version: int
    enforcement: str
    triggered: bool
    violated: bool
    reason: str
    violation_type: ViolationType | None = None
    risk_score: float | None = None


@dataclass(frozen=True)
class RuntimeVerdict:
    """Final runtime decision for one proposed seller action."""

    allowed: bool
    decision: EnforcementDecision
    matched_rule_id: str | None = None
    reason: str | None = None
    violation_type: ViolationType | None = None
    evaluations: tuple[RuleEvaluation, ...] = field(default_factory=tuple)

    @property
    def requires_self_reflection(self) -> bool:
        return self.decision == EnforcementDecision.SELF_REFLECT


class AgentSpecCommerceRuntime:
    """Evaluate seller actions against AgentSpec-Commerce policy rules."""

    def __init__(
        self,
        rules: Sequence[ParsedRule] | None = None,
    ) -> None:
        loaded_rules = list(rules) if rules is not None else load_policy()

        if not loaded_rules:
            raise AgentSpecRuntimeError(
                "AgentSpec-Commerce runtime requires at least one rule."
            )

        self._rules = tuple(loaded_rules)
        self._validate_rules()

    @property
    def rules(self) -> tuple[ParsedRule, ...]:
        return self._rules

    def evaluate(
        self,
        *,
        action: RawAction,
        buyer_max_price: float | None,
        seller_min_price: float | None,
        event: str = SELLER_ACTION_EVENT,
    ) -> RuntimeVerdict:
        """Evaluate one seller proposal in policy order.

        AgentSpec enforcement is applied to the first triggered rule whose
        predicate evaluates to true.
        """
        context = PredicateContext(
            action=action,
            buyer_max_price=buyer_max_price,
            seller_min_price=seller_min_price,
        )

        evaluations: list[RuleEvaluation] = []

        for rule in self._rules:
            triggered = rule.event in {"any", event}

            if not triggered:
                evaluations.append(
                    RuleEvaluation(
                        rule_id=rule.rule_id,
                        event=rule.event,
                        predicate_name=rule.predicate_name,
                        predicate_version=rule.predicate_version,
                        enforcement=rule.enforcement,
                        triggered=False,
                        violated=False,
                        reason="Rule trigger did not match the current event.",
                    )
                )
                continue

            predicate_verdict = evaluate_predicate(
                rule.predicate_name,
                rule.predicate_version,
                context,
            )

            evaluation = RuleEvaluation(
                rule_id=rule.rule_id,
                event=rule.event,
                predicate_name=rule.predicate_name,
                predicate_version=rule.predicate_version,
                enforcement=rule.enforcement,
                triggered=True,
                violated=predicate_verdict.violated,
                reason=predicate_verdict.reason,
                violation_type=predicate_verdict.violation_type,
                risk_score=predicate_verdict.risk_score,
            )
            evaluations.append(evaluation)

            if not predicate_verdict.violated:
                continue

            decision = self._decision_for_enforcement(rule.enforcement)

            if decision == EnforcementDecision.CONTINUE:
                continue

            return RuntimeVerdict(
                allowed=False,
                decision=decision,
                matched_rule_id=rule.rule_id,
                reason=predicate_verdict.reason,
                violation_type=predicate_verdict.violation_type,
                evaluations=tuple(evaluations),
            )

        return RuntimeVerdict(
            allowed=True,
            decision=EnforcementDecision.CONTINUE,
            evaluations=tuple(evaluations),
        )

    def _validate_rules(self) -> None:
        """Fail closed when the policy contains unsupported enforcement."""
        for rule in self._rules:
            self._decision_for_enforcement(rule.enforcement)

    @staticmethod
    def _decision_for_enforcement(
        enforcement: str,
    ) -> EnforcementDecision:
        mapping = {
            "none": EnforcementDecision.CONTINUE,
            "llm_self_reflect": EnforcementDecision.SELF_REFLECT,
            "skip": EnforcementDecision.SKIP,
            "stop": EnforcementDecision.STOP,
        }

        decision = mapping.get(enforcement)

        if decision is None:
            raise UnsupportedEnforcementError(
                "Unsupported AgentSpec-Commerce enforcement: "
                + repr(enforcement)
                + "."
            )

        return decision
