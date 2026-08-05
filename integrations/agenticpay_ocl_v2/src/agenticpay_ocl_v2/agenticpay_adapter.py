"""Thin observable-text adapter for the AgenticPay seller action boundary."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from aocl_core.contracts import (
    CheckLevel,
    CheckResult,
    ControlDecision,
    DecisionType,
    ObservableContext,
    ObservedOutcome,
    ProposedAction,
)
from aocl_core.runtime import IntegrationOCLRuntime


_FORBIDDEN_CONTEXT_KEYS = {
    "answer",
    "buyer_max_price",
    "label",
    "oracle",
    "persona_type",
    "profile_id",
    "reward",
    "seller_min_price",
    "user_profile",
}


def _forbidden_key(key: object) -> bool:
    normalized = str(key).casefold()
    return normalized in _FORBIDDEN_CONTEXT_KEYS or any(
        marker in normalized
        for marker in ("oracle", "reward", "score", "hidden", "ground_truth")
    )


def _sanitized_visible_state(state: Mapping[str, Any] | None) -> dict[str, Any]:
    if state is None:
        return {}
    result: dict[str, Any] = {}
    for key, value in state.items():
        if _forbidden_key(key):
            continue
        if isinstance(value, Mapping):
            result[key] = _sanitized_visible_state(value)
        elif isinstance(value, (tuple, list)):
            result[key] = [
                _sanitized_visible_state(item) if isinstance(item, Mapping) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def _visible_dialogue(history: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    allowed = {"role", "content", "text", "round"}
    return tuple(
        {key: item[key] for key in allowed if key in item}
        for item in history
    )


@dataclass(frozen=True, slots=True)
class HostActionDisposition:
    action_id: str
    decision: ControlDecision
    execute: bool
    seller_text: str | None
    requires_revision: bool
    requires_escalation: bool


@dataclass(slots=True)
class AgenticPayOCLAdapter:
    runtime: IntegrationOCLRuntime
    action_type: str = "commerce.respond"

    def evaluate_seller_text(
        self,
        *,
        episode_id: str,
        step_id: int,
        actor_id: str,
        seller_text: str,
        dialogue: Sequence[Mapping[str, Any]],
        visible_state: Mapping[str, Any] | None = None,
        attempt: int = 0,
    ) -> HostActionDisposition:
        action_id = f"{episode_id}:{step_id}:{actor_id}:{attempt}"
        action = ProposedAction(
            action_id=action_id,
            actor_id=actor_id,
            action_type=self.action_type,
            visible_text=seller_text,
            payload={"text": seller_text},
        )
        context = ObservableContext(
            episode_id=episode_id,
            step_id=step_id,
            actor_id=actor_id,
            dialogue=_visible_dialogue(dialogue),
            visible_state=_sanitized_visible_state(visible_state),
        )
        decision = self.runtime.evaluate(action=action, context=context)
        execute = decision.decision in {DecisionType.APPROVE, DecisionType.WARN}
        return HostActionDisposition(
            action_id=action_id,
            decision=decision,
            execute=execute,
            seller_text=seller_text if execute else None,
            requires_revision=decision.decision is DecisionType.REVISE,
            requires_escalation=decision.decision is DecisionType.ESCALATE,
        )

    def observe(
        self,
        disposition: HostActionDisposition,
        *,
        status: str,
        visible_result: Mapping[str, Any] | None = None,
    ) -> None:
        self.runtime.observe(
            ObservedOutcome(
                action_id=disposition.action_id,
                executed=disposition.execute,
                status=status,
                visible_result=visible_result or {},
            )
        )


@dataclass(frozen=True, slots=True)
class SellerPriceBoundsValidator:
    """Platform-owned hard check; bounds are not exposed in ObservableContext."""

    seller_min_price: float | None = None
    buyer_max_price: float | None = None

    def validate(
        self,
        action: ProposedAction,
        context: ObservableContext,
    ) -> tuple[CheckResult, ...]:
        del context
        text = action.visible_text or ""
        matches = re.findall(r"\$([\d,]+(?:\.\d+)?)", text)
        if not matches:
            return (CheckResult(check_id="price_bounds", passed=True),)
        price = float(matches[-1].replace(",", ""))
        violations: list[str] = []
        if self.seller_min_price is not None and price < self.seller_min_price:
            violations.append("seller floor")
        if self.buyer_max_price is not None and price > self.buyer_max_price:
            violations.append("buyer cap")
        return (
            CheckResult(
                check_id="price_bounds",
                passed=not violations,
                level=CheckLevel.ERROR if violations else CheckLevel.INFO,
                reason=(f"Price violates: {', '.join(violations)}" if violations else ""),
                source="agenticpay_validator",
                recommended_decision=(DecisionType.REVISE if violations else None),
                metadata={"price": price},
            ),
        )
