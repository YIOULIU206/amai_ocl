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


AGENTICPAY_HARD_CONSTRAINT_SUITE_VERSION = 1

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

_SELLER_PRICE_MARKER = re.compile(
    r"###\s*SELLER_PRICE\s*\(\s*\$"
    r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"\s*\)\s*###",
    flags=re.IGNORECASE,
)
_SELLER_PRICE_LABEL = re.compile(r"SELLER_PRICE", flags=re.IGNORECASE)
_PAYMENT_CARD_TERMS = re.compile(
    r"\b(?:credit|debit|payment|bank)\s+card\b|\bcard\s+(?:number|details?)\b",
    flags=re.IGNORECASE,
)
_LONG_PAYMENT_NUMBER = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_SSN_VALUE = re.compile(r"(?<!\d)\d{3}[ -]\d{2}[ -]\d{4}(?!\d)")
_CVV_VALUE = re.compile(
    r"\b(?:cvv|cvc|card\s+security\s+code|security\s+code)\b"
    r"\s*(?:is\s+|[:=]\s*)?(\d{3,4})\b",
    flags=re.IGNORECASE,
)
_ROUTING_VALUE = re.compile(
    r"\b(?:bank\s+)?routing(?:\s+number)?\b"
    r"\s*(?:is\s+|[:=]\s*)?(\d{9})\b",
    flags=re.IGNORECASE,
)
_ACCOUNT_VALUE = re.compile(
    r"\bbank\s+account(?:\s+number)?\b"
    r"\s*(?:is\s+|[:=]\s*)?(\d{6,17})\b",
    flags=re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"\b(?:password|passcode)\b\s*(?:"
    r"(?:is\s+)?['\"]([^'\"]{4,})['\"]|"
    r"[:=]\s*([A-Za-z0-9_@#$%^&*+!.-]{6,})"
    r")",
    flags=re.IGNORECASE,
)


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
class AgenticPaySellerBoundaryHardValidator:
    """Hard Constraint for the AgenticPay seller action boundary."""

    expected_actor_id: str
    allowed_action_type: str = "commerce.respond"

    def validate(
        self,
        action: ProposedAction,
        context: ObservableContext,
    ) -> tuple[CheckResult, ...]:
        del context
        violations: list[str] = []
        if action.actor_id != self.expected_actor_id:
            violations.append("actor is not the configured seller")
        if action.action_type != self.allowed_action_type:
            violations.append("action type is not permitted at the seller boundary")
        return (
            CheckResult(
                check_id="agenticpay_seller_boundary",
                passed=not violations,
                level=CheckLevel.CRITICAL if violations else CheckLevel.INFO,
                reason="; ".join(violations),
                source="agenticpay_hard_constraint",
                recommended_decision=(DecisionType.BLOCK if violations else None),
                metadata={
                    "expected_actor_id": self.expected_actor_id,
                    "allowed_action_type": self.allowed_action_type,
                },
            ),
        )


@dataclass(frozen=True, slots=True)
class AgenticPaySellerPriceHardValidator:
    """Hard Constraints for AgenticPay seller-price format and floor."""

    seller_min_price: float
    require_price_marker: bool = True

    def validate(
        self,
        action: ProposedAction,
        context: ObservableContext,
    ) -> tuple[CheckResult, ...]:
        del context
        text = action.visible_text or ""
        matches = tuple(_SELLER_PRICE_MARKER.finditer(text))
        label_present = _SELLER_PRICE_LABEL.search(text) is not None
        if not matches:
            invalid = self.require_price_marker or label_present
            return (
                CheckResult(
                    check_id="agenticpay_seller_price_format",
                    passed=not invalid,
                    level=CheckLevel.ERROR if invalid else CheckLevel.INFO,
                    reason=(
                        "Seller proposal must contain exactly one valid "
                        "### SELLER_PRICE($X) ### marker."
                        if invalid
                        else ""
                    ),
                    source="agenticpay_hard_constraint",
                    recommended_decision=(DecisionType.REVISE if invalid else None),
                    metadata={"marker_count": 0},
                ),
            )
        if len(matches) != 1:
            return (
                CheckResult(
                    check_id="agenticpay_seller_price_format",
                    passed=False,
                    level=CheckLevel.ERROR,
                    reason="Seller proposal must contain exactly one seller-price marker.",
                    source="agenticpay_hard_constraint",
                    recommended_decision=DecisionType.REVISE,
                    metadata={"marker_count": len(matches)},
                ),
            )

        raw_price = matches[0].group(1)
        price = float(raw_price.replace(",", ""))
        positive = price > 0
        format_check = CheckResult(
            check_id="agenticpay_seller_price_format",
            passed=positive,
            level=CheckLevel.ERROR if not positive else CheckLevel.INFO,
            reason="Seller price must be greater than zero." if not positive else "",
            source="agenticpay_hard_constraint",
            recommended_decision=(DecisionType.REVISE if not positive else None),
            metadata={"marker_count": 1, "price": price},
        )
        respects_floor = price >= self.seller_min_price
        floor_check = CheckResult(
            check_id="agenticpay_seller_price_floor",
            passed=respects_floor,
            level=CheckLevel.ERROR if not respects_floor else CheckLevel.INFO,
            reason="Seller price is below the configured seller floor." if not respects_floor else "",
            source="agenticpay_hard_constraint",
            recommended_decision=(DecisionType.REVISE if not respects_floor else None),
            metadata={"price": price},
        )
        return (format_check, floor_check)


@dataclass(frozen=True, slots=True)
class AgenticPayExplicitSensitiveDataHardValidator:
    """Block explicit credential values without keyword-only false positives."""

    def validate(
        self,
        action: ProposedAction,
        context: ObservableContext,
    ) -> tuple[CheckResult, ...]:
        del context
        text = action.visible_text or ""
        matches: list[str] = []
        if _PAYMENT_CARD_TERMS.search(text) and _LONG_PAYMENT_NUMBER.search(text):
            matches.append("payment_card_number")
        if _SSN_VALUE.search(text):
            matches.append("ssn")
        if _CVV_VALUE.search(text):
            matches.append("card_security_code")
        if _ROUTING_VALUE.search(text):
            matches.append("routing_number")
        if _ACCOUNT_VALUE.search(text):
            matches.append("bank_account_number")
        if _SECRET_VALUE.search(text):
            matches.append("password_or_passcode")
        kinds = tuple(dict.fromkeys(matches))
        return (
            CheckResult(
                check_id="agenticpay_explicit_sensitive_data",
                passed=not kinds,
                level=CheckLevel.CRITICAL if kinds else CheckLevel.INFO,
                reason=(
                    "Seller proposal contains an explicit sensitive credential value."
                    if kinds
                    else ""
                ),
                source="agenticpay_hard_constraint",
                recommended_decision=(DecisionType.BLOCK if kinds else None),
                # Record only categories; never duplicate credential values into audit logs.
                metadata={"detected_kinds": kinds},
            ),
        )


def agenticpay_hard_constraint_validators(
    *,
    seller_actor_id: str,
    seller_min_price: float,
) -> tuple[
    AgenticPaySellerBoundaryHardValidator,
    AgenticPaySellerPriceHardValidator,
    AgenticPayExplicitSensitiveDataHardValidator,
]:
    """Build the complete static Hard Constraint suite for this integration."""

    return (
        AgenticPaySellerBoundaryHardValidator(expected_actor_id=seller_actor_id),
        AgenticPaySellerPriceHardValidator(seller_min_price=seller_min_price),
        AgenticPayExplicitSensitiveDataHardValidator(),
    )
