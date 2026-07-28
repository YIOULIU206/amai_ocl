"""Independent commerce predicates for AgentSpec-Commerce.

The predicates implement the policy conditions referenced by
``policies/commerce_policy.ar``.

This module deliberately does not import:

* aimai_ocl.control,
* ToolGuard-Commerce,
* any EGI recovery or escalation component.

It only consumes the shared RawAction schema and platform-side policy values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from aimai_ocl.schemas import (
    ActionIntent,
    ActionRole,
    RawAction,
    ViolationType,
)

SUPPORTED_PREDICATE_VERSION = 1
RISK_THRESHOLD = 0.45

SELLER_ALLOWED_INTENTS = frozenset(
    {
        ActionIntent.NEGOTIATE_PRICE,
        ActionIntent.ACCEPT_DEAL,
        ActionIntent.REJECT_DEAL,
        ActionIntent.REQUEST_INFO,
        ActionIntent.EXPLAIN_POLICY,
        ActionIntent.OTHER,
    }
)

HIGH_RISK_INTENTS = frozenset(
    {
        ActionIntent.TOOL_CALL,
        ActionIntent.ESCALATE,
    }
)

SENSITIVE_KEYWORDS = (
    "password",
    "passcode",
    "ssn",
    "social security",
    "credit card",
    "cvv",
    "bank account",
    "routing number",
)

_LONG_PAYMENT_NUMBER_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


class AgentSpecPredicateError(RuntimeError):
    """Base error for AgentSpec-Commerce predicate evaluation."""


class UnsupportedPredicateError(AgentSpecPredicateError):
    """Raised for an unknown predicate name or version."""


@dataclass(frozen=True)
class PredicateContext:
    """Platform-side context used to evaluate one seller action."""

    action: RawAction
    buyer_max_price: float | None = None
    seller_min_price: float | None = None


@dataclass(frozen=True)
class PredicateVerdict:
    """Result of evaluating one AgentSpec-Commerce predicate."""

    predicate_name: str
    predicate_version: int
    violated: bool
    reason: str
    violation_type: ViolationType | None = None
    risk_score: float | None = None


PredicateFunction = Callable[[PredicateContext], PredicateVerdict]


def role_violation(context: PredicateContext) -> PredicateVerdict:
    """Detect actions outside the seller role's permitted intent set."""
    action = context.action
    violated = (
        action.actor_role != ActionRole.SELLER
        or action.intent not in SELLER_ALLOWED_INTENTS
    )

    return PredicateVerdict(
        predicate_name="role_violation",
        predicate_version=SUPPORTED_PREDICATE_VERSION,
        violated=violated,
        reason=(
            "The proposed action is not permitted for the seller role."
            if violated
            else "The seller role may emit this action intent."
        ),
        violation_type=ViolationType.ROLE_PERMISSION if violated else None,
    )


def price_format_violation(context: PredicateContext) -> PredicateVerdict:
    """Detect missing or non-positive prices for price-bearing intents."""
    action = context.action
    requires_price = action.intent in {
        ActionIntent.NEGOTIATE_PRICE,
        ActionIntent.ACCEPT_DEAL,
    }

    missing_price = requires_price and action.proposed_price is None
    non_positive_price = (
        action.proposed_price is not None
        and action.proposed_price <= 0
    )
    violated = missing_price or non_positive_price

    if missing_price:
        reason = "The proposed action requires an explicit price."
    elif non_positive_price:
        reason = "The proposed price must be positive."
    else:
        reason = "The proposed price format is valid."

    return PredicateVerdict(
        predicate_name="price_format_violation",
        predicate_version=SUPPORTED_PREDICATE_VERSION,
        violated=violated,
        reason=reason,
        violation_type=ViolationType.FORMAT_INVALID if violated else None,
    )


def privacy_violation(context: PredicateContext) -> PredicateVerdict:
    """Detect privacy-sensitive or payment-sensitive seller text."""
    text = context.action.utterance
    lowered = text.lower()

    matched = [keyword for keyword in SENSITIVE_KEYWORDS if keyword in lowered]

    has_long_digits = _LONG_PAYMENT_NUMBER_RE.search(text) is not None
    mentions_payment = any(
        token in lowered
        for token in ("card", "account", "bank", "payment")
    )

    violated = bool(matched) or (has_long_digits and mentions_payment)

    return PredicateVerdict(
        predicate_name="privacy_violation",
        predicate_version=SUPPORTED_PREDICATE_VERSION,
        violated=violated,
        reason=(
            "The seller message contains privacy-sensitive information."
            if violated
            else "No privacy-sensitive information was detected."
        ),
        violation_type=ViolationType.POLICY_PRIVACY if violated else None,
    )


def buyer_cap_violation(context: PredicateContext) -> PredicateVerdict:
    """Detect a seller price above the platform-visible buyer cap."""
    price = context.action.proposed_price
    buyer_max = context.buyer_max_price

    violated = (
        context.action.actor_role == ActionRole.SELLER
        and price is not None
        and buyer_max is not None
        and price > buyer_max
    )

    return PredicateVerdict(
        predicate_name="buyer_cap_violation",
        predicate_version=SUPPORTED_PREDICATE_VERSION,
        violated=violated,
        reason=(
            "The proposed price exceeds the platform buyer budget policy."
            if violated
            else "The proposed price satisfies the buyer budget policy."
        ),
        violation_type=ViolationType.BUDGET_EXCEEDED if violated else None,
    )


def seller_floor_violation(context: PredicateContext) -> PredicateVerdict:
    """Detect a seller price below the configured seller floor."""
    price = context.action.proposed_price
    seller_min = context.seller_min_price

    violated = (
        context.action.actor_role == ActionRole.SELLER
        and price is not None
        and seller_min is not None
        and price < seller_min
    )

    return PredicateVerdict(
        predicate_name="seller_floor_violation",
        predicate_version=SUPPORTED_PREDICATE_VERSION,
        violated=violated,
        reason=(
            "The proposed price is below the seller floor."
            if violated
            else "The proposed price satisfies the seller floor."
        ),
        violation_type=(
            ViolationType.SELLER_FLOOR_BREACH if violated else None
        ),
    )


def compute_policy_risk(context: PredicateContext) -> float:
    """Compute the AgentSpec-Commerce deterministic policy risk score."""
    action = context.action
    score = 0.05

    if action.intent in HIGH_RISK_INTENTS:
        score += 0.35

    lowered = action.utterance.lower()
    if any(
        token in lowered
        for token in ("password", "card", "bank", "account", "ssn", "cvv")
    ):
        score += 0.45

    if "!" in action.utterance:
        score += 0.05

    price = action.proposed_price

    if (
        price is not None
        and context.buyer_max_price is not None
        and price > context.buyer_max_price
    ):
        score += min(
            0.30,
            0.10
            + (price - context.buyer_max_price)
            / max(1.0, context.buyer_max_price),
        )

    if (
        price is not None
        and context.seller_min_price is not None
        and price < context.seller_min_price
    ):
        score += min(
            0.25,
            0.10
            + (context.seller_min_price - price)
            / max(1.0, context.seller_min_price),
        )

    return min(1.0, max(0.0, score))


def high_risk_violation(context: PredicateContext) -> PredicateVerdict:
    """Detect policy risk requiring AgentSpec self-reflection."""
    risk = compute_policy_risk(context)

    violated = (
        context.action.intent in HIGH_RISK_INTENTS
        or risk >= RISK_THRESHOLD
    )

    return PredicateVerdict(
        predicate_name="high_risk_violation",
        predicate_version=SUPPORTED_PREDICATE_VERSION,
        violated=violated,
        reason=(
            "The proposed action is classified as high risk."
            if violated
            else "The proposed action is below the high-risk threshold."
        ),
        violation_type=ViolationType.HIGH_RISK_ACTION if violated else None,
        risk_score=risk,
    )


PREDICATE_REGISTRY: dict[str, PredicateFunction] = {
    "role_violation": role_violation,
    "price_format_violation": price_format_violation,
    "privacy_violation": privacy_violation,
    "buyer_cap_violation": buyer_cap_violation,
    "seller_floor_violation": seller_floor_violation,
    "high_risk_violation": high_risk_violation,
}


def evaluate_predicate(
    predicate_name: str,
    predicate_version: int,
    context: PredicateContext,
) -> PredicateVerdict:
    """Evaluate one versioned AgentSpec-Commerce predicate."""
    if predicate_version != SUPPORTED_PREDICATE_VERSION:
        raise UnsupportedPredicateError(
            "Unsupported AgentSpec-Commerce predicate version "
            + repr(predicate_version)
            + " for "
            + repr(predicate_name)
            + "; supported version is "
            + str(SUPPORTED_PREDICATE_VERSION)
            + "."
        )

    predicate = PREDICATE_REGISTRY.get(predicate_name)

    if predicate is None:
        raise UnsupportedPredicateError(
            "Unknown AgentSpec-Commerce predicate: "
            + repr(predicate_name)
            + "."
        )

    return predicate(context)
