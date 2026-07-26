"""Platform-side policy state, redaction, and deterministic proposal ids.

The platform guard sees the buyer budget cap; the seller LLM never does. This
module is the single place where that asymmetry is implemented, together with
the deterministic proposal id used to key all ToolGuard-Commerce metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping

from aimai_ocl.baselines.toolguard_commerce.tools import (
    SELLER_HIDDEN_ARGUMENT_NAMES,
    build_tool_arguments,
)

# Formal configuration value of toolguard_buyer_max_price_visibility.
BUYER_MAX_VISIBILITY_PLATFORM_VISIBLE = "platform_visible"
SUPPORTED_BUYER_MAX_VISIBILITIES = (BUYER_MAX_VISIBILITY_PLATFORM_VISIBLE,)

PROPOSAL_ID_PREFIX = "tgc-"
PROPOSAL_ID_DIGEST_CHARS = 20
_FIELD_SEPARATOR = "\x1f"

REDACTED_TOKEN = "[redacted-platform-only]"
GENERIC_VIOLATION_REASON = "The draft violated the platform seller policy."
MAX_FEEDBACK_CHARS = 600

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_SECRET_NAME_RE = re.compile(
    r"buyer[_\s-]*(?:max|maximum|budget)(?:[_\s-]*price)?"
    r"|budget[_\s-]*cap"
    r"|buyer[_\s-]*willingness[_\s-]*to[_\s-]*pay",
    re.IGNORECASE,
)
_SECRET_NAME_REPLACEMENT = "the platform budget policy"


class BuyerMaxVisibilityError(ValueError):
    """Raised when an unsupported buyer_max visibility is configured."""


def normalize_buyer_max_visibility(value: Any) -> str:
    """Validate the configured buyer_max visibility, failing closed.

    Input:
        value: Raw configuration value.

    Output:
        The normalized visibility string.

    Raises:
        BuyerMaxVisibilityError: If the value is missing or unsupported. The
            baseline never falls back to a default, because a wrong value here
            would decide whether the buyer budget cap leaks to the seller.
    """
    if value is None:
        raise BuyerMaxVisibilityError(
            "toolguard_buyer_max_price_visibility is not configured; "
            "expected one of: " + ", ".join(SUPPORTED_BUYER_MAX_VISIBILITIES)
        )
    candidate = str(value).strip().lower()
    if candidate not in SUPPORTED_BUYER_MAX_VISIBILITIES:
        raise BuyerMaxVisibilityError(
            "Unsupported toolguard_buyer_max_price_visibility "
            + repr(value)
            + "; supported values: "
            + ", ".join(SUPPORTED_BUYER_MAX_VISIBILITIES)
        )
    return candidate


@dataclass(frozen=True)
class PlatformPolicyState:
    """Platform-side view of one negotiation round.

    buyer_max_price is platform-visible only: guard_arguments() forwards it to
    the guard, while seller_visible_state() and sanitize_violation_feedback()
    keep it away from the seller LLM.
    """

    round_id: int
    seller_min_price: float | None = None
    buyer_max_price: float | None = None
    product_price: float | None = None
    product_name: str | None = None
    max_rounds: int | None = None
    buyer_max_price_visibility: str = BUYER_MAX_VISIBILITY_PLATFORM_VISIBLE

    def __post_init__(self) -> None:
        normalize_buyer_max_visibility(self.buyer_max_price_visibility)

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any] | None,
        *,
        round_id: int,
        buyer_max_price_visibility: Any = BUYER_MAX_VISIBILITY_PLATFORM_VISIBLE,
    ) -> "PlatformPolicyState":
        """Build the platform state from the runner control state."""
        source: Mapping[str, Any] = state or {}
        return cls(
            round_id=int(round_id),
            seller_min_price=_as_float(source.get("seller_min_price")),
            buyer_max_price=_as_float(source.get("buyer_max_price")),
            product_price=_as_float(source.get("product_price")),
            product_name=_as_str(source.get("product_name")),
            max_rounds=_as_int(source.get("max_rounds")),
            buyer_max_price_visibility=normalize_buyer_max_visibility(
                buyer_max_price_visibility
            ),
        )

    def guard_arguments(
        self,
        *,
        intent: str,
        proposed_price: float | None,
        actor_role: str,
        message: str,
    ) -> dict[str, Any]:
        """Build platform-side guard arguments, including buyer_max_price."""
        return build_tool_arguments(
            intent=intent,
            proposed_price=proposed_price,
            actor_role=actor_role,
            message=message,
            round_id=self.round_id,
            seller_min_price=self.seller_min_price,
            buyer_max_price=self.buyer_max_price,
            product_price=self.product_price,
        )

    def seller_visible_state(self, observation: Mapping[str, Any] | None) -> dict[str, Any]:
        """Return the observation with platform-only keys removed.

        For AgenticPay observations this is a plain copy, because the raw
        observation does not carry the buyer budget cap. The filter exists so
        that the cap can never reach the seller LLM even if a future
        observation started to include it.
        """
        return strip_platform_only_keys(observation)

    def secret_values(self) -> tuple[float, ...]:
        """Numeric values that must not appear in seller-facing text."""
        return tuple(value for value in (self.buyer_max_price,) if value is not None)


def strip_platform_only_keys(
    value: Any,
    *,
    hidden_keys: Iterable[str] = SELLER_HIDDEN_ARGUMENT_NAMES,
) -> dict[str, Any]:
    """Recursively drop platform-only keys from a mapping."""
    hidden = {str(key).lower() for key in hidden_keys}

    def _clean(node: Any) -> Any:
        if isinstance(node, Mapping):
            return {
                key: _clean(item)
                for key, item in node.items()
                if str(key).lower() not in hidden
            }
        if isinstance(node, list):
            return [_clean(item) for item in node]
        if isinstance(node, tuple):
            return tuple(_clean(item) for item in node)
        return node

    cleaned = _clean(value)
    return cleaned if isinstance(cleaned, dict) else {}


def sanitize_violation_feedback(
    reason: str | None,
    *,
    buyer_max_price: float | None = None,
    extra_secret_values: Iterable[float] = (),
    max_chars: int = MAX_FEEDBACK_CHARS,
) -> str:
    """Turn a raw guard message into seller-safe violation feedback.

    Platform-only vocabulary is replaced and every number matching a
    platform-only value (the buyer budget cap) is redacted, so the seller LLM
    can never read the cap out of the feedback it receives on retry.
    """
    text = (reason or "").strip()
    if not text:
        text = GENERIC_VIOLATION_REASON
    text = _SECRET_NAME_RE.sub(_SECRET_NAME_REPLACEMENT, text)

    secrets: list[float] = []
    if buyer_max_price is not None:
        secrets.append(float(buyer_max_price))
    for value in extra_secret_values:
        if value is not None:
            secrets.append(float(value))

    if secrets:

        def _redact(match: "re.Match[str]") -> str:
            try:
                number = float(match.group(0))
            except ValueError:  # pragma: no cover - regex only matches numbers
                return match.group(0)
            for secret in secrets:
                if abs(number - secret) <= 1e-6:
                    return REDACTED_TOKEN
            return match.group(0)

        text = _NUMBER_RE.sub(_redact, text)

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        text = GENERIC_VIOLATION_REASON
    if len(text) > max_chars:
        text = text[: max(0, max_chars - 3)].rstrip() + "..."
    return text


def make_proposal_id(
    *,
    episode_key: str,
    arm: str,
    round_id: int,
    attempt: int,
    actor_id: str,
    text: str | None,
) -> str:
    """Return a deterministic proposal id.

    The id is a truncated sha256 digest over episode key, arm, round, attempt,
    actor, and the exact candidate text. It is deliberately *not* uuid4, so an
    identical replay produces identical proposal ids.
    """
    payload = _FIELD_SEPARATOR.join(
        [
            str(episode_key),
            str(arm),
            str(int(round_id)),
            str(int(attempt)),
            str(actor_id),
            text or "",
        ]
    )
    digest = sha256(payload.encode("utf-8")).hexdigest()
    return PROPOSAL_ID_PREFIX + digest[:PROPOSAL_ID_DIGEST_CHARS]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
