"""Synthetic ToolGuard fixtures for the ToolGuard-Commerce baseline tests.

These fixtures deliberately avoid importing toolguard: they reproduce only the
contract that the pinned ToolGuard runtime exposes to the baseline, namely

* check_toolcall(tool_name, args, delegate),
* a PolicyViolationException raised from inside the guard,
* the possibility that a guard asks its delegate for an auxiliary lookup.

Detection in the baseline is by exception class *name*, because generated
guards import PolicyViolationException from the copied runtime package inside
the generated guard folder rather than from toolguard itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

COMMERCE_TOOL_NAME = "execute_seller_action"

SENSITIVE_TOKENS = (
    "credit card",
    "cvv",
    "bank account",
    "routing number",
    "password",
)


class PolicyViolationException(Exception):
    """Stand-in for toolguard.data_types.PolicyViolationException."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self._msg = message

    @property
    def message(self) -> str:
        return self._msg


class SyntheticCommerceGuard:
    """Deterministic stand-in for the generated execute_seller_action guard.

    The refusal messages intentionally mention the platform-only argument name
    and its numeric value, so tests can prove that the seller-facing feedback
    is sanitized before the seller LLM ever sees it.
    """

    def __init__(self, *, tool_name: str = COMMERCE_TOOL_NAME) -> None:
        self.tool_name = tool_name
        self.calls: list[dict[str, Any]] = []

    def check_toolcall(self, tool_name: str, args: dict[str, Any], delegate: Any) -> None:
        self.calls.append({"tool_name": tool_name, "args": dict(args), "delegate": delegate})
        if tool_name != self.tool_name:
            return None

        message = str(args.get("message") or "")
        lowered = message.lower()
        for token in SENSITIVE_TOKENS:
            if token in lowered:
                raise PolicyViolationException(
                    "Seller message requests private payment data: " + token + "."
                )

        if str(args.get("actor_role") or "").lower() != "seller":
            raise PolicyViolationException(
                "Only the seller role may execute " + self.tool_name + "."
            )

        price = args.get("proposed_price")
        seller_min = args.get("seller_min_price")
        buyer_max = args.get("buyer_max_price")
        if price is not None:
            if price <= 0:
                raise PolicyViolationException("Proposed price must be positive.")
            if seller_min is not None and price < seller_min:
                raise PolicyViolationException(
                    "Proposed price "
                    + format(float(price), ".2f")
                    + " is below the seller floor "
                    + format(float(seller_min), ".2f")
                    + "."
                )
            if buyer_max is not None and price > buyer_max:
                raise PolicyViolationException(
                    "Proposed price "
                    + format(float(price), ".2f")
                    + " exceeds buyer_max_price "
                    + format(float(buyer_max), ".2f")
                    + "."
                )
        return None


class ScriptedGuard:
    """Guard that replays a fixed sequence of verdicts.

    Each verdict is either None (allow) or a violation reason string (block).
    """

    def __init__(self, verdicts: Sequence[str | None]) -> None:
        self._verdicts = list(verdicts)
        self.calls: list[dict[str, Any]] = []

    def check_toolcall(self, tool_name: str, args: dict[str, Any], delegate: Any) -> None:
        self.calls.append(dict(args))
        if not self._verdicts:
            return None
        reason = self._verdicts.pop(0)
        if reason is None:
            return None
        raise PolicyViolationException(reason)


class LookupRequiringGuard:
    """Guard that needs an auxiliary API lookup, which must fail closed."""

    def __init__(self, *, lookup_tool: str = "get_buyer_profile") -> None:
        self.lookup_tool = lookup_tool
        self.calls: list[dict[str, Any]] = []

    def check_toolcall(self, tool_name: str, args: dict[str, Any], delegate: Any) -> None:
        self.calls.append(dict(args))
        delegate.invoke(self.lookup_tool, {"round_id": args.get("round_id")}, dict)


class BrokenGuard:
    """Guard that fails for a non-policy reason (must not be treated as ALLOW)."""

    def check_toolcall(self, tool_name: str, args: dict[str, Any], delegate: Any) -> None:
        raise RuntimeError("synthetic guard crashed")


GUARD_MODULE_SOURCE = '''"""Synthetic generated guard module (test fixture)."""


class PolicyViolationException(Exception):
    pass


def guard_execute_seller_action(api, **kwargs):
    return None
'''


def write_synthetic_guard_dir(
    directory: str | Path,
    *,
    tool_name: str = COMMERCE_TOOL_NAME,
    include_tool_entry: bool = True,
    include_guard_module: bool = True,
) -> Path:
    """Create a minimal generated-guard folder layout for loader tests."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    guard_file_name = "guard_" + tool_name + ".py"
    if include_guard_module:
        (root / guard_file_name).write_text(GUARD_MODULE_SOURCE, encoding="utf-8")

    tools: dict[str, Any] = {}
    if include_tool_entry:
        tools[tool_name] = {
            "tool": {"tool_name": tool_name, "policy_items": []},
            "guard_fn_name": "guard_" + tool_name,
            "guard_file": {"file_name": guard_file_name, "content": GUARD_MODULE_SOURCE},
            "item_guard_files": [],
            "test_files": [],
        }
    payload = {
        "domain": {
            "app_name": "commerce_negotiation",
            "app_api_class_name": "I_CommerceNegotiation",
            "app_api_impl_class_name": "CommerceNegotiationImpl",
        },
        "tools": tools,
    }
    (root / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return root


class RecordingSeller:
    """Seller stub that records every generation call."""

    def __init__(self, replies: Iterable[Any] = (), *, name: str = "seller") -> None:
        self.name = name
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def respond(self, conversation_history: Any, current_state: Any) -> Any:
        self.calls.append(
            {"conversation_history": conversation_history, "current_state": current_state}
        )
        if self.replies:
            return self.replies.pop(0)
        return None
