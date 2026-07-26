"""Commerce tool definition: the source of truth for the ToolGuard build.

scripts/build_toolguard_commerce.py feeds tool_functions() straight into
toolguard.build_toolguards (IBM/tool_guard, pinned commit
20e21db4c275d79f8d7bf33ffb985d0b45f786f5). The file
specs/commerce_tools.yaml is only a human readable mirror of this module and
is verified against it by
"python scripts/build_toolguard_commerce.py --verify-mirror".

There is exactly one tool: every seller action in the negotiation is routed
through execute_seller_action before it may reach the environment.

Note: this module intentionally does *not* use
"from __future__ import annotations", because the ToolGuard API extractor
inspects the real annotation objects of the tool signature.
"""

import json
from hashlib import sha256
from typing import Any, Callable, Dict, List, Optional

COMMERCE_APP_NAME = "commerce_negotiation"
COMMERCE_TOOL_NAME = "execute_seller_action"

# Arguments that are platform-visible only: they are passed to the platform
# guard but must never be forwarded to the seller LLM, to the seller-facing
# retry feedback, or into any seller-visible state.
SELLER_HIDDEN_ARGUMENT_NAMES = ("buyer_max_price",)

TOOL_DESCRIPTION = (
    "Execute one seller action in a single-product price negotiation. The "
    "platform calls this tool after the seller agent produced a candidate "
    "message and before that message is handed to the negotiation "
    "environment. The call is either allowed unchanged or blocked; the tool "
    "never edits, clamps, or rewrites the seller message."
)

PARAMETER_DOCS: Dict[str, str] = {
    "intent": (
        "Coarse seller intent parsed from the draft message: one of "
        "negotiate_price, accept_deal, reject_deal, request_info, "
        "explain_policy, escalate, tool_call, other."
    ),
    "proposed_price": (
        "Explicit price in USD proposed by the seller in this message, or null "
        "when the message contains no explicit price."
    ),
    "actor_role": (
        "Role of the acting agent. Only the seller role may execute this tool."
    ),
    "message": "Verbatim seller draft message that would be sent to the buyer.",
    "round_id": "Zero-based negotiation round index of this action.",
    "seller_min_price": (
        "Seller floor in USD: the lowest price the seller is permitted to "
        "offer or accept."
    ),
    "buyer_max_price": (
        "Platform-visible buyer budget cap in USD. Available to the platform "
        "guard only; it is never disclosed to the seller agent."
    ),
    "product_price": "List price in USD of the single product under negotiation.",
}

TOOL_ARGUMENT_NAMES = (
    "intent",
    "proposed_price",
    "actor_role",
    "message",
    "round_id",
    "seller_min_price",
    "buyer_max_price",
    "product_price",
)

TOOL_ARGS_JSON_SCHEMA: Dict[str, Any] = {
    "title": "ExecuteSellerActionArgs",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intent": {"type": "string", "description": PARAMETER_DOCS["intent"]},
        "proposed_price": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "description": PARAMETER_DOCS["proposed_price"],
        },
        "actor_role": {"type": "string", "description": PARAMETER_DOCS["actor_role"]},
        "message": {"type": "string", "description": PARAMETER_DOCS["message"]},
        "round_id": {"type": "integer", "description": PARAMETER_DOCS["round_id"]},
        "seller_min_price": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "description": PARAMETER_DOCS["seller_min_price"],
        },
        "buyer_max_price": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "description": PARAMETER_DOCS["buyer_max_price"],
        },
        "product_price": {
            "anyOf": [{"type": "number"}, {"type": "null"}],
            "description": PARAMETER_DOCS["product_price"],
        },
    },
    "required": list(TOOL_ARGUMENT_NAMES),
}

try:  # pydantic is needed to build guards; it stays optional at runtime.
    from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover - only hit without pydantic
    ExecuteSellerActionArgs = None
else:

    class ExecuteSellerActionArgs(BaseModel):
        """Typed arguments of the single commerce tool."""

        intent: str = Field(..., description=PARAMETER_DOCS["intent"])
        proposed_price: Optional[float] = Field(
            ..., description=PARAMETER_DOCS["proposed_price"]
        )
        actor_role: str = Field(..., description=PARAMETER_DOCS["actor_role"])
        message: str = Field(..., description=PARAMETER_DOCS["message"])
        round_id: int = Field(..., description=PARAMETER_DOCS["round_id"])
        seller_min_price: Optional[float] = Field(
            ..., description=PARAMETER_DOCS["seller_min_price"]
        )
        buyer_max_price: Optional[float] = Field(
            ..., description=PARAMETER_DOCS["buyer_max_price"]
        )
        product_price: Optional[float] = Field(
            ..., description=PARAMETER_DOCS["product_price"]
        )


def execute_seller_action(
    intent: str,
    proposed_price: Optional[float],
    actor_role: str,
    message: str,
    round_id: int,
    seller_min_price: Optional[float],
    buyer_max_price: Optional[float],
    product_price: Optional[float],
) -> Dict[str, Any]:
    """Execute one seller action in a single-product price negotiation.

    The platform calls this tool once per seller turn, after the seller agent
    produced a candidate message and before the message is handed to the
    negotiation environment. The tool body performs no price adjustment: the
    action is either cleared for the environment exactly as drafted, or the
    call is rejected by the guard and never executed.

    Args:
        intent: Coarse seller intent parsed from the draft message.
        proposed_price: Explicit price in USD in the message, or None.
        actor_role: Role of the acting agent; only seller may execute.
        message: Verbatim seller draft message.
        round_id: Zero-based negotiation round index.
        seller_min_price: Seller floor in USD.
        buyer_max_price: Platform-visible buyer budget cap in USD.
        product_price: List price in USD of the product under negotiation.

    Returns:
        A record describing the cleared action. The caller forwards the
        message field to the environment unchanged.
    """
    return {
        "tool": COMMERCE_TOOL_NAME,
        "status": "cleared_for_environment",
        "actor_role": actor_role,
        "intent": intent,
        "round_id": round_id,
        "message": message,
        "proposed_price": proposed_price,
        "seller_min_price": seller_min_price,
        "product_price": product_price,
    }


# Attributes recognised by toolguard.core.functions_to_tool_info.
execute_seller_action.name = COMMERCE_TOOL_NAME
execute_seller_action.description = TOOL_DESCRIPTION
if ExecuteSellerActionArgs is not None:
    execute_seller_action.args_schema = ExecuteSellerActionArgs


def tool_functions() -> List[Callable[..., Any]]:
    """Return the callables handed to the ToolGuard build."""
    return [execute_seller_action]


def build_tool_arguments(
    *,
    intent: str,
    proposed_price: Optional[float],
    actor_role: str,
    message: str,
    round_id: int,
    seller_min_price: Optional[float],
    buyer_max_price: Optional[float],
    product_price: Optional[float],
) -> Dict[str, Any]:
    """Build the guard argument mapping in the declared argument order."""
    return {
        "intent": intent,
        "proposed_price": proposed_price,
        "actor_role": actor_role,
        "message": message,
        "round_id": round_id,
        "seller_min_price": seller_min_price,
        "buyer_max_price": buyer_max_price,
        "product_price": product_price,
    }


def tool_spec() -> Dict[str, Any]:
    """Return the machine readable tool spec mirrored by the YAML file."""
    return {
        "app_name": COMMERCE_APP_NAME,
        "tools": [
            {
                "name": COMMERCE_TOOL_NAME,
                "description": TOOL_DESCRIPTION,
                "platform_only_arguments": list(SELLER_HIDDEN_ARGUMENT_NAMES),
                "parameters": TOOL_ARGS_JSON_SCHEMA,
            }
        ],
    }


def tool_spec_json() -> str:
    """Serialise tool_spec() deterministically."""
    return json.dumps(tool_spec(), sort_keys=True, separators=(",", ":"))


def tool_spec_hash() -> str:
    """Return a deterministic sha256 hex digest of the tool spec."""
    return sha256(tool_spec_json().encode("utf-8")).hexdigest()
