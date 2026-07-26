"""ToolGuard-Commerce: an external ToolGuard baseline for AgenticPay.

The baseline inserts IBM ToolGuard between seller generation and environment
execution:

    seller_agent.respond()
      -> ToolGuard-Commerce check (ALLOW / BLOCK)
      -> executed_seller_actions
      -> adapter.step()
      -> env.step()

Deliberate design constraints, kept strictly weaker than OCL:

* the guard may only ALLOW or BLOCK, never rewrite, clamp, or escalate,
* a blocked draft may be revised by the same seller LLM exactly once,
* a second violation becomes a no-op (None) for that round,
* the platform-visible buyer budget cap reaches the guard but never the seller
  LLM nor the seller-facing violation feedback,
* missing guards, unloadable guards, and guards that request auxiliary API
  lookups fail closed.

External dependency, pinned:
    IBM/tool_guard @ 20e21db4c275d79f8d7bf33ffb985d0b45f786f5

tools.py is the source of truth for the ToolGuard build;
specs/commerce_tools.yaml is only a readable mirror of it.
"""

from __future__ import annotations

from pathlib import Path

from aimai_ocl.baselines.toolguard_commerce.adapter import (
    GUARD_DECISION_ALLOW,
    GUARD_DECISION_BLOCK,
    GuardVerdict,
    SellerTurnOutcome,
    ToolGuardCommerceAdapter,
    create_adapter,
)
from aimai_ocl.baselines.toolguard_commerce.event_logger import (
    BASELINE_NAME,
    INTERCEPTED_DEFINITION,
    PROPOSAL_METRIC_KEYS,
    ProposalRecord,
    ToolGuardEventLogger,
)
from aimai_ocl.baselines.toolguard_commerce.policy_state import (
    BUYER_MAX_VISIBILITY_PLATFORM_VISIBLE,
    SUPPORTED_BUYER_MAX_VISIBILITIES,
    BuyerMaxVisibilityError,
    PlatformPolicyState,
    make_proposal_id,
    normalize_buyer_max_visibility,
    sanitize_violation_feedback,
    strip_platform_only_keys,
)
from aimai_ocl.baselines.toolguard_commerce.retry_policy import (
    MAX_SUPPORTED_RETRY_BUDGET,
    RetryBudgetError,
    RetryPolicy,
    coerce_retry_budget,
)
from aimai_ocl.baselines.toolguard_commerce.runtime import (
    GUARD_RESULT_FILENAME,
    TOOLGUARD_PINNED_COMMIT,
    TOOLGUARD_PIP_INSTALL_ARG,
    TOOLGUARD_PIP_SPEC,
    TOOLGUARD_REPO_URL,
    GuardExecutionError,
    GuardUnavailableError,
    NoLookupInvoker,
    ToolGuardCommerceError,
    UnsupportedGuardLookupError,
    guard_directory_hash,
    is_policy_violation,
    load_commerce_guards,
    policy_violation_message,
    preflight_report,
    toolguard_import_status,
)
from aimai_ocl.baselines.toolguard_commerce.tools import (
    COMMERCE_APP_NAME,
    COMMERCE_TOOL_NAME,
    SELLER_HIDDEN_ARGUMENT_NAMES,
    TOOL_ARGS_JSON_SCHEMA,
    TOOL_DESCRIPTION,
    build_tool_arguments,
    execute_seller_action,
    tool_functions,
    tool_spec,
    tool_spec_hash,
    tool_spec_json,
)

PACKAGE_ROOT = Path(__file__).resolve().parent
SELLER_POLICY_PATH = PACKAGE_ROOT / "policies" / "seller_policy.md"
TOOL_SPEC_MIRROR_PATH = PACKAGE_ROOT / "specs" / "commerce_tools.yaml"

__all__ = [
    "BASELINE_NAME",
    "BUYER_MAX_VISIBILITY_PLATFORM_VISIBLE",
    "COMMERCE_APP_NAME",
    "COMMERCE_TOOL_NAME",
    "GUARD_DECISION_ALLOW",
    "GUARD_DECISION_BLOCK",
    "GUARD_RESULT_FILENAME",
    "INTERCEPTED_DEFINITION",
    "MAX_SUPPORTED_RETRY_BUDGET",
    "PACKAGE_ROOT",
    "PROPOSAL_METRIC_KEYS",
    "SELLER_HIDDEN_ARGUMENT_NAMES",
    "SELLER_POLICY_PATH",
    "SUPPORTED_BUYER_MAX_VISIBILITIES",
    "TOOLGUARD_PINNED_COMMIT",
    "TOOLGUARD_PIP_INSTALL_ARG",
    "TOOLGUARD_PIP_SPEC",
    "TOOLGUARD_REPO_URL",
    "TOOL_ARGS_JSON_SCHEMA",
    "TOOL_DESCRIPTION",
    "TOOL_SPEC_MIRROR_PATH",
    "BuyerMaxVisibilityError",
    "GuardExecutionError",
    "GuardUnavailableError",
    "GuardVerdict",
    "NoLookupInvoker",
    "PlatformPolicyState",
    "ProposalRecord",
    "RetryBudgetError",
    "RetryPolicy",
    "SellerTurnOutcome",
    "ToolGuardCommerceAdapter",
    "ToolGuardCommerceError",
    "ToolGuardEventLogger",
    "UnsupportedGuardLookupError",
    "build_tool_arguments",
    "coerce_retry_budget",
    "create_adapter",
    "execute_seller_action",
    "guard_directory_hash",
    "is_policy_violation",
    "load_commerce_guards",
    "make_proposal_id",
    "normalize_buyer_max_visibility",
    "policy_violation_message",
    "preflight_report",
    "sanitize_violation_feedback",
    "strip_platform_only_keys",
    "tool_functions",
    "tool_spec",
    "tool_spec_hash",
    "tool_spec_json",
    "toolguard_import_status",
]
