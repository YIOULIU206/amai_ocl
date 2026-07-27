"""Mocked tests for the ToolGuard-Commerce external baseline core.

Everything here runs with a synthetic guard fixture: no toolguard install, no
LLM credentials, and no AgenticPay environment are required. Runner-level and
statistics-level behaviour is covered by
tests/test_toolguard_commerce_integration.py.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from toolguard_commerce_fixtures import (  # noqa: E402
    BrokenGuard,
    LookupRequiringGuard,
    RecordingSeller,
    ScriptedGuard,
    SyntheticCommerceGuard,
    write_synthetic_guard_dir,
)

from aimai_ocl.baselines.toolguard_commerce import (  # noqa: E402
    COMMERCE_TOOL_NAME,
    GUARD_DECISION_ALLOW,
    GUARD_DECISION_BLOCK,
    PROPOSAL_METRIC_KEYS,
    SELLER_POLICY_PATH,
    TOOLGUARD_PINNED_COMMIT,
    TOOL_SPEC_MIRROR_PATH,
    GuardExecutionError,
    GuardUnavailableError,
    NoLookupInvoker,
    PlatformPolicyState,
    RetryBudgetError,
    RetryPolicy,
    ToolGuardCommerceAdapter,
    ToolGuardEventLogger,
    UnsupportedGuardLookupError,
    execute_seller_action,
    load_commerce_guards,
    make_proposal_id,
    sanitize_violation_feedback,
    tool_functions,
)
from aimai_ocl.baselines.toolguard_commerce import tools as tools_mod  # noqa: E402
from aimai_ocl.schemas import AuditEventType, ControlDecision  # noqa: E402

BUYER_MAX = 120.0
SELLER_MIN = 90.0
EPISODE_KEY = "unit-episode"
ARM = "toolguard_commerce"

COMPLIANT_TEXT = "Happy to close this today at $110 for the jacket."
BELOW_FLOOR_TEXT = "Final offer, I can do $80 right now."
DEEPER_BELOW_FLOOR_TEXT = "Fine, $70 then."
PRIVACY_TEXT = "Send me your credit card number and I can do $110."

BASE_HISTORY = [
    {"role": "buyer", "content": "Can you go lower?", "round": 0},
]
OBSERVATION = {"current_round": 0, "conversation_history": list(BASE_HISTORY)}


def make_policy_state(round_id: int = 0, buyer_max: float | None = BUYER_MAX) -> PlatformPolicyState:
    return PlatformPolicyState(
        round_id=round_id,
        seller_min_price=SELLER_MIN,
        buyer_max_price=buyer_max,
        product_price=180.0,
        product_name="Winter Jacket",
        max_rounds=10,
    )


def make_adapter(guard: Any, *, retry_budget: int = 1) -> ToolGuardCommerceAdapter:
    logger = ToolGuardEventLogger(
        retry_budget=retry_budget, arm=ARM, episode_key=EPISODE_KEY
    )
    return ToolGuardCommerceAdapter(
        guard_runtime=guard,
        invoker=NoLookupInvoker(),
        retry_policy=RetryPolicy(budget=retry_budget),
        logger=logger,
        episode_key=EPISODE_KEY,
        arm=ARM,
    )


class ToolGuardCommerceAllowPathTests(unittest.TestCase):
    """ALLOW path: compliant seller actions are executed verbatim."""

    def test_compliant_action_passes_through_unchanged(self) -> None:
        """Input: compliant draft. Output: identical text, no extra generation."""
        guard = SyntheticCommerceGuard()
        seller = RecordingSeller(["should not be used"])
        adapter = make_adapter(guard)

        outcome = adapter.process_seller_turn(
            text=COMPLIANT_TEXT,
            seller_history=list(BASE_HISTORY),
            observation=dict(OBSERVATION),
            policy_state=make_policy_state(),
            actor_id="seller",
            round_id=0,
            seller_respond=seller.respond,
        )

        self.assertEqual(COMPLIANT_TEXT, outcome.text)
        self.assertEqual([GUARD_DECISION_ALLOW], outcome.decisions)
        self.assertFalse(outcome.no_op)
        self.assertEqual(0, outcome.retry_seller_generation_calls)
        self.assertEqual([], seller.calls)
        self.assertEqual(1, len(guard.calls))

    def test_guard_receives_all_declared_tool_arguments(self) -> None:
        """Input: one guarded turn. Output: full declared argument payload."""
        guard = SyntheticCommerceGuard()
        adapter = make_adapter(guard)
        adapter.process_seller_turn(
            text=COMPLIANT_TEXT,
            seller_history=list(BASE_HISTORY),
            observation=dict(OBSERVATION),
            policy_state=make_policy_state(),
            actor_id="seller",
            round_id=3,
            seller_respond=None,
        )

        args = guard.calls[0]["args"]
        self.assertEqual(COMMERCE_TOOL_NAME, guard.calls[0]["tool_name"])
        self.assertEqual(list(tools_mod.TOOL_ARGUMENT_NAMES), list(args))
        self.assertEqual("seller", args["actor_role"])
        self.assertEqual(3, args["round_id"])
        self.assertEqual(110.0, args["proposed_price"])
        self.assertEqual(SELLER_MIN, args["seller_min_price"])
        self.assertEqual(180.0, args["product_price"])
        self.assertEqual(COMPLIANT_TEXT, args["message"])

    def test_tool_body_does_not_clamp_or_rewrite(self) -> None:
        """Input: below-floor price. Output: tool echoes it unchanged."""
        result = execute_seller_action(
            intent="negotiate_price",
            proposed_price=80.0,
            actor_role="seller",
            message=BELOW_FLOOR_TEXT,
            round_id=0,
            seller_min_price=SELLER_MIN,
            buyer_max_price=BUYER_MAX,
            product_price=180.0,
        )
        self.assertEqual(80.0, result["proposed_price"])
        self.assertEqual(BELOW_FLOOR_TEXT, result["message"])
        self.assertEqual([execute_seller_action], tool_functions())


class ToolGuardCommerceBlockAndRetryTests(unittest.TestCase):
    """BLOCK path: one sanitized retry, then a no-op."""

    def test_violating_action_is_blocked(self) -> None:
        """Input: below-floor draft, no seller available for revision."""
        guard = SyntheticCommerceGuard()
        adapter = make_adapter(guard)

        outcome = adapter.process_seller_turn(
            text=BELOW_FLOOR_TEXT,
            seller_history=list(BASE_HISTORY),
            observation=dict(OBSERVATION),
            policy_state=make_policy_state(),
            actor_id="seller",
            round_id=0,
            seller_respond=None,
        )

        self.assertIsNone(outcome.text)
        self.assertTrue(outcome.no_op)
        self.assertEqual([GUARD_DECISION_BLOCK], outcome.decisions)
        self.assertEqual(0, outcome.retry_seller_generation_calls)

    def test_violation_reason_reaches_retry_and_history_is_a_copy(self) -> None:
        """Input: privacy violation. Output: sanitized reason in a history copy."""
        guard = SyntheticCommerceGuard()
        seller = RecordingSeller([COMPLIANT_TEXT])
        adapter = make_adapter(guard)
        history = list(BASE_HISTORY)
        observation = dict(OBSERVATION)

        outcome = adapter.process_seller_turn(
            text=PRIVACY_TEXT,
            seller_history=history,
            observation=observation,
            policy_state=make_policy_state(),
            actor_id="seller",
            round_id=0,
            seller_respond=seller.respond,
        )

        self.assertEqual(COMPLIANT_TEXT, outcome.text)
        self.assertEqual([GUARD_DECISION_BLOCK, GUARD_DECISION_ALLOW], outcome.decisions)
        self.assertEqual(1, len(seller.calls))

        retry_history = seller.calls[0]["conversation_history"]
        self.assertEqual(len(history) + 1, len(retry_history))
        self.assertEqual(1, len(history), "permanent history must not grow")
        self.assertIsNot(history, retry_history)
        feedback = retry_history[-1]["content"]
        self.assertEqual("platform", retry_history[-1]["role"])
        self.assertIn("credit card", feedback)
        self.assertIn("blocked", feedback.lower())
        self.assertEqual(observation, seller.calls[0]["current_state"])

    def test_compliant_revision_is_executed_without_clamping(self) -> None:
        """Input: below-floor draft then compliant revision at $95."""
        guard = SyntheticCommerceGuard()
        revision = "I can meet you at $95, that is my best."
        seller = RecordingSeller([revision])
        adapter = make_adapter(guard)

        outcome = adapter.process_seller_turn(
            text=BELOW_FLOOR_TEXT,
            seller_history=list(BASE_HISTORY),
            observation=dict(OBSERVATION),
            policy_state=make_policy_state(),
            actor_id="seller",
            round_id=0,
            seller_respond=seller.respond,
        )

        self.assertEqual(revision, outcome.text)
        self.assertNotIn("90.00", outcome.text)
        self.assertNotIn("$90", outcome.text)
        self.assertEqual(1, outcome.retry_seller_generation_calls)

    def test_second_violation_becomes_a_no_op(self) -> None:
        """Input: two below-floor drafts. Output: None and a single retry."""
        guard = SyntheticCommerceGuard()
        seller = RecordingSeller([DEEPER_BELOW_FLOOR_TEXT, "third attempt $60"])
        adapter = make_adapter(guard)

        outcome = adapter.process_seller_turn(
            text=BELOW_FLOOR_TEXT,
            seller_history=list(BASE_HISTORY),
            observation=dict(OBSERVATION),
            policy_state=make_policy_state(),
            actor_id="seller",
            round_id=0,
            seller_respond=seller.respond,
        )

        self.assertIsNone(outcome.text)
        self.assertTrue(outcome.no_op)
        self.assertEqual([GUARD_DECISION_BLOCK, GUARD_DECISION_BLOCK], outcome.decisions)
        self.assertEqual(1, len(seller.calls), "at most one extra seller generation")
        self.assertEqual(1, outcome.retry_seller_generation_calls)
        self.assertEqual(2, len(guard.calls))

    def test_empty_revision_becomes_a_no_op(self) -> None:
        """Input: blocked draft and an empty revision. Output: no-op."""
        guard = SyntheticCommerceGuard()
        seller = RecordingSeller(["   "])
        adapter = make_adapter(guard)

        outcome = adapter.process_seller_turn(
            text=BELOW_FLOOR_TEXT,
            seller_history=list(BASE_HISTORY),
            observation=dict(OBSERVATION),
            policy_state=make_policy_state(),
            actor_id="seller",
            round_id=0,
            seller_respond=seller.respond,
        )

        self.assertIsNone(outcome.text)
        self.assertTrue(outcome.no_op)
        self.assertEqual(1, len(seller.calls))

    def test_zero_retry_budget_blocks_without_regeneration(self) -> None:
        """Input: retry budget 0. Output: no seller revision at all."""
        guard = SyntheticCommerceGuard()
        seller = RecordingSeller([COMPLIANT_TEXT])
        adapter = make_adapter(guard, retry_budget=0)

        outcome = adapter.process_seller_turn(
            text=BELOW_FLOOR_TEXT,
            seller_history=list(BASE_HISTORY),
            observation=dict(OBSERVATION),
            policy_state=make_policy_state(),
            actor_id="seller",
            round_id=0,
            seller_respond=seller.respond,
        )

        self.assertIsNone(outcome.text)
        self.assertEqual([], seller.calls)

    def test_retry_budget_above_one_is_rejected(self) -> None:
        """Input: retry budget 2. Output: configuration error."""
        with self.assertRaises(RetryBudgetError):
            RetryPolicy(budget=2)


class ToolGuardCommerceIsolationTests(unittest.TestCase):
    """The buyer budget cap is platform-visible only."""

    def test_buyer_max_reaches_guard_but_not_the_seller(self) -> None:
        """Input: cap-breaching draft. Output: cap in guard args, not in feedback."""
        guard = SyntheticCommerceGuard()
        seller = RecordingSeller([COMPLIANT_TEXT])
        adapter = make_adapter(guard)
        observation = {
            "current_round": 0,
            "conversation_history": list(BASE_HISTORY),
            "buyer_max_price": BUYER_MAX,
        }

        adapter.process_seller_turn(
            text="This jacket cannot go under $130.",
            seller_history=list(BASE_HISTORY),
            observation=observation,
            policy_state=make_policy_state(),
            actor_id="seller",
            round_id=0,
            seller_respond=seller.respond,
        )

        self.assertEqual(BUYER_MAX, guard.calls[0]["args"]["buyer_max_price"])
        seller_state = seller.calls[0]["current_state"]
        self.assertNotIn("buyer_max_price", seller_state)
        feedback = seller.calls[0]["conversation_history"][-1]["content"]
        self.assertNotIn("buyer_max_price", feedback)
        self.assertNotIn("120", feedback)
        self.assertIn("[redacted-platform-only]", feedback)

    def test_sanitizer_redacts_cap_names_and_values(self) -> None:
        """Input: raw guard reason. Output: no cap name and no cap value."""
        sanitized = sanitize_violation_feedback(
            "Proposed price 130.00 exceeds buyer_max_price 120.00.",
            buyer_max_price=BUYER_MAX,
        )
        self.assertNotIn("buyer_max_price", sanitized)
        self.assertNotIn("120", sanitized)
        self.assertIn("130.00", sanitized)

    def test_baseline_package_does_not_depend_on_ocl_control(self) -> None:
        """Input: package sources. Output: no OCL control usage anywhere."""
        package_dir = REPO_ROOT / "aimai_ocl" / "baselines"
        sources = sorted(package_dir.rglob("*.py"))
        self.assertTrue(sources)
        forbidden = ("aimai_ocl.control", "apply_control", "resolve_escalation", "ControlConfig")
        for path in sources:
            text = path.read_text(encoding="utf-8")
            for needle in forbidden:
                self.assertNotIn(needle, text, str(path) + " must not use " + needle)

    def test_no_escalation_rewrite_or_replan_events(self) -> None:
        """Input: blocked then allowed chain. Output: only approve/block events."""
        guard = SyntheticCommerceGuard()
        seller = RecordingSeller([COMPLIANT_TEXT])
        adapter = make_adapter(guard)
        adapter.process_seller_turn(
            text=BELOW_FLOOR_TEXT,
            seller_history=list(BASE_HISTORY),
            observation=dict(OBSERVATION),
            policy_state=make_policy_state(),
            actor_id="seller",
            round_id=0,
            seller_respond=seller.respond,
        )

        allowed_types = {
            AuditEventType.CONSTRAINT_EVALUATED,
            AuditEventType.ACTION_EXECUTED,
        }
        decisions = set()
        for event in adapter.events:
            self.assertIn(event.event_type, allowed_types)
            if event.executable_action is not None:
                decisions.add(event.executable_action.decision)
        self.assertTrue(decisions.issubset({ControlDecision.APPROVE, ControlDecision.BLOCK}))


class ToolGuardCommerceFailClosedTests(unittest.TestCase):
    """Missing guards and auxiliary lookups stop the run."""

    def test_missing_guard_directory_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "not-generated"
            with self.assertRaises(GuardUnavailableError):
                load_commerce_guards(missing)

    def test_missing_result_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(GuardUnavailableError):
                load_commerce_guards(tmp)

    def test_result_without_commerce_tool_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_synthetic_guard_dir(tmp, include_tool_entry=False)
            with self.assertRaises(GuardUnavailableError):
                load_commerce_guards(tmp)

    def test_missing_guard_module_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_synthetic_guard_dir(tmp, include_guard_module=False)
            with self.assertRaises(GuardUnavailableError):
                load_commerce_guards(tmp)

    def test_adapter_requires_a_guard_runtime(self) -> None:
        with self.assertRaises(GuardUnavailableError):
            ToolGuardCommerceAdapter(guard_runtime=None)
        with self.assertRaises(GuardUnavailableError):
            ToolGuardCommerceAdapter(guard_runtime=object())

    def test_auxiliary_lookup_fails_closed(self) -> None:
        """Input: guard requiring a lookup. Output: explicit error, no verdict."""
        adapter = make_adapter(LookupRequiringGuard())
        with self.assertRaises(UnsupportedGuardLookupError):
            adapter.process_seller_turn(
                text=COMPLIANT_TEXT,
                seller_history=list(BASE_HISTORY),
                observation=dict(OBSERVATION),
                policy_state=make_policy_state(),
                actor_id="seller",
                round_id=0,
                seller_respond=None,
            )

    def test_no_lookup_invoker_never_returns_data(self) -> None:
        with self.assertRaises(UnsupportedGuardLookupError):
            NoLookupInvoker().invoke("get_buyer_profile", {"round_id": 0}, dict)

    def test_non_policy_guard_error_is_not_an_allow(self) -> None:
        adapter = make_adapter(BrokenGuard())
        with self.assertRaises(GuardExecutionError):
            adapter.process_seller_turn(
                text=COMPLIANT_TEXT,
                seller_history=list(BASE_HISTORY),
                observation=dict(OBSERVATION),
                policy_state=make_policy_state(),
                actor_id="seller",
                round_id=0,
                seller_respond=None,
            )


class ToolGuardCommerceMetricsTests(unittest.TestCase):
    """Proposal ids are deterministic and metrics are keyed by them."""

    def _run_blocked_then_allowed(self) -> ToolGuardCommerceAdapter:
        adapter = make_adapter(ScriptedGuard(["below the seller floor 90.00", None]))
        adapter.process_seller_turn(
            text=BELOW_FLOOR_TEXT,
            seller_history=list(BASE_HISTORY),
            observation=dict(OBSERVATION),
            policy_state=make_policy_state(),
            actor_id="seller",
            round_id=0,
            seller_respond=RecordingSeller([COMPLIANT_TEXT]).respond,
        )
        return adapter

    def test_proposal_id_is_deterministic_and_not_uuid4(self) -> None:
        first = make_proposal_id(
            episode_key=EPISODE_KEY, arm=ARM, round_id=0, attempt=0,
            actor_id="seller", text=BELOW_FLOOR_TEXT,
        )
        second = make_proposal_id(
            episode_key=EPISODE_KEY, arm=ARM, round_id=0, attempt=0,
            actor_id="seller", text=BELOW_FLOOR_TEXT,
        )
        other = make_proposal_id(
            episode_key=EPISODE_KEY, arm=ARM, round_id=0, attempt=1,
            actor_id="seller", text=BELOW_FLOOR_TEXT,
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertRegex(first, r"^tgc-[0-9a-f]{20}$")

    def test_repeated_runs_produce_identical_proposal_ids(self) -> None:
        ids_first = [p["proposal_id"] for p in self._run_blocked_then_allowed().export()["proposals"]]
        ids_second = [p["proposal_id"] for p in self._run_blocked_then_allowed().export()["proposals"]]
        self.assertEqual(ids_first, ids_second)
        self.assertEqual(2, len(ids_first))

    def test_metrics_are_keyed_by_proposal_and_track_the_chain(self) -> None:
        adapter = self._run_blocked_then_allowed()
        payload = adapter.export()
        proposals = payload["proposals"]
        self.assertEqual(2, len(proposals))
        blocked, revision = proposals

        for record in proposals:
            for key in PROPOSAL_METRIC_KEYS:
                self.assertIn(key, record)
        self.assertEqual(blocked["chain_id"], revision["chain_id"])
        self.assertEqual(blocked["proposal_id"], blocked["chain_id"])

        self.assertEqual(1, blocked["detected"])
        self.assertEqual(1, blocked["blocked"])
        self.assertEqual(1, blocked["retry_attempted"])
        self.assertEqual(1, blocked["retry_seller_generation_calls"])
        self.assertEqual(0, blocked["candidate_selected_for_execution"])
        self.assertEqual(0, blocked["reached_env"])
        self.assertEqual(1, blocked["intercepted"])

        self.assertEqual(0, revision["detected"])
        self.assertEqual(0, revision["blocked"])
        self.assertEqual(1, revision["revision_passed"])
        self.assertEqual(1, revision["candidate_selected_for_execution"])
        self.assertEqual(0, revision["reached_env"])
        self.assertEqual(0, revision["intercepted"])
        self.assertGreaterEqual(revision["guard_runtime_sec"], 0.0)

        adapter.mark_reached_env(revision["proposal_id"])
        totals = adapter.export()["totals"]
        self.assertEqual(2, totals["proposals"])
        self.assertEqual(1, totals["detected"])
        self.assertEqual(1, totals["blocked"])
        self.assertEqual(1, totals["reached_env"])
        self.assertEqual(1, totals["intercepted"])
        self.assertEqual(1, totals["retry_seller_generation_calls"])

    def test_exhausted_chain_marks_retry_exhausted(self) -> None:
        adapter = make_adapter(ScriptedGuard(["floor breach", "floor breach again"]))
        adapter.process_seller_turn(
            text=BELOW_FLOOR_TEXT,
            seller_history=list(BASE_HISTORY),
            observation=dict(OBSERVATION),
            policy_state=make_policy_state(),
            actor_id="seller",
            round_id=0,
            seller_respond=RecordingSeller([DEEPER_BELOW_FLOOR_TEXT]).respond,
        )
        payload = adapter.export()
        self.assertEqual(0, payload["totals"]["candidate_selected_for_execution"])
        self.assertEqual(0, payload["totals"]["reached_env"])
        self.assertEqual(2, payload["totals"]["intercepted"])
        self.assertEqual(1, payload["proposals"][-1]["retry_exhausted"])
        self.assertEqual(1, payload["retry_budget"])


class ToolGuardCommerceSpecTests(unittest.TestCase):
    """The YAML spec is a mirror of tools.py and the pin is recorded."""

    def _load_build_script(self) -> Any:
        path = REPO_ROOT / "scripts" / "build_toolguard_commerce.py"
        spec = importlib.util.spec_from_file_location("build_toolguard_commerce", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_spec_mirror_matches_tools_module(self) -> None:
        module = self._load_build_script()
        ok, message = module.verify_mirror()
        self.assertTrue(ok, message)

    def test_policy_and_spec_files_exist(self) -> None:
        self.assertTrue(SELLER_POLICY_PATH.is_file())
        self.assertTrue(TOOL_SPEC_MIRROR_PATH.is_file())

    def test_toolguard_commit_is_pinned(self) -> None:
        from aimai_ocl.baselines.toolguard_commerce import (
            TOOLGUARD_PIP_INSTALL_ARG,
            TOOLGUARD_PIP_SPEC,
        )

        self.assertEqual(
            "20e21db4c275d79f8d7bf33ffb985d0b45f786f5", TOOLGUARD_PINNED_COMMIT
        )
        self.assertIn(TOOLGUARD_PINNED_COMMIT, TOOLGUARD_PIP_SPEC)
        self.assertTrue(TOOLGUARD_PIP_INSTALL_ARG.endswith(TOOLGUARD_PINNED_COMMIT))
        self.assertIn("github.com/IBM/tool_guard", TOOLGUARD_PIP_INSTALL_ARG)


if __name__ == "__main__":
    unittest.main()
