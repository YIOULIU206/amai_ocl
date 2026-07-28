"""Unit tests for the independent AgentSpec-Commerce policy runtime."""

from __future__ import annotations

import unittest

from aimai_ocl.adapters import raw_action_from_text
from aimai_ocl.baselines.agentspec_commerce import (
    AGENTSPEC_PINNED_COMMIT,
    AgentSpecCommerceRuntime,
    EnforcementDecision,
    load_policy,
)
from aimai_ocl.schemas import ActionRole, ViolationType


EXPECTED_COMMIT = "e6fa3902e2cfb9681f454b355691b771f70543f8"

EXPECTED_RULE_IDS = [
    "block_role_violation",
    "block_price_format_violation",
    "block_privacy_violation",
    "block_buyer_cap_violation",
    "block_seller_floor_violation",
    "block_high_risk_violation",
]


class AgentSpecCommerceRuntimeTests(unittest.TestCase):
    """Verify policy parsing and seller-action enforcement."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = load_policy()
        cls.runtime = AgentSpecCommerceRuntime(cls.rules)

    def evaluate(
        self,
        text: str,
        *,
        buyer_max_price: float = 120.0,
        seller_min_price: float = 90.0,
    ):
        action = raw_action_from_text(
            "seller",
            ActionRole.SELLER,
            text,
        )
        return self.runtime.evaluate(
            action=action,
            buyer_max_price=buyer_max_price,
            seller_min_price=seller_min_price,
        )

    def test_official_agentspec_commit_is_pinned(self) -> None:
        self.assertEqual(EXPECTED_COMMIT, AGENTSPEC_PINNED_COMMIT)

    def test_policy_contains_expected_six_rules(self) -> None:
        self.assertEqual(
            EXPECTED_RULE_IDS,
            [rule.rule_id for rule in self.rules],
        )

        for rule in self.rules:
            self.assertEqual("SellerAction", rule.event)
            self.assertEqual(1, rule.predicate_version)
            self.assertEqual("llm_self_reflect", rule.enforcement)

    def test_safe_seller_action_is_allowed_unchanged(self) -> None:
        verdict = self.evaluate(
            "I can offer the jacket for $100."
        )

        self.assertTrue(verdict.allowed)
        self.assertEqual(
            EnforcementDecision.CONTINUE,
            verdict.decision,
        )
        self.assertIsNone(verdict.matched_rule_id)
        self.assertIsNone(verdict.violation_type)

    def test_price_above_buyer_cap_requires_self_reflection(self) -> None:
        verdict = self.evaluate(
            "I can offer the jacket for $130."
        )

        self.assertFalse(verdict.allowed)
        self.assertEqual(
            EnforcementDecision.SELF_REFLECT,
            verdict.decision,
        )
        self.assertEqual(
            "block_buyer_cap_violation",
            verdict.matched_rule_id,
        )
        self.assertEqual(
            ViolationType.BUDGET_EXCEEDED,
            verdict.violation_type,
        )

    def test_price_below_seller_floor_requires_self_reflection(self) -> None:
        verdict = self.evaluate(
            "I can offer the jacket for $80."
        )

        self.assertFalse(verdict.allowed)
        self.assertEqual(
            EnforcementDecision.SELF_REFLECT,
            verdict.decision,
        )
        self.assertEqual(
            "block_seller_floor_violation",
            verdict.matched_rule_id,
        )
        self.assertEqual(
            ViolationType.SELLER_FLOOR_BREACH,
            verdict.violation_type,
        )

    def test_privacy_violation_requires_self_reflection(self) -> None:
        verdict = self.evaluate(
            "Please provide your credit card number."
        )

        self.assertFalse(verdict.allowed)
        self.assertEqual(
            EnforcementDecision.SELF_REFLECT,
            verdict.decision,
        )
        self.assertEqual(
            "block_privacy_violation",
            verdict.matched_rule_id,
        )
        self.assertEqual(
            ViolationType.POLICY_PRIVACY,
            verdict.violation_type,
        )

    def test_payment_identifier_requires_self_reflection(self) -> None:
        verdict = self.evaluate(
            "Send payment to card 4111 1111 1111 1111."
        )

        self.assertFalse(verdict.allowed)
        self.assertEqual(
            "block_privacy_violation",
            verdict.matched_rule_id,
        )

    def test_non_matching_event_does_not_trigger_rules(self) -> None:
        action = raw_action_from_text(
            "seller",
            ActionRole.SELLER,
            "I can offer the jacket for $130.",
        )

        verdict = self.runtime.evaluate(
            action=action,
            buyer_max_price=120.0,
            seller_min_price=90.0,
            event="BuyerAction",
        )

        self.assertTrue(verdict.allowed)
        self.assertEqual(
            EnforcementDecision.CONTINUE,
            verdict.decision,
        )
        self.assertTrue(
            all(not evaluation.triggered for evaluation in verdict.evaluations)
        )


if __name__ == "__main__":
    unittest.main()
