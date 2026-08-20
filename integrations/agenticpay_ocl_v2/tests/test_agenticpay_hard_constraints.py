"""Hard Constraint coverage for the AgenticPay V2 action boundary."""

from __future__ import annotations

import pytest

from aocl_core.contracts import DecisionType, ObservableContext, ProposedAction
from aocl_core.policies import ControlMode
from aocl_core.runtime import IntegrationOCLRuntime
from agenticpay_ocl_v2.agenticpay_adapter import (
    AgenticPayExplicitSensitiveDataHardValidator,
    AgenticPayOCLAdapter,
    AgenticPaySellerBoundaryHardValidator,
    AgenticPaySellerPriceHardValidator,
    agenticpay_hard_constraint_validators,
)
from agenticpay_ocl_v2.agenticpay_runner import run_agenticpay_episode


def _action(
    text: str,
    *,
    actor_id: str = "Seller",
    action_type: str = "commerce.respond",
) -> ProposedAction:
    return ProposedAction(
        action_id="episode:0:Seller:0",
        actor_id=actor_id,
        action_type=action_type,
        visible_text=text,
        payload={"text": text},
    )


def _context(*, actor_id: str = "Seller") -> ObservableContext:
    return ObservableContext(
        episode_id="episode",
        step_id=0,
        actor_id=actor_id,
    )


def _failed_ids(results: tuple[object, ...]) -> set[str]:
    return {item.check_id for item in results if not item.passed}  # type: ignore[attr-defined]


def test_seller_boundary_rejects_wrong_actor_or_action_type() -> None:
    validator = AgenticPaySellerBoundaryHardValidator(expected_actor_id="Seller")

    wrong_actor = validator.validate(
        _action("### SELLER_PRICE($150) ###", actor_id="Buyer"),
        _context(actor_id="Buyer"),
    )[0]
    wrong_action = validator.validate(
        _action("### SELLER_PRICE($150) ###", action_type="commerce.refund"),
        _context(),
    )[0]

    assert wrong_actor.passed is False
    assert wrong_actor.recommended_decision is DecisionType.BLOCK
    assert wrong_action.passed is False
    assert wrong_action.recommended_decision is DecisionType.BLOCK


def test_price_validator_reads_only_the_agenticpay_seller_marker() -> None:
    validator = AgenticPaySellerPriceHardValidator(seller_min_price=90)
    text = (
        "Your $40 offer is too low. I can offer "
        "### SELLER_PRICE($1,250.50) ###. The product code is $12."
    )

    results = validator.validate(_action(text), _context())

    assert not _failed_ids(results)
    assert results[-1].metadata["price"] == 1250.5


@pytest.mark.parametrize(
    "text",
    (
        "I can make an offer, but omitted the required marker.",
        "I can offer SELLER_PRICE($150).",
        "### SELLER_PRICE($150) ### or ### SELLER_PRICE($140) ###",
        "### SELLER_PRICE($0) ###",
        "### SELLER_PRICE($-5) ###",
    ),
)
def test_price_validator_revises_missing_malformed_multiple_or_nonpositive_price(
    text: str,
) -> None:
    result = AgenticPaySellerPriceHardValidator(seller_min_price=90).validate(
        _action(text),
        _context(),
    )[0]

    assert result.passed is False
    assert result.check_id == "agenticpay_seller_price_format"
    assert result.recommended_decision is DecisionType.REVISE


def test_price_validator_revises_offer_below_seller_floor() -> None:
    results = AgenticPaySellerPriceHardValidator(seller_min_price=90).validate(
        _action("I can offer ### SELLER_PRICE($80) ###."),
        _context(),
    )

    assert _failed_ids(results) == {"agenticpay_seller_price_floor"}
    assert results[-1].recommended_decision is DecisionType.REVISE


@pytest.mark.parametrize(
    ("text", "expected_kind"),
    (
        (
            "Use credit card number 4000-1234-5678-9010. "
            "### SELLER_PRICE($150) ###",
            "payment_card_number",
        ),
        ("The SSN is 123-45-6789. ### SELLER_PRICE($150) ###", "ssn"),
        ("The CVV is 123. ### SELLER_PRICE($150) ###", "card_security_code"),
        (
            "My routing number is 123456789. ### SELLER_PRICE($150) ###",
            "routing_number",
        ),
        (
            "My bank account number is 1234567890. ### SELLER_PRICE($150) ###",
            "bank_account_number",
        ),
        (
            "The password is 'hunter2'. ### SELLER_PRICE($150) ###",
            "password_or_passcode",
        ),
    ),
)
def test_explicit_sensitive_values_are_hard_blocked(
    text: str,
    expected_kind: str,
) -> None:
    result = AgenticPayExplicitSensitiveDataHardValidator().validate(
        _action(text),
        _context(),
    )[0]

    assert result.passed is False
    assert result.recommended_decision is DecisionType.BLOCK
    assert expected_kind in result.metadata["detected_kinds"]
    assert "4000" not in str(result.metadata)
    assert "hunter2" not in str(result.metadata)


@pytest.mark.parametrize(
    "text",
    (
        "Never share your credit card or CVV here. ### SELLER_PRICE($150) ###",
        "I cannot accept a bank account transfer. ### SELLER_PRICE($150) ###",
        "Your password is not required. ### SELLER_PRICE($150) ###",
        "Please use secure platform checkout. ### SELLER_PRICE($150) ###",
    ),
)
def test_keyword_only_safe_redirection_is_not_hard_blocked(text: str) -> None:
    result = AgenticPayExplicitSensitiveDataHardValidator().validate(
        _action(text),
        _context(),
    )[0]

    assert result.passed is True


def test_complete_hard_constraint_suite_controls_adapter_execution() -> None:
    validators = agenticpay_hard_constraint_validators(
        seller_actor_id="Seller",
        seller_min_price=90,
    )
    runtime = IntegrationOCLRuntime(mode=ControlMode.BLOCKING, validators=validators)
    adapter = AgenticPayOCLAdapter(runtime)

    allowed = adapter.evaluate_seller_text(
        episode_id="episode",
        step_id=0,
        actor_id="Seller",
        seller_text="Please use platform checkout. ### SELLER_PRICE($150) ###",
        dialogue=(),
    )
    below_floor = adapter.evaluate_seller_text(
        episode_id="episode",
        step_id=1,
        actor_id="Seller",
        seller_text="I can offer ### SELLER_PRICE($80) ###.",
        dialogue=(),
    )
    explicit_secret = adapter.evaluate_seller_text(
        episode_id="episode",
        step_id=2,
        actor_id="Seller",
        seller_text=(
            "Use credit card number 4000-1234-5678-9010. "
            "### SELLER_PRICE($150) ###"
        ),
        dialogue=(),
    )

    assert allowed.execute is True
    assert allowed.decision.decision is DecisionType.APPROVE
    assert below_floor.execute is False
    assert below_floor.requires_revision is True
    assert explicit_secret.execute is False
    assert explicit_secret.decision.decision is DecisionType.BLOCK


class _ScriptedAgent:
    def __init__(self, name: str, *responses: str) -> None:
        self.name = name
        self.responses = list(responses)

    def respond(self, **_: object) -> str:
        return self.responses.pop(0)


class _OneStepEnv:
    def __init__(self) -> None:
        self.seller_action: str | None = "not-called"

    def reset(self, **_: object) -> tuple[dict[str, object], dict[str, object]]:
        return (
            {"current_round": 0, "conversation_history": [], "status": "ongoing"},
            {},
        )

    def step(
        self,
        *,
        buyer_action: str | None,
        seller_action: str | None,
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        del buyer_action
        self.seller_action = seller_action
        return ({}, 0.0, True, False, {"status": "timeout"})

    def close(self) -> None:
        return None


def test_episode_runner_revises_hard_price_failure_before_env_step() -> None:
    env = _OneStepEnv()
    seller = _ScriptedAgent(
        "Seller",
        "I can offer ### SELLER_PRICE($80) ###.",
        "I can offer ### SELLER_PRICE($100) ###.",
    )
    runtime = IntegrationOCLRuntime(
        validators=agenticpay_hard_constraint_validators(
            seller_actor_id=seller.name,
            seller_min_price=90,
        )
    )

    result = run_agenticpay_episode(
        episode_id="episode",
        env_id="fake",
        buyer_agent=_ScriptedAgent("Buyer", "Can you lower the price?"),
        seller_agent=seller,
        ocl_adapter=AgenticPayOCLAdapter(runtime),
        reset_kwargs={},
        maximum_revision_attempts=1,
        make_env=lambda *_args, **_kwargs: env,
    )

    assert [item.decision for item in result.turns[0].proposals] == [
        "revise",
        "approve",
    ]
    assert result.turns[0].proposals[0].executed is False
    assert env.seller_action == "I can offer ### SELLER_PRICE($100) ###."


def test_episode_runner_never_executes_explicit_secret() -> None:
    env = _OneStepEnv()
    seller = _ScriptedAgent(
        "Seller",
        "Use credit card 4000-1234-5678-9010. ### SELLER_PRICE($150) ###",
    )
    runtime = IntegrationOCLRuntime(
        validators=agenticpay_hard_constraint_validators(
            seller_actor_id=seller.name,
            seller_min_price=90,
        )
    )

    result = run_agenticpay_episode(
        episode_id="episode",
        env_id="fake",
        buyer_agent=_ScriptedAgent("Buyer", "How do I pay?"),
        seller_agent=seller,
        ocl_adapter=AgenticPayOCLAdapter(runtime),
        reset_kwargs={},
        make_env=lambda *_args, **_kwargs: env,
    )

    assert result.turns[0].proposals[0].decision == "block"
    assert result.turns[0].proposals[0].executed is False
    assert env.seller_action is None
