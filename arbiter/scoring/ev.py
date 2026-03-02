"""Expected value calculation for prediction market contracts."""

from __future__ import annotations

from dataclasses import dataclass, field

from arbiter.ingestion.base import Contract
from arbiter.scoring.kelly import kelly_criterion


@dataclass
class ScoredOpportunity:
    """A contract scored with expected value and Kelly sizing.

    Wraps the original Contract via composition, adding computed fields.
    For consistency arbs, ``anchor_contract`` holds the sibling that proves
    the price floor (the second leg of the trade).
    """

    contract: Contract
    direction: str  # "yes" or "no"
    market_price: float  # price we'd pay (yes_price or no_price)
    model_probability: float  # our estimated true probability for this direction
    expected_value: float  # EV per dollar, after execution costs
    kelly_size: float  # full Kelly fraction (caller applies fractional Kelly)
    anchor_contract: Contract | None = field(default=None)  # sibling for consistency arbs
    strategy_name: str = field(default="")  # which strategy produced this opportunity


def compute_ev(
    contract: Contract,
    model_prob_yes: float,
    fee_rate: float = 0.0,
) -> list[ScoredOpportunity]:
    """Compute expected value for both YES and NO sides of a contract.

    EV per direction = model_prob - (market_price + fee_rate)

    The fee_rate accounts for execution costs (platform fees, half-spread).
    A positive EV means the model believes the contract is mispriced in our favor.

    Args:
        contract: The normalized market contract.
        model_prob_yes: Model's estimated probability of YES outcome, in (0, 1).
        fee_rate: Execution cost as a fraction of contract price (default 0).

    Returns:
        List of ScoredOpportunity for both YES and NO sides.
    """
    results: list[ScoredOpportunity] = []

    # YES side
    yes_cost = contract.yes_price + fee_rate
    if 0 < yes_cost < 1:
        ev_yes = model_prob_yes - yes_cost
        payout_ratio = (1.0 / yes_cost) - 1.0
        kelly_yes = kelly_criterion(model_prob_yes, payout_ratio)
        results.append(
            ScoredOpportunity(
                contract=contract,
                direction="yes",
                market_price=contract.yes_price,
                model_probability=model_prob_yes,
                expected_value=ev_yes,
                kelly_size=kelly_yes,
            )
        )

    # NO side
    model_prob_no = 1.0 - model_prob_yes
    no_cost = contract.no_price + fee_rate
    if 0 < no_cost < 1:
        ev_no = model_prob_no - no_cost
        payout_ratio = (1.0 / no_cost) - 1.0
        kelly_no = kelly_criterion(model_prob_no, payout_ratio)
        results.append(
            ScoredOpportunity(
                contract=contract,
                direction="no",
                market_price=contract.no_price,
                model_probability=model_prob_no,
                expected_value=ev_no,
                kelly_size=kelly_no,
            )
        )

    return results
