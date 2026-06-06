"""Desire / utility helpers built on top of stock-level belief.

This module is the deterministic backbone after the Belief layer. It combines
``belief_score`` with account-state features to produce rule-based buy, sell,
and hold desires. Optional LLM desire adjustments should be small, ordinal, and
applied on top of this backbone rather than replacing it.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


PROSPECT_ALPHA = 0.88
PROSPECT_BETA = 0.88
BASE_LOSS_AVERSION = 2.25
BASE_RISK_APPETITE = 0.60
BASE_DISPOSITION_STRENGTH = 0.55

GAIN_POSITION_WEIGHT_BASE = 0.60
GAIN_POSITION_WEIGHT_SCALE = 0.40
OVERHEAT_PROFIT_TAKING_SCALE = 0.40
LOSS_RESISTANCE_SELL_OFFSET = 0.50
CAPITULATION_BEARISH_SCALE = 1.00

BUY_UNCERTAINTY_PENALTY = 0.50
HOLD_BASE = 0.25
HOLD_UNCERTAINTY_WEIGHT = 0.35
HOLD_LOSS_RESISTANCE_WEIGHT = 0.25
HOLD_WEAK_BELIEF_WEIGHT = 0.15
HOLD_ACTIVE_DESIRE_PENALTY = 0.20

ADJUSTMENT_DELTAS = {
    "strongly_down": -0.20,
    "moderately_down": -0.12,
    "slightly_down": -0.06,
    "unchanged": 0.0,
    "slightly_up": 0.06,
    "moderately_up": 0.12,
    "strongly_up": 0.20,
}

PRESSURE_BONUSES = {
    "none": 0.0,
    "low": 0.03,
    "medium": 0.06,
    "high": 0.10,
    "extreme": 0.14,
}

CONFIDENCE_SCALES = {
    "low": 0.50,
    "medium": 1.00,
    "high": 1.00,
}


def build_rule_desires(
    *,
    belief_score: float,
    account_features: Mapping[str, Any],
    cognition: Optional[Mapping[str, Any]] = None,
    market_features: Optional[Mapping[str, Any]] = None,
    utility_params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, float]:
    """Build deterministic buy/sell/hold desire scores.

    This function intentionally uses account state, so it belongs after the
    account-free Belief layer. The output scores are clipped to ``[0, 1]`` and
    should later be passed to an Action layer with cash/position constraints.
    """

    params = dict(utility_params or {})
    risk_appetite = _clip_float(params.get("risk_appetite"), 0.0, 1.0, BASE_RISK_APPETITE)
    loss_aversion = max(0.0, _to_float(params.get("loss_aversion"), BASE_LOSS_AVERSION))
    disposition_strength = _clip_float(
        params.get("disposition_strength"),
        0.0,
        1.0,
        BASE_DISPOSITION_STRENGTH,
    )

    cognition = dict(cognition or {})
    market_features = dict(market_features or {})
    belief = _clip_float(belief_score, -1.0, 1.0, 0.0)
    bullish = max(belief, 0.0)
    bearish = max(-belief, 0.0)

    cash_ratio = _clip_float(account_features.get("cash_ratio"), 0.0, 1.0, 0.0)
    has_position = 1.0 if _to_float(account_features.get("has_position"), 0.0) > 0 else 0.0
    position_weight = _clip_float(account_features.get("position_weight"), 0.0, 1.0, 0.0)
    pnl_return = _clip_float(account_features.get("pnl_return"), -1.0, 3.0, 0.0)
    gain = max(pnl_return, 0.0)
    loss = max(-pnl_return, 0.0)
    uncertainty = _clip_float(cognition.get("uncertainty"), 0.0, 1.0, 0.5)
    overheat = _overheat_score(market_features)

    profit_taking_pressure = (
        has_position
        * disposition_strength
        * _power(gain, PROSPECT_ALPHA)
        * (GAIN_POSITION_WEIGHT_BASE + GAIN_POSITION_WEIGHT_SCALE * position_weight)
        * (1.0 + OVERHEAT_PROFIT_TAKING_SCALE * overheat)
    )

    loss_realization_resistance = (
        has_position
        * loss_aversion
        * _power(loss, PROSPECT_BETA)
        * _clip(1.0 - CAPITULATION_BEARISH_SCALE * bearish, 0.0, 1.0)
    )

    buy_desire = (
        bullish
        * risk_appetite
        * cash_ratio
        * (1.0 - position_weight)
        * (1.0 - BUY_UNCERTAINTY_PENALTY * uncertainty)
    )

    bearish_sell_pressure = (
        has_position
        * bearish
        * (0.7 + 0.3 * min(loss_aversion / BASE_LOSS_AVERSION, 1.5))
    )
    sell_desire = (
        bearish_sell_pressure
        + profit_taking_pressure
        - LOSS_RESISTANCE_SELL_OFFSET * loss_realization_resistance
    )

    hold_desire = (
        HOLD_BASE
        + HOLD_UNCERTAINTY_WEIGHT * uncertainty
        + HOLD_LOSS_RESISTANCE_WEIGHT * loss_realization_resistance
        + HOLD_WEAK_BELIEF_WEIGHT * (1.0 - abs(belief))
        - HOLD_ACTIVE_DESIRE_PENALTY * max(buy_desire, sell_desire)
    )

    return {
        "buy_desire": round(_clip(buy_desire, 0.0, 1.0), 6),
        "sell_desire": round(_clip(sell_desire, 0.0, 1.0), 6),
        "hold_desire": round(_clip(hold_desire, 0.0, 1.0), 6),
        "profit_taking_pressure": round(_clip(profit_taking_pressure, 0.0, 1.0), 6),
        "loss_realization_resistance": round(_clip(loss_realization_resistance, 0.0, 1.0), 6),
        "bearish_sell_pressure": round(_clip(bearish_sell_pressure, 0.0, 1.0), 6),
        "overheat_score": round(_clip(overheat, 0.0, 1.0), 6),
    }


def apply_desire_adjustment(
    rule_desires: Mapping[str, Any],
    adjustment: Mapping[str, Any],
) -> Dict[str, float | str]:
    """Apply bounded ordinal LLM desire adjustments to rule desires."""

    confidence = str(adjustment.get("confidence", "medium"))
    scale = CONFIDENCE_SCALES.get(confidence, CONFIDENCE_SCALES["medium"])

    buy_delta = _adjustment_delta(adjustment.get("buy_adjustment")) * scale
    sell_delta = _adjustment_delta(adjustment.get("sell_adjustment")) * scale
    hold_delta = _adjustment_delta(adjustment.get("hold_adjustment")) * scale

    profit_bonus = _pressure_bonus(adjustment.get("profit_taking_pressure")) * scale
    loss_hold_bonus = _pressure_bonus(adjustment.get("loss_hold_pressure")) * scale

    buy_desire = _clip_float(rule_desires.get("buy_desire"), 0.0, 1.0, 0.0) + buy_delta
    sell_desire = (
        _clip_float(rule_desires.get("sell_desire"), 0.0, 1.0, 0.0)
        + sell_delta
        + profit_bonus
    )
    hold_desire = (
        _clip_float(rule_desires.get("hold_desire"), 0.0, 1.0, 0.0)
        + hold_delta
        + loss_hold_bonus
    )

    result = {
        "buy_desire": round(_clip(buy_desire, 0.0, 1.0), 6),
        "sell_desire": round(_clip(sell_desire, 0.0, 1.0), 6),
        "hold_desire": round(_clip(hold_desire, 0.0, 1.0), 6),
        "buy_delta": round(buy_delta, 6),
        "sell_delta": round(sell_delta + profit_bonus, 6),
        "hold_delta": round(hold_delta + loss_hold_bonus, 6),
        "confidence_scale": round(scale, 6),
    }
    result["dominant_desire"] = max(
        ("buy", "sell", "hold"),
        key=lambda key: float(result[f"{key}_desire"]),
    )
    return result


def _overheat_score(market_features: Mapping[str, Any]) -> float:
    overbought = max(-_clip_float(market_features.get("rsi_score"), -1.0, 1.0, 0.0), 0.0)
    high_anchor = max(-_clip_float(market_features.get("anchor_score"), -1.0, 1.0, 0.0), 0.0)
    return _clip(0.5 * overbought + 0.5 * high_anchor, 0.0, 1.0)


def _adjustment_delta(value: Any) -> float:
    return float(ADJUSTMENT_DELTAS.get(str(value or "unchanged"), 0.0))


def _pressure_bonus(value: Any) -> float:
    return float(PRESSURE_BONUSES.get(str(value or "none"), 0.0))


def _power(value: float, exponent: float) -> float:
    return max(value, 0.0) ** exponent


def _clip_float(value: Any, low: float, high: float, default: float) -> float:
    return _clip(_to_float(value, default), low, high)


def _to_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if numeric != numeric:
        return float(default)
    return numeric


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
