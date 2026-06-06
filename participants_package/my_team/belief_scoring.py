"""Rule-based stock belief scoring from LLM cognition components.

This module is the demo Belief-score layer. It turns account-free LLM cognition
fields and market features into:

* ``belief_score``: continuous stock-level bullish / bearish belief in [-1, 1].
* ``sentiment_class``: discrete class following ``belief_score`` with a neutral
  band.

It intentionally does not use cash, position, average cost, unrealized PnL, or
prospect utility. Those belong to the later Desire / Utility layer.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


SENTIMENT_NEUTRAL_THRESHOLD = 0.10

MACRO_TO_STOCK_DISCOUNT = 0.35
ATTENTION_TO_BELIEF_SCALE = 0.30
UNCERTAINTY_PENALTY_SCALE = 0.45

MICRO_WEIGHT = 0.45
MACRO_WEIGHT = 0.20
TECHNICAL_WEIGHT = 0.25
ATTENTION_WEIGHT = 0.10

TECHNICAL_PRIOR_WEIGHT = 0.65
LLM_TECHNICAL_WEIGHT = 0.35

TECHNICAL_MA_WEIGHT = 0.35
TECHNICAL_MOMENTUM_WEIGHT = 0.30
TECHNICAL_RECENT_RETURN_WEIGHT = 0.15
TECHNICAL_RSI_WEIGHT = 0.10
TECHNICAL_ANCHOR_WEIGHT = 0.10
TECHNICAL_VOLATILITY_PENALTY = 0.15


def build_belief_score(
    cognition: Mapping[str, Any],
    market_features: Mapping[str, Any],
    *,
    neutral_threshold: float = SENTIMENT_NEUTRAL_THRESHOLD,
) -> Dict[str, float | int]:
    """Build final stock-level belief fields.

    Formula
    -------
    ``micro_component``:
        ``micro_direction * micro_strength``.

    ``macro_component``:
        ``macro_direction * macro_strength * 0.35``. Macro evidence is
        discounted because broad market news should not become a full
        current-symbol signal unless there is direct linkage.

    ``technical_prior``:
        Rule-based technical score from market features.

    ``technical_component``:
        ``0.65 * technical_prior + 0.35 * llm_technical_bias``.

    ``attention_component``:
        ``attention_bias * 0.30``. Attention can amplify belief but should not
        dominate it.

    ``raw_score``:
        Weighted sum of micro, macro, technical, and attention components.

    ``belief_score``:
        ``clip(raw_score * (1 - 0.45 * uncertainty), -1, 1)``.

    ``sentiment_class``:
        ``1`` if score is above ``neutral_threshold``, ``-1`` if below
        ``-neutral_threshold``, otherwise ``0``.
    """

    micro_component = _direction(cognition.get("micro_direction")) * _clip_float(
        cognition.get("micro_strength"),
        0.0,
        1.0,
        0.0,
    )
    macro_component = (
        _direction(cognition.get("macro_direction"))
        * _clip_float(cognition.get("macro_strength"), 0.0, 1.0, 0.0)
        * MACRO_TO_STOCK_DISCOUNT
    )
    technical_prior = build_technical_prior(market_features)
    llm_technical_bias = _clip_float(cognition.get("technical_bias"), -1.0, 1.0, technical_prior)
    technical_component = _clip(
        TECHNICAL_PRIOR_WEIGHT * technical_prior + LLM_TECHNICAL_WEIGHT * llm_technical_bias,
        -1.0,
        1.0,
    )
    attention_component = (
        _clip_float(cognition.get("attention_bias"), -1.0, 1.0, 0.0)
        * ATTENTION_TO_BELIEF_SCALE
    )
    uncertainty = _clip_float(cognition.get("uncertainty"), 0.0, 1.0, 0.5)
    uncertainty_multiplier = _clip(1.0 - UNCERTAINTY_PENALTY_SCALE * uncertainty, 0.0, 1.0)

    raw_score = _clip(
        MICRO_WEIGHT * micro_component
        + MACRO_WEIGHT * macro_component
        + TECHNICAL_WEIGHT * technical_component
        + ATTENTION_WEIGHT * attention_component,
        -1.0,
        1.0,
    )
    belief_score = _clip(raw_score * uncertainty_multiplier, -1.0, 1.0)
    sentiment_class = sentiment_class_from_belief_score(
        belief_score,
        neutral_threshold=neutral_threshold,
    )

    return {
        "belief_score": round(float(belief_score), 6),
        "sentiment_class": int(sentiment_class),
        "raw_belief_score": round(float(raw_score), 6),
        "uncertainty_multiplier": round(float(uncertainty_multiplier), 6),
        "micro_component": round(float(micro_component), 6),
        "macro_component": round(float(macro_component), 6),
        "technical_prior": round(float(technical_prior), 6),
        "technical_component": round(float(technical_component), 6),
        "attention_component": round(float(attention_component), 6),
    }


def build_technical_prior(market_features: Mapping[str, Any]) -> float:
    """Build a rule-based technical belief prior from engineered features."""

    score = (
        TECHNICAL_MA_WEIGHT * _clip_float(market_features.get("ma_score"), -1.0, 1.0, 0.0)
        + TECHNICAL_MOMENTUM_WEIGHT * _clip_float(market_features.get("momentum_score"), -1.0, 1.0, 0.0)
        + TECHNICAL_RECENT_RETURN_WEIGHT * _clip_float(market_features.get("recent_return_score"), -1.0, 1.0, 0.0)
        + TECHNICAL_RSI_WEIGHT * _clip_float(market_features.get("rsi_score"), -1.0, 1.0, 0.0)
        + TECHNICAL_ANCHOR_WEIGHT * _clip_float(market_features.get("anchor_score"), -1.0, 1.0, 0.0)
        - TECHNICAL_VOLATILITY_PENALTY * _clip_float(market_features.get("volatility_score"), 0.0, 1.0, 0.0)
    )
    return _clip(score, -1.0, 1.0)


def sentiment_class_from_belief_score(
    belief_score: float,
    *,
    neutral_threshold: float = SENTIMENT_NEUTRAL_THRESHOLD,
) -> int:
    """Map continuous belief to {-1, 0, 1} with a tunable neutral band."""

    threshold = max(0.0, float(neutral_threshold))
    score = float(belief_score)
    if score > threshold:
        return 1
    if score < -threshold:
        return -1
    return 0


def _direction(value: Any) -> int:
    numeric = _clip_float(value, -1.0, 1.0, 0.0)
    if numeric > 0.25:
        return 1
    if numeric < -0.25:
        return -1
    return 0


def _clip_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if numeric != numeric:
        return float(default)
    return _clip(numeric, low, high)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
