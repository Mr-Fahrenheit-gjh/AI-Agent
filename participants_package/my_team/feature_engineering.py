"""Feature engineering helpers for team investment agents.

Data source
-----------
This module is designed around the official competition interface:
``MarketObservation.klines``, ``MarketObservation.cash``,
``MarketObservation.position``, and ``MarketObservation.avg_cost``.

Design principles
-----------------
1. Short-history fallback: every feature returns a neutral value when the
   requested lookback window is unavailable.
2. No judge interruption: invalid, empty, or partial inputs should not raise
   during scoring.
3. No decision logic: this module only emits reusable scalar features that can
   later be concatenated into a rules engine or LLM prompt.

Code map
--------
* ``build_market_features`` implements technical features from OHLCV rows.
* ``build_account_features`` implements cash, holding, anchor, and prospect
  theory features from the account state.
* ``build_observation_features`` joins both feature families for one official
  ``MarketObservation``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional


EPS = 1e-9
LOT_SIZE = 100

ATTENTION_WINDOW = 20
ATTENTION_MIN = 0.5
ATTENTION_MAX = 3.0

ANCHOR_WINDOW = 30

MA_SHORT_WINDOW = 5
MA_LONG_WINDOW = 20
MA_FULL_SIGNAL = 0.05

RSI_WINDOW = 14
RSI_MIN_HISTORY = 6
RSI_SCORE_SCALE = 30.0

VOLATILITY_WINDOW = 20
VOLATILITY_SCALE = 20.0

MOMENTUM_WINDOW = 5
MOMENTUM_FULL_SIGNAL = 0.10

RECENT_RETURN_FULL_SIGNAL = 0.05

PROSPECT_ALPHA = 0.88
PROSPECT_BETA = 0.88
PROSPECT_LOSS_LAMBDA = 2.25


def build_market_features(klines: Iterable[Any]) -> Dict[str, float]:
    """Build technical features from official ``KLine`` rows or dict rows.

    Features, formulas, and meanings
    --------------------------------
    ``attention_multiplier``:
        Formula: ``V_t / mean(V_{t-W:t-1})`` with ``W=20`` by default.
        Meaning: limited attention / abnormal volume. Large values represent
        unusual attention that can amplify FOMO or panic.
        Code parameters: ``ATTENTION_WINDOW``, ``ATTENTION_MIN``,
        ``ATTENTION_MAX``.

    ``attention_score``:
        Formula: ``clip(log(attention_multiplier) / log(3), -1, 1)``.
        Meaning: normalized attention feature, positive for volume expansion.

    ``anchor_position``:
        Formula: ``(C_t - Low_W) / (High_W - Low_W + eps)`` with ``W=30``.
        Meaning: current close's percentile in a recent high-low anchor range.
        Code parameter: ``ANCHOR_WINDOW``.

    ``anchor_score``:
        Formula: ``clip(1 - 2 * anchor_position, -1, 1)``.
        Meaning: low-price anchoring is positive, high-price anchoring is
        negative.

    ``ma_short`` and ``ma_long``:
        Formula: moving averages of recent closes over 5 and 20 bars.
        Meaning: raw MA levels for debugging and prompt construction.
        Code parameters: ``MA_SHORT_WINDOW``, ``MA_LONG_WINDOW``.

    ``ma_score``:
        Formula: ``clip((MA5 / MA20 - 1) / 0.05, -1, 1)``.
        Meaning: trend-following MA feature. Positive means short-term trend is
        stronger than the longer anchor.
        Code parameter: ``MA_FULL_SIGNAL``.

    ``rsi``:
        Formula: standard RSI over 14 periods. If ``6 <= n < 15`` use
        ``n - 1``; if ``n < 6`` return neutral ``50``.
        Meaning: overbought / oversold oscillator.
        Code parameters: ``RSI_WINDOW``, ``RSI_MIN_HISTORY``.

    ``rsi_score``:
        Formula: ``clip((50 - RSI) / 30, -1, 1)``.
        Meaning: oversold is positive, overbought is negative.
        Code parameter: ``RSI_SCORE_SCALE``.

    ``volatility_score``:
        Formula: ``clip(std(returns[-20:]) * 20, 0, 1)``.
        Meaning: risk and uncertainty from realized close-to-close returns.
        Code parameters: ``VOLATILITY_WINDOW``, ``VOLATILITY_SCALE``.

    ``momentum_score``:
        Formula: ``clip((C_t / C_{t-k} - 1) / 0.10, -1, 1)`` with ``k=5``.
        Meaning: medium-short cumulative return / momentum.
        Code parameters: ``MOMENTUM_WINDOW``, ``MOMENTUM_FULL_SIGNAL``.

    ``recent_return_score``:
        Formula: ``clip((C_t / C_{t-1} - 1) / 0.05, -1, 1)``.
        Meaning: most recent price shock.
        Code parameter: ``RECENT_RETURN_FULL_SIGNAL``.

    ``current_price`` and ``kline_count``:
        Latest close and number of usable rows.
    """

    rows = list(klines or [])
    count = len(rows)
    neutral = _neutral_market_features(count=count)
    if not rows:
        return neutral

    closes = [_row_value(row, "close", 0.0) for row in rows]
    highs = [_row_value(row, "high", close) for row, close in zip(rows, closes)]
    lows = [_row_value(row, "low", close) for row, close in zip(rows, closes)]
    volumes = [_row_value(row, "volume", 0.0) for row in rows]

    valid_closes = [value for value in closes if value > 0]
    if not valid_closes:
        return neutral

    current_price = valid_closes[-1]

    attention_multiplier = 1.0
    if len(volumes) >= 2:
        past_volumes = [value for value in volumes[-(ATTENTION_WINDOW + 1) : -1] if value > 0]
        avg_volume = _mean(past_volumes)
        if avg_volume > 0:
            attention_multiplier = _clip(
                _safe_div(volumes[-1], avg_volume, default=1.0),
                ATTENTION_MIN,
                ATTENTION_MAX,
            )
    attention_score = _clip(math.log(max(attention_multiplier, EPS)) / math.log(3.0), -1.0, 1.0)

    anchor_highs = [value for value in highs[-ANCHOR_WINDOW:] if value > 0]
    anchor_lows = [value for value in lows[-ANCHOR_WINDOW:] if value > 0]
    if anchor_highs and anchor_lows:
        high_w = max(anchor_highs)
        low_w = min(anchor_lows)
        anchor_position = _clip(_safe_div(current_price - low_w, high_w - low_w + EPS), 0.0, 1.0)
    else:
        anchor_position = 0.5
    anchor_score = _clip(1.0 - 2.0 * anchor_position, -1.0, 1.0)

    ma_short = _mean(valid_closes[-MA_SHORT_WINDOW:])
    ma_long = _mean(valid_closes[-MA_LONG_WINDOW:])
    ma_score = _clip(_safe_div(ma_short, ma_long, default=1.0) - 1.0, -MA_FULL_SIGNAL, MA_FULL_SIGNAL)
    ma_score = _safe_div(ma_score, MA_FULL_SIGNAL, default=0.0)

    rsi = _rsi(valid_closes)
    rsi_score = _clip((50.0 - rsi) / RSI_SCORE_SCALE, -1.0, 1.0)

    returns = _returns(valid_closes)
    volatility_score = _clip(_std(returns[-VOLATILITY_WINDOW:]) * VOLATILITY_SCALE, 0.0, 1.0)

    momentum_score = 0.0
    if len(valid_closes) >= 2:
        lookback_index = max(0, len(valid_closes) - 1 - MOMENTUM_WINDOW)
        base_price = valid_closes[lookback_index]
        momentum_score = _clip(
            _safe_div(current_price, base_price, default=1.0) - 1.0,
            -MOMENTUM_FULL_SIGNAL,
            MOMENTUM_FULL_SIGNAL,
        )
        momentum_score = _safe_div(momentum_score, MOMENTUM_FULL_SIGNAL, default=0.0)

    recent_return_score = 0.0
    if len(valid_closes) >= 2:
        recent_return_score = _clip(
            _safe_div(current_price, valid_closes[-2], default=1.0) - 1.0,
            -RECENT_RETURN_FULL_SIGNAL,
            RECENT_RETURN_FULL_SIGNAL,
        )
        recent_return_score = _safe_div(recent_return_score, RECENT_RETURN_FULL_SIGNAL, default=0.0)

    return {
        "attention_multiplier": float(attention_multiplier),
        "attention_score": float(attention_score),
        "anchor_position": float(anchor_position),
        "anchor_score": float(anchor_score),
        "ma_short": float(ma_short),
        "ma_long": float(ma_long),
        "ma_score": float(ma_score),
        "rsi": float(rsi),
        "rsi_score": float(rsi_score),
        "volatility_score": float(volatility_score),
        "momentum_score": float(momentum_score),
        "recent_return_score": float(recent_return_score),
        "current_price": float(current_price),
        "kline_count": float(count),
    }


def build_account_features(
    *,
    cash: float,
    symbol: str,
    position: int,
    avg_cost: float,
    current_price: float,
    latest_prices_by_symbol: Optional[Mapping[str, float]] = None,
    positions_by_symbol: Optional[Mapping[str, int]] = None,
    avg_cost_by_symbol: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Build cash, holding, anchor, and prospect-theory features.

    Features, formulas, and meanings
    --------------------------------
    ``cash_raw``:
        Formula: official ``MarketObservation.cash``.
        Meaning: current available cash.

    ``estimated_equity``:
        Formula: ``cash + sum(position_i * latest_price_i)``. If no portfolio
        dictionaries are supplied, fall back to ``cash + position * price``.
        Meaning: estimated portfolio value used for normalized account state.

    ``cash_ratio``:
        Formula: ``cash / estimated_equity``.
        Meaning: liquidity reserve and remaining buying capacity.

    ``buying_power_lots``:
        Formula: ``floor(cash / current_price / 100)``.
        Meaning: number of 100-share lots affordable at the current price.
        Code parameter: ``LOT_SIZE``.

    ``position_qty`` and ``has_position``:
        Formula: current official position and a 0/1 holding flag.
        Meaning: whether disposition-effect features should be active.

    ``position_notional`` and ``position_weight``:
        Formula: ``position * current_price`` and notional over estimated
        equity.
        Meaning: concentration in the current symbol.

    ``avg_cost``:
        Formula: official ``MarketObservation.avg_cost``.
        Meaning: average holding cost and psychological anchor. The evaluator
        keeps this anchor unchanged on sells while shares remain.

    ``pnl_return``:
        Formula: ``current_price / avg_cost - 1``.
        Meaning: unrealized gain/loss relative to the cost anchor.

    ``cost_distance``:
        Formula: ``log(current_price / avg_cost)``.
        Meaning: symmetric log distance from the cost anchor.

    ``gain_flag`` and ``loss_flag``:
        Formula: signs of ``pnl_return``.
        Meaning: binary state for gain-domain versus loss-domain behavior.

    ``prospect_value``:
        Formula: ``x^alpha`` for gains and ``-lambda * (-x)^beta`` for losses,
        where ``x = pnl_return``.
        Meaning: prospect-theory subjective value. Losses are amplified by
        loss aversion.
        Code parameters: ``PROSPECT_ALPHA``, ``PROSPECT_BETA``,
        ``PROSPECT_LOSS_LAMBDA``.

    ``prospect_value_clipped``:
        Formula: ``clip(prospect_value, -1, 1)``.
        Meaning: bounded psychological value for stable downstream use.
    """

    cash_value = float(cash or 0.0)
    current_price = float(current_price or 0.0)
    position_qty = max(0, int(position or 0))
    avg_cost_value = float(avg_cost or 0.0)

    estimated_equity = _estimated_equity(
        cash=cash_value,
        symbol=symbol,
        position=position_qty,
        current_price=current_price,
        latest_prices_by_symbol=latest_prices_by_symbol,
        positions_by_symbol=positions_by_symbol,
        avg_cost_by_symbol=avg_cost_by_symbol,
    )
    position_notional = position_qty * max(current_price, 0.0)

    has_position = 1.0 if position_qty > 0 else 0.0
    pnl_return = 0.0
    cost_distance = 0.0
    prospect_value = 0.0
    if position_qty > 0 and avg_cost_value > 0 and current_price > 0:
        ratio = _safe_div(current_price, avg_cost_value, default=1.0)
        pnl_return = ratio - 1.0
        cost_distance = math.log(max(ratio, EPS))
        prospect_value = _prospect_value(pnl_return)

    buying_power_lots = 0.0
    if current_price > 0:
        buying_power_lots = float(int(cash_value / current_price // LOT_SIZE))

    return {
        "cash_raw": float(cash_value),
        "estimated_equity": float(estimated_equity),
        "cash_ratio": float(_safe_div(cash_value, estimated_equity, default=0.0)),
        "buying_power_lots": float(buying_power_lots),
        "position_qty": float(position_qty),
        "has_position": has_position,
        "position_notional": float(position_notional),
        "position_weight": float(_safe_div(position_notional, estimated_equity, default=0.0)),
        "avg_cost": float(avg_cost_value),
        "pnl_return": float(pnl_return),
        "cost_distance": float(cost_distance),
        "gain_flag": 1.0 if pnl_return > 0 else 0.0,
        "loss_flag": 1.0 if pnl_return < 0 else 0.0,
        "prospect_value": float(prospect_value),
        "prospect_value_clipped": float(_clip(prospect_value, -1.0, 1.0)),
    }


def build_observation_features(
    observation: Any,
    *,
    latest_prices_by_symbol: Optional[Mapping[str, float]] = None,
    positions_by_symbol: Optional[Mapping[str, int]] = None,
    avg_cost_by_symbol: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """Build market and account features from one official observation.

    ``observation`` is expected to follow the official ``MarketObservation``
    shape. If ``observation.klines`` is empty, market features are neutral and
    account features use ``current_price=0``.
    """

    klines = list(getattr(observation, "klines", []) or [])
    market_features = build_market_features(klines)
    current_price = market_features.get("current_price", 0.0)
    account_features = build_account_features(
        cash=float(getattr(observation, "cash", 0.0) or 0.0),
        symbol=str(getattr(observation, "symbol", "")),
        position=int(getattr(observation, "position", 0) or 0),
        avg_cost=float(getattr(observation, "avg_cost", 0.0) or 0.0),
        current_price=current_price,
        latest_prices_by_symbol=latest_prices_by_symbol,
        positions_by_symbol=positions_by_symbol,
        avg_cost_by_symbol=avg_cost_by_symbol,
    )
    return {**market_features, **account_features}


def _neutral_market_features(count: int = 0) -> Dict[str, float]:
    return {
        "attention_multiplier": 1.0,
        "attention_score": 0.0,
        "anchor_position": 0.5,
        "anchor_score": 0.0,
        "ma_short": 0.0,
        "ma_long": 0.0,
        "ma_score": 0.0,
        "rsi": 50.0,
        "rsi_score": 0.0,
        "volatility_score": 0.0,
        "momentum_score": 0.0,
        "recent_return_score": 0.0,
        "current_price": 0.0,
        "kline_count": float(count),
    }


def _row_value(row: Any, key: str, default: float = 0.0) -> float:
    value: Any = default
    if isinstance(row, Mapping):
        value = row.get(key, default)
    else:
        value = getattr(row, key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _mean(values: List[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else 0.0


def _std(values: List[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) < 2:
        return 0.0
    mean = _mean(clean)
    return math.sqrt(sum((value - mean) ** 2 for value in clean) / (len(clean) - 1))


def _returns(closes: List[float]) -> List[float]:
    return [
        closes[index] / closes[index - 1] - 1.0
        for index in range(1, len(closes))
        if closes[index - 1] > 0
    ]


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) <= EPS:
        return default
    return numerator / denominator


def _prospect_value(
    x: float,
    alpha: float = PROSPECT_ALPHA,
    beta: float = PROSPECT_BETA,
    loss_lambda: float = PROSPECT_LOSS_LAMBDA,
) -> float:
    if x >= 0:
        return float(x**alpha)
    return float(-loss_lambda * ((-x) ** beta))


def _rsi(closes: List[float]) -> float:
    if len(closes) < RSI_MIN_HISTORY:
        return 50.0

    window = RSI_WINDOW if len(closes) >= RSI_WINDOW + 1 else len(closes) - 1
    if window <= 0:
        return 50.0

    recent = closes[-(window + 1) :]
    deltas = [recent[index] - recent[index - 1] for index in range(1, len(recent))]
    gains = [max(delta, 0.0) for delta in deltas]
    losses = [max(-delta, 0.0) for delta in deltas]
    avg_gain = _mean(gains)
    avg_loss = _mean(losses)

    if avg_gain <= 0 and avg_loss <= 0:
        return 50.0
    if avg_loss <= EPS:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def _estimated_equity(
    *,
    cash: float,
    symbol: str,
    position: int,
    current_price: float,
    latest_prices_by_symbol: Optional[Mapping[str, float]],
    positions_by_symbol: Optional[Mapping[str, int]],
    avg_cost_by_symbol: Optional[Mapping[str, float]],
) -> float:
    if not positions_by_symbol:
        return max(cash + position * max(current_price, 0.0), EPS)

    total = cash
    for item_symbol, item_position in positions_by_symbol.items():
        qty = max(0, int(item_position or 0))
        if qty <= 0:
            continue
        price = 0.0
        if latest_prices_by_symbol and item_symbol in latest_prices_by_symbol:
            price = float(latest_prices_by_symbol[item_symbol] or 0.0)
        elif item_symbol == symbol:
            price = current_price
        elif avg_cost_by_symbol and item_symbol in avg_cost_by_symbol:
            price = float(avg_cost_by_symbol[item_symbol] or 0.0)
        if price > 0:
            total += qty * price
    return max(total, EPS)
