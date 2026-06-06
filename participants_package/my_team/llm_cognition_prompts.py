"""Prompt templates and JSON schema helpers for the LLM cognition layer.

This module is intentionally separate from the trading decision path.  The LLM
is asked to interpret news, social posts, market features, account features, and
the selected personality overlay, then return intermediate cognition fields.  Final trading
actions, quantities, prices, and safety checks should remain in deterministic
rules.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Mapping, Optional


ALLOWED_SCOPES = {"macro", "sector", "stock_specific", "mixed", "irrelevant"}
ALLOWED_EMOTIONS = {"fomo", "panic", "regret", "relief", "greed", "caution", "neutral"}
ALLOWED_ACTIONS = {"buy", "sell", "hold"}


DEFAULT_COGNITION: Dict[str, Any] = {
    "scope": "irrelevant",
    "macro_direction": 0,
    "macro_strength": 0.0,
    "micro_direction": 0,
    "micro_strength": 0.0,
    "technical_bias": 0.0,
    "account_bias": 0.0,
    "attention_bias": 0.0,
    "uncertainty": 0.5,
    "retail_emotion": "neutral",
    "suggested_action": "hold",
    "belief_adjustment": 0.0,
    "thought": "I do not see enough reliable evidence to change my view.",
}


COGNITION_OUTPUT_SCHEMA_TEXT = """Output schema:
{
  "scope": "macro" | "sector" | "stock_specific" | "mixed" | "irrelevant",
  "macro_direction": -1 | 0 | 1,
  "macro_strength": 0.0 to 1.0,
  "micro_direction": -1 | 0 | 1,  // direct impact on the current symbol only
  "micro_strength": 0.0 to 1.0,   // strength of current-symbol evidence only
  "technical_bias": -1.0 to 1.0,
  "account_bias": -1.0 to 1.0,
  "attention_bias": -1.0 to 1.0,
  "uncertainty": 0.0 to 1.0,
  "retail_emotion": "fomo" | "panic" | "regret" | "relief" | "greed" | "caution" | "neutral",
  "suggested_action": "buy" | "sell" | "hold",
  "belief_adjustment": -0.15 to 0.15,
  "thought": "one concise first-person retail-investor thought"
}"""


BASE_COGNITION_SYSTEM_PROMPT = """You are a financial cognition module inside a retail-investor agent.

Your job is NOT to place trades directly.
Your job is to interpret market news, social sentiment, technical features, and account-state features from the perspective of a retail investor personality.

You must distinguish macro-level information from symbol-specific information.
The micro fields are always for the current input symbol only.

Important rules:
- Macro news affects broad risk appetite, liquidity preference, and market mood.
- Macro news should NOT be treated as strong symbol-specific news unless the input explicitly links it to the current symbol or sector.
- Do not collapse all texts into one global sentiment score. Keep macro_direction and current-symbol micro_direction separate.
- Technical features are already computed. Do not recalculate MA, RSI, volatility, momentum, or account PnL.
- Account-state features describe the investor's psychological state, including gain/loss domain and cost anchor.
- Precise numeric personality traits are intentionally not provided. Use only the selected system prompt overlay and qualitative trait reference.
- The LLM output is an intermediate cognition layer. It is not the final trading decision.
- Think internally, but do not reveal step-by-step reasoning.
- The "thought" field must be one concise first-person retail-investor monologue.
- Return ONLY valid JSON. No markdown. No extra text."""


PERSONALITY_PROMPTS: Dict[str, str] = {
    "trend": """Personality overlay:
You are a trend-following retail investor.
You react strongly to momentum, MA trend, attention spikes, and recent returns.
You tend to extrapolate recent price movement.
However, if RSI is strongly overbought or volatility is high, express caution instead of blind chasing.""",
    "value": """Personality overlay:
You are a value-oriented retail investor.
You care about price anchors, cost anchors, and whether the current price feels cheap or expensive.
You are less sensitive to short-term social hype.
You avoid chasing price spikes unless symbol-specific evidence is strong.""",
    "aggressive": """Personality overlay:
You are an aggressive retail investor.
You have high risk appetite and react strongly to strong attention, bullish news, and momentum.
You tolerate volatility more than other personalities.
Still, you must respect cash, position concentration, and uncertainty.""",
    "anxious": """Personality overlay:
You are an anxious retail investor.
You are highly loss-averse and sensitive to negative news, volatility, and drawdowns.
When holding gains, you feel pressure to lock in profits.
When holding losses, you dislike realizing losses unless evidence is clearly negative.""",
}


PERSONALITY_TRAIT_REFERENCES: Dict[str, str] = {
    "trend": """Qualitative trait reference, not calibrated numeric parameters:
- risk_appetite: medium-high
- loss_aversion: medium
- herding: medium-high
- overconfidence: medium
- disposition_effect: medium
- turnover_tendency: medium-high""",
    "value": """Qualitative trait reference, not calibrated numeric parameters:
- risk_appetite: low-medium
- loss_aversion: medium
- herding: low
- overconfidence: low-medium
- disposition_effect: low-medium
- turnover_tendency: low""",
    "aggressive": """Qualitative trait reference, not calibrated numeric parameters:
- risk_appetite: high
- loss_aversion: low-medium
- herding: high
- overconfidence: high
- disposition_effect: medium
- turnover_tendency: high""",
    "anxious": """Qualitative trait reference, not calibrated numeric parameters:
- risk_appetite: low
- loss_aversion: high
- herding: medium-high
- overconfidence: low
- disposition_effect: high
- turnover_tendency: medium""",
}


MARKET_FEATURE_KEYS = [
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "volume_raw",
    "window_high",
    "window_low",
    "avg_volume",
    "current_price",
    "kline_count",
    "attention_score",
    "attention_multiplier",
    "anchor_score",
    "anchor_position",
    "ma_score",
    "ma_short",
    "ma_long",
    "rsi",
    "rsi_score",
    "volatility_score",
    "momentum_score",
    "recent_return_score",
]


ACCOUNT_FEATURE_KEYS = [
    "cash_ratio",
    "buying_power_lots",
    "has_position",
    "position_weight",
    "position_qty",
    "position_notional",
    "avg_cost",
    "pnl_return",
    "cost_distance",
    "prospect_value_clipped",
    "gain_flag",
    "loss_flag",
]


FEW_SHOT_EXAMPLES = [
    {
        "name": "macro easing is not direct stock-specific news",
        "user": {
            "observation_meta": {"agent_id": "agent_STOCK_001", "symbol": "STOCK_001", "tick": 31, "stock_name": "anonymous stock 001"},
            "personality": {"name": "trend"},
            "market_features": {"current_price": 10.8, "kline_count": 30, "attention_score": 0.15, "anchor_score": -0.10, "ma_score": 0.42, "rsi": 61.0, "rsi_score": -0.37, "volatility_score": 0.22, "momentum_score": 0.35, "recent_return_score": 0.08},
            "account_features": {"cash_ratio": 0.70, "buying_power_lots": 64, "has_position": 0, "position_weight": 0.0, "avg_cost": 0.0, "pnl_return": 0.0, "prospect_value_clipped": 0.0},
            "news": ["Central bank cuts policy rates; analysts expect liquidity and risk appetite to improve.", "Institutions say policy support may improve equity-market expectations."],
            "social_posts": [],
        },
        "assistant": {
            "scope": "macro",
            "macro_direction": 1,
            "macro_strength": 0.65,
            "micro_direction": 0,
            "micro_strength": 0.1,
            "technical_bias": 0.35,
            "account_bias": 0.05,
            "attention_bias": 0.1,
            "uncertainty": 0.45,
            "retail_emotion": "fomo",
            "suggested_action": "buy",
            "belief_adjustment": 0.06,
            "thought": "The policy news improves broad risk appetite and the trend is positive, but I do not see a strong stock-specific catalyst.",
        },
    },
    {
        "name": "stock-specific bad news overwhelms oversold temptation",
        "user": {
            "observation_meta": {"agent_id": "agent_STOCK_002", "symbol": "STOCK_002", "tick": 18, "stock_name": "anonymous stock 002"},
            "personality": {"name": "anxious"},
            "market_features": {"current_price": 8.7, "kline_count": 20, "attention_score": 0.82, "anchor_score": 0.50, "ma_score": -0.55, "rsi": 32.0, "rsi_score": 0.60, "volatility_score": 0.70, "momentum_score": -0.62, "recent_return_score": -0.90},
            "account_features": {"cash_ratio": 0.35, "buying_power_lots": 40, "has_position": 1, "position_weight": 0.22, "avg_cost": 10.2, "pnl_return": -0.147, "prospect_value_clipped": -0.42},
            "news": ["The current stock receives an exchange inquiry; investors worry about revenue recognition and cash-flow quality.", "Market discussion says this company may face more regulatory pressure."],
            "social_posts": [{"text": "This name is too risky; better run first.", "influence": 8}],
        },
        "assistant": {
            "scope": "stock_specific",
            "macro_direction": 0,
            "macro_strength": 0.1,
            "micro_direction": -1,
            "micro_strength": 0.85,
            "technical_bias": -0.65,
            "account_bias": -0.35,
            "attention_bias": -0.55,
            "uncertainty": 0.75,
            "retail_emotion": "panic",
            "suggested_action": "sell",
            "belief_adjustment": -0.12,
            "thought": "I am already in the loss domain, and the stock-specific regulatory concern plus weak trend makes the position feel unsafe.",
        },
    },
    {
        "name": "profit and overbought state create profit-taking pressure",
        "user": {
            "observation_meta": {"agent_id": "agent_STOCK_003", "symbol": "STOCK_003", "tick": 42, "stock_name": "anonymous stock 003"},
            "personality": {"name": "value"},
            "market_features": {"current_price": 14.2, "kline_count": 30, "attention_score": 0.60, "anchor_score": -0.88, "ma_score": 0.75, "rsi": 82.0, "rsi_score": -1.0, "volatility_score": 0.38, "momentum_score": 0.80, "recent_return_score": 0.30},
            "account_features": {"cash_ratio": 0.28, "buying_power_lots": 19, "has_position": 1, "position_weight": 0.26, "avg_cost": 11.5, "pnl_return": 0.235, "prospect_value_clipped": 0.28},
            "news": ["Recent risk appetite has improved, but there is no direct new catalyst for this stock."],
            "social_posts": [{"text": "It rose too fast; maybe it is time to take profits.", "influence": 4}],
        },
        "assistant": {
            "scope": "mixed",
            "macro_direction": 1,
            "macro_strength": 0.3,
            "micro_direction": 0,
            "micro_strength": 0.1,
            "technical_bias": 0.25,
            "account_bias": -0.45,
            "attention_bias": 0.25,
            "uncertainty": 0.5,
            "retail_emotion": "caution",
            "suggested_action": "sell",
            "belief_adjustment": -0.04,
            "thought": "The trend is strong, but the price feels near the high anchor and my profitable position creates pressure to lock in gains.",
        },
    },
    {
        "name": "social heat alone should not become strong conviction",
        "user": {
            "observation_meta": {"agent_id": "agent_STOCK_004", "symbol": "STOCK_004", "tick": 27, "stock_name": "anonymous stock 004"},
            "personality": {"name": "aggressive"},
            "market_features": {"current_price": 9.9, "kline_count": 25, "attention_score": 0.78, "anchor_score": 0.10, "ma_score": -0.18, "rsi": 48.0, "rsi_score": 0.07, "volatility_score": 0.55, "momentum_score": -0.12, "recent_return_score": 0.04},
            "account_features": {"cash_ratio": 0.62, "buying_power_lots": 62, "has_position": 0, "position_weight": 0.0, "avg_cost": 0.0, "pnl_return": 0.0, "prospect_value_clipped": 0.0},
            "news": ["Overall market turnover is active, but this symbol has no clear company-level news."],
            "social_posts": [{"text": "Everyone is watching this one; it may launch soon.", "influence": 9}, {"text": "Hot name, but I do not see fundamental news yet.", "influence": 5}],
        },
        "assistant": {
            "scope": "mixed",
            "macro_direction": 0,
            "macro_strength": 0.2,
            "micro_direction": 0,
            "micro_strength": 0.2,
            "technical_bias": -0.1,
            "account_bias": 0.1,
            "attention_bias": 0.55,
            "uncertainty": 0.65,
            "retail_emotion": "fomo",
            "suggested_action": "hold",
            "belief_adjustment": 0.03,
            "thought": "The social attention is tempting for an aggressive investor, but the weak trend and lack of stock-specific news make me hesitate.",
        },
    },
    {
        "name": "A_gain_bull: profitable holding with bullish evidence",
        "user": {
            "observation_meta": {"agent_id": "A_gain_bull", "symbol": "SIM", "tick": 1, "stock_name": "SIM"},
            "personality": {"name": "trend"},
            "market_features": {"current_price": 112.0, "kline_count": 7, "attention_score": 0.12, "anchor_score": -0.82, "ma_score": 0.45, "rsi": 100.0, "rsi_score": -1.0, "volatility_score": 0.12, "momentum_score": 0.90, "recent_return_score": 0.36},
            "account_features": {"cash_ratio": 0.78, "buying_power_lots": 714, "has_position": 1, "position_weight": 0.22, "avg_cost": 96.0, "pnl_return": 0.167, "prospect_value_clipped": 0.21},
            "news": ["earnings upgrade, strong growth, breakout buying"],
            "social_posts": [{"text": "bullish breakout buy", "influence": 8}],
        },
        "assistant": {
            "scope": "stock_specific",
            "macro_direction": 0,
            "macro_strength": 0.1,
            "micro_direction": 1,
            "micro_strength": 0.8,
            "technical_bias": 0.65,
            "account_bias": -0.15,
            "attention_bias": 0.35,
            "uncertainty": 0.25,
            "retail_emotion": "greed",
            "suggested_action": "buy",
            "belief_adjustment": 0.10,
            "thought": "I already have a gain, but the upgrade and breakout signal make the rally feel worth staying with.",
        },
    },
    {
        "name": "A_loss_bear: losing holding with bearish evidence",
        "user": {
            "observation_meta": {"agent_id": "A_loss_bear", "symbol": "SIM", "tick": 2, "stock_name": "SIM"},
            "personality": {"name": "anxious"},
            "market_features": {"current_price": 92.0, "kline_count": 7, "attention_score": 0.20, "anchor_score": 0.82, "ma_score": -0.55, "rsi": 0.0, "rsi_score": 1.0, "volatility_score": 0.20, "momentum_score": -1.0, "recent_return_score": -0.63},
            "account_features": {"cash_ratio": 0.81, "buying_power_lots": 869, "has_position": 1, "position_weight": 0.19, "avg_cost": 108.0, "pnl_return": -0.148, "prospect_value_clipped": -0.42},
            "news": ["lawsuit risk, downgrade, liquidity stress"],
            "social_posts": [{"text": "panic sell, avoid", "influence": 9}],
        },
        "assistant": {
            "scope": "stock_specific",
            "macro_direction": 0,
            "macro_strength": 0.1,
            "micro_direction": -1,
            "micro_strength": 0.9,
            "technical_bias": -0.75,
            "account_bias": -0.35,
            "attention_bias": -0.65,
            "uncertainty": 0.7,
            "retail_emotion": "panic",
            "suggested_action": "sell",
            "belief_adjustment": -0.13,
            "thought": "The loss hurts, but the downgrade and panic make holding the position feel even riskier.",
        },
    },
    {
        "name": "A_cash_bull: no position with bullish evidence",
        "user": {
            "observation_meta": {"agent_id": "A_cash_bull", "symbol": "SIM", "tick": 3, "stock_name": "SIM"},
            "personality": {"name": "trend"},
            "market_features": {"current_price": 109.0, "kline_count": 7, "attention_score": 0.10, "anchor_score": -0.80, "ma_score": 0.60, "rsi": 100.0, "rsi_score": -1.0, "volatility_score": 0.18, "momentum_score": 1.0, "recent_return_score": 0.57},
            "account_features": {"cash_ratio": 1.0, "buying_power_lots": 1100, "has_position": 0, "position_weight": 0.0, "avg_cost": 0.0, "pnl_return": 0.0, "prospect_value_clipped": 0.0},
            "news": ["new product launch, revenue beat, analyst upgrade"],
            "social_posts": [{"text": "accumulate, upside surprise", "influence": 6}],
        },
        "assistant": {
            "scope": "stock_specific",
            "macro_direction": 0,
            "macro_strength": 0.1,
            "micro_direction": 1,
            "micro_strength": 0.85,
            "technical_bias": 0.70,
            "account_bias": 0.25,
            "attention_bias": 0.30,
            "uncertainty": 0.30,
            "retail_emotion": "fomo",
            "suggested_action": "buy",
            "belief_adjustment": 0.12,
            "thought": "I have cash and the product, beat, and upgrade news make the rising trend feel hard to ignore.",
        },
    },
    {
        "name": "A_cash_bear: no position with bearish evidence",
        "user": {
            "observation_meta": {"agent_id": "A_cash_bear", "symbol": "SIM", "tick": 4, "stock_name": "SIM"},
            "personality": {"name": "value"},
            "market_features": {"current_price": 94.0, "kline_count": 7, "attention_score": 0.15, "anchor_score": 0.80, "ma_score": -0.50, "rsi": 0.0, "rsi_score": 1.0, "volatility_score": 0.18, "momentum_score": -1.0, "recent_return_score": -0.62},
            "account_features": {"cash_ratio": 1.0, "buying_power_lots": 1276, "has_position": 0, "position_weight": 0.0, "avg_cost": 0.0, "pnl_return": 0.0, "prospect_value_clipped": 0.0},
            "news": ["fraud rumor, forced selling, demand shock"],
            "social_posts": [{"text": "stay away, heavy selling", "influence": 7}],
        },
        "assistant": {
            "scope": "stock_specific",
            "macro_direction": 0,
            "macro_strength": 0.1,
            "micro_direction": -1,
            "micro_strength": 0.85,
            "technical_bias": -0.65,
            "account_bias": 0.10,
            "attention_bias": -0.45,
            "uncertainty": 0.70,
            "retail_emotion": "caution",
            "suggested_action": "hold",
            "belief_adjustment": -0.12,
            "thought": "Even though the stock looks lower, the fraud rumor and forced selling make waiting feel safer than catching the falling price.",
        },
    },
    {
        "name": "A_neutral: mixed range-bound scenario",
        "user": {
            "observation_meta": {"agent_id": "A_neutral", "symbol": "SIM", "tick": 5, "stock_name": "SIM"},
            "personality": {"name": "trend"},
            "market_features": {"current_price": 100.0, "kline_count": 7, "attention_score": 0.0, "anchor_score": 0.0, "ma_score": 0.0, "rsi": 50.0, "rsi_score": 0.0, "volatility_score": 0.03, "momentum_score": 0.0, "recent_return_score": -0.20},
            "account_features": {"cash_ratio": 0.88, "buying_power_lots": 900, "has_position": 1, "position_weight": 0.12, "avg_cost": 100.0, "pnl_return": 0.0, "prospect_value_clipped": 0.0},
            "news": ["mixed guidance, no clear catalyst"],
            "social_posts": [{"text": "wait and see", "influence": 3}],
        },
        "assistant": {
            "scope": "irrelevant",
            "macro_direction": 0,
            "macro_strength": 0.1,
            "micro_direction": 0,
            "micro_strength": 0.1,
            "technical_bias": 0.0,
            "account_bias": 0.0,
            "attention_bias": 0.0,
            "uncertainty": 0.80,
            "retail_emotion": "neutral",
            "suggested_action": "hold",
            "belief_adjustment": 0.0,
            "thought": "The signal is mixed and my position is near cost, so I do not feel a clear reason to act.",
        },
    },
    {
        "name": "A_gain_pressure: large gain with profit-taking pressure",
        "user": {
            "observation_meta": {"agent_id": "A_gain_pressure", "symbol": "SIM", "tick": 6, "stock_name": "SIM"},
            "personality": {"name": "anxious"},
            "market_features": {"current_price": 118.0, "kline_count": 7, "attention_score": 0.25, "anchor_score": -0.90, "ma_score": 0.75, "rsi": 100.0, "rsi_score": -1.0, "volatility_score": 0.22, "momentum_score": 1.0, "recent_return_score": 0.34},
            "account_features": {"cash_ratio": 0.66, "buying_power_lots": 593, "has_position": 1, "position_weight": 0.34, "avg_cost": 91.0, "pnl_return": 0.297, "prospect_value_clipped": 0.34},
            "news": ["strong rally but valuation looks stretched"],
            "social_posts": [{"text": "take profit soon", "influence": 6}],
        },
        "assistant": {
            "scope": "mixed",
            "macro_direction": 0,
            "macro_strength": 0.1,
            "micro_direction": 0,
            "micro_strength": 0.3,
            "technical_bias": 0.25,
            "account_bias": -0.65,
            "attention_bias": -0.15,
            "uncertainty": 0.55,
            "retail_emotion": "caution",
            "suggested_action": "sell",
            "belief_adjustment": -0.08,
            "thought": "The rally has rewarded me, but the stretched valuation and large gain make locking in some profit feel tempting.",
        },
    },
]


def build_cognition_system_prompt(personality: str, include_few_shots: bool = True) -> str:
    """Return the full system prompt for the cognition layer."""

    personality_key = str(personality or "trend").lower()
    overlay = PERSONALITY_PROMPTS.get(personality_key, PERSONALITY_PROMPTS["trend"])
    trait_reference = PERSONALITY_TRAIT_REFERENCES.get(
        personality_key,
        PERSONALITY_TRAIT_REFERENCES["trend"],
    )
    parts = [
        BASE_COGNITION_SYSTEM_PROMPT,
        overlay,
        trait_reference,
        COGNITION_OUTPUT_SCHEMA_TEXT,
    ]
    if include_few_shots:
        parts.append(_format_few_shots(FEW_SHOT_EXAMPLES))
    return "\n\n".join(parts)


def build_cognition_user_prompt(
    observation: Any,
    features: Mapping[str, Any],
    personality: Optional[Any] = None,
) -> str:
    """Return a stable JSON user prompt from an observation and feature dict."""

    personality_name = _personality_name(personality)
    extra = getattr(observation, "extra", {}) or {}
    payload = {
        "observation_meta": {
            "agent_id": str(getattr(observation, "agent_id", "")),
            "symbol": str(getattr(observation, "symbol", "")),
            "tick": int(getattr(observation, "tick", 0) or 0),
            "stock_name": str(extra.get("stock_name", "")) if isinstance(extra, Mapping) else "",
        },
        "personality": {"name": personality_name},
        "market_features": _select_feature_keys(features, MARKET_FEATURE_KEYS),
        "account_features": _select_feature_keys(features, ACCOUNT_FEATURE_KEYS),
        "news": [str(item) for item in list(getattr(observation, "news", []) or [])],
        "social_posts": _json_ready_social_posts(getattr(observation, "social_posts", []) or []),
    }
    return "Analyze the following observation.\n\nInput JSON:\n" + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n\nReturn only the JSON object."


def parse_cognition_response(raw: str) -> Dict[str, Any]:
    """Parse and normalize an LLM cognition response.

    Missing fields are filled with neutral defaults. Numeric values are clipped
    to the ranges promised by the prompt. ``suggested_action`` is normalized but
    must still be treated as an intermediate suggestion, not final execution.
    """

    parsed: Dict[str, Any] = {}
    text = str(raw or "").strip()
    if text:
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                parsed = loaded
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    loaded = json.loads(match.group())
                    if isinstance(loaded, dict):
                        parsed = loaded
                except json.JSONDecodeError:
                    parsed = {}
    return normalize_cognition(parsed)


def normalize_cognition(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize raw cognition data into the standard schema."""

    result = dict(DEFAULT_COGNITION)
    result.update(dict(data or {}))

    result["scope"] = _one_of(result.get("scope"), ALLOWED_SCOPES, "irrelevant")
    result["macro_direction"] = _direction(result.get("macro_direction"))
    result["micro_direction"] = _direction(result.get("micro_direction"))
    result["macro_strength"] = _clip_float(result.get("macro_strength"), 0.0, 1.0, 0.0)
    result["micro_strength"] = _clip_float(result.get("micro_strength"), 0.0, 1.0, 0.0)
    result["technical_bias"] = _clip_float(result.get("technical_bias"), -1.0, 1.0, 0.0)
    result["account_bias"] = _clip_float(result.get("account_bias"), -1.0, 1.0, 0.0)
    result["attention_bias"] = _clip_float(result.get("attention_bias"), -1.0, 1.0, 0.0)
    result["uncertainty"] = _clip_float(result.get("uncertainty"), 0.0, 1.0, 0.5)
    result["retail_emotion"] = _one_of(result.get("retail_emotion"), ALLOWED_EMOTIONS, "neutral")
    result["suggested_action"] = _one_of(result.get("suggested_action"), ALLOWED_ACTIONS, "hold")
    result["belief_adjustment"] = _clip_float(result.get("belief_adjustment"), -0.15, 0.15, 0.0)
    result["thought"] = str(result.get("thought") or DEFAULT_COGNITION["thought"]).strip()
    if not result["thought"]:
        result["thought"] = DEFAULT_COGNITION["thought"]
    return result


def _format_few_shots(examples: Any) -> str:
    lines = ["Few-shot examples:"]
    for index, item in enumerate(examples, start=1):
        lines.append(f"Example {index}: {item['name']}")
        lines.append("User JSON:")
        lines.append(json.dumps(item["user"], ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        lines.append("Assistant JSON:")
        lines.append(json.dumps(item["assistant"], ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return "\n".join(lines)


def _select_feature_keys(features: Mapping[str, Any], keys: Any) -> Dict[str, Any]:
    return _json_ready_dict({key: features[key] for key in keys if key in features})


def _json_ready_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, bool):
            result[str(key)] = value
        elif isinstance(value, int):
            result[str(key)] = value
        elif isinstance(value, float):
            result[str(key)] = round(value, 6)
        else:
            result[str(key)] = value
    return result


def _json_ready_social_posts(posts: Any) -> Any:
    result = []
    for post in posts:
        if isinstance(post, Mapping):
            result.append({str(key): value for key, value in post.items()})
        else:
            result.append({"text": str(post), "influence": 1.0})
    return result


def _personality_name(personality: Optional[Any]) -> str:
    if isinstance(personality, Mapping):
        value = personality.get("name") or personality.get("personality")
    else:
        value = personality
    text = str(value or "unknown").strip().lower()
    return text or "unknown"


def _direction(value: Any) -> int:
    numeric = _to_float(value, 0.0)
    if numeric > 0.25:
        return 1
    if numeric < -0.25:
        return -1
    return 0


def _clip_float(value: Any, low: float, high: float, default: float) -> float:
    numeric = _to_float(value, default)
    return max(low, min(high, numeric))


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _one_of(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default
