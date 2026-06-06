"""Prompt templates and JSON helpers for the LLM belief layer.

This module is intentionally separate from the trading decision path. The LLM
only interprets the current symbol's market features, news, social posts, and
personality overlay, then returns stock-level belief fields. Account state,
position PnL, utility, final action, quantity, and limit price belong to later
deterministic Desire / Action layers.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Mapping, Optional


ALLOWED_SCOPES = {"macro", "sector", "stock_specific", "mixed", "irrelevant"}
ALLOWED_EMOTIONS = {"fomo", "panic", "regret", "relief", "greed", "caution", "neutral"}


DEFAULT_COGNITION: Dict[str, Any] = {
    "scope": "irrelevant",
    "macro_direction": 0,
    "macro_strength": 0.0,
    "micro_direction": 0,
    "micro_strength": 0.0,
    "technical_direction": 0,
    "technical_strength": 0.0,
    "technical_bias": 0.0,
    "attention_bias": 0.0,
    "uncertainty": 0.5,
    "retail_emotion": "neutral",
    "belief_reason": "I do not see enough reliable evidence to form a strong view.",
}


COGNITION_OUTPUT_SCHEMA_TEXT = """Output schema:
{
  "scope": "macro" | "sector" | "stock_specific" | "mixed" | "irrelevant",
  "macro_direction": -1 | 0 | 1,
  "macro_strength": 0.0 to 1.0,
  "micro_direction": -1 | 0 | 1,       // direct text impact on the current symbol only
  "micro_strength": 0.0 to 1.0,        // strength of current-symbol text evidence only
  "technical_direction": -1 | 0 | 1,
  "technical_strength": 0.0 to 1.0,
  "technical_bias": -1.0 to 1.0,
  "attention_bias": -1.0 to 1.0,
  "uncertainty": 0.0 to 1.0,
  "retail_emotion": "fomo" | "panic" | "regret" | "relief" | "greed" | "caution" | "neutral",
  "belief_reason": "one concise first-person belief about this stock"
}"""


BASE_COGNITION_SYSTEM_PROMPT = """You are the Belief layer inside a retail-investor agent.

Your job is NOT to place trades directly.
Your job is NOT to decide quantity, limit price, or whether account PnL should trigger profit-taking.
Your job is to interpret market features, news, and social sentiment for the current symbol.

You must distinguish macro-level information from symbol-specific information.
The micro fields are always for the current input symbol only.

Important rules:
- This layer is account-state free. Do not reason about cash, position size, avg cost, unrealized PnL, or utility.
- Macro news affects broad risk appetite, liquidity preference, and market mood.
- Macro news should NOT be treated as strong symbol-specific news unless the input explicitly links it to the current symbol or sector.
- Do not collapse all texts into one global sentiment score. Keep macro_direction and current-symbol micro_direction separate.
- Technical features are already computed. Do not recalculate MA, RSI, volatility, momentum, or returns.
- Precise numeric personality traits are intentionally not provided. Use only the selected system prompt overlay and qualitative trait reference.
- Personality may shape interpretation intensity, but it must not reverse clear facts.
- Think internally, but do not reveal step-by-step reasoning.
- The "belief_reason" field must be one concise first-person stock-level belief.
- Return ONLY valid JSON. No markdown. No extra text."""


PERSONALITY_PROMPTS: Dict[str, str] = {
    "trend": """Personality overlay:
You are a trend-following retail investor.
You react strongly to momentum, MA trend, attention spikes, and recent returns.
You tend to extrapolate recent price movement.
However, if RSI is strongly overbought or volatility is high, express caution instead of blind chasing.""",
    "value": """Personality overlay:
You are a value-oriented retail investor.
You care about price anchors and whether the current price feels cheap or expensive.
You are less sensitive to short-term social hype.
You avoid chasing price spikes unless symbol-specific evidence is strong.""",
    "aggressive": """Personality overlay:
You are an aggressive retail investor.
You have high risk appetite and react strongly to strong attention, bullish news, and momentum.
You tolerate volatility more than other personalities.
Still, you must respect uncertainty and should not turn unclear evidence into certainty.""",
    "anxious": """Personality overlay:
You are an anxious retail investor.
You are highly sensitive to negative news, volatility, and drawdowns.
You may discount weak bullish signals when uncertainty is high.
Still, you must not turn clearly bullish facts into bearish facts.""",
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


FEW_SHOT_EXAMPLES = [
    {
        "name": "macro easing is not direct stock-specific news",
        "user": {
            "observation_meta": {"agent_id": "agent_STOCK_001", "symbol": "STOCK_001", "tick": 31, "stock_name": "anonymous stock 001"},
            "personality": {"name": "trend"},
            "market_features": {"current_price": 10.8, "kline_count": 30, "attention_score": 0.15, "anchor_score": -0.10, "ma_score": 0.42, "rsi": 61.0, "rsi_score": -0.37, "volatility_score": 0.22, "momentum_score": 0.35, "recent_return_score": 0.08},
            "news": ["Central bank cuts policy rates; analysts expect liquidity and risk appetite to improve."],
            "social_posts": [],
        },
        "assistant": {"scope": "macro", "macro_direction": 1, "macro_strength": 0.65, "micro_direction": 0, "micro_strength": 0.1, "technical_direction": 1, "technical_strength": 0.45, "technical_bias": 0.35, "attention_bias": 0.1, "uncertainty": 0.45, "retail_emotion": "fomo", "belief_reason": "The policy news improves broad risk appetite and the trend looks positive, but I do not see a strong stock-specific catalyst."},
    },
    {
        "name": "stock-specific bad news overwhelms oversold temptation",
        "user": {
            "observation_meta": {"agent_id": "agent_STOCK_002", "symbol": "STOCK_002", "tick": 18, "stock_name": "anonymous stock 002"},
            "personality": {"name": "anxious"},
            "market_features": {"current_price": 8.7, "kline_count": 20, "attention_score": 0.82, "anchor_score": 0.50, "ma_score": -0.55, "rsi": 32.0, "rsi_score": 0.60, "volatility_score": 0.70, "momentum_score": -0.62, "recent_return_score": -0.90},
            "news": ["The current stock receives an exchange inquiry; investors worry about revenue recognition and cash-flow quality."],
            "social_posts": [{"text": "This name is too risky; better run first.", "influence": 8}],
        },
        "assistant": {"scope": "stock_specific", "macro_direction": 0, "macro_strength": 0.1, "micro_direction": -1, "micro_strength": 0.85, "technical_direction": -1, "technical_strength": 0.70, "technical_bias": -0.65, "attention_bias": -0.55, "uncertainty": 0.75, "retail_emotion": "panic", "belief_reason": "The stock-specific regulatory concern and weak trend make this stock feel unsafe despite the oversold signal."},
    },
    {
        "name": "profit-pressure words are not account state in belief layer",
        "user": {
            "observation_meta": {"agent_id": "agent_STOCK_003", "symbol": "STOCK_003", "tick": 42, "stock_name": "anonymous stock 003"},
            "personality": {"name": "value"},
            "market_features": {"current_price": 14.2, "kline_count": 30, "attention_score": 0.60, "anchor_score": -0.88, "ma_score": 0.75, "rsi": 82.0, "rsi_score": -1.0, "volatility_score": 0.38, "momentum_score": 0.80, "recent_return_score": 0.30},
            "news": ["Recent risk appetite has improved, but there is no direct new catalyst for this stock."],
            "social_posts": [{"text": "It rose too fast; maybe it is time to take profits.", "influence": 4}],
        },
        "assistant": {"scope": "mixed", "macro_direction": 1, "macro_strength": 0.3, "micro_direction": 0, "micro_strength": 0.1, "technical_direction": 1, "technical_strength": 0.55, "technical_bias": 0.25, "attention_bias": 0.2, "uncertainty": 0.55, "retail_emotion": "caution", "belief_reason": "The stock still has positive trend evidence, but the high anchor and overbought signal make my stock-level view only mildly bullish."},
    },
    {
        "name": "social heat alone should not become strong conviction",
        "user": {
            "observation_meta": {"agent_id": "agent_STOCK_004", "symbol": "STOCK_004", "tick": 27, "stock_name": "anonymous stock 004"},
            "personality": {"name": "aggressive"},
            "market_features": {"current_price": 9.9, "kline_count": 25, "attention_score": 0.78, "anchor_score": 0.10, "ma_score": -0.18, "rsi": 48.0, "rsi_score": 0.07, "volatility_score": 0.55, "momentum_score": -0.12, "recent_return_score": 0.04},
            "news": ["Overall market turnover is active, but this symbol has no clear company-level news."],
            "social_posts": [{"text": "Everyone is watching this one; it may launch soon.", "influence": 9}],
        },
        "assistant": {"scope": "mixed", "macro_direction": 0, "macro_strength": 0.2, "micro_direction": 0, "micro_strength": 0.2, "technical_direction": 0, "technical_strength": 0.2, "technical_bias": -0.1, "attention_bias": 0.55, "uncertainty": 0.65, "retail_emotion": "fomo", "belief_reason": "The social attention is tempting, but the weak trend and lack of stock-specific news keep my stock view close to neutral."},
    },
    {
        "name": "A_gain_bull: bullish evidence independent of account profit",
        "user": {
            "observation_meta": {"agent_id": "A_gain_bull", "symbol": "SIM", "tick": 1, "stock_name": "SIM"},
            "personality": {"name": "trend"},
            "market_features": {"current_price": 112.0, "kline_count": 7, "attention_score": 0.12, "anchor_score": -0.82, "ma_score": 0.45, "rsi": 100.0, "rsi_score": -1.0, "volatility_score": 0.12, "momentum_score": 0.90, "recent_return_score": 0.36},
            "news": ["earnings upgrade, strong growth, breakout buying"],
            "social_posts": [{"text": "bullish breakout buy", "influence": 8}],
        },
        "assistant": {"scope": "stock_specific", "macro_direction": 0, "macro_strength": 0.1, "micro_direction": 1, "micro_strength": 0.8, "technical_direction": 1, "technical_strength": 0.75, "technical_bias": 0.65, "attention_bias": 0.35, "uncertainty": 0.25, "retail_emotion": "greed", "belief_reason": "The upgrade, growth language, and breakout trend make this stock look strongly bullish."},
    },
    {
        "name": "A_loss_bear: bearish evidence independent of account loss",
        "user": {
            "observation_meta": {"agent_id": "A_loss_bear", "symbol": "SIM", "tick": 2, "stock_name": "SIM"},
            "personality": {"name": "anxious"},
            "market_features": {"current_price": 92.0, "kline_count": 7, "attention_score": 0.20, "anchor_score": 0.82, "ma_score": -0.55, "rsi": 0.0, "rsi_score": 1.0, "volatility_score": 0.20, "momentum_score": -1.0, "recent_return_score": -0.63},
            "news": ["lawsuit risk, downgrade, liquidity stress"],
            "social_posts": [{"text": "panic sell, avoid", "influence": 9}],
        },
        "assistant": {"scope": "stock_specific", "macro_direction": 0, "macro_strength": 0.1, "micro_direction": -1, "micro_strength": 0.9, "technical_direction": -1, "technical_strength": 0.8, "technical_bias": -0.75, "attention_bias": -0.65, "uncertainty": 0.70, "retail_emotion": "panic", "belief_reason": "The lawsuit, downgrade, liquidity stress, and falling trend make this stock look clearly bearish."},
    },
    {
        "name": "A_cash_bull: bullish evidence independent of available cash",
        "user": {
            "observation_meta": {"agent_id": "A_cash_bull", "symbol": "SIM", "tick": 3, "stock_name": "SIM"},
            "personality": {"name": "trend"},
            "market_features": {"current_price": 109.0, "kline_count": 7, "attention_score": 0.10, "anchor_score": -0.80, "ma_score": 0.60, "rsi": 100.0, "rsi_score": -1.0, "volatility_score": 0.18, "momentum_score": 1.0, "recent_return_score": 0.57},
            "news": ["new product launch, revenue beat, analyst upgrade"],
            "social_posts": [{"text": "accumulate, upside surprise", "influence": 6}],
        },
        "assistant": {"scope": "stock_specific", "macro_direction": 0, "macro_strength": 0.1, "micro_direction": 1, "micro_strength": 0.85, "technical_direction": 1, "technical_strength": 0.8, "technical_bias": 0.70, "attention_bias": 0.30, "uncertainty": 0.30, "retail_emotion": "fomo", "belief_reason": "The product launch, revenue beat, upgrade, and strong trend make this stock look bullish."},
    },
    {
        "name": "A_cash_bear: bearish evidence independent of no position",
        "user": {
            "observation_meta": {"agent_id": "A_cash_bear", "symbol": "SIM", "tick": 4, "stock_name": "SIM"},
            "personality": {"name": "value"},
            "market_features": {"current_price": 94.0, "kline_count": 7, "attention_score": 0.15, "anchor_score": 0.80, "ma_score": -0.50, "rsi": 0.0, "rsi_score": 1.0, "volatility_score": 0.18, "momentum_score": -1.0, "recent_return_score": -0.62},
            "news": ["fraud rumor, forced selling, demand shock"],
            "social_posts": [{"text": "stay away, heavy selling", "influence": 7}],
        },
        "assistant": {"scope": "stock_specific", "macro_direction": 0, "macro_strength": 0.1, "micro_direction": -1, "micro_strength": 0.85, "technical_direction": -1, "technical_strength": 0.75, "technical_bias": -0.65, "attention_bias": -0.45, "uncertainty": 0.70, "retail_emotion": "caution", "belief_reason": "The fraud rumor, demand shock, and heavy selling make this stock look bearish even if the price is low."},
    },
    {
        "name": "A_neutral: mixed range-bound scenario",
        "user": {
            "observation_meta": {"agent_id": "A_neutral", "symbol": "SIM", "tick": 5, "stock_name": "SIM"},
            "personality": {"name": "trend"},
            "market_features": {"current_price": 100.0, "kline_count": 7, "attention_score": 0.0, "anchor_score": 0.0, "ma_score": 0.0, "rsi": 50.0, "rsi_score": 0.0, "volatility_score": 0.03, "momentum_score": 0.0, "recent_return_score": -0.20},
            "news": ["mixed guidance, no clear catalyst"],
            "social_posts": [{"text": "wait and see", "influence": 3}],
        },
        "assistant": {"scope": "irrelevant", "macro_direction": 0, "macro_strength": 0.1, "micro_direction": 0, "micro_strength": 0.1, "technical_direction": 0, "technical_strength": 0.1, "technical_bias": 0.0, "attention_bias": 0.0, "uncertainty": 0.80, "retail_emotion": "neutral", "belief_reason": "The signal is mixed and range-bound, so my stock-level view stays neutral."},
    },
    {
        "name": "A_gain_pressure: stretched valuation affects belief, not account profit",
        "user": {
            "observation_meta": {"agent_id": "A_gain_pressure", "symbol": "SIM", "tick": 6, "stock_name": "SIM"},
            "personality": {"name": "anxious"},
            "market_features": {"current_price": 118.0, "kline_count": 7, "attention_score": 0.25, "anchor_score": -0.90, "ma_score": 0.75, "rsi": 100.0, "rsi_score": -1.0, "volatility_score": 0.22, "momentum_score": 1.0, "recent_return_score": 0.34},
            "news": ["strong rally but valuation looks stretched"],
            "social_posts": [{"text": "take profit soon", "influence": 6}],
        },
        "assistant": {"scope": "mixed", "macro_direction": 0, "macro_strength": 0.1, "micro_direction": 0, "micro_strength": 0.3, "technical_direction": 1, "technical_strength": 0.55, "technical_bias": 0.25, "attention_bias": -0.15, "uncertainty": 0.55, "retail_emotion": "caution", "belief_reason": "The trend remains positive, but the stretched valuation and overbought signal make my stock-level bullishness weak."},
    },
]


def build_cognition_system_prompt(personality: str, include_few_shots: bool = True) -> str:
    """Return the full system prompt for the account-free Belief layer."""

    personality_key = str(personality or "trend").lower()
    overlay = PERSONALITY_PROMPTS.get(personality_key, PERSONALITY_PROMPTS["trend"])
    trait_reference = PERSONALITY_TRAIT_REFERENCES.get(personality_key, PERSONALITY_TRAIT_REFERENCES["trend"])
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
    """Return a stable account-free JSON user prompt.

    Account fields such as cash, position, avg_cost, and prospect-value features
    are intentionally excluded from this Belief-layer prompt.
    """

    extra = getattr(observation, "extra", {}) or {}
    payload = {
        "observation_meta": {
            "agent_id": str(getattr(observation, "agent_id", "")),
            "symbol": str(getattr(observation, "symbol", "")),
            "tick": int(getattr(observation, "tick", 0) or 0),
            "stock_name": str(extra.get("stock_name", "")) if isinstance(extra, Mapping) else "",
        },
        "personality": {"name": _personality_name(personality)},
        "market_features": _select_feature_keys(features, MARKET_FEATURE_KEYS),
        "news": [str(item) for item in list(getattr(observation, "news", []) or [])],
        "social_posts": _json_ready_social_posts(getattr(observation, "social_posts", []) or []),
    }
    return "Analyze the following account-free stock observation.\n\nInput JSON:\n" + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n\nReturn only the JSON object."


def parse_cognition_response(raw: str) -> Dict[str, Any]:
    """Parse and normalize an LLM belief response."""

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
    """Normalize raw belief data into the standard account-free schema."""

    result = dict(DEFAULT_COGNITION)
    raw = dict(data or {})
    for key in DEFAULT_COGNITION:
        if key in raw:
            result[key] = raw[key]

    result["scope"] = _one_of(result.get("scope"), ALLOWED_SCOPES, "irrelevant")
    result["macro_direction"] = _direction(result.get("macro_direction"))
    result["micro_direction"] = _direction(result.get("micro_direction"))
    result["technical_direction"] = _direction(result.get("technical_direction"))
    result["macro_strength"] = _clip_float(result.get("macro_strength"), 0.0, 1.0, 0.0)
    result["micro_strength"] = _clip_float(result.get("micro_strength"), 0.0, 1.0, 0.0)
    result["technical_strength"] = _clip_float(result.get("technical_strength"), 0.0, 1.0, 0.0)
    result["technical_bias"] = _clip_float(result.get("technical_bias"), -1.0, 1.0, 0.0)
    result["attention_bias"] = _clip_float(result.get("attention_bias"), -1.0, 1.0, 0.0)
    result["uncertainty"] = _clip_float(result.get("uncertainty"), 0.0, 1.0, 0.5)
    result["retail_emotion"] = _one_of(result.get("retail_emotion"), ALLOWED_EMOTIONS, "neutral")
    result["belief_reason"] = str(result.get("belief_reason") or DEFAULT_COGNITION["belief_reason"]).strip()
    if not result["belief_reason"]:
        result["belief_reason"] = DEFAULT_COGNITION["belief_reason"]
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
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if numeric != numeric:
        return default
    return numeric


def _one_of(value: Any, allowed: Any, default: Any) -> Any:
    return value if value in allowed else default
