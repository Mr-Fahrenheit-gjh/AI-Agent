"""Final thought helpers for the BDI trading chain.

The final thought layer runs after Belief, Desire, and Action. It explains both
the stock-level belief and the final trade action. If those diverge, the thought
should explicitly bridge the gap with account psychology such as profit-taking,
loss realization resistance, uncertainty, or execution constraints.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Mapping, Optional


MAX_THOUGHT_CHARS = 280

DEFAULT_THOUGHT = (
    "I do not see enough reliable evidence for a strong move, so I keep the trade restrained."
)


THOUGHT_OUTPUT_SCHEMA_TEXT = """Output schema:
{
  "thought": "one concise first-person retail-investor thought"
}"""


BASE_THOUGHT_SYSTEM_PROMPT = """You are the Final Thought layer inside a retail-investor agent.

You run after Belief, Desire, and Action have already been produced.
Your job is NOT to change the action, quantity, price, belief score, or desires.
Your job is to explain the final decision as a concise first-person retail-investor thought.

Important rules:
- Explain both the stock-level belief and the final action.
- If belief and action diverge, explicitly explain the psychological bridge.
- Common bridges include profit-taking, loss realization resistance, uncertainty, cash limits, no position, heavy position, overbought price, or weak conviction.
- Do not invent facts that are not in the input.
- Do not reveal step-by-step reasoning.
- Keep the thought concise, natural, and trader-like.
- Return ONLY valid JSON. No markdown. No extra text."""


PERSONALITY_THOUGHT_PROMPTS: Dict[str, str] = {
    "trend": """Personality style:
You sound like a trend follower. Mention momentum, breakout, fading trend, or letting winners run when relevant.""",
    "value": """Personality style:
You sound like a value-oriented investor. Mention price anchors, expensive/cheap feeling, patience, or avoiding stretched prices when relevant.""",
    "aggressive": """Personality style:
You sound like an aggressive investor. Mention willingness to participate, confidence in strong evidence, or controlled risk-taking when relevant.""",
    "anxious": """Personality style:
You sound like an anxious investor. Mention caution, locking in gains, avoiding pain, uncertainty, or reluctance to realize losses when relevant.""",
}


PERSONALITY_TRAIT_REFERENCES: Dict[str, str] = {
    "trend": "Qualitative reference: medium-high risk appetite, medium loss aversion, medium-high turnover.",
    "value": "Qualitative reference: low-medium risk appetite, medium loss aversion, low turnover, anchor-sensitive.",
    "aggressive": "Qualitative reference: high risk appetite, high confidence, high turnover, bounded caution.",
    "anxious": "Qualitative reference: low risk appetite, high loss aversion, high profit-taking pressure.",
}


BELIEF_KEYS = [
    "belief_score",
    "sentiment_class",
    "belief_reason",
    "uncertainty",
    "macro_direction",
    "micro_direction",
    "technical_bias",
    "attention_bias",
]

DESIRE_KEYS = [
    "buy_desire",
    "sell_desire",
    "hold_desire",
    "dominant_desire",
    "profit_taking_pressure",
    "loss_realization_resistance",
    "bearish_sell_pressure",
    "desire_reason",
]

ACTION_KEYS = ["action", "quantity", "limit_price", "official_belief_score", "official_sentiment_class"]

ACCOUNT_KEYS = [
    "cash_ratio",
    "has_position",
    "position_weight",
    "position_qty",
    "avg_cost",
    "pnl_return",
    "prospect_value_clipped",
    "gain_flag",
    "loss_flag",
]

MARKET_KEYS = [
    "current_price",
    "anchor_score",
    "rsi",
    "rsi_score",
    "volatility_score",
    "momentum_score",
    "recent_return_score",
]


FEW_SHOT_EXAMPLES = [
    {
        "name": "bullish belief but sell to lock profit",
        "user": {
            "personality": {"name": "anxious"},
            "belief": {"belief_score": 0.62, "sentiment_class": 1, "belief_reason": "The stock still has bullish evidence but looks stretched.", "uncertainty": 0.30},
            "desire": {"buy_desire": 0.08, "sell_desire": 0.41, "hold_desire": 0.37, "dominant_desire": "sell", "profit_taking_pressure": 0.48, "loss_realization_resistance": 0.0, "desire_reason": "The large gain and stretched price make locking in profit more tempting."},
            "action": {"action": "sell", "quantity": 100, "limit_price": 117.3},
            "account_features": {"has_position": 1, "position_weight": 0.34, "pnl_return": 0.30, "gain_flag": 1, "loss_flag": 0},
            "market_context": {"anchor_score": -0.90, "rsi_score": -1.0, "volatility_score": 0.22},
        },
        "assistant": {"thought": "I still read the stock as bullish, but my gain is large and the price feels stretched, so I choose to lock in part of the profit."},
    },
    {
        "name": "bearish belief but hold due to loss realization resistance",
        "user": {
            "personality": {"name": "value"},
            "belief": {"belief_score": -0.18, "sentiment_class": -1, "belief_reason": "The stock is mildly bearish but evidence is mixed.", "uncertainty": 0.55},
            "desire": {"buy_desire": 0.02, "sell_desire": 0.10, "hold_desire": 0.58, "dominant_desire": "hold", "profit_taking_pressure": 0.0, "loss_realization_resistance": 0.42, "desire_reason": "The bearish evidence is not decisive, and I am reluctant to realize the loss."},
            "action": {"action": "hold", "quantity": 0, "limit_price": 92.0},
            "account_features": {"has_position": 1, "position_weight": 0.22, "pnl_return": -0.09, "gain_flag": 0, "loss_flag": 1},
            "market_context": {"anchor_score": 0.45, "rsi_score": 0.30, "volatility_score": 0.35},
        },
        "assistant": {"thought": "The stock looks a little weak, but the signal is mixed and I dislike realizing this loss near a lower anchor, so I hold for now."},
    },
    {
        "name": "strong bullish belief with cash leads to buy",
        "user": {
            "personality": {"name": "aggressive"},
            "belief": {"belief_score": 0.78, "sentiment_class": 1, "belief_reason": "The stock has strong bullish news and momentum.", "uncertainty": 0.25},
            "desire": {"buy_desire": 0.54, "sell_desire": 0.0, "hold_desire": 0.16, "dominant_desire": "buy", "profit_taking_pressure": 0.0, "loss_realization_resistance": 0.0, "desire_reason": "Cash is available and the bullish evidence supports participation."},
            "action": {"action": "buy", "quantity": 100, "limit_price": 109.6},
            "account_features": {"cash_ratio": 1.0, "has_position": 0, "position_weight": 0.0, "pnl_return": 0.0, "gain_flag": 0, "loss_flag": 0},
            "market_context": {"anchor_score": -0.80, "rsi_score": -1.0, "volatility_score": 0.18},
        },
        "assistant": {"thought": "The bullish news and momentum are strong, and I still have cash, so I am willing to buy despite the price already running hot."},
    },
    {
        "name": "clear bearish belief with loss still sells",
        "user": {
            "personality": {"name": "anxious"},
            "belief": {"belief_score": -0.82, "sentiment_class": -1, "belief_reason": "The stock-specific news and trend are clearly bearish.", "uncertainty": 0.35},
            "desire": {"buy_desire": 0.0, "sell_desire": 0.68, "hold_desire": 0.24, "dominant_desire": "sell", "profit_taking_pressure": 0.0, "loss_realization_resistance": 0.08, "desire_reason": "The evidence is too negative to justify simply holding the loss."},
            "action": {"action": "sell", "quantity": 100, "limit_price": 91.4},
            "account_features": {"has_position": 1, "position_weight": 0.18, "pnl_return": -0.15, "gain_flag": 0, "loss_flag": 1},
            "market_context": {"anchor_score": 0.82, "rsi_score": 1.0, "volatility_score": 0.60},
        },
        "assistant": {"thought": "Realizing the loss hurts, but the stock-specific evidence and trend are too bearish, so I choose to cut some risk."},
    },
    {
        "name": "neutral belief and uncertainty leads to hold",
        "user": {
            "personality": {"name": "trend"},
            "belief": {"belief_score": 0.03, "sentiment_class": 0, "belief_reason": "The stock is range-bound with no clear catalyst.", "uncertainty": 0.80},
            "desire": {"buy_desire": 0.03, "sell_desire": 0.02, "hold_desire": 0.70, "dominant_desire": "hold", "profit_taking_pressure": 0.0, "loss_realization_resistance": 0.0, "desire_reason": "Uncertainty is high and neither side has a strong edge."},
            "action": {"action": "hold", "quantity": 0, "limit_price": 100.0},
            "account_features": {"has_position": 1, "position_weight": 0.12, "pnl_return": 0.0, "gain_flag": 0, "loss_flag": 0},
            "market_context": {"anchor_score": 0.0, "rsi_score": 0.0, "volatility_score": 0.03},
        },
        "assistant": {"thought": "The stock is stuck in a range and the signal is unclear, so I would rather wait than force a trade."},
    },
]


def build_rule_thought(
    *,
    personality: Any,
    belief: Mapping[str, Any],
    desire: Mapping[str, Any],
    action: Mapping[str, Any],
    account_features: Mapping[str, Any],
    market_context: Optional[Mapping[str, Any]] = None,
) -> str:
    """Build a deterministic fallback thought."""

    action_name = str(action.get("action", "hold"))
    belief_score = _to_float(belief.get("belief_score"), 0.0)
    pnl_return = _to_float(account_features.get("pnl_return"), 0.0)
    uncertainty = _to_float(belief.get("uncertainty"), 0.5)
    desire_reason = str(desire.get("desire_reason") or "").strip()
    belief_reason = str(belief.get("belief_reason") or "").strip()
    market_context = dict(market_context or {})

    bridge = ""
    if action_name == "sell" and belief_score > 0.1 and pnl_return > 0:
        bridge = "but my gain is meaningful, so I want to lock in part of the profit"
    elif action_name == "hold" and belief_score < -0.1 and pnl_return < 0:
        bridge = "but the signal is not decisive enough for me to realize the loss"
    elif action_name == "buy" and belief_score > 0.1:
        bridge = "and the desire to participate is strong enough for me to buy"
    elif action_name == "sell" and belief_score < -0.1:
        bridge = "and the bearish evidence makes me reduce risk"
    elif action_name == "hold" and uncertainty >= 0.6:
        bridge = "so I prefer to wait rather than force a trade"
    elif action_name == "hold":
        bridge = "so I choose to stay patient"
    else:
        bridge = f"so I choose to {action_name}"

    overheat = ""
    if _to_float(market_context.get("rsi_score"), 0.0) < -0.7 or _to_float(market_context.get("anchor_score"), 0.0) < -0.7:
        overheat = " The price also feels stretched."

    text = belief_reason or DEFAULT_THOUGHT
    if desire_reason:
        text = f"{text} {desire_reason}"
    text = f"{text} {bridge[0].upper() + bridge[1:]}.{overheat}"
    return _clean_thought(text)


def build_thought_system_prompt(personality: str, include_few_shots: bool = True) -> str:
    """Return the system prompt for final thought generation."""

    personality_key = str(personality or "trend").lower()
    style = PERSONALITY_THOUGHT_PROMPTS.get(personality_key, PERSONALITY_THOUGHT_PROMPTS["trend"])
    trait_reference = PERSONALITY_TRAIT_REFERENCES.get(personality_key, PERSONALITY_TRAIT_REFERENCES["trend"])
    parts = [BASE_THOUGHT_SYSTEM_PROMPT, style, trait_reference, THOUGHT_OUTPUT_SCHEMA_TEXT]
    if include_few_shots:
        parts.append(_format_few_shots(FEW_SHOT_EXAMPLES))
    return "\n\n".join(parts)


def build_thought_user_prompt(
    *,
    personality: Any,
    belief: Mapping[str, Any],
    desire: Mapping[str, Any],
    action: Mapping[str, Any],
    account_features: Mapping[str, Any],
    market_context: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return a stable JSON user prompt for final thought generation."""

    payload = {
        "personality": {"name": _personality_name(personality)},
        "belief": _select_keys(belief, BELIEF_KEYS),
        "desire": _select_keys(desire, DESIRE_KEYS),
        "action": _select_keys(action, ACTION_KEYS),
        "account_features": _select_keys(account_features, ACCOUNT_KEYS),
        "market_context": _select_keys(market_context or {}, MARKET_KEYS),
    }
    return "Write the final trading thought for this completed decision.\n\nInput JSON:\n" + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n\nReturn only the JSON object."


def parse_thought_response(raw: str) -> Dict[str, str]:
    """Parse an LLM final-thought response."""

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
    thought = _clean_thought(str(parsed.get("thought") or DEFAULT_THOUGHT))
    return {"thought": thought}


def _format_few_shots(examples: Any) -> str:
    lines = ["Few-shot examples:"]
    for index, item in enumerate(examples, start=1):
        lines.append(f"Example {index}: {item['name']}")
        lines.append("User JSON:")
        lines.append(json.dumps(item["user"], ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        lines.append("Assistant JSON:")
        lines.append(json.dumps(item["assistant"], ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return "\n".join(lines)


def _select_keys(data: Mapping[str, Any], keys: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if isinstance(value, float):
            result[str(key)] = round(value, 6)
        else:
            result[str(key)] = value
    return result


def _personality_name(personality: Any) -> str:
    if isinstance(personality, Mapping):
        value = personality.get("name") or personality.get("personality")
    else:
        value = personality
    text = str(value or "unknown").strip().lower()
    return text or "unknown"


def _clean_thought(text: str) -> str:
    cleaned = " ".join(str(text or DEFAULT_THOUGHT).split())
    if len(cleaned) > MAX_THOUGHT_CHARS:
        cleaned = cleaned[: MAX_THOUGHT_CHARS - 1].rstrip(" ,;") + "."
    return cleaned or DEFAULT_THOUGHT


def _to_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if numeric != numeric:
        return float(default)
    return numeric
