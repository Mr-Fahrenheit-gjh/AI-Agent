"""Prompt templates for LLM desire-adjustment layer.

The Desire LLM layer is allowed to see account state because it runs after the
account-free Belief layer. It must not output final actions or free-form numeric
desire scores. It only returns bounded ordinal adjustments that are mapped to
small numeric deltas by ``desire_utility.py``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Mapping, Optional


ADJUSTMENTS = {
    "strongly_down",
    "moderately_down",
    "slightly_down",
    "unchanged",
    "slightly_up",
    "moderately_up",
    "strongly_up",
}
PRESSURES = {"none", "low", "medium", "high", "extreme"}
CONFIDENCES = {"low", "medium", "high"}


DEFAULT_DESIRE_ADJUSTMENT: Dict[str, str] = {
    "buy_adjustment": "unchanged",
    "sell_adjustment": "unchanged",
    "hold_adjustment": "unchanged",
    "profit_taking_pressure": "none",
    "loss_hold_pressure": "none",
    "confidence": "medium",
    "desire_reason": "I do not see a strong personality-based reason to alter the rule desires.",
}


DESIRE_OUTPUT_SCHEMA_TEXT = """Output schema:
{
  "buy_adjustment": "strongly_down" | "moderately_down" | "slightly_down" | "unchanged" | "slightly_up" | "moderately_up" | "strongly_up",
  "sell_adjustment": "strongly_down" | "moderately_down" | "slightly_down" | "unchanged" | "slightly_up" | "moderately_up" | "strongly_up",
  "hold_adjustment": "strongly_down" | "moderately_down" | "slightly_down" | "unchanged" | "slightly_up" | "moderately_up" | "strongly_up",
  "profit_taking_pressure": "none" | "low" | "medium" | "high" | "extreme",
  "loss_hold_pressure": "none" | "low" | "medium" | "high" | "extreme",
  "confidence": "low" | "medium" | "high",
  "desire_reason": "one concise first-person reason for the desire adjustment"
}"""


BASE_DESIRE_SYSTEM_PROMPT = """You are the Desire-adjustment layer inside a retail-investor agent.

You run after the Belief layer and after deterministic rule desires have been computed.
Your job is NOT to choose the final trading action.
Your job is NOT to output absolute numeric desire scores.
Your job is to make small ordinal personality-based adjustments to the given rule desires.

Important rules:
- The rule desires are the scale anchor. Do not ignore them.
- Output only ordinal adjustments such as slightly_up or moderately_down.
- Account state is allowed here: cash, position, avg cost, PnL, and prospect features describe psychological pressure.
- Belief describes the stock-level view; account state describes why final desire may diverge from belief.
- Personality may shape the adjustment, but it should not overturn clear evidence with no reason.
- If evidence is ambiguous, prefer unchanged or slight adjustments.
- Think internally, but do not reveal step-by-step reasoning.
- Return ONLY valid JSON. No markdown. No extra text."""


PERSONALITY_DESIRE_PROMPTS: Dict[str, str] = {
    "trend": """Personality overlay:
You are trend-following. If belief and trend remain strong, you are less eager to take profits too early. If trend weakens, you can reduce quickly.""",
    "value": """Personality overlay:
You are value-oriented. You dislike buying stretched prices and prefer patience. You may take profits when price looks expensive relative to anchors.""",
    "aggressive": """Personality overlay:
You are aggressive and confident. You tolerate volatility and may increase buy desire when belief is strong, but should still respect large losses and uncertainty.""",
    "anxious": """Personality overlay:
You are anxious and loss-averse. Gains create stronger profit-taking pressure; losses create stronger reluctance to realize losses unless belief is clearly bearish.""",
}


BELIEF_KEYS = [
    "belief_score",
    "sentiment_class",
    "uncertainty",
    "belief_reason",
    "macro_direction",
    "macro_strength",
    "micro_direction",
    "micro_strength",
    "technical_bias",
    "attention_bias",
]

RULE_DESIRE_KEYS = [
    "buy_desire",
    "sell_desire",
    "hold_desire",
    "profit_taking_pressure",
    "loss_realization_resistance",
    "bearish_sell_pressure",
    "overheat_score",
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

MARKET_CONTEXT_KEYS = [
    "current_price",
    "anchor_score",
    "anchor_position",
    "rsi",
    "rsi_score",
    "volatility_score",
    "momentum_score",
    "recent_return_score",
]


FEW_SHOT_EXAMPLES = [
    {
        "name": "bullish stock but large gain creates profit-taking pressure",
        "user": {
            "personality": {"name": "anxious"},
            "belief": {"belief_score": 0.62, "sentiment_class": 1, "uncertainty": 0.30, "belief_reason": "The stock still has bullish evidence but looks stretched."},
            "rule_desires": {"buy_desire": 0.18, "sell_desire": 0.36, "hold_desire": 0.34, "profit_taking_pressure": 0.48, "loss_realization_resistance": 0.0},
            "account_features": {"cash_ratio": 0.66, "has_position": 1, "position_weight": 0.34, "pnl_return": 0.30, "prospect_value_clipped": 0.34, "gain_flag": 1, "loss_flag": 0},
            "market_context": {"anchor_score": -0.90, "rsi_score": -1.0, "volatility_score": 0.22},
        },
        "assistant": {"buy_adjustment": "slightly_down", "sell_adjustment": "moderately_up", "hold_adjustment": "unchanged", "profit_taking_pressure": "high", "loss_hold_pressure": "none", "confidence": "medium", "desire_reason": "I still see positive evidence, but the large gain and stretched price make locking in profit more tempting."},
    },
    {
        "name": "loss with weak bearish belief creates holding resistance",
        "user": {
            "personality": {"name": "value"},
            "belief": {"belief_score": -0.18, "sentiment_class": -1, "uncertainty": 0.55, "belief_reason": "The stock is mildly bearish but evidence is mixed."},
            "rule_desires": {"buy_desire": 0.02, "sell_desire": 0.10, "hold_desire": 0.58, "profit_taking_pressure": 0.0, "loss_realization_resistance": 0.42},
            "account_features": {"cash_ratio": 0.40, "has_position": 1, "position_weight": 0.22, "pnl_return": -0.09, "prospect_value_clipped": -0.27, "gain_flag": 0, "loss_flag": 1},
            "market_context": {"anchor_score": 0.45, "rsi_score": 0.30, "volatility_score": 0.35},
        },
        "assistant": {"buy_adjustment": "unchanged", "sell_adjustment": "slightly_down", "hold_adjustment": "slightly_up", "profit_taking_pressure": "none", "loss_hold_pressure": "medium", "confidence": "medium", "desire_reason": "The bearish evidence is not decisive, and as a value investor I am reluctant to realize a loss near a lower anchor."},
    },
    {
        "name": "clear bearish belief can overcome loss holding",
        "user": {
            "personality": {"name": "anxious"},
            "belief": {"belief_score": -0.82, "sentiment_class": -1, "uncertainty": 0.35, "belief_reason": "The stock-specific news and trend are clearly bearish."},
            "rule_desires": {"buy_desire": 0.0, "sell_desire": 0.62, "hold_desire": 0.28, "profit_taking_pressure": 0.0, "loss_realization_resistance": 0.08},
            "account_features": {"cash_ratio": 0.75, "has_position": 1, "position_weight": 0.18, "pnl_return": -0.15, "prospect_value_clipped": -0.42, "gain_flag": 0, "loss_flag": 1},
            "market_context": {"anchor_score": 0.82, "rsi_score": 1.0, "volatility_score": 0.60},
        },
        "assistant": {"buy_adjustment": "strongly_down", "sell_adjustment": "slightly_up", "hold_adjustment": "slightly_down", "profit_taking_pressure": "none", "loss_hold_pressure": "low", "confidence": "medium", "desire_reason": "Even though realizing the loss hurts, the stock-level evidence is too negative to justify simply holding."},
    },
    {
        "name": "cash and strong belief support buy but adjustment stays bounded",
        "user": {
            "personality": {"name": "aggressive"},
            "belief": {"belief_score": 0.78, "sentiment_class": 1, "uncertainty": 0.25, "belief_reason": "The stock has strong bullish news and momentum."},
            "rule_desires": {"buy_desire": 0.42, "sell_desire": 0.0, "hold_desire": 0.22, "profit_taking_pressure": 0.0, "loss_realization_resistance": 0.0},
            "account_features": {"cash_ratio": 1.0, "has_position": 0, "position_weight": 0.0, "pnl_return": 0.0, "prospect_value_clipped": 0.0, "gain_flag": 0, "loss_flag": 0},
            "market_context": {"anchor_score": -0.80, "rsi_score": -1.0, "volatility_score": 0.18},
        },
        "assistant": {"buy_adjustment": "moderately_up", "sell_adjustment": "unchanged", "hold_adjustment": "slightly_down", "profit_taking_pressure": "none", "loss_hold_pressure": "none", "confidence": "medium", "desire_reason": "With cash available and strong bullish belief, my aggressive side makes buying more appealing, though the adjustment should stay bounded."},
    },
]


def build_desire_system_prompt(personality: str, include_few_shots: bool = True) -> str:
    """Return the system prompt for ordinal LLM desire adjustment."""

    personality_key = str(personality or "trend").lower()
    overlay = PERSONALITY_DESIRE_PROMPTS.get(personality_key, PERSONALITY_DESIRE_PROMPTS["trend"])
    parts = [BASE_DESIRE_SYSTEM_PROMPT, overlay, DESIRE_OUTPUT_SCHEMA_TEXT]
    if include_few_shots:
        parts.append(_format_few_shots(FEW_SHOT_EXAMPLES))
    return "\n\n".join(parts)


def build_desire_user_prompt(
    *,
    personality: Any,
    belief: Mapping[str, Any],
    rule_desires: Mapping[str, Any],
    account_features: Mapping[str, Any],
    market_context: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return a stable JSON user prompt for desire adjustment."""

    payload = {
        "personality": {"name": _personality_name(personality)},
        "belief": _select_keys(belief, BELIEF_KEYS),
        "rule_desires": _select_keys(rule_desires, RULE_DESIRE_KEYS),
        "account_features": _select_keys(account_features, ACCOUNT_FEATURE_KEYS),
        "market_context": _select_keys(market_context or {}, MARKET_CONTEXT_KEYS),
    }
    return "Adjust the rule desires using personality and account psychology.\n\nInput JSON:\n" + json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n\nReturn only the JSON object."


def parse_desire_adjustment(raw: str) -> Dict[str, str]:
    """Parse and normalize an LLM desire-adjustment response."""

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
    return normalize_desire_adjustment(parsed)


def normalize_desire_adjustment(data: Mapping[str, Any]) -> Dict[str, str]:
    """Normalize raw adjustment fields into bounded labels."""

    raw = dict(data or {})
    result = dict(DEFAULT_DESIRE_ADJUSTMENT)
    result["buy_adjustment"] = _one_of(raw.get("buy_adjustment"), ADJUSTMENTS, "unchanged")
    result["sell_adjustment"] = _one_of(raw.get("sell_adjustment"), ADJUSTMENTS, "unchanged")
    result["hold_adjustment"] = _one_of(raw.get("hold_adjustment"), ADJUSTMENTS, "unchanged")
    result["profit_taking_pressure"] = _one_of(raw.get("profit_taking_pressure"), PRESSURES, "none")
    result["loss_hold_pressure"] = _one_of(raw.get("loss_hold_pressure"), PRESSURES, "none")
    result["confidence"] = _one_of(raw.get("confidence"), CONFIDENCES, "medium")
    reason = str(raw.get("desire_reason") or result["desire_reason"]).strip()
    result["desire_reason"] = reason or DEFAULT_DESIRE_ADJUSTMENT["desire_reason"]
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


def _one_of(value: Any, allowed: Any, default: str) -> str:
    text = str(value or default).strip()
    return text if text in allowed else default
