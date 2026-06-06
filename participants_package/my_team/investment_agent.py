# ============================================================
# 比赛任务一：投资 Agent（BDI 信念-欲望-意图）建模
# ============================================================
# 🔒 KEEP（不可修改）:
#   - ingest_market() / ingest_news() / ingest_social() 签名
#   - update_beliefs() / decide() / apply_fill() 签名
#   - Belief / Position / Decision 数据类结构
#
# 🔧 MODIFIABLE（可修改）:
#   - PERSONALITY_PRESETS（性格预设参数）
#   - POSITIVE_WORDS / NEGATIVE_WORDS（情感词典）
#   - _build_thought()（内心独白生成逻辑）
#   - _text_sentiment() / _social_sentiment()（情感计算方法）
#   - decide() 中的决策阈值（buy_threshold / sell_threshold）
#   - Belief.score 的权重配比
#
# 🚀 ADVANCED（高级玩法）:
#   - 用 LLM 替换规则式决策: 设置 llm_client 参数，取消注释 decide() 中的 LLM 分支
#   - 接入多模态信息: 扩展 MarketObservation.extra 字段
#   - 实现记忆机制: 利用 self.memory 存储历史决策
#
# 💡 TIP（提示）:
#   - 使用 llm_helper.py 快速接入 DeepSeek/OpenAI:
#       from llm_helper import create_llm_client
#       agent = InvestmentAgent("id", llm_client=create_llm_client("config.yaml"))
#   - 性格参数对行为影响极大，建议先在小规模模拟中调优
#   - 处置效应（DE）指标目标区间: [0.05, 0.15]
# ============================================================

"""BDI investment agent for the Silicon Finance Market challenge.

The class is intentionally usable without network access.  A callable LLM
adapter can be injected, but the deterministic behavioural model remains the
source of truth for reproducible scoring and tests.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from belief_scoring import build_belief_score
from desire_utility import apply_desire_adjustment, build_rule_desires
from feature_engineering import build_account_features, build_market_features
from final_thought import (
    build_rule_thought,
    build_thought_system_prompt,
    build_thought_user_prompt,
    parse_thought_response,
)
from llm_cognition_prompts import (
    build_cognition_system_prompt,
    build_cognition_user_prompt,
    parse_cognition_response,
)
from llm_desire_prompts import (
    build_desire_system_prompt,
    build_desire_user_prompt,
    parse_desire_adjustment,
)


LLMClient = Callable[[str, str], str]

# ============================================================
# 🔧 MODIFIABLE: 情感词典
# 扩展这些词表以改进情感分析准确度。
# 支持中英文混合。
# ============================================================

POSITIVE_WORDS = {
    "beat",
    "growth",
    "upgrade",
    "bull",
    "surge",
    "profit",
    "strong",
    "record",
    "buy",
    "breakout",
    "利好",
    "增长",
    "上调",
    "突破",
    "盈利",
    "买入",
}

NEGATIVE_WORDS = {
    "miss",
    "fraud",
    "downgrade",
    "bear",
    "crash",
    "loss",
    "weak",
    "sell",
    "risk",
    "panic",
    "利空",
    "亏损",
    "下调",
    "暴跌",
    "卖出",
    "风险",
    "恐慌",
}


# ============================================================
# 🔧 MODIFIABLE: 性格预设参数
# 每个性格由 6 个维度定义（均为 0.0 ~ 1.0 或更高）:
#   risk_appetite   — 风险偏好（越高越敢冒险）
#   loss_aversion   — 损失厌恶（越高越不愿止损，>1.0 为过度厌恶）
#   herding         — 羊群效应（越高越跟随社交信号）
#   overconfidence  — 过度自信（越高越相信自己的判断）
#   disposition     — 处置效应强度（越高越倾向于过早止盈/过晚止损）
#   turnover        — 交易频率倾向（越高换手越频繁）
# 💡 TIP: 调整这些参数是塑造 Agent 行为的最直接方式。
# ============================================================
PERSONALITY_PRESETS: Dict[str, Dict[str, float]] = {
    "aggressive": {
        "risk_appetite": 0.82,
        "loss_aversion": 1.65,
        "herding": 0.66,
        "overconfidence": 0.68,
        "disposition": 0.18,
        "turnover": 0.72,
    },
    "value": {
        "risk_appetite": 0.38,
        "loss_aversion": 1.20,
        "herding": 0.20,
        "overconfidence": 0.28,
        "disposition": 0.08,
        "turnover": 0.23,
    },
    "trend": {
        "risk_appetite": 0.60,
        "loss_aversion": 1.35,
        "herding": 0.55,
        "overconfidence": 0.48,
        "disposition": 0.12,
        "turnover": 0.48,
    },
    "anxious": {
        "risk_appetite": 0.30,
        "loss_aversion": 2.05,
        "herding": 0.72,
        "overconfidence": 0.18,
        "disposition": 0.28,
        "turnover": 0.58,
    },
}


@dataclass
class Belief:
    """Internal belief state over one symbol."""

    fair_value: float = 0.0
    momentum: float = 0.0
    sentiment: float = 0.0
    volatility: float = 0.0
    confidence: float = 0.35
    social_pressure: float = 0.0

    @property
    def score(self) -> float:
        # 🔧 MODIFIABLE: 信念综合得分的权重配比
        # 调整 sentiment/momentum/social_pressure/volatility 的权重
        # 以改变 Agent 对各类信息的敏感度
        return (
            0.36 * self.sentiment
            + 0.28 * self.momentum
            + 0.22 * self.social_pressure
            - 0.14 * self.volatility
        ) * (0.55 + self.confidence)


@dataclass
class Position:
    quantity: int
    avg_cost: float


@dataclass
class Decision:
    agent_id: str
    symbol: str
    action: str
    quantity: int
    limit_price: float
    thought: str
    belief_score: float
    sentiment_class: int

    def as_order_dict(self) -> Dict[str, Any]:
        direction = "hold" if self.action == "hold" else self.action
        return {
            "user_id": self.agent_id,
            "stock_code": self.symbol,
            "direction": direction,
            "amount": self.quantity,
            "target_price": self.limit_price,
            "thought": self.thought,
            "belief_score": self.belief_score,
        }


@dataclass
class InvestmentAgent:
    """A personality-aware BDI investor.

    Public workflow:
    1. ingest_market(...)
    2. ingest_news(...)
    3. ingest_social(...)
    4. update_beliefs(...)
    5. decide(...)
    """

    agent_id: str
    personality: str = "value"
    cash: float = 100_000.0
    positions: Dict[str, Position] = field(default_factory=dict)
    llm_client: Optional[LLMClient] = None
    seed: Optional[int] = None
    llm_desire_enabled: bool = False
    llm_thought_enabled: bool = False
    traits: Dict[str, float] = field(default_factory=dict)
    beliefs: Dict[str, Belief] = field(default_factory=dict)

    def __post_init__(self) -> None:
        preset = PERSONALITY_PRESETS.get(self.personality, PERSONALITY_PRESETS["value"])
        merged = dict(preset)
        merged.update(self.traits)
        self.traits = merged
        base_seed = self.seed
        if base_seed is None:
            digest = hashlib.sha256(self.agent_id.encode("utf-8")).hexdigest()
            base_seed = int(digest[:8], 16)
        self._rng = random.Random(base_seed)
        self._market: Dict[str, List[Dict[str, float]]] = {}
        self._news: Dict[str, List[str]] = {}
        self._social: Dict[str, List[Mapping[str, Any]]] = {}
        self.memory: List[Dict[str, Any]] = []

    def ingest_market(self, symbol: str, klines: Iterable[Mapping[str, Any]]) -> None:
        rows = []
        for row in klines:
            rows.append(
                {
                    "open": float(row.get("open", row.get("open_price", 0.0))),
                    "high": float(row.get("high", 0.0)),
                    "low": float(row.get("low", 0.0)),
                    "close": float(row.get("close", row.get("close_price", 0.0))),
                    "volume": float(row.get("volume", row.get("vol", 0.0))),
                }
            )
        if rows:
            self._market[symbol] = rows[-90:]

    def ingest_news(self, symbol: str, summaries: Iterable[str]) -> None:
        self._news.setdefault(symbol, []).extend(str(item) for item in summaries)
        self._news[symbol] = self._news[symbol][-50:]

    def ingest_social(self, symbol: str, posts: Iterable[Mapping[str, Any] | str]) -> None:
        normalized = []
        for post in posts:
            if isinstance(post, str):
                normalized.append({"text": post, "influence": 1.0})
            else:
                normalized.append(post)
        self._social.setdefault(symbol, []).extend(normalized)
        self._social[symbol] = self._social[symbol][-80:]

    def update_beliefs(self, symbol: str) -> Belief:
        market = self._market.get(symbol, [])
        closes = [row["close"] for row in market if row["close"] > 0]
        if not closes:
            belief = self.beliefs.get(symbol, Belief())
            self.beliefs[symbol] = belief
            return belief

        last = closes[-1]
        short = _mean(closes[-5:])
        long = _mean(closes[-20:]) if len(closes) >= 20 else _mean(closes)
        momentum = _clip((short / long - 1.0) * 10.0, -1.0, 1.0) if long else 0.0
        volatility = _clip(_std(_returns(closes[-20:])) * 20.0, 0.0, 1.0)
        sentiment = self._text_sentiment(self._news.get(symbol, []))
        social_pressure = self._social_sentiment(self._social.get(symbol, []))

        prior = self.beliefs.get(symbol, Belief(fair_value=last))
        value_anchor = prior.fair_value or last
        fair_value = 0.82 * value_anchor + 0.18 * last * (1.0 + 0.07 * sentiment)
        confidence = _clip(
            0.30
            + 0.25 * min(len(self._news.get(symbol, [])) / 10.0, 1.0)
            + 0.25 * min(len(market) / 30.0, 1.0)
            + 0.20 * self.traits["overconfidence"],
            0.05,
            0.95,
        )

        belief = Belief(
            fair_value=fair_value,
            momentum=momentum,
            sentiment=sentiment,
            volatility=volatility,
            confidence=confidence,
            social_pressure=social_pressure,
        )
        self.beliefs[symbol] = belief
        return belief

    # region Legacy rule-based decision engine
    def _legacy_rule_decide(self, symbol: str) -> Decision:
        belief = self.update_beliefs(symbol)
        price = self.current_price(symbol)
        if price <= 0:
            return Decision(self.agent_id, symbol, "hold", 0, 0.0, "No tradable price.", 0, 0)

        position = self.positions.get(symbol)
        unrealized = 0.0
        if position and position.avg_cost > 0:
            unrealized = price / position.avg_cost - 1.0

        value_gap = _clip((belief.fair_value / price - 1.0) * 8.0, -1.0, 1.0)
        market_return = self._recent_return(symbol)
        evidence_score = _clip(
            0.45 * belief.sentiment
            + 0.35 * belief.social_pressure
            + 0.35 * belief.momentum
            + 0.15 * value_gap
            + 0.20 * market_return,
            -1.0,
            1.0,
        )

        action = "hold"
        quantity = 0
        limit_price = price
        score = _clip(evidence_score, -1.0, 1.0)

        has_position = bool(position and position.quantity > 0)
        strong_profit = has_position and unrealized >= 0.18
        modest_profit = has_position and unrealized >= 0.05
        meaningful_loss = has_position and unrealized <= -0.05

        # BDI decision table: evidence creates belief, while P/L shapes retail intent.
        if not has_position:
            if evidence_score >= 0.20:
                action = "buy"
                score = max(0.35, evidence_score)
            else:
                action = "hold"
                score = 0.0
        elif strong_profit and evidence_score <= 0.55:
            action = "sell"
            score = min(-0.35, evidence_score - 0.45)
        elif modest_profit and evidence_score < -0.05:
            action = "sell"
            score = min(-0.30, evidence_score)
        elif meaningful_loss:
            if evidence_score <= -0.65:
                action = "sell"
                score = min(-0.65, evidence_score)
            else:
                action = "hold"
                score = 0.0
        elif evidence_score >= 0.35:
            action = "buy"
            score = max(0.35, evidence_score)
        elif evidence_score <= -0.35:
            action = "sell"
            score = min(-0.35, evidence_score)
        else:
            action = "hold"
            score = 0.0

        if action == "buy":
            equity = max(self.cash + (position.quantity * price if position else 0.0), 1.0)
            budget = min(self.cash, equity * 0.12)
            quantity = _lot_size(budget / max(price, 1e-9))
            limit_price = price * (1.0 + 0.006)
        elif action == "sell" and position:
            target_qty = _lot_size((self.cash + position.quantity * price) * 0.10 / max(price, 1e-9))
            quantity = min(position.quantity, max(100, target_qty))
            quantity = _lot_size(quantity)
            limit_price = price * (1.0 - 0.006)

        if action == "buy" and quantity * limit_price > self.cash:
            quantity = _lot_size(self.cash / max(limit_price, 1e-9))
        if action == "sell" and position:
            quantity = min(quantity, _lot_size(position.quantity))
        if quantity <= 0:
            action = "hold"
            quantity = 0
            score = 0.0
            limit_price = price

        sentiment_class = 1 if action == "buy" else -1 if action == "sell" else 0

        thought = self._build_thought(symbol, belief, action, unrealized, value_gap)
        decision = Decision(
            agent_id=self.agent_id,
            symbol=symbol,
            action=action,
            quantity=int(quantity),
            limit_price=round(float(limit_price), 4),
            thought=thought,
            belief_score=round(float(score), 4),
            sentiment_class=sentiment_class,
        )
        self.memory.append({"symbol": symbol, "belief": belief, "decision": decision})
        self.memory = self.memory[-200:]
        return decision
    # endregion

    def decide(self, symbol: str) -> Decision:
        """Return a decision for one symbol.

        Default path remains the legacy deterministic agent. When ``llm_client``
        is injected, use the new BDI demo chain:
        feature engineering -> LLM belief -> belief scoring -> desire utility
        -> LLM desire adjustment -> action -> final thought.
        """

        if self.llm_client is None:
            return self._legacy_rule_decide(symbol)
        try:
            return self._bdi_llm_decide(symbol)
        except Exception as exc:
            fallback = self._legacy_rule_decide(symbol)
            fallback.thought = f"{fallback.thought} LLM pipeline fallback: {type(exc).__name__}."
            return fallback

    def _bdi_llm_decide(self, symbol: str) -> Decision:
        price = self.current_price(symbol)
        if price <= 0:
            return Decision(self.agent_id, symbol, "hold", 0, 0.0, "No tradable price.", 0.0, 0)

        market_rows = self._market.get(symbol, [])
        market_features = build_market_features(market_rows)
        position = self.positions.get(symbol)
        position_qty = position.quantity if position else 0
        avg_cost = position.avg_cost if position else 0.0
        account_features = build_account_features(
            cash=self.cash,
            symbol=symbol,
            position=position_qty,
            avg_cost=avg_cost,
            current_price=price,
        )
        observation = SimpleNamespace(
            agent_id=self.agent_id,
            symbol=symbol,
            tick=len(self.memory) + 1,
            extra={},
            news=list(self._news.get(symbol, [])),
            social_posts=list(self._social.get(symbol, [])),
        )

        cognition_system = build_cognition_system_prompt(self.personality)
        cognition_user = build_cognition_user_prompt(observation, market_features, self.personality)
        cognition_raw = self.llm_client(cognition_system, cognition_user)  # type: ignore[misc]
        if not str(cognition_raw or "").strip():
            raise RuntimeError("empty LLM belief response")
        cognition = parse_cognition_response(cognition_raw)
        if self._cognition_is_empty(cognition, observation):
            cognition = self._heuristic_cognition(symbol, market_features)

        belief_fields = build_belief_score(cognition, market_features)
        belief_context: Dict[str, Any] = {**cognition, **belief_fields}

        rule_desires = build_rule_desires(
            belief_score=float(belief_fields["belief_score"]),
            account_features=account_features,
            cognition=belief_context,
            market_features=market_features,
        )

        if self.llm_desire_enabled:
            desire_system = build_desire_system_prompt(self.personality)
            desire_user = build_desire_user_prompt(
                personality=self.personality,
                belief=belief_context,
                rule_desires=rule_desires,
                account_features=account_features,
                market_context=market_features,
            )
            desire_raw = self.llm_client(desire_system, desire_user)  # type: ignore[misc]
            desire_adjustment = parse_desire_adjustment(desire_raw)
        else:
            desire_adjustment = parse_desire_adjustment("")
        final_desires = apply_desire_adjustment(rule_desires, desire_adjustment)
        desire_context: Dict[str, Any] = {**rule_desires, **final_desires, "desire_reason": desire_adjustment.get("desire_reason", "")}

        action, quantity, limit_price, official_score, official_sentiment = self._action_from_desires(
            symbol=symbol,
            price=price,
            account_features=account_features,
            final_desires=final_desires,
        )

        action_context = {
            "action": action,
            "quantity": quantity,
            "limit_price": limit_price,
            "official_belief_score": official_score,
            "official_sentiment_class": official_sentiment,
        }
        thought = build_rule_thought(
            personality=self.personality,
            belief=belief_context,
            desire=desire_context,
            action=action_context,
            account_features=account_features,
            market_context=market_features,
        )
        if self.llm_thought_enabled:
            thought_system = build_thought_system_prompt(self.personality)
            thought_user = build_thought_user_prompt(
                personality=self.personality,
                belief=belief_context,
                desire=desire_context,
                action=action_context,
                account_features=account_features,
                market_context=market_features,
            )
            try:
                thought_raw = self.llm_client(thought_system, thought_user)  # type: ignore[misc]
                thought = parse_thought_response(thought_raw)["thought"]
            except Exception:
                pass

        decision = Decision(
            agent_id=self.agent_id,
            symbol=symbol,
            action=action,
            quantity=int(quantity),
            limit_price=round(float(limit_price), 4),
            thought=thought,
            belief_score=round(float(official_score), 4),
            sentiment_class=int(official_sentiment),
        )
        self.memory.append(
            {
                "symbol": symbol,
                "market_features": market_features,
                "account_features": account_features,
                "cognition": cognition,
                "belief": belief_fields,
                "rule_desires": rule_desires,
                "desire_adjustment": desire_adjustment,
                "final_desires": final_desires,
                "decision": decision,
            }
        )
        self.memory = self.memory[-200:]
        return decision

    def _cognition_is_empty(self, cognition: Mapping[str, Any], observation: Any) -> bool:
        has_text = bool(getattr(observation, "news", None) or getattr(observation, "social_posts", None))
        return (
            has_text
            and cognition.get("scope") == "irrelevant"
            and abs(float(cognition.get("micro_strength", 0.0) or 0.0)) <= 1e-9
            and abs(float(cognition.get("technical_bias", 0.0) or 0.0)) <= 1e-9
        )

    def _heuristic_cognition(self, symbol: str, market_features: Mapping[str, Any]) -> Dict[str, Any]:
        text_sentiment = self._text_sentiment(self._news.get(symbol, []))
        social_sentiment = self._social_sentiment(self._social.get(symbol, []))
        text_score = _clip(0.65 * text_sentiment + 0.35 * social_sentiment, -1.0, 1.0)
        technical_bias = _clip(
            0.40 * float(market_features.get("ma_score", 0.0))
            + 0.30 * float(market_features.get("momentum_score", 0.0))
            + 0.15 * float(market_features.get("recent_return_score", 0.0))
            + 0.10 * float(market_features.get("rsi_score", 0.0))
            + 0.05 * float(market_features.get("anchor_score", 0.0)),
            -1.0,
            1.0,
        )
        micro_direction = 1 if text_score > 0.08 else -1 if text_score < -0.08 else 0
        technical_direction = 1 if technical_bias > 0.08 else -1 if technical_bias < -0.08 else 0
        uncertainty = _clip(0.55 - 0.25 * abs(text_score) + 0.25 * float(market_features.get("volatility_score", 0.0)), 0.15, 0.85)
        emotion = "neutral"
        if text_score > 0.25:
            emotion = "fomo"
        elif text_score < -0.25:
            emotion = "panic"
        return {
            "scope": "stock_specific" if micro_direction else "mixed",
            "macro_direction": 0,
            "macro_strength": 0.1,
            "micro_direction": micro_direction,
            "micro_strength": _clip(abs(text_score), 0.0, 1.0),
            "technical_direction": technical_direction,
            "technical_strength": _clip(abs(technical_bias), 0.0, 1.0),
            "technical_bias": technical_bias,
            "attention_bias": _clip(float(market_features.get("attention_score", 0.0)) + 0.25 * social_sentiment, -1.0, 1.0),
            "uncertainty": uncertainty,
            "retail_emotion": emotion,
            "belief_reason": "The fallback cognition uses news, social sentiment, and technical features because the LLM belief response was empty.",
        }

    def _action_from_desires(
        self,
        *,
        symbol: str,
        price: float,
        account_features: Mapping[str, Any],
        final_desires: Mapping[str, Any],
    ) -> tuple[str, int, float, float, int]:
        position = self.positions.get(symbol)
        has_position = bool(position and position.quantity > 0)
        buy_desire = float(final_desires.get("buy_desire", 0.0))
        sell_desire = float(final_desires.get("sell_desire", 0.0))
        hold_desire = float(final_desires.get("hold_desire", 0.0))

        if not has_position:
            sell_desire = 0.0
        if self.cash < price * 100:
            buy_desire = 0.0

        action = max(
            ("buy", "sell", "hold"),
            key=lambda name: {"buy": buy_desire, "sell": sell_desire, "hold": hold_desire}[name],
        )
        if action == "buy" and buy_desire < max(0.22, hold_desire + 0.03):
            action = "hold"
        if action == "sell" and sell_desire < max(0.24, hold_desire + 0.03):
            action = "hold"

        quantity = 0
        limit_price = price
        if action == "buy":
            equity = max(self.cash + (position.quantity * price if position else 0.0), 1.0)
            budget_fraction = _clip(0.05 + 0.12 * buy_desire, 0.04, 0.16)
            budget = min(self.cash, equity * budget_fraction)
            quantity = _lot_size(budget / max(price, 1e-9))
            limit_price = price * (1.0 + 0.006)
        elif action == "sell" and position:
            sell_fraction = _clip(0.20 + 0.55 * sell_desire, 0.20, 0.75)
            quantity = _lot_size(position.quantity * sell_fraction)
            if quantity <= 0 and position.quantity >= 100:
                quantity = 100
            quantity = min(quantity, _lot_size(position.quantity))
            limit_price = price * (1.0 - 0.006)

        if action == "buy" and quantity * limit_price > self.cash:
            quantity = _lot_size(self.cash / max(limit_price, 1e-9))
        if action == "sell" and position:
            quantity = min(quantity, _lot_size(position.quantity))
        if quantity <= 0:
            action = "hold"
            quantity = 0
            limit_price = price

        if action == "buy":
            official_score = max(0.35, buy_desire)
            official_sentiment = 1
        elif action == "sell":
            official_score = -max(0.35, sell_desire)
            official_sentiment = -1
        else:
            official_score = 0.0
            official_sentiment = 0
        return action, int(quantity), round(float(limit_price), 4), _clip(official_score, -1.0, 1.0), official_sentiment

    def current_price(self, symbol: str) -> float:
        rows = self._market.get(symbol, [])
        return float(rows[-1]["close"]) if rows else 0.0

    def _recent_return(self, symbol: str) -> float:
        rows = self._market.get(symbol, [])
        closes = [row["close"] for row in rows if row["close"] > 0]
        if len(closes) < 2:
            return 0.0
        lookback = closes[-6] if len(closes) >= 6 else closes[0]
        return _clip((closes[-1] / lookback - 1.0) * 4.0, -1.0, 1.0) if lookback > 0 else 0.0

    def apply_fill(self, symbol: str, side: str, price: float, quantity: int) -> None:
        if quantity <= 0:
            return
        if side == "buy":
            cost = price * quantity
            if cost > self.cash + 1e-9:
                return
            old = self.positions.get(symbol)
            if old:
                total_qty = old.quantity + quantity
                avg_cost = (old.avg_cost * old.quantity + cost) / total_qty
                self.positions[symbol] = Position(total_qty, avg_cost)
            else:
                self.positions[symbol] = Position(quantity, price)
            self.cash -= cost
        elif side == "sell":
            old = self.positions.get(symbol)
            if not old:
                return
            sold = min(quantity, old.quantity)
            self.cash += price * sold
            remaining = old.quantity - sold
            if remaining > 0:
                self.positions[symbol] = Position(remaining, old.avg_cost)
            else:
                self.positions.pop(symbol, None)

    def prompt(self, symbol: str) -> str:
        belief = self.beliefs.get(symbol) or self.update_beliefs(symbol)
        return (
            f"Role: {self.personality} retail investor.\n"
            f"Belief: fair_value={belief.fair_value:.4f}, momentum={belief.momentum:.3f}, "
            f"sentiment={belief.sentiment:.3f}, social_pressure={belief.social_pressure:.3f}, "
            f"volatility={belief.volatility:.3f}, confidence={belief.confidence:.3f}.\n"
            "Return JSON with thought, action in [buy, sell, hold], quantity, limit_price."
        )

    def _llm_decide(self, symbol: str) -> Decision:
        """🚀 ADVANCED: 使用 LLM 生成决策（替代规则式决策）。

        调用 self.llm_client(system_prompt, user_prompt) 获取 LLM 响应，
        解析为 Decision 对象。如果 LLM 调用失败或返回无效结果，
        自动回退到规则式 decide()。

        🔧 MODIFIABLE: 修改 system_prompt 和 user_prompt 以调整 LLM 行为。
        💡 TIP: 可通过 prompt engineering 注入性格特征到 system_prompt 中。

        Returns:
            Decision 对象，包含 LLM 生成的决策。
        """
        # ---- 构建系统提示词 ----
        # 🔧 MODIFIABLE: 调整提示词以改变 Agent 行为风格
        system_prompt = (
            f"You are a {self.personality} retail investor in a financial market simulation. "
            f"Your task is to decide whether to buy, sell, or hold a stock based on market data, "
            f"news, and social sentiment.\n\n"
            f"Investment philosophy:\n"
            f"- You have a risk appetite of {self.traits['risk_appetite']:.2f} (0=conservative, 1=aggressive)\n"
            f"- You exhibit disposition effect: tendency to sell winners too early and hold losers too long\n"
            f"- Your decision should reflect your personality type: {self.personality}\n\n"
            f"Response format: Return ONLY a JSON object (no markdown, no extra text):\n"
            f'{{"action": "buy"|"sell"|"hold", "quantity": <int>, "limit_price": <float>, '
            f'"thought": "<your reasoning as a trader>", "belief_score": <-1.0 to 1.0>, '
            f'"sentiment_class": <-1|0|1>}}\n'
        )

        # ---- 构建用户提示词 ----
        user_prompt = self.prompt(symbol)

        # ---- 调用 LLM ----
        try:
            raw = self.llm_client(system_prompt, user_prompt)  # type: ignore[misc]
        except Exception:
            # LLM 调用失败，回退到规则式决策
            return self._rule_decide_fallback(symbol)

        if not raw:
            return self._rule_decide_fallback(symbol)

        # ---- 解析 LLM 响应 ----
        parsed = self._parse_llm_response(raw)

        # ---- 验证并构建 Decision ----
        action = parsed.get("action", "hold")
        if action not in {"buy", "sell", "hold"}:
            action = "hold"

        quantity = int(parsed.get("quantity", 0))
        limit_price = float(parsed.get("limit_price", self.current_price(symbol)))
        thought = str(parsed.get("thought", "LLM-generated decision."))
        belief_score = float(parsed.get("belief_score", 0.0))
        sentiment_class = int(parsed.get("sentiment_class", 0))

        # 安全约束
        price = self.current_price(symbol)
        if action == "buy":
            budget = self.cash * 0.28
            max_qty = int(budget // max(limit_price, 1e-9))
            quantity = min(quantity, max_qty)
        elif action == "sell":
            position = self.positions.get(symbol)
            if position:
                quantity = min(quantity, position.quantity)
            else:
                action = "hold"
                quantity = 0

        if quantity <= 0:
            action = "hold"
            quantity = 0

        return Decision(
            agent_id=self.agent_id,
            symbol=symbol,
            action=action,
            quantity=int(quantity),
            limit_price=round(float(limit_price), 4),
            thought=thought,
            belief_score=round(float(belief_score), 4),
            sentiment_class=int(sentiment_class),
        )

    def _rule_decide_fallback(self, symbol: str) -> Decision:
        return self._legacy_rule_decide(symbol)

    def _parse_llm_response(self, raw: str) -> dict:
        """🔧 MODIFIABLE: 解析 LLM 原始响应为字典。

        LLM 有时会在 JSON 外包裹 markdown 代码块或额外文本。
        此方法尝试多种策略提取 JSON。

        Args:
            raw: LLM 原始响应文本。

        Returns:
            解析后的字典；如果完全无法解析则返回空字典。
        """
        import json as _json
        import re as _re

        text = raw.strip()
        # 策略1: 直接解析
        try:
            return _json.loads(text)
        except _json.JSONDecodeError:
            pass

        # 策略2: 去除 markdown 代码块标记
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
            try:
                return _json.loads(text)
            except _json.JSONDecodeError:
                pass

        # 策略3: 正则提取 JSON 对象
        match = _re.search(r"\{[^{}]*\"action\"[^{}]*\}", text, _re.DOTALL)
        if match:
            try:
                return _json.loads(match.group())
            except _json.JSONDecodeError:
                pass

        return {}

    def _build_thought(
        self, symbol: str, belief: Belief, action: str, unrealized: float, value_gap: float
    ) -> str:
        direction = "bullish" if action == "buy" else "bearish or profit-taking" if action == "sell" else "mixed"
        parts = [
            f"{symbol} evidence feels {direction}",
            f"momentum {belief.momentum:.2f}",
            f"news sentiment {belief.sentiment:.2f}",
            f"social pressure {belief.social_pressure:.2f}",
        ]
        if action == "sell" and unrealized > 0:
            parts.append(f"my unrealized gain is {unrealized:.1%}, so I want to lock in part of it")
        elif action == "hold" and unrealized < 0:
            parts.append(f"my position is down {abs(unrealized):.1%}, and I dislike realizing the loss without stronger evidence")
        elif action == "hold":
            parts.append("the signal is not strong enough to justify trading")
        elif action == "buy":
            parts.append("the positive evidence and available cash make participation attractive")
        if abs(value_gap) > 0.15:
            parts.append("price is away from my value anchor")
        parts.append(f"therefore I choose {action}")
        return "; ".join(parts) + "."

    def _text_sentiment(self, texts: Iterable[str]) -> float:
        text = " ".join(texts).lower()
        if not text:
            return 0.0
        pos = sum(text.count(word.lower()) for word in POSITIVE_WORDS)
        neg = sum(text.count(word.lower()) for word in NEGATIVE_WORDS)
        return _clip((pos - neg) / max(pos + neg + 2, 1), -1.0, 1.0)

    def _social_sentiment(self, posts: Iterable[Mapping[str, Any]]) -> float:
        total_weight = 0.0
        score = 0.0
        for post in posts:
            text = str(post.get("text", ""))
            influence = float(post.get("influence", post.get("likes", 1.0)) or 1.0)
            influence = math.log1p(max(influence, 0.0))
            score += influence * self._text_sentiment([text])
            total_weight += influence
        if total_weight <= 0:
            return 0.0
        return _clip(score / total_weight * (0.35 + self.traits["herding"]), -1.0, 1.0)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((value - mu) ** 2 for value in values) / (len(values) - 1))


def _returns(closes: List[float]) -> List[float]:
    return [
        closes[i] / closes[i - 1] - 1.0
        for i in range(1, len(closes))
        if closes[i - 1] > 0
    ]


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _lot_size(quantity: float, lot: int = 100) -> int:
    if quantity <= 0:
        return 0
    return int(quantity // lot) * lot
