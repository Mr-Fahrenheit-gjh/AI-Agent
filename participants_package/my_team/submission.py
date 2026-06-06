"""Team submission entry point. DO NOT change method signatures."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# Validator loads this file as a standalone module (not a package), so sibling
# modules in this directory must be reachable via sys.path.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from competition_solution.investment_agent import InvestmentAgent, Position
from submission_interface.api import (
    AgentDecision,
    AlertRecord,
    CompetitionSubmission,
    MarketObservation,
    MatchResult,
    OrderRequest,
    TradeRecord,
)

from my_exchange import build_exchange  # noqa: E402


class TeamSubmission(CompetitionSubmission):
    def __init__(self, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config = dict(config or {})
        self.agents: Dict[str, InvestmentAgent] = {}
        self.exchange = build_exchange()
        self.seed = 0

    def reset(self, seed: int = 0, config: Optional[Mapping[str, Any]] = None) -> None:
        self.config.update(dict(config or {}))
        self.agents = {}
        self.exchange = build_exchange()
        self.seed = seed

    def decide(self, observation: MarketObservation) -> AgentDecision:
        agent = self.agents.get(observation.agent_id)
        if agent is None:
            personality = self.config.get("default_personality", "trend")
            agent = InvestmentAgent(
                agent_id=observation.agent_id,
                personality=personality,
                cash=observation.cash,
                seed=self.seed + len(self.agents),
            )
            self.agents[observation.agent_id] = agent
        agent.cash = observation.cash
        if observation.position > 0:
            agent.positions[observation.symbol] = Position(observation.position, observation.avg_cost)
        else:
            agent.positions.pop(observation.symbol, None)
        agent.ingest_market(observation.symbol, [item.to_dict() for item in observation.klines])
        agent.ingest_news(observation.symbol, observation.news)
        agent.ingest_social(observation.symbol, observation.social_posts)
        decision = agent.decide(observation.symbol)
        return AgentDecision(
            agent_id=decision.agent_id,
            symbol=decision.symbol,
            action=decision.action,
            quantity=decision.quantity,
            limit_price=decision.limit_price,
            thought=decision.thought,
            belief_score=decision.belief_score,
            sentiment_class=decision.sentiment_class,
        )

    def match_orders(
        self,
        orders: List[OrderRequest],
        last_prices: Mapping[str, float],
        tick: int,
    ) -> MatchResult:
        self.exchange.regulator._last_tick = tick
        accepted: List[str] = []
        rejected: List[str] = []
        trades: List[TradeRecord] = []
        alerts: List[AlertRecord] = []
        close_prices = dict(last_prices)

        for order in orders:
            result = self.exchange.submit_order(
                agent_id=order.agent_id,
                symbol=order.symbol,
                side=order.side,
                price=order.price,
                quantity=order.quantity,
                timestamp=order.timestamp,
                entity_id=order.entity_id,
            )
            if result["accepted"]:
                accepted.append(order.order_id)
            else:
                rejected.append(order.order_id)
            for item in result["trades"]:
                trades.append(TradeRecord(**item))
                close_prices[item["symbol"]] = item["price"]
            for item in result["alerts"]:
                alerts.append(AlertRecord(**item))

        return MatchResult(
            trades=trades,
            accepted_order_ids=accepted,
            rejected_order_ids=rejected,
            alerts=alerts,
            close_prices=close_prices,
        )


def create_submission(config: Optional[Mapping[str, Any]] = None) -> CompetitionSubmission:
    return TeamSubmission(config)
