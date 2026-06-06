"""Custom regulator wired into the baseline ExchangeAgent.

撮合引擎复用 baseline 的 LimitOrderBook（价格-时间优先已正确实现）。
监管层用本文件的 MyRegulator 替换 baseline 的 RegulatoryAgent，
内部委托给 detectors.py 的三个检测器。
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional

from competition_solution.exchange import (
    Alert,
    ExchangeAgent,
    LimitOrderBook,
    Order,
    RegulatoryAgent,
    Trade,
)

from detectors import PumpDumpDetector, SpoofDetector, WashDetector


class MyRegulator(RegulatoryAgent):
    """Replaces baseline RegulatoryAgent. Delegates to detector classes."""

    def __init__(self) -> None:
        super().__init__()
        self.wash = WashDetector()
        self.spoof = SpoofDetector()
        self.pump = PumpDumpDetector()
        self._last_tick: int = 0

    def pre_submit(self, order: Order, book: LimitOrderBook) -> Optional[Alert]:
        alert = self.wash.on_pre_submit(order, book)
        if alert is not None:
            self.alerts.append(alert)
            return alert
        alert = self.spoof.on_pre_submit(order, book)
        if alert is not None:
            self.alerts.append(alert)
            return alert
        self.open_orders[order.order_id] = order
        return None

    def on_cancel(
        self,
        order: Order,
        timestamp: int,
        book: LimitOrderBook,
    ) -> Optional[Alert]:
        self.open_orders.pop(order.order_id, None)
        alert = self.spoof.on_cancel(order, timestamp, book)
        if alert is not None:
            self.alerts.append(alert)
        return alert

    def on_trades(
        self,
        trades: Iterable[Trade],
        orders: Mapping[str, Order],
    ) -> List[Alert]:
        trade_list = list(trades)
        out: List[Alert] = []
        for alert in self.wash.on_trades(trade_list, orders):
            self.alerts.append(alert)
            out.append(alert)
        for alert in self.pump.on_trades(trade_list, orders, self._last_tick):
            self.alerts.append(alert)
            out.append(alert)
        return out


def build_exchange() -> ExchangeAgent:
    return ExchangeAgent(regulator=MyRegulator())
