"""Surveillance detectors. Each detector is a stateful class with one `check` method.

骨架阶段：三个 check 全部返回 None。后续在这里填规则/LLM 逻辑即可，
不需要再动 submission.py 或 my_exchange.py。
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Tuple

from competition_solution.exchange import Alert, LimitOrderBook, Order, Trade


def _average_depth(levels: List[Tuple[float, int]]) -> float:
    if not levels:
        return 0.0
    return sum(qty for _, qty in levels) / len(levels)


# ============================================================
# WashDetector — 洗售/对倒检测（纯规则）
# ------------------------------------------------------------
# 两条规则，都是高置信度直判：
#   R1 (pre-submit): 新单将与盘口前 5 档某挂单成交，且双方 entity_id 相同
#                    → 自成交，撮合前直接 block，避免脏成交污染 trades 流。
#   R2 (on-trades): 撮合后兜底——若买卖双方 entity 相同（极端情况下绕过 R1），
#                    仍发 wash_trading 告警；同时维护 (buyer, seller) 对的滑动
#                    窗口计数，同一对手在 wash_window tick 内成交 ≥4 次即视为
#                    对倒环 wash_trading_ring。
# 设计点：同 tick 出 alert，无外部调用，latency ≤ 3 tick 锁满分。
# ============================================================
WASH_WINDOW = 8           # 对倒环滑动窗口（tick）
WASH_RING_THRESHOLD = 4   # 同一对手方在窗口内成交次数阈值


class WashDetector:
    """Detect wash trading (same beneficial owner on both sides)."""

    def __init__(self) -> None:
        self._recent_trades: Deque[Trade] = deque(maxlen=500)

    def on_pre_submit(self, order: Order, book: LimitOrderBook) -> Optional[Alert]:
        # R1: 同 entity 自买自卖的预交易拦截
        book._sort_books()
        contra = book.sell if order.side == "buy" else book.buy
        for resting in contra[:5]:
            crosses = (
                order.side == "buy" and order.price >= resting.price
            ) or (
                order.side == "sell" and order.price <= resting.price
            )
            if crosses and resting.entity_id == order.entity_id:
                return Alert(
                    timestamp=order.timestamp,
                    alert_type="wash_trading",
                    severity=0.98,
                    order_id=f"{order.order_id},{resting.order_id}",
                    entity_id=str(order.entity_id),
                    symbol=order.symbol,
                    action="block_order_and_freeze_entity",
                    thought=(
                        f"Incoming {order.side} from entity {order.entity_id} would cross "
                        f"a resting order from the same beneficial owner at price "
                        f"{resting.price}; blocked pre-trade to prevent self-cross."
                    ),
                )
        return None

    def on_trades(
        self,
        trades: Iterable[Trade],
        order_snapshots: Mapping[str, Order],
    ) -> List[Alert]:
        # R2: 撮合后同 entity 兜底 + 对倒环检测
        alerts: List[Alert] = []
        for trade in trades:
            self._recent_trades.append(trade)
            buyer = order_snapshots.get(trade.buy_order_id)
            seller = order_snapshots.get(trade.sell_order_id)
            buyer_entity = buyer.entity_id if buyer else trade.buyer_id
            seller_entity = seller.entity_id if seller else trade.seller_id

            # 同 entity 兜底
            if buyer_entity == seller_entity:
                alerts.append(Alert(
                    timestamp=trade.timestamp,
                    alert_type="wash_trading",
                    severity=0.98,
                    order_id=f"{trade.buy_order_id},{trade.sell_order_id}",
                    entity_id=str(buyer_entity),
                    symbol=trade.symbol,
                    action="block_trade_and_freeze_entity",
                    thought=(
                        f"Trade {trade.buy_order_id}/{trade.sell_order_id} resolves to "
                        f"the same beneficial owner {buyer_entity}; no economic risk transfer."
                    ),
                ))
                continue

            # 对倒环：同一对手方在窗口内反复成交
            pair = frozenset({buyer_entity, seller_entity})
            count = sum(
                1 for t in self._recent_trades
                if t.symbol == trade.symbol
                and t.timestamp >= trade.timestamp - WASH_WINDOW
                and frozenset({
                    (order_snapshots.get(t.buy_order_id).entity_id
                     if order_snapshots.get(t.buy_order_id) else t.buyer_id),
                    (order_snapshots.get(t.sell_order_id).entity_id
                     if order_snapshots.get(t.sell_order_id) else t.seller_id),
                }) == pair
            )
            if count >= WASH_RING_THRESHOLD:
                alerts.append(Alert(
                    timestamp=trade.timestamp,
                    alert_type="wash_trading_ring",
                    severity=0.85,
                    order_id=f"{trade.buy_order_id},{trade.sell_order_id}",
                    entity_id=f"{buyer_entity}|{seller_entity}",
                    symbol=trade.symbol,
                    action="warn_and_sample_for_review",
                    thought=(
                        f"Entities {buyer_entity} and {seller_entity} reversed risk "
                        f"{count} times within {WASH_WINDOW} ticks on {trade.symbol}; "
                        f"consistent with circular wash-trading."
                    ),
                ))
        return alerts


# ============================================================
# SpoofDetector — 幌骗/虚假申报检测（纯规则）
# ------------------------------------------------------------
# 两个触发口：
#
# A) on_pre_submit — 预交易识别"挂虚假深度的大单"
#    AND 条件：
#      1. 单量远超同侧深度：qty ≥ avg_depth × LARGE_ORDER_RATIO（深度为空时直接看绝对量）
#      2. 远离对手价：买单价 < 卖一价 且偏离 ≥ FAR_FROM_TOUCH_BPS（或反向）
#    这类订单不可能立即成交，挂在远端只为制造假深度诱导对手 → 典型 spoof/layer。
#
# B) on_cancel — 事后识别"快速撤大单"
#    AND 条件：
#      1. age ≤ SPOOF_CANCEL_WINDOW tick
#      2. 大单（同 A.1）
#      3. 同 entity 在窗口内累计撤单 ≥ SPOOF_SAME_ENTITY_RECENT 次
#    用于补 A 漏网的"贴近盘口下大单后秒撤"形态。
#
# 两个口都同 tick 出 alert，latency 锁 0。
# ============================================================
SPOOF_CANCEL_WINDOW = 3
LARGE_ORDER_RATIO = 3.0
LARGE_ORDER_ABS_MIN = 500       # 当盘口为空 / 深度极薄时的兜底绝对阈值
SPOOF_SAME_ENTITY_RECENT = 2
FAR_FROM_TOUCH_BPS = 100.0      # 远离对手价的最小幅度（1% = 100bps）


class SpoofDetector:
    """Detect spoofing / quote stuffing (large orders cancelled fast)."""

    def __init__(self) -> None:
        self._cancelled: Deque[Dict] = deque(maxlen=200)

    def on_pre_submit(self, order: Order, book: LimitOrderBook) -> Optional[Alert]:
        # 大单判定（看同侧深度，深度为 0 用绝对量兜底）
        avg_depth = _average_depth(book.depth(order.side, levels=5))
        is_large = (
            (avg_depth > 0 and order.quantity >= avg_depth * LARGE_ORDER_RATIO)
            or (avg_depth == 0 and order.quantity >= LARGE_ORDER_ABS_MIN)
        )
        if not is_large:
            return None

        # 远离盘口：买单远低于卖一价 / 卖单远高于买一价
        bid, ask = book.best_bid_ask()
        touch = ask if order.side == "buy" else bid
        if touch is None or touch <= 0:
            return None
        away = (order.side == "buy" and order.price < touch) or \
               (order.side == "sell" and order.price > touch)
        if not away:
            return None
        distance_bps = abs(order.price - touch) / touch * 10000.0
        if distance_bps < FAR_FROM_TOUCH_BPS:
            return None

        return Alert(
            timestamp=order.timestamp,
            alert_type="spoofing",
            severity=0.90,
            order_id=order.order_id,
            entity_id=order.entity_id or order.agent_id,
            symbol=order.symbol,
            action="block_order_and_flag_entity",
            thought=(
                f"Entity {order.entity_id} submitted oversized {order.side} order "
                f"(qty {order.quantity}, depth≈{avg_depth:.0f}) at price {order.price}, "
                f"{distance_bps:.0f}bps away from touch {touch}; cannot fill — "
                f"pattern matches spoofing/layering."
            ),
        )

    def on_cancel(
        self,
        order: Order,
        timestamp: int,
        book: LimitOrderBook,
    ) -> Optional[Alert]:
        age = timestamp - order.timestamp
        self._cancelled.append({"entity_id": order.entity_id, "timestamp": timestamp})

        # 条件 1：撤单速度
        if age > SPOOF_CANCEL_WINDOW:
            return None

        # 条件 2：单量
        avg_depth = _average_depth(book.depth(order.side, levels=5))
        is_large = (
            (avg_depth > 0 and order.quantity >= max(1, avg_depth) * LARGE_ORDER_RATIO)
            or (avg_depth == 0 and order.quantity >= LARGE_ORDER_ABS_MIN)
        )
        if not is_large:
            return None

        # 条件 3：同 entity 高频撤
        same_entity_recent = sum(
            1 for item in self._cancelled
            if item["entity_id"] == order.entity_id
            and item["timestamp"] >= timestamp - SPOOF_CANCEL_WINDOW
        )
        if same_entity_recent < SPOOF_SAME_ENTITY_RECENT:
            return None

        # 条件 4：远离盘口（用对手价做基准）
        bid, ask = book.best_bid_ask()
        touch = ask if order.side == "buy" else bid
        if touch is not None and touch > 0:
            distance_bps = abs(order.price - touch) / touch * 10000.0
            # 买单挂在对手价下方 / 卖单挂在对手价上方 才算"远离"
            away = (order.side == "buy" and order.price < touch) or \
                   (order.side == "sell" and order.price > touch)
            if not (away and distance_bps >= FAR_FROM_TOUCH_BPS):
                return None

        return Alert(
            timestamp=timestamp,
            alert_type="spoofing",
            severity=0.92,
            order_id=order.order_id,
            entity_id=order.entity_id or order.agent_id,
            symbol=order.symbol,
            action="intervene_cancel_and_throttle",
            thought=(
                f"Entity {order.entity_id} cancelled an oversized {order.side} order "
                f"(qty {order.quantity} vs avg depth {avg_depth:.0f}) within {age} ticks "
                f"at price {order.price} away from the touch; pattern consistent with spoofing."
            ),
        )


# ============================================================
# PumpDumpDetector — 拉抬出货检测（纯规则）
# ------------------------------------------------------------
# 维护每个 symbol 的滑动 trade 窗口（最近 PUMP_WINDOW tick），
# 收到新成交后检查三条 AND 信号：
#   1. 价格异动：窗口内 (max_price - min_price)/min_price ≥ PUMP_RISE_PCT
#   2. 集中卖出：窗口里某一 seller_entity 卖出量占比 ≥ DUMP_CONCENTRATION
#   3. 多方参与拉升：窗口内不同 buyer_entity 数 ≥ PUMP_BUYERS_MIN
# 三条全中 → fire pump_and_dump，alert 与触发 trade 同 tick 发出（latency=0）。
# 阈值偏宽松，靠 AND 组合控 precision。
# ============================================================
PUMP_WINDOW = 5             # 滑动窗口（tick）
PUMP_RISE_PCT = 0.03        # 窗口价格涨幅阈值（3%）
DUMP_CONCENTRATION = 0.5    # 集中卖出占比（≥50%）
PUMP_BUYERS_MIN = 3         # 不同买方 entity 数下限


class PumpDumpDetector:
    """Detect coordinated pump-and-dump across multiple entities.

    Baseline 完全未实现，是任务二的最大提分点。
    """

    def __init__(self) -> None:
        self._recent: Dict[str, Deque[Trade]] = {}

    def on_trades(
        self,
        trades: Iterable[Trade],
        order_snapshots: Mapping[str, Order],
        tick: int,
    ) -> List[Alert]:
        alerts: List[Alert] = []
        for trade in trades:
            buf = self._recent.setdefault(trade.symbol, deque(maxlen=200))
            buf.append(trade)
            # 清理过期 trade
            while buf and buf[0].timestamp < trade.timestamp - PUMP_WINDOW:
                buf.popleft()
            if len(buf) < 3:
                continue

            # 1) 价格异动
            prices = [t.price for t in buf]
            lo, hi = min(prices), max(prices)
            if lo <= 0 or (hi - lo) / lo < PUMP_RISE_PCT:
                continue

            # 2) 集中卖出
            seller_qty: Dict[str, int] = {}
            total_qty = 0
            for t in buf:
                sell_o = order_snapshots.get(t.sell_order_id)
                ent = sell_o.entity_id if sell_o else t.seller_id
                seller_qty[ent] = seller_qty.get(ent, 0) + t.quantity
                total_qty += t.quantity
            if total_qty == 0:
                continue
            top_seller, top_qty = max(seller_qty.items(), key=lambda kv: kv[1])
            if top_qty / total_qty < DUMP_CONCENTRATION:
                continue

            # 3) 多方买入
            buyer_entities = set()
            for t in buf:
                buy_o = order_snapshots.get(t.buy_order_id)
                buyer_entities.add(buy_o.entity_id if buy_o else t.buyer_id)
            if len(buyer_entities) < PUMP_BUYERS_MIN:
                continue

            alerts.append(Alert(
                timestamp=trade.timestamp,
                alert_type="pump_and_dump",
                severity=0.93,
                order_id=f"{trade.buy_order_id},{trade.sell_order_id}",
                entity_id=str(top_seller),
                symbol=trade.symbol,
                action="halt_symbol_and_review",
                thought=(
                    f"Symbol {trade.symbol} ran up {(hi-lo)/lo*100:.1f}% over "
                    f"{PUMP_WINDOW} ticks driven by {len(buyer_entities)} buying entities, "
                    f"while seller {top_seller} captured {top_qty/total_qty*100:.0f}% of "
                    f"sell volume — pattern consistent with pump-and-dump distribution."
                ),
            ))
            # 同一 symbol 一个窗口里只发一次，清空缓冲避免重复告警
            buf.clear()
        return alerts
