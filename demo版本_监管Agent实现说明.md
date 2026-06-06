# Demo 版本监管 Agent 实现说明

本文档记录当前 `my_team` 目录中 demo 版本交易所与监管 Agent 的具体实现。它描述的是当前代码实际怎么做，不代表最终工业级市场监控方案。

对应文件：

```text
participants_package/my_team/submission.py
participants_package/my_team/exchange.py
```

## 1. 在 submission.py 中如何调用

评测系统调用：

```python
TeamSubmission.match_orders(orders, last_prices, tick)
```

当前 `submission.py` 对每个 `OrderRequest` 逐个调用：

```python
self.exchange.submit_order(
    agent_id=order.agent_id,
    symbol=order.symbol,
    side=order.side,
    price=order.price,
    quantity=order.quantity,
    timestamp=order.timestamp,
    entity_id=order.entity_id,
)
```

然后把内部返回结果转换成官方 `MatchResult`：

```python
MatchResult(
    trades=...,
    accepted_order_ids=...,
    rejected_order_ids=...,
    alerts=...,
    close_prices=...,
)
```

输入稳定性保护：

```python
try:
    result = self.exchange.submit_order(...)
except (TypeError, ValueError):
    rejected.append(order.order_id)
    continue
```

因此非法订单不会导致整轮评测崩溃，而是进入 `rejected_order_ids`。

## 2. 整体结构

`exchange.py` 中有三层：

```text
Order / Trade / Alert 数据结构
LimitOrderBook 撮合引擎
RegulatoryAgent 监管逻辑
ExchangeAgent 对外包装
```

调用流程：

```text
OrderRequest
  -> ExchangeAgent.submit_order()
  -> Order(...)
  -> RegulatoryAgent.pre_submit()
  -> LimitOrderBook.submit()
  -> RegulatoryAgent.on_trades()
  -> dict result
  -> submission.py 包装成 MatchResult
```

## 3. Order 数据结构

内部订单：

```python
Order(
    order_id,
    agent_id,
    symbol,
    side,
    price,
    quantity,
    timestamp,
    entity_id,
    visible=True,
    remaining=quantity,
)
```

校验：

```python
side in {"buy", "sell"}
quantity > 0
price > 0
```

如果 `entity_id is None`：

```python
entity_id = agent_id
```

`entity_id` 用于识别同一实益所有人或关联账户。

## 4. Trade 数据结构

成交记录：

```python
Trade(
    symbol,
    price,
    quantity,
    buy_order_id,
    sell_order_id,
    buyer_id,
    seller_id,
    timestamp,
)
```

成交价格使用挂单方价格。

## 5. LimitOrderBook 撮合机制

当前撮合引擎是标准限价订单簿。

### 5.1 订单簿状态

每个 symbol 一个 `LimitOrderBook`：

```python
self.buy: List[Order]
self.sell: List[Order]
self.orders: Dict[str, Order]
self.trades: List[Trade]
```

### 5.2 排序规则

买盘：

```python
self.buy.sort(key=lambda item: (-item.price, item.timestamp, item.order_id))
```

含义：

```text
价格高优先
同价时间早优先
再按 order_id 稳定排序
```

卖盘：

```python
self.sell.sort(key=lambda item: (item.price, item.timestamp, item.order_id))
```

含义：

```text
价格低优先
同价时间早优先
再按 order_id 稳定排序
```

### 5.3 成交条件

新买单：

```python
buy_price >= best_ask
```

新卖单：

```python
sell_price <= best_bid
```

如果不交叉，则停止撮合，剩余订单进入订单簿。

### 5.4 成交数量

```python
quantity = min(incoming.remaining, resting.remaining)
```

支持：

```text
完全成交
部分成交
一笔订单连续吃多档
未成交剩余挂簿
```

### 5.5 成交价格

```python
trade_price = resting.price
```

即使用被动挂单方价格。

### 5.6 撤单

`LimitOrderBook.cancel(order_id)` 会：

```text
从 self.orders 删除订单
从 buy 或 sell 列表中删除订单
返回被撤订单
```

当前正式 `match_orders()` 输入没有撤单请求，但 `ExchangeAgent.cancel_order()` 和监管 `on_cancel()` 已保留。

## 6. ExchangeAgent

### 6.1 状态

```python
self.books: Dict[str, LimitOrderBook]
self.regulator: RegulatoryAgent
self._ids = itertools.count(1)
self.order_snapshots: Dict[str, Order]
```

`books` 按 symbol 隔离，不同标的订单簿互不影响。

### 6.2 内部 order_id

内部订单 ID：

```python
order_id = f"O{next(self._ids)}"
```

注意：

```text
内部 order_id 和外部 OrderRequest.order_id 不同。
submission.py 返回 accepted/rejected 时使用外部 order_id。
TradeRecord 中的 buy_order_id/sell_order_id 使用内部 order_id。
```

### 6.3 提交流程

```python
order = Order(...)
book = self.books.setdefault(symbol, LimitOrderBook(symbol))
alert = self.regulator.pre_submit(order, book)
```

如果预提交监管返回 block 类 alert：

```python
if alert.action.startswith("block_"):
    return accepted=False
```

否则：

```python
保存 order snapshot
trades = book.submit(order)
alerts = regulator.on_trades(trades, snapshots)
return accepted=True
```

非拦截型 pre-submit alert 会和成交后 alerts 一起返回：

```python
"alerts": pre_alerts + trade_alerts
```

## 7. RegulatoryAgent 参数

默认参数：

```python
wash_window = 8
spoof_cancel_window = 3
large_order_ratio = 4.0
pump_window = 12
```

含义：

```text
wash_window: 反复互成交检测窗口
spoof_cancel_window: 快速撤单检测窗口
large_order_ratio: 大单阈值，订单量至少为参考深度的 4 倍
pump_window: pump-and-dump 滚动检测窗口
```

监管状态：

```python
events: 最近 submit/cancel 事件，最多 2000 条
open_orders: 当前开放订单
cancelled: 撤单记录
alerts: 历史预警
entity_trades: 最近成交记录，最多 2000 条
symbol_trade_windows: 每个 symbol 的成交窗口，最多 200 条
pump_alerted_symbols: 近期已报 pump 的 symbol，避免重复报警
```

## 8. Wash Trading 检测

### 8.1 提交前同实体自成交拦截

方法：

```python
_same_entity_cross_alert(order, book)
```

检查对手盘前 5 档：

```python
contra = book.sell if incoming is buy else book.buy
```

判断价格交叉：

```python
buy incoming: order.price >= resting.price
sell incoming: order.price <= resting.price
```

判断同实体：

```python
resting.entity_id == order.entity_id
```

触发条件：

```python
crosses and same_entity
```

输出 alert：

```python
Alert(
    alert_type="wash_trading",
    severity=0.98,
    action="block_order_and_freeze_entity",
)
```

因为 action 以 `block_` 开头，`ExchangeAgent` 会拒绝该订单。

### 8.2 成交后同实体检测

在 `on_trades()` 中：

```python
buyer_entity == seller_entity
```

触发：

```python
alert_type = "wash_trading"
severity = 0.98
action = "block_trade_and_freeze_entity"
```

### 8.3 反复互成交检测

在 `wash_window = 8` 内统计同一对手方组合：

```python
recent = trades where trade.timestamp >= current.timestamp - wash_window
pair_count = count({buyer_id, seller_id} == current_pair)
```

触发：

```python
pair_count >= 4
```

输出：

```python
alert_type = "wash_trading_ring"
severity = 0.84
action = "warn_and_sample_for_review"
```

## 9. Spoofing / Layering 检测

当前 demo 有两类 spoofing 检测：

```text
submit-stage 远离盘口大单检测
cancel-stage 快速撤单检测
```

### 9.1 Submit-stage 远离盘口大单

方法：

```python
_large_far_order_alert(order, book)
```

获取盘口：

```python
bid, ask = book.best_bid_ask()
```

计算深度：

```python
same_side_depth = average(book.depth(order.side, levels=5))
opposite_depth = average(book.depth(opposite_side, levels=5))
reference_depth = max(same_side_depth, opposite_depth, 1.0)
```

大单判定：

```python
is_large = order.quantity >= reference_depth * large_order_ratio
```

当前：

```python
large_order_ratio = 4.0
```

远离盘口：

```python
buy order:  order.price < ask * 0.92
sell order: order.price > bid * 1.08
```

即：

```text
买单价格比最优卖价低 8% 以上
卖单价格比最优买价高 8% 以上
```

不立即成交：

```python
not crosses
```

触发条件：

```python
is_large and far_from_touch and not crosses
```

输出：

```python
alert_type = "spoofing"
severity = 0.78
action = "monitor_or_throttle"
```

该 alert 不拦截订单，只预警。

### 9.2 Cancel-stage 快速撤单

方法：

```python
on_cancel(order, timestamp, book)
```

计算订单年龄：

```python
age = cancel_timestamp - order.timestamp
```

同实体近期撤单：

```python
same_entity_recent = cancels from same entity within spoof_cancel_window
```

大单：

```python
is_large = avg_depth == 0 or order.quantity >= avg_depth * large_order_ratio
```

触发条件：

```python
age <= spoof_cancel_window
and is_large
and len(same_entity_recent) >= 2
```

当前：

```python
spoof_cancel_window = 3
```

输出：

```python
alert_type = "spoofing"
severity = 0.91
action = "intervene_cancel_and_throttle"
```

注意：

```text
当前正式 match_orders 输入没有撤单请求，所以本地固定评测主要依赖 submit-stage spoofing。
```

## 10. Pump and Dump 检测

### 10.1 成交窗口记录

每笔成交后记录：

```python
{
    "timestamp": trade.timestamp,
    "price": trade.price,
    "quantity": trade.quantity,
    "buyer_entity": buyer_entity,
    "seller_entity": seller_entity,
}
```

按 symbol 分开保存：

```python
symbol_trade_windows[symbol]
```

每个 symbol 最多保留 200 条。

### 10.2 检测窗口

```python
recent = trades where timestamp >= current_timestamp - pump_window
```

当前：

```python
pump_window = 12
```

至少需要：

```python
len(recent) >= 4
```

### 10.3 价格拉升

```python
return_window = last_price / first_price - 1.0
```

触发阈值：

```python
return_window >= 0.025
```

即窗口内价格上涨至少 2.5%。

### 10.4 卖方集中度

统计每个 seller_entity 的成交量：

```python
seller_concentration = top_seller_qty / total_qty
```

触发阈值：

```python
seller_concentration >= 0.45
```

即最大卖方贡献至少 45% 成交量。

### 10.5 买方分散度

统计不同 buyer_entity 数量：

```python
distinct_buyers = len(buyer_volume)
```

触发阈值：

```python
distinct_buyers >= 3
```

### 10.6 多数价格上行

```python
mostly_rising = count(next_price >= prev_price) >= len(recent) - 2
```

允许少数不连续，但大多数成交价格应不下降。

### 10.7 最终触发条件

```python
if (
    return_window >= 0.025
    and seller_concentration >= 0.45
    and distinct_buyers >= 3
    and mostly_rising
):
    alert_type = "pump_and_dump"
```

输出：

```python
Alert(
    alert_type="pump_and_dump",
    severity=0.88,
    entity_id=top_seller,
    action="warn_and_sample_for_review",
)
```

### 10.8 重复告警抑制

如果同一 symbol 在 `pump_window` 内已经报过 pump：

```python
do not alert again
```

避免同一操纵事件重复刷 alert。

## 11. Alert 命名

当前 alert_type 使用明确关键词：

```text
wash_trading
wash_trading_ring
spoofing
pump_and_dump
```

原因：

```text
本地评测脚本通过关键词 wash / spoof / layer / pump / dump 判断是否检测到对应异常。
```

## 12. close_prices 更新

在 `submission.py` 中：

```python
close_prices = dict(last_prices)
```

每产生一笔成交：

```python
close_prices[item["symbol"]] = item["price"]
```

因此返回的 `MatchResult.close_prices` 反映每个 symbol 最新成交价；没有成交的 symbol 保留输入 `last_prices`。

## 13. 当前 demo 的已知局限

1. 撮合引擎较标准，但没有市场订单、撤单输入流、批量取消等复杂功能。
2. spoofing submit-stage 检测依赖大单和远离盘口，可能漏掉贴近盘口但意图虚假的 spoofing。
3. cancel-stage spoofing 逻辑保留了，但当前正式 `match_orders()` 没有撤单请求，所以很少触发。
4. Pump-and-dump 检测是短窗口规则，不是真正统计异常模型。
5. 没有账户图、净仓位变化、环形交易图搜索等更复杂 wash trading 检测。
6. 没有区分正常大宗挂单、做市补流动性和操纵挂单的更细证据链。
7. 阈值如 `4.0`、`0.92`、`1.08`、`0.025`、`0.45` 都是 demo 启发式参数，后续需要用更多正常/异常订单流调优。
