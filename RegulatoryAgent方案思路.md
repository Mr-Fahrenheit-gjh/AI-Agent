# RegulatoryAgent 方案思路记录

## 核心定位

RegulatoryAgent 的目标不是重写撮合引擎，而是在现有 Limit Order Book 之上识别并解释异常交易行为。

主要覆盖三类异常：

- Wash Trading: 自买自卖、关联账户对倒，制造虚假成交量。
- Spoofing / Layering: 大额或多层虚假订单制造盘口压力，随后撤单或不真实成交。
- Pump and Dump: 短窗口内拉高价格后集中卖出。

评价重点：

- 异常检测 Precision / Recall / F1。
- 响应延迟，越早 alert 越好。
- 正常订单误报控制。
- 每条 alert 要有清晰 thought。

## 总体方法

采用：

```text
规则检测 + 统计异常检测 + 账户图关系 + 可解释证据模板
```

不优先使用黑箱监督模型，原因：

- 比赛数据量有限，缺少大量标注样本。
- 三类异常都有明确订单流机制，可以用规则和统计特征稳定捕捉。
- 监管任务重视可解释性。
- 本地评测场景有限，规则法更稳定。

推荐拆成三个 detector：

```text
WashTradeDetector
SpoofingDetector
PumpDumpDetector
```

每个 detector 产生：

```text
risk_score
alert_type
severity
evidence
thought
```

## Wash Trading

### 核心机制

Wash Trading 的核心是没有真实 beneficial ownership 或 risk transfer，却制造了成交外观或成交量。

### 可落地特征

1. 单笔强规则

```text
incoming order 与 resting contra order 价格交叉
且 incoming_order.entity_id == resting_order.entity_id
```

这是最强信号，可以直接 block。

2. 成交后同 owner 检查

```text
buyer_entity == seller_entity
```

如果成交双方最终属于同一 beneficial owner，发 high severity alert。

3. 短窗口反复互成交

```text
count(trades between A and B within wash_window) >= threshold
```

可以标记为：

```text
wash_trading_ring
```

4. 高成交量、低净仓位变化

```text
net_position_change_ratio =
    abs(net_position_change) / total_traded_volume
```

如果成交量高但净仓位变化很低，则可疑。

### 推荐规则

```text
if incoming_crosses_resting and same_entity:
    block_order
    alert_type = "wash_trading"

elif trade_executed and buyer_entity == seller_entity:
    alert_type = "wash_trading"

elif repeated_pair_trades >= threshold and net_position_change_ratio is low:
    alert_type = "wash_trading_ring"
```

### Thought 模板

```text
Incoming order would match with a resting order from the same beneficial owner. This would create reported volume without genuine ownership transfer, so the exchange blocks the order.
```

```text
The same counterparties repeatedly traded with little net position change inside a short window, which is consistent with wash trading or coordinated volume creation.
```

## Spoofing / Layering

### 核心机制

Spoofing 的核心不是单纯大单，而是订单生命周期：

```text
提交大额可见订单
显著改变盘口深度或买卖不平衡
没有真实成交意图
短时间撤单或几乎不成交
同实体/关联实体可能在反方向成交
```

Layering 是同一实体在多个价位层级提交订单，制造流动性或压力假象。

### 可落地特征

1. 大单异常

```text
order_size_ratio = order.quantity / average_depth_same_side
```

或：

```text
order.quantity >= large_order_ratio * average_depth
```

2. 距离盘口

```text
distance_from_touch = abs(order.price - best_touch) / mid_price
```

远离盘口但数量巨大的订单更像展示压力，而不是真实成交需求。

3. 盘口冲击

```text
depth_impact = order.quantity / top_k_depth
```

4. 快速撤单

```text
cancel_latency = cancel_tick - submit_tick
```

5. 低成交率

```text
fill_ratio = executed_quantity / submitted_quantity
```

6. 撤单后反向成交

```text
same_entity trades opposite side within H ticks after cancel
```

7. 多层挂单

```text
same_entity places large orders across multiple price levels
```

### 分层检测

#### Level 1: submit 阶段可疑预警

```text
if large_order and far_from_touch and depth_impact_high:
    alert_type = "spoofing_suspicion" 或 "spoofing"
    severity = medium
    action = "monitor_or_throttle"
```

用途是抢响应延迟。当前本地评测没有撤单输入，因此必须在 `pre_submit()` 阶段识别远离盘口的大额订单。

#### Level 2: cancel 阶段确认

```text
if large_order and quick_cancel and low_fill:
    alert_type = "spoofing"
    severity = high
```

#### Level 3: 反向成交强化

```text
if quick_cancel and opposite_side_trade_after_cancel:
    severity = very_high
    action = "intervene_cancel_and_throttle"
```

### 推荐风险分

```text
S_spoof =
    f(order_size_abnormality)
  + f(depth_impact)
  + f(cancel_latency)
  + f(low_fill_ratio)
  + f(opposite_side_trade)
  + f(repeated_cancel_burst)
```

实际实现可先用分段规则，不必训练模型。

### Thought 模板

```text
Entity {entity_id} submitted an oversized order far from the touch that materially changed displayed depth without a plausible execution path. This pattern is consistent with spoofing or layering intent, so the exchange flags the order for throttling.
```

```text
Entity {entity_id} submitted oversized orders that materially changed displayed depth, then cancelled them within {age} ticks with little or no execution. This pattern is consistent with spoofing or layering intent.
```

## Pump and Dump

### 核心机制

Pump and Dump 是：

```text
短窗口内通过异常买入或协同订单流拉高价格
随后同一实体或关联实体集中卖出
```

### 订单流阶段

1. Accumulation / 协同建仓

```text
少数实体或关联账户在拉升前买入
```

2. Pump / 拉升

```text
短窗口价格快速上涨
成交量放大
主动买入压力上升
```

3. Dump / 出货

```text
价格拉升后，同一实体或关联实体集中卖出
```

### 可落地特征

1. 短窗口收益异常

```text
return_window = last_price / price_{t-k} - 1
```

2. 成交量异常

```text
window_volume 或 volume_zscore
```

3. 主动买入压力

```text
buy_pressure = buy_side_trade_volume / total_trade_volume
```

4. 买方集中度

```text
buyer_concentration = top_k_buyer_volume / total_buy_volume
```

5. 卖方集中度

```text
seller_concentration = top_k_seller_volume_after_pump / total_sell_volume_after_pump
```

6. 同组先买后卖

```text
entity buys during pump window and sells during dump window
```

### 推荐规则

#### Early Pump Alert

```text
if short_window_return high and volume high and buy_pressure high:
    alert_type = "pump_suspicion"
    severity = medium
```

#### Confirmed Pump and Dump

```text
if pump_suspicion_recent and seller_concentration_high and same_group_buy_then_sell:
    alert_type = "pump_and_dump"
    severity = high
```

### 适配本地评测的简化规则

当前本地 pump 场景近似为：

```text
连续多个 tick:
    同一 seller 挂出价格逐步升高的 sell order
    多个 buyer 用明显高价 buy order 吃单
```

因此可以先识别：

```text
同一 symbol 短窗口内成交价连续上升
同一 seller 高频卖出
多个 buyer 用明显高价吃单
```

触发：

```text
alert_type = "pump_and_dump"
```

### Thought 模板

```text
The symbol experienced abnormal short-window price expansion, while the same seller repeatedly supplied liquidity to multiple aggressive buyers. This concentrated sell-into-rising-prices pattern is consistent with pump-and-dump manipulation.
```

```text
The symbol experienced abnormal short-window price and volume expansion, followed by concentrated selling from entity {entity_id}. This buy-then-sell pattern is consistent with pump-and-dump manipulation.
```

## 通用监控框架

每个 tick 更新：

```text
order submissions
cancellations
trades
entity positions
entity pair graph
symbol price/volume windows
```

每个 detector 输出：

```text
score
evidence
alert
```

最终合并：

```text
deduplicate alerts
choose severity
generate thought
```

## 结合当前代码的扩展点

当前 `exchange.py` 已有：

- `RegulatoryAgent.pre_submit()`
- `_same_entity_cross_alert()`
- `on_cancel()`
- `on_trades()`
- `events`
- `open_orders`
- `cancelled`
- `entity_trades`
- `alerts`

推荐结构：

```text
RegulatoryAgent
|
|-- pre_submit(order, book)
|   |-- WashDetector.same_entity_cross
|   |-- SpoofDetector.large_far_depth_impact_prealert
|   `-- record open order
|
|-- on_cancel(order, timestamp, book)
|   |-- SpoofDetector.quick_cancel_low_fill
|   |-- SpoofDetector.layering_cancel_burst
|   `-- update cancellation history
|
`-- on_trades(trades, orders)
    |-- WashDetector.same_owner_trade
    |-- WashDetector.repeated_pair_or_cycle
    |-- PumpDumpDetector.update_price_volume_window
    |-- PumpDumpDetector.detect_pump_suspicion
    `-- PumpDumpDetector.detect_dump_confirmation
```

## 实现优先级

1. 保留 `LimitOrderBook`，不要重写撮合逻辑。
2. 在 `pre_submit()` 增加 submit-stage spoofing 检测。
3. 在 `on_trades()` 增加 pump-and-dump rolling window 检测。
4. 增强 wash trading：重复 pair、entity 净仓位、账户关系图。
5. 最后完善 `on_cancel()` 的 lifecycle spoofing / layering 检测。

## 本地评测适配提醒

本地评测检测 alert 是否存在时看关键词：

```text
wash
spoof 或 layer
pump 或 dump
```

因此 `alert_type` 必须直接包含这些词，例如：

```text
wash_trading
spoofing
layering
pump_and_dump
```

不要只写：

```text
market_abuse
abnormal_depth
manipulation
```

否则解释正确也可能不被评测脚本识别。

## 一句话总结

不要把监管系统做成“看到大单就报警”或“价格涨了就报警”。每个 alert 都应该有证据链：谁、在什么窗口、通过什么订单行为、制造了什么市场假象、为什么需要干预。先实现适配比赛接口的轻量规则版，再逐步增加账户图、净仓位和订单生命周期检测。
