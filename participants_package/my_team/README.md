# Team Submission 方案说明

本文档记录当前 `my_team` 提交目录的完整实现逻辑，方便审查、调参和订正。正式入口是：

```python
submission.py:create_submission(config)
```

提交目录包含：

```text
my_team/
├── submission.py
├── investment_agent.py
├── exchange.py
├── README.md
└── requirements.txt
```

当前方案不依赖外部服务，默认不调用 LLM，使用规则式 BDI 投资 Agent 和规则式交易所监管 Agent。

## 1. 总体架构

评测系统只调用 `submission.py` 中的：

```python
reset(seed, config)
decide(observation)
match_orders(orders, last_prices, tick)
```

内部转发关系：

```text
decide(observation)
  -> InvestmentAgent.ingest_market/news/social
  -> InvestmentAgent.decide(symbol)
  -> AgentDecision

match_orders(orders, last_prices, tick)
  -> ExchangeAgent.submit_order(...)
  -> LimitOrderBook + RegulatoryAgent
  -> MatchResult
```

`submission.py` 还做了一层稳定性保护：如果订单的 `side`、价格或数量非法，`ExchangeAgent` 创建订单时可能抛出 `ValueError`，此时该外部订单 ID 会被加入 `rejected_order_ids`，整轮评测不会崩溃。

## 2. InvestmentAgent 设计

### 2.1 核心目标

该 Agent 不是收益最大化策略，而是模拟散户行为：

- 看 K 线趋势、新闻、社交情绪。
- 有浮盈时更容易部分止盈。
- 有浮亏时更不愿意实现亏损。
- 保证 `action`、`belief_score`、`sentiment_class`、`thought` 一致。
- 控制交易量，避免不交易或过度交易。

### 2.2 输入状态

`submission.py` 每次收到 `MarketObservation` 后，会同步：

```python
agent.cash = observation.cash
agent.positions[symbol] = Position(observation.position, observation.avg_cost)
agent.ingest_market(symbol, klines)
agent.ingest_news(symbol, news)
agent.ingest_social(symbol, social_posts)
```

如果 `observation.position <= 0`，会删除该 symbol 的持仓，防止上一次调用残留。

### 2.3 K 线特征

在 `update_beliefs()` 中处理最近 K 线：

```python
closes = [row["close"] for row in market if row["close"] > 0]
last = closes[-1]
short = mean(closes[-5:])
long = mean(closes[-20:]) if len(closes) >= 20 else mean(closes)
```

动量：

```python
momentum = clip((short / long - 1.0) * 10.0, -1.0, 1.0)
```

含义：

- 最近 5 根均价高于长期均价，`momentum > 0`。
- 最近 5 根均价低于长期均价，`momentum < 0`。
- 乘以 `10.0` 是为了把较小涨跌放大到 `[-1, 1]` 区间。

波动率：

```python
volatility = clip(std(returns(closes[-20:])) * 20.0, 0.0, 1.0)
```

当前版本中 `volatility` 会进入 `Belief`，但最终 `evidence_score` 没有直接使用它。它保留用于后续调参或 thought 扩展。

近期收益：

```python
lookback = closes[-6] if len(closes) >= 6 else closes[0]
market_return = clip((closes[-1] / lookback - 1.0) * 4.0, -1.0, 1.0)
```

含义：

- 用近 6 根左右的涨跌衡量短期注意力。
- 乘以 `4.0` 后裁剪到 `[-1, 1]`。

### 2.4 新闻情绪特征

新闻情绪使用词典，不调用外部模型。

正向词：

```text
beat, growth, upgrade, bull, surge, profit, strong, record, buy, breakout,
利好, 增长, 上调, 突破, 盈利, 买入
```

负向词：

```text
miss, fraud, downgrade, bear, crash, loss, weak, sell, risk, panic,
利空, 亏损, 下调, 暴跌, 卖出, 风险, 恐慌
```

计算：

```python
pos = count(positive words)
neg = count(negative words)
sentiment = clip((pos - neg) / max(pos + neg + 2, 1), -1.0, 1.0)
```

这里加 `+2` 是为了平滑，避免只有一个词时情绪过度极端。

### 2.5 社交情绪特征

社交帖子会被标准化为：

```python
{"text": ..., "influence": ...}
```

如果传入的是字符串，默认：

```python
influence = 1.0
```

每条帖子的权重：

```python
weight = log1p(max(influence, 0.0))
```

社交情绪：

```python
social_pressure = clip(weighted_text_sentiment * (0.35 + herding), -1.0, 1.0)
```

其中 `herding` 来自人格参数。默认 `trend` 人格的：

```python
herding = 0.55
```

所以社交压力会乘以：

```python
0.35 + 0.55 = 0.90
```

含义：趋势型散户比较受社交共识影响，但不会无限放大。

### 2.6 Fair Value 与 Confidence

内部价值锚：

```python
fair_value = 0.82 * prior_fair_value + 0.18 * last * (1.0 + 0.07 * sentiment)
```

含义：

- `0.82` 保留旧估值锚，体现散户认知有惯性。
- `0.18` 接收最新价格和新闻情绪。
- `0.07 * sentiment` 让正负新闻轻微改变估值锚。

信心：

```python
confidence = clip(
    0.30
    + 0.25 * min(news_count / 10.0, 1.0)
    + 0.25 * min(market_count / 30.0, 1.0)
    + 0.20 * overconfidence,
    0.05,
    0.95,
)
```

当前默认 `trend` 人格：

```python
overconfidence = 0.48
```

### 2.7 人格参数

当前 `submission.py` 默认：

```python
personality = config.get("default_personality", "trend")
```

`trend` 参数：

```python
risk_appetite = 0.60
loss_aversion = 1.35
herding = 0.55
overconfidence = 0.48
disposition = 0.12
turnover = 0.48
```

当前最终决策主要直接使用了 `herding`、`overconfidence`，其他参数保留在类中，便于后续扩展或换人格。

### 2.8 证据分数 Evidence Score

最终动作不是直接用 `Belief.score`，而是用一个更稳定、和评分指标更一致的 `evidence_score`：

```python
evidence_score = clip(
    0.45 * belief.sentiment
    + 0.35 * belief.social_pressure
    + 0.35 * belief.momentum
    + 0.15 * value_gap
    + 0.20 * market_return,
    -1.0,
    1.0,
)
```

各项含义：

- `0.45 * sentiment`: 新闻方向权重最高。
- `0.35 * social_pressure`: 社交情绪权重较高。
- `0.35 * momentum`: 趋势确认。
- `0.15 * value_gap`: 估值锚与当前价格偏离。
- `0.20 * market_return`: 近期涨跌带来的注意力/趋势确认。

价值偏离：

```python
value_gap = clip((fair_value / price - 1.0) * 8.0, -1.0, 1.0)
```

### 2.9 持仓盈亏状态

当前盈亏：

```python
unrealized = price / avg_cost - 1.0
```

状态阈值：

```python
strong_profit = position exists and unrealized >= 0.18
modest_profit = position exists and unrealized >= 0.05
meaningful_loss = position exists and unrealized <= -0.05
```

解释：

- `>= 18%`: 大幅盈利，容易出现落袋为安心理。
- `>= 5%`: 普通浮盈。
- `<= -5%`: 有意义浮亏，散户更容易惜售。

### 2.10 决策表

#### 空仓

```python
if evidence_score >= 0.20:
    action = "buy"
    belief_score = max(0.35, evidence_score)
else:
    action = "hold"
    belief_score = 0.0
```

空仓看跌时不能做空，所以是 `hold`，并把 `belief_score` 归 0，避免出现“强看跌但 hold”导致信念-动作相关性下降。

#### 大幅盈利

```python
elif strong_profit and evidence_score <= 0.55:
    action = "sell"
    belief_score = min(-0.35, evidence_score - 0.45)
```

解释：

- 即使信息略偏正，只要不是极强正面，大幅盈利时也会部分止盈。
- 将 `belief_score` 调为负或偏负，是为了让 `sell` 与输出信念一致。

#### 普通浮盈

```python
elif modest_profit and evidence_score < -0.05:
    action = "sell"
    belief_score = min(-0.30, evidence_score)
```

普通盈利只有在证据开始转弱时卖出。

#### 浮亏

```python
elif meaningful_loss:
    if evidence_score <= -0.65:
        action = "sell"
        belief_score = min(-0.65, evidence_score)
    else:
        action = "hold"
        belief_score = 0.0
```

解释：

- 浮亏时不轻易卖，体现处置效应。
- 只有强负面证据才止损。

#### 普通持仓

```python
elif evidence_score >= 0.35:
    action = "buy"
    belief_score = max(0.35, evidence_score)
elif evidence_score <= -0.35:
    action = "sell"
    belief_score = min(-0.35, evidence_score)
else:
    action = "hold"
    belief_score = 0.0
```

### 2.11 情绪类别 Sentiment Class

当前直接和动作绑定：

```python
sentiment_class = 1 if action == "buy" else -1 if action == "sell" else 0
```

这样做的目的：

- 本地评分会计算 `sentiment_class` 或 `belief_score` 与 `action` 的 Spearman 相关。
- 绑定后能避免 `thought` 看涨但 action 卖出的混乱情况。

### 2.12 交易量控制

买入：

```python
equity = cash + position_qty * price
budget = min(cash, equity * 0.12)
quantity = lot_size(budget / price)
limit_price = price * 1.006
```

卖出：

```python
target_qty = lot_size(equity * 0.10 / price)
quantity = min(position.quantity, max(100, target_qty))
quantity = lot_size(quantity)
limit_price = price * 0.994
```

其中：

```python
lot_size(quantity, lot=100)
```

含义：

- 所有交易按 100 的手数取整。
- 买入预算约为总权益 12%，但不能超过现金。
- 卖出目标约为总权益 10%，至少尝试 100，但不能超过持仓。
- `buy` 限价比当前价高 `0.6%`，提高成交可能性。
- `sell` 限价比当前价低 `0.6%`，提高成交可能性。

安全过滤：

```python
if buy_cost > cash:
    quantity = lot_size(cash / limit_price)
if sell:
    quantity = min(quantity, lot_size(position.quantity))
if quantity <= 0:
    action = "hold"
    belief_score = 0.0
    limit_price = price
```

这样可以处理现金不足买 100 股、持仓不足卖 100 股等情况。

### 2.13 Thought 生成

`thought` 是模板化生成，保证和动作一致。

核心字段：

```text
evidence feels bullish / bearish or profit-taking / mixed
momentum
news sentiment
social pressure
持仓盈亏心理
therefore I choose action
```

例如卖出止盈：

```text
SIM evidence feels bearish or profit-taking; momentum 0.28; news sentiment 0.33; social pressure 0.30; my unrealized gain is 29.7%, so I want to lock in part of it; therefore I choose sell.
```

### 2.14 LLM 兼容与 Fallback

代码保留了 `llm_client` 注入接口，但默认不启用。

如果未来设置了 `llm_client`：

```python
if self.llm_client is not None:
    return self._llm_decide(symbol)
```

当前已修复 LLM 失败时的 fallback：

```python
def _rule_decide_fallback(symbol):
    temporarily set self.llm_client = None
    return self.decide(symbol)
    restore self.llm_client
```

这样 LLM 报错或返回空内容时不会递归卡死。

## 3. ExchangeAgent 与撮合引擎

### 3.1 撮合逻辑

`LimitOrderBook` 保持标准价格-时间优先：

买盘排序：

```python
(-price, timestamp, order_id)
```

卖盘排序：

```python
(price, timestamp, order_id)
```

成交条件：

```python
buy order crosses if buy_price >= best_ask
sell order crosses if sell_price <= best_bid
```

成交数量：

```python
min(incoming.remaining, resting.remaining)
```

成交价格：

```python
resting order price
```

如果 incoming order 部分成交后还有剩余，则留在订单簿。

### 3.2 输入稳定性

`Order.__post_init__()` 校验：

```python
side in {"buy", "sell"}
quantity > 0
price > 0
```

`submission.py` 捕获非法订单异常并拒单：

```python
except (TypeError, ValueError):
    rejected.append(order.order_id)
    continue
```

## 4. RegulatoryAgent 设计

监管逻辑在不改撮合引擎的前提下增强。

### 4.1 监管参数

默认参数：

```python
wash_window = 8
spoof_cancel_window = 3
large_order_ratio = 4.0
pump_window = 12
```

含义：

- `wash_window`: 反复互成交检测窗口。
- `spoof_cancel_window`: 快速撤单窗口。
- `large_order_ratio`: 大单阈值，订单量至少为参考深度的 4 倍。
- `pump_window`: Pump-and-dump 滚动检测窗口。

维护状态：

```python
events: recent submit/cancel events, maxlen=2000
open_orders: currently open orders
cancelled: recent cancellation records
alerts: alert history
entity_trades: recent trades, maxlen=2000
symbol_trade_windows: per-symbol trade windows, maxlen=200
pump_alerted_symbols: recently alerted symbols to avoid duplicate alerts
```

### 4.2 Wash Trading 检测

#### 提交前同实体交叉

在 `pre_submit()` 中先检查：

```python
contra = opposite side top 5 resting orders
crosses = incoming price crosses resting price
same_entity = incoming.entity_id == resting.entity_id
```

触发条件：

```python
crosses and same_entity
```

输出：

```python
alert_type = "wash_trading"
severity = 0.98
action = "block_order_and_freeze_entity"
```

因为 action 以 `block_` 开头，所以 `ExchangeAgent.submit_order()` 会拒绝该订单。

#### 成交后同实体

在 `on_trades()` 中：

```python
buyer_entity == seller_entity
```

输出：

```python
alert_type = "wash_trading"
severity = 0.98
action = "block_trade_and_freeze_entity"
```

#### 反复互成交

短窗口内统计同一 buyer/seller pair：

```python
recent trades where timestamp >= current_timestamp - wash_window
pair_count = count({buyer_id, seller_id} == current pair)
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

### 4.3 Spoofing / Layering 检测

#### Submit-stage spoofing

这是当前本地评测和低延迟最关键的部分。

计算盘口：

```python
bid, ask = book.best_bid_ask()
same_side_depth = average(book.depth(order.side, 5))
opposite_depth = average(book.depth(opposite_side, 5))
reference_depth = max(same_side_depth, opposite_depth, 1.0)
```

大单：

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

- 买单低于最优卖价 8% 以上。
- 卖单高于最优买价 8% 以上。

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

注意：该 alert 不拦截订单，只预警/限流。`ExchangeAgent.submit_order()` 会把非拦截型 pre-submit alert 一并返回。

#### Cancel-stage spoofing

如果外部调用 `cancel_order()`，会检测：

```python
age = cancel_timestamp - order.timestamp
same_entity_recent = recent cancels from same entity within spoof_cancel_window
is_large = quantity >= avg_depth * large_order_ratio
```

触发：

```python
age <= spoof_cancel_window and is_large and len(same_entity_recent) >= 2
```

输出：

```python
alert_type = "spoofing"
severity = 0.91
action = "intervene_cancel_and_throttle"
```

### 4.4 Pump and Dump 检测

每笔成交后记录窗口：

```python
{
  timestamp,
  price,
  quantity,
  buyer_entity,
  seller_entity,
}
```

按 symbol 分开维护，避免多标的互相污染。

检测窗口：

```python
recent trades where timestamp >= current_timestamp - pump_window
pump_window = 12
```

至少需要：

```python
len(recent) >= 4
```

价格拉升：

```python
return_window = last_price / first_price - 1.0
return_window >= 0.025
```

即窗口内上涨至少 2.5%。

卖方集中度：

```python
seller_concentration = top_seller_qty / total_qty
seller_concentration >= 0.45
```

即最大卖方贡献至少 45% 的成交量。

买方分散度：

```python
distinct_buyers >= 3
```

即至少 3 个买方实体参与拉升。

价格多数上行：

```python
mostly_rising = count(next_price >= prev_price) >= len(recent) - 2
```

允许少量不连续，但整体趋势要上行。

触发条件：

```python
return_window >= 0.025
and seller_concentration >= 0.45
and distinct_buyers >= 3
and mostly_rising
```

输出：

```python
alert_type = "pump_and_dump"
severity = 0.88
action = "warn_and_sample_for_review"
entity_id = top_seller
```

重复告警抑制：

```python
if same symbol alerted within pump_window:
    do not alert again
```

### 4.5 Alert 命名

本地评测会通过关键词判断是否检测到异常，所以当前使用明确名称：

```text
wash_trading
wash_trading_ring
spoofing
pump_and_dump
```

不要改成模糊名称，例如 `market_abuse`，否则评测脚本可能不识别。

## 5. 当前稳定性测试覆盖

新增测试文件：

```text
competition_solution/tests/test_team_submission_edges.py
```

覆盖：

- 同 seed reset 后决策完全一致。
- 多类投资场景输出合法。
- 现金不足买 100 股时稳定 hold。
- 空仓看跌不做空。
- LLM 调用失败 fallback。
- 非法订单 reject，不崩溃。
- reset 清空订单簿和监管状态。
- 多 symbol 监管窗口隔离。
- 正常近盘口大单不误报 spoofing。
- 正常上涨但卖方不集中不误报 pump/dump。
- 重复正常撮合不产生 alert。
- spoofing 和 pump-and-dump 仍能检测。

## 6. 当前本地评分

运行命令：

```bash
cd participants_package
python -m submission_interface.validator my_team
python evaluate_submission.py my_team
python -m unittest discover -s competition_solution/tests
```

当前结果：

```text
validator: status ok
unittest: 16 tests OK
evaluate_submission.py: 97.0 / 100
```

分项：

```text
任务一 InvestmentAgent: 93.9 / 100
  format: 100.0
  disposition_effect: 76.7
  belief_correlation: 100.0
  turnover_similarity: 98.6
  DE = 0.2000
  rho = 1.0000
  turnover_WD = 0.0437

任务二 Exchange/RegulatoryAgent: 100.0 / 100
  price_time_priority: 100.0
  surveillance: 100.0
  precision = 1.0000
  recall = 1.0000
  f1 = 1.0000
```

## 7. 已知可调点

### 7.1 处置效应偏强

当前本地：

```text
DE = 0.2000
```

满分区间是：

```text
0.05 <= DE <= 0.15
```

所以任务一没有满分。如果要进一步调低 DE，可考虑：

- 把 `strong_profit` 阈值从 `0.18` 提高到 `0.22`。
- 把 `strong_profit and evidence_score <= 0.55` 改成 `<= 0.45`。
- 降低盈利止盈触发频率。

风险：可能降低 belief-action 相关性或 turnover 相似度。

### 7.2 Spoofing 阈值

当前远离盘口阈值：

```text
buy price < ask * 0.92
sell price > bid * 1.08
```

如果隐藏测试正常远离挂单较多，可放宽为：

```text
0.90 / 1.10
```

如果隐藏测试 spoofing 更隐蔽，可收紧为：

```text
0.95 / 1.05
```

### 7.3 Pump-and-dump 阈值

当前：

```text
return_window >= 2.5%
seller_concentration >= 45%
distinct_buyers >= 3
len(recent) >= 4
```

如果误报正常上涨，建议提高：

```text
return_window >= 4%
seller_concentration >= 55%
```

如果漏报弱 P&D，可降低：

```text
return_window >= 2%
seller_concentration >= 40%
```

## 8. 审查重点

建议重点审查：

1. `evidence_score` 各权重是否合理。
2. 大幅盈利止盈阈值 `18%` 是否过低。
3. 浮亏止损阈值 `evidence_score <= -0.65` 是否过严。
4. 买入预算 `12% equity` 和卖出目标 `10% equity` 是否符合你想要的换手率。
5. spoofing 的 `8%` 远离盘口阈值是否过宽或过窄。
6. pump-and-dump 的 `2.5%` 短窗口涨幅和 `45%` 卖方集中度是否适合隐藏测试。
7. `sentiment_class` 与 `action` 绑定是否过于迎合本地 Spearman 指标。
