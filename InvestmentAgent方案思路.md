# InvestmentAgent 方案思路记录

## 核心定位

InvestmentAgent 不是收益最大化策略，也不是纯技术指标策略，而是一个受市场信息、新闻、社交情绪、持仓盈亏和人格影响的散户决策 Agent。

目标是让 `action`、`thought`、`belief_score`、`sentiment_class` 和持仓处境保持一致，并体现适度的散户行为偏差，尤其是处置效应。

## 总体架构

```text
MarketObservation
  -> 信息感知层
  -> LLM/规则认知层
  -> 散户心理层
  -> BDI 意图层
  -> 交易执行层
  -> AgentDecision
```

设计原则：

- LLM 负责把新闻、社交和市场状态转成结构化认知。
- 规则负责最终交易动作、交易量、价格和合法性过滤。
- LLM 不直接决定最终 `buy/sell/hold`，避免输出不稳定或违反现金/持仓约束。
- 必须有规则 fallback，保证无 LLM 或 LLM 失败时仍能稳定运行。

## 信息感知层

从 observation 中提取四类信息：

1. 市场走势
   - `trend_state`: up / down / flat
   - `return_state`: big_up / small_up / flat / small_down / big_down
   - `volatility_state`: low / medium / high
   - `volume_attention`: low / medium / high
   - `drawdown_state`: normal / drawdown / severe_drawdown

2. 新闻信息
   - `news_direction`: bullish / bearish / neutral / mixed
   - `news_strength`: weak / medium / strong
   - `news_event_type`: earnings / regulation / product / scandal / macro / other
   - `news_uncertainty`: low / medium / high

3. 社交信息
   - `social_direction`: bullish / bearish / neutral / mixed
   - `social_strength`: weak / medium / strong
   - `social_attention`: low / medium / high
   - `social_emotion`: FOMO / panic / calm / regret / overconfidence
   - `social_consensus`: high / low

4. 持仓心理
   - `position_state`: empty / holding
   - `pnl_state`: gain / loss / flat
   - `gain_level`: small_gain / large_gain
   - `loss_level`: small_loss / large_loss
   - `near_breakeven`: true / false
   - `position_pressure`: low / medium / high
   - `cash_pressure`: low / medium / high

## LLM/规则认知层

推荐让 LLM 输出结构化认知 JSON，而不是最终交易动作。

示例字段：

```json
{
  "news_direction": "bullish",
  "news_strength": "strong",
  "social_direction": "mixed",
  "social_strength": "medium",
  "market_attention": "high",
  "uncertainty": "medium",
  "retail_emotion": "FOMO",
  "dominant_opportunity": "positive news and strong attention",
  "dominant_risk": "high volatility and mixed social reactions",
  "belief_summary": "External information is mostly positive, but social disagreement and volatility create hesitation.",
  "suggested_intention": "lean_buy"
}
```

LLM 失败时，用词典和规则生成同样结构的认知结果。

## 散户心理层

至少建模五类散户心理机制：

1. 注意力驱动
   - 强新闻、社交热度、成交量放大、暴涨暴跌会提高行动倾向。

2. 羊群效应
   - 社交一致看多时更容易 FOMO。
   - 社交一致看空时更容易 panic。

3. 处置效应
   - 浮盈时更容易卖出或部分止盈。
   - 浮亏时更容易持有，不愿实现亏损。
   - 注意不要过强，目标 DE 是适度正值。

4. 回本心理
   - 接近成本价但仍亏损时倾向 hold。
   - 刚回本或小幅盈利时更容易犹豫或小幅卖出。

5. 过度交易约束
   - 弱信号 hold。
   - 中等信号小量交易。
   - 强信号中等交易。
   - 避免完全不交易，也避免高频大额刷单。

## BDI 落地

```text
Belief: 市场偏多、偏空、混合或不确定
Desire: 赚钱、避免亏损、锁定收益、跟随群体、控制风险
Intention: 倾向买、卖或持有
Action: 经过现金、持仓、换手过滤后的最终动作
```

Belief 建议采用有序状态，并映射到连续分数：

```text
strong_bullish -> 约 0.8 到 1.0
bullish        -> 约 0.3 到 0.7
neutral        -> 约 -0.2 到 0.2
bearish        -> 约 -0.7 到 -0.3
strong_bearish -> 约 -1.0 到 -0.8
```

为了 Spearman 相关性，最终动作要尽量和信念方向一致：

```text
buy  -> belief_score > 0, sentiment_class = 1
sell -> belief_score < 0, sentiment_class = -1
hold -> belief_score 接近 0, sentiment_class = 0
```

如果因为处置效应触发止盈卖出，应同步把 belief 调整为中性或轻微负面，避免 thought 看涨但 action 卖出。

## 状态依赖决策表

### 空仓

```text
strong_bullish -> buy
bullish        -> buy 或 hold
neutral        -> hold
bearish        -> hold
strong_bearish -> hold
```

### 持仓浮盈

```text
strong_bullish -> hold 或小幅 sell
bullish        -> hold 或小幅 sell
neutral        -> sell
bearish        -> sell
strong_bearish -> sell
```

### 持仓浮亏

```text
strong_bullish -> hold 或 buy
bullish        -> hold
neutral        -> hold
bearish        -> hold 或小幅 sell
strong_bearish -> sell
```

### 接近成本价

```text
bullish -> hold，等更高
neutral -> hold，想回本或刚回本后犹豫
bearish -> 小幅 sell 或 hold
```

## 交易执行层

合法性过滤必须保证：

- `action in {"buy", "sell", "hold"}`
- `belief_score in [-1, 1]`
- `sentiment_class in {-1, 0, 1}`
- `limit_price > 0`
- `quantity >= 0`
- 卖出数量不能超过持仓
- 买入金额不能超过现金或评测允许范围

交易量建议用档位控制：

```text
hold          -> 0
weak signal   -> 100
medium signal -> 100 到 200
strong signal -> 200 到 300
```

再根据现金和持仓裁剪：

```text
buy_qty = min(desired_qty, cash // limit_price)
sell_qty = min(desired_qty, current_position)
```

限价：

```text
buy  -> current_price * slightly_above
sell -> current_price * slightly_below
hold -> current_price
```

## Thought 模板

Thought 要解释四件事：

1. 看到了什么信息。
2. 当前信念和情绪是什么。
3. 持仓盈亏如何影响心理。
4. 为什么最终选择 buy/sell/hold。

示例：

```text
News and social posts are mostly bullish, and recent price action confirms the positive sentiment. Since I have enough cash and no heavy position, I feel pressure to participate rather than miss the move, so I choose buy.
```

```text
The signal is no longer clearly bullish, and I already have an unrealized gain. I feel tempted to lock in the profit before it disappears, so I choose sell.
```

```text
The market signal is weak or mixed, but my position is currently at a loss. I dislike realizing the loss now and prefer to wait for clearer evidence, so I choose hold.
```

```text
The news and social signals are bearish, but I do not currently hold the asset. Since selling is not feasible, I choose hold and avoid entering the position.
```

## 实现优先级

1. 先实现纯规则版，保证接口、合规和本地评测稳定。
2. 再增加结构化 LLM cognition，可开关、可 fallback。
3. 调 `belief_score/action/sentiment_class` 的一致性。
4. 调处置效应，目标是适度正 DE，而不是越大越好。
5. 调交易量和换手率，避免不交易或过度交易。

## 一句话总结

让 LLM 或规则做散户式认知解释，把新闻、社交和市场状态转成结构化信念；再用 BDI 决策表结合持仓盈亏、处置效应、人格和交易约束生成动作；最后用 thought 把信念、情绪、持仓心理和动作原因串起来。
