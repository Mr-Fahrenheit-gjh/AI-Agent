# Demo 版本 InvestmentAgent 实现说明

本文档记录当前 `my_team` 目录中 demo 版本 `InvestmentAgent` 的具体实现。它描述的是当前代码实际怎么做，不代表最终理论最优方案。

对应文件：

```text
participants_package/my_team/submission.py
participants_package/my_team/investment_agent.py
```

## 1. 在 submission.py 中如何调用

评测系统调用：

```python
TeamSubmission.decide(observation)
```

当前 `submission.py` 做三件事：

1. 根据 `observation.agent_id` 获取或创建一个 `InvestmentAgent`。
2. 把 observation 中的现金、持仓、K 线、新闻和社交帖子同步给 Agent。
3. 调用 `agent.decide(symbol)`，再包装成官方 `AgentDecision`。

创建 Agent：

```python
personality = self.config.get("default_personality", "trend")
agent = InvestmentAgent(
    agent_id=observation.agent_id,
    personality=personality,
    cash=observation.cash,
    seed=self.seed + len(self.agents),
)
```

默认人格是：

```text
trend
```

每次决策前同步持仓：

```python
agent.cash = observation.cash

if observation.position > 0:
    agent.positions[observation.symbol] = Position(
        observation.position,
        observation.avg_cost,
    )
else:
    agent.positions.pop(observation.symbol, None)
```

这保证评测器传入的持仓状态优先，不依赖 Agent 自己的旧状态。

同步输入数据：

```python
agent.ingest_market(symbol, [kline.to_dict() for kline in observation.klines])
agent.ingest_news(symbol, observation.news)
agent.ingest_social(symbol, observation.social_posts)
```

## 2. Agent 内部状态

`InvestmentAgent` 保存：

```python
agent_id
personality
cash
positions
llm_client
seed
traits
beliefs
_market
_news
_social
memory
```

其中：

```text
_market: 每个 symbol 最近最多 90 根 K 线
_news: 每个 symbol 最近最多 50 条新闻文本
_social: 每个 symbol 最近最多 80 条社交帖子
memory: 最近最多 200 条决策记录
```

## 3. 人格参数

当前有四类人格：

```text
aggressive
value
trend
anxious
```

默认使用 `trend`：

```python
{
    "risk_appetite": 0.60,
    "loss_aversion": 1.35,
    "herding": 0.55,
    "overconfidence": 0.48,
    "disposition": 0.12,
    "turnover": 0.48,
}
```

当前 demo 实现里，实际直接影响计算的主要是：

```text
herding: 放大社交情绪
overconfidence: 参与 confidence 计算
```

其他参数保留在 traits 中，后续可用于更完整的人格建模。

## 4. 市场数据处理

### 4.1 ingest_market

K 线被标准化为：

```python
{
    "open": float(...),
    "high": float(...),
    "low": float(...),
    "close": float(...),
    "volume": float(...),
}
```

最多保留最近 90 根：

```python
self._market[symbol] = rows[-90:]
```

### 4.2 当前价格

```python
price = self.current_price(symbol)
```

即最后一根 K 线的：

```python
close
```

### 4.3 动量 momentum

在 `update_beliefs()` 中：

```python
short = mean(closes[-5:])
long = mean(closes[-20:]) if len(closes) >= 20 else mean(closes)
momentum = clip((short / long - 1.0) * 10.0, -1.0, 1.0)
```

含义：

```text
最近 5 根均价高于长期均价 -> momentum 为正
最近 5 根均价低于长期均价 -> momentum 为负
```

### 4.4 波动率 volatility

```python
volatility = clip(std(returns(closes[-20:])) * 20.0, 0.0, 1.0)
```

当前 demo 中，`volatility` 会进入 `Belief`，但没有直接进入最终 `evidence_score`。

### 4.5 近期收益 market_return

在 `_recent_return()` 中：

```python
lookback = closes[-6] if len(closes) >= 6 else closes[0]
market_return = clip((closes[-1] / lookback - 1.0) * 4.0, -1.0, 1.0)
```

含义：

```text
近 6 根 K 线涨幅越大，market_return 越正
近 6 根 K 线跌幅越大，market_return 越负
```

## 5. 新闻文本处理

当前 demo 使用词典匹配。

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

`+2` 是平滑项，避免少数关键词导致极端分数。

局限：

```text
这是非常初级的文本处理，适合 demo 和手写关键词测试，不适合充分理解 60 天包中的长中文宏观新闻。
```

## 6. 社交文本处理

如果帖子是字符串，则转成：

```python
{"text": post, "influence": 1.0}
```

每条帖子权重：

```python
weight = log1p(influence)
```

社交压力：

```python
social_pressure = clip(
    weighted_text_sentiment * (0.35 + self.traits["herding"]),
    -1.0,
    1.0,
)
```

默认 `trend.herding = 0.55`，因此放大系数为：

```text
0.35 + 0.55 = 0.90
```

注意：

```text
60 天匿名市场包没有社交数据，social_posts=[]，所以该模块在 60 天测试中基本不起作用。
```

## 7. Belief 构建

`Belief` 数据结构：

```python
Belief(
    fair_value,
    momentum,
    sentiment,
    volatility,
    confidence,
    social_pressure,
)
```

### 7.1 fair_value

```python
prior = previous belief fair_value or last price
fair_value = 0.82 * prior + 0.18 * last * (1.0 + 0.07 * sentiment)
```

含义：

```text
内部价值锚有惯性
最新价格和新闻情绪轻微修正价值锚
```

### 7.2 confidence

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

含义：

```text
新闻越多、K线越多、人格越自信，confidence 越高
```

### 7.3 Belief.score

类里保留了一个 `Belief.score`：

```python
Belief.score = (
    0.36 * sentiment
    + 0.28 * momentum
    + 0.22 * social_pressure
    - 0.14 * volatility
) * (0.55 + confidence)
```

但当前 demo 的最终决策不是直接使用 `Belief.score`，而是使用下面的 `evidence_score`。

## 8. Evidence Score

当前 demo 的核心信号：

```python
value_gap = clip((fair_value / price - 1.0) * 8.0, -1.0, 1.0)

evidence_score = clip(
    0.45 * sentiment
    + 0.35 * social_pressure
    + 0.35 * momentum
    + 0.15 * value_gap
    + 0.20 * market_return,
    -1.0,
    1.0,
)
```

解释：

```text
新闻情绪权重: 0.45
社交压力权重: 0.35
K线动量权重: 0.35
价值偏离权重: 0.15
近期收益权重: 0.20
```

局限：

```text
这些参数是 demo 启发式设置，不是经过文献或网格搜索严格优化的最终参数。
```

## 9. 持仓盈亏状态

当前浮动盈亏：

```python
unrealized = price / avg_cost - 1.0
```

状态阈值：

```python
strong_profit = unrealized >= 0.18
modest_profit = unrealized >= 0.05
meaningful_loss = unrealized <= -0.05
```

含义：

```text
浮盈 >= 18%: 大幅盈利，触发较强止盈心理
浮盈 >= 5%: 普通浮盈
浮亏 <= -5%: 有意义亏损，触发惜售心理
```

## 10. 决策表

### 10.1 空仓

```python
if not has_position:
    if evidence_score >= 0.20:
        action = "buy"
        belief_score = max(0.35, evidence_score)
    else:
        action = "hold"
        belief_score = 0.0
```

空仓看跌不会做空。

### 10.2 大幅盈利

```python
elif strong_profit and evidence_score <= 0.55:
    action = "sell"
    belief_score = min(-0.35, evidence_score - 0.45)
```

含义：

```text
大幅盈利时，只要正面证据不是特别强，就部分止盈。
```

### 10.3 普通浮盈

```python
elif modest_profit and evidence_score < -0.05:
    action = "sell"
    belief_score = min(-0.30, evidence_score)
```

含义：

```text
普通盈利状态下，证据转弱才卖。
```

### 10.4 浮亏

```python
elif meaningful_loss:
    if evidence_score <= -0.65:
        action = "sell"
        belief_score = min(-0.65, evidence_score)
    else:
        action = "hold"
        belief_score = 0.0
```

含义：

```text
亏损时不轻易卖，只有极强负面证据才止损。
```

### 10.5 普通持仓

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

## 11. 交易量和限价

### 11.1 买入

```python
equity = cash + position_qty * price
budget = min(cash, equity * 0.12)
quantity = lot_size(budget / price)
limit_price = price * 1.006
```

### 11.2 卖出

```python
target_qty = lot_size(equity * 0.10 / price)
quantity = min(position.quantity, max(100, target_qty))
quantity = lot_size(quantity)
limit_price = price * 0.994
```

### 11.3 手数和安全约束

手数：

```python
lot_size = 100
```

安全约束：

```python
买入不能超过现金
卖出不能超过持仓
quantity <= 0 时改成 hold
```

当数量归零时：

```python
action = "hold"
belief_score = 0.0
limit_price = current_price
```

## 12. sentiment_class

当前直接与动作绑定：

```python
sentiment_class = 1 if action == "buy" else -1 if action == "sell" else 0
```

优点：

```text
保证 sentiment_class 与 action 高度一致。
```

缺点：

```text
不够真实，偏向服务本地 Spearman 相关性指标。
```

## 13. Thought 生成

当前使用模板生成：

```text
{symbol} evidence feels bullish/mixed/bearish or profit-taking;
momentum ...;
news sentiment ...;
social pressure ...;
持仓盈亏心理;
therefore I choose {action}.
```

卖出止盈时会加入：

```text
my unrealized gain is X%, so I want to lock in part of it
```

浮亏 hold 时会加入：

```text
my position is down X%, and I dislike realizing the loss without stronger evidence
```

买入时会加入：

```text
the positive evidence and available cash make participation attractive
```

## 14. LLM 分支

当前代码保留 `_llm_decide()`，但 `submission.py` 没有注入 `llm_client`，所以默认不使用 LLM。

如果未来设置 `llm_client`，当前 `_llm_decide()` 会让 LLM 直接输出：

```json
{
  "action": "buy|sell|hold",
  "quantity": 100,
  "limit_price": 10.0,
  "thought": "...",
  "belief_score": 0.5,
  "sentiment_class": 1
}
```

并做基本安全约束：

```text
买入数量不超过 28% 现金预算
卖出数量不超过持仓
quantity <= 0 时改 hold
```

LLM 调用失败时，会临时关闭 `llm_client` 回退到规则决策。

当前建议：

```text
后续不应让 LLM 直接下单，而应改成 LLM cognition，即让 LLM 输出新闻理解、情绪、风险偏好解释，再由规则层执行。
```

## 15. 当前 demo 的已知局限

1. 新闻处理是词典匹配，无法充分理解中文宏观新闻。
2. 60 天包没有社交舆论数据，因此 social_pressure 在该数据中为 0。
3. 60 天包新闻是全市场新闻，不是个股新闻，当前新闻权重可能偏高。
4. K 线只用了动量、近期收益和波动率，没有使用成交量放大、突破、回撤、区间位置等特征。
5. 人格参数没有完整参与 BDI，只是部分参与计算。
6. `sentiment_class` 与动作绑定，有指标化倾向。
7. `evidence_score` 权重和阈值是启发式设置，不是最终理论最优。
