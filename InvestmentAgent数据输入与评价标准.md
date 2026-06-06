# InvestmentAgent 数据输入与评价标准

本文档只讨论任务一 `InvestmentAgent`，用于梳理当前项目中 Agent 能看到什么数据、哪些数据是真实长周期数据、哪些只是本地测试脚本构造的信号，以及各项评价指标如何计算。

## 1. InvestmentAgent 的官方接口输入

评测系统调用：

```python
decision = submission.decide(observation)
```

其中 `observation` 是 `MarketObservation`，定义在：

```text
participants_package/submission_interface/api.py
```

字段如下：

```python
MarketObservation(
    agent_id: str,
    tick: int,
    symbol: str,
    klines: List[KLine],
    news: List[str],
    social_posts: List[Dict[str, Any]],
    cash: float,
    position: int,
    avg_cost: float,
    extra: Dict[str, Any],
)
```

### 1.1 KLine 字段

每根 K 线是：

```python
KLine(
    symbol: str,
    timestamp: str,
    open: float,
    high: float,
    low: float,
    close: float,
    volume: float,
)
```

也就是说 Agent 至少能看到：

```text
开盘价 open
最高价 high
最低价 low
收盘价 close
成交量 volume
```

当前项目里没有盘口深度、逐笔成交、买卖方向成交量等更细数据传给 `InvestmentAgent`。

### 1.2 新闻字段

```python
news: List[str]
```

含义是当前决策点可见的新闻摘要或新闻文本。

注意：不同测试包中的新闻质量不同：

- 原 `evaluate_submission.py` 中是手写关键词短句。
- 60 天包中是较长的中文宏观/市场新闻。

### 1.3 社交字段

```python
social_posts: List[Dict[str, Any]]
```

题目接口支持社交帖子，例如：

```python
{"text": "bullish breakout buy", "influence": 8}
```

但当前 60 天数据包没有真实社交舆论数据，评测器传入的是：

```python
social_posts=[]
```

所以社交模块需要兼容，但不能作为当前 60 天数据调参的主要依据。

### 1.4 持仓字段

```python
cash: 当前现金
position: 当前持仓数量
avg_cost: 当前持仓平均成本
```

这三个字段非常关键，因为处置效应、回本心理、止盈/惜售都依赖当前持仓盈亏。

当前浮动盈亏可计算为：

```python
unrealized_return = current_price / avg_cost - 1.0
```

其中：

```python
current_price = observation.klines[-1].close
```

## 2. 原始本地评测 evaluate_submission.py 的输入

文件：

```text
participants_package/evaluate_submission.py
```

任务一使用 `build_task1_observations()` 构造 6 个固定场景。

### 2.1 这 6 个场景是什么

场景包括：

```text
1. 盈利持仓 + 看涨文本
2. 亏损持仓 + 看跌文本
3. 空仓 + 看涨文本
4. 空仓 + 看跌文本
5. 中性震荡行情
6. 大幅盈利 + 止盈压力
```

每个场景都手写：

```text
K 线 close 序列
news 短句
social_posts 短句
cash
position
avg_cost
```

例如盈利看涨：

```python
closes = [100, 101, 103, 105, 108, 110, 112]
news = ["earnings upgrade, strong growth, breakout buying"]
social_posts = [{"text": "bullish breakout buy", "influence": 8}]
cash = 80_000
position = 200
avg_cost = 96.0
```

此时：

```text
last_price = 112
unrealized_return = 112 / 96 - 1 = 16.7%
```

例如亏损看跌：

```python
closes = [112, 110, 107, 103, 99, 95, 92]
news = ["lawsuit risk, downgrade, liquidity stress"]
social_posts = [{"text": "panic sell, avoid", "influence": 9}]
cash = 80_000
position = 200
avg_cost = 108.0
```

此时：

```text
last_price = 92
unrealized_return = 92 / 108 - 1 = -14.8%
```

### 2.2 这些文本算不算真实新闻/社交

严格来说不算。

原始本地评测中的 `news` 和 `social_posts` 是手写关键词信号，例如：

```text
growth upgrade breakout buy
panic sell avoid
take profit soon
```

它们更像是告诉 Agent 当前场景偏多或偏空的标签文本，不是真实新闻数据，也不是真实社交舆论数据。

### 2.3 原始本地评测有没有真实交易

没有。

原 `evaluate_submission.py` 只调用：

```python
decision = submission.decide(observation)
```

然后记录动作、信念、换手率和 DE 样本。它不会真正执行买卖，也不会更新组合现金和持仓。

也就是说：

```text
底仓是评测器手写给定的
你的 buy/sell 不会影响下一轮 observation
没有真实组合收益率
没有夏普率、最大回撤、最终净值等收益指标
```

## 3. 60 天匿名市场数据包的输入

新增数据包路径可能位于：

```text
market_60d_package/
```

或：

```text
participants_package/new_data/market_60d_package/
```

核心文件：

```text
data/market_60d.csv
data/calendar_60d.csv
data/profile_60d.csv
data/news_60d.jsonl
data/metadata.json
evaluate_60d_submission.py
```

### 3.1 数据规模

`metadata.json` 中说明：

```text
train_days: 60
symbols: 10
market_rows: 600
news_days: 55
news_items: 437
```

即：

```text
10 个匿名标的
60 天行情
共 600 行行情
55 天有新闻
437 条新闻
```

### 3.2 market_60d.csv

行情格式：

```text
symbol, day, open, high, low, close, volume, name
```

示例：

```text
STOCK_001, 第1天, 9.65, 9.81, 9.65, 9.81, 7952096.18, 匿名标的001
```

每个标的每天一行。

### 3.3 news_60d.jsonl

新闻格式：

```json
{"day": "第1天", "news": ["新闻1", "新闻2", "..."]}
```

这些新闻是真实长度更接近新闻摘要的中文文本，主要是：

```text
宏观政策
央行操作
美联储
市场情绪
经济数据
监管政策
```

重要注意：

```text
这些新闻是按天给全市场的，不是逐个股票单独给的。
```

也就是说，同一天：

```text
STOCK_001 看到同一批新闻
STOCK_002 也看到同一批新闻
...
STOCK_010 也看到同一批新闻
```

因此，60 天包里的新闻更适合理解为：

```text
市场整体风险偏好 / 宏观情绪 / 政策背景
```

不应简单当作个股专属利好/利空。

### 3.4 60 天包有没有社交舆论

没有。

`evaluate_60d_submission.py` 构造 observation 时：

```python
social_posts=[]
```

所以在 60 天包中：

```text
social_pressure 恒为 0
herding 参数基本不起作用
社交模块不能作为主要调参依据
```

## 4. 60 天评测中的交易机制

60 天评测器和原 6 场景不同，它会模拟组合交易。

### 4.1 初始状态

默认：

```python
initial_cash = 100_000.0
portfolio.holdings = {}
```

也就是说：

```text
一开始没有底仓
只有 100000 现金
```

### 4.2 每天如何调用 Agent

评测器按天遍历，每个 symbol 维护自己的 K 线历史。

当历史长度足够后，构造：

```python
MarketObservation(
    agent_id=f"agent_{symbol}",
    tick=tick,
    symbol=symbol,
    klines=list(history),
    news=当天新闻[:max_news],
    social_posts=[],
    cash=portfolio.cash,
    position=holding.quantity,
    avg_cost=holding.avg_cost,
    extra={
        "day": day,
        "stock_name": name,
        "news_count": len(news),
    },
)
```

然后调用：

```python
decision = submission.decide(observation)
```

### 4.3 成交价格

60 天包不走交易所撮合，不调用 `match_orders()`。

如果 `decision.action == "buy"` 或 `"sell"`，成交价格直接使用当天该标的：

```python
price = close
```

也就是：

```text
用当天收盘价模拟成交
```

### 4.4 买入执行

买入时：

```python
max_quantity = cash / (price * (1 + fee_rate))
executed = min(decision.quantity, max_quantity)
```

默认手续费：

```python
fee_rate = 0.0003
```

买入后：

```python
notional = executed * price
fee = notional * fee_rate
avg_cost = (old_avg_cost * old_qty + notional + fee) / new_total_qty
cash -= notional + fee
```

### 4.5 卖出执行

卖出时：

```python
executed = min(decision.quantity, holding.quantity)
notional = executed * price
fee = notional * fee_rate
cash += notional - fee
holding.quantity -= executed
```

如果持仓卖完，则删除该 symbol 的 holding。

### 4.6 组合价值

组合价值：

```python
portfolio_value = cash + sum(position_qty * latest_price)
```

但当前评测没有把最终收益率作为主要评分项。组合价值主要用于：

```text
计算换手率
更新真实持仓和成本
形成真实 DE 样本
```

## 5. InvestmentAgent 的评价指标

当前任务一主要评价：

```text
格式合法性
处置效应
信念-动作相关性
换手率分布相似度
60 天包额外有跨标的稳定性和动作多样性
```

## 6. 格式合法性 Format

评测会检查：

```text
返回值必须是 AgentDecision
action 必须是 buy / sell / hold
thought 必须存在且长度足够
belief_score 必须在 [-1, 1]
sentiment_class 必须是 -1 / 0 / 1
limit_price 必须大于 0
quantity 必须非负
sell 数量不能超过当前 position
buy 金额不能严重超过现金/权益
```

原始本地评测中买入约束为：

```python
notional <= max(cash, equity) * 1.20
```

其中：

```python
notional = decision.limit_price * quantity
equity = cash + position * last_price
```

## 7. 处置效应 DE

### 7.1 定义

处置效应衡量散户是否倾向：

```text
卖出盈利资产
持有亏损资产
```

核心指标：

```text
PGR = Realized Gains / Possible Gains
PLR = Realized Losses / Possible Losses
DE = PGR - PLR
```

### 7.2 在代码中如何判断盈利/亏损

`metrics.py` 中：

```python
is_gain = price >= avg_cost
```

如果当前价格高于成本：

```text
这是 possible gain
```

如果当前价格低于成本：

```text
这是 possible loss
```

如果 action 是 `sell`：

```text
盈利状态卖出 -> realized gain
亏损状态卖出 -> realized loss
```

### 7.3 DE 例子

假设：

```text
盈利机会出现 10 次，卖出 3 次
亏损机会出现 10 次，卖出 1 次
```

则：

```text
PGR = 3 / 10 = 0.30
PLR = 1 / 10 = 0.10
DE = 0.20
```

这表示更愿意卖盈利，不愿卖亏损。

### 7.4 DE 评分

本地评分逻辑：

```text
0.05 <= DE <= 0.15 -> 满分
DE <= 0            -> 0 分
DE < 0.05         -> 按比例给分
0.15 < DE <= 0.30 -> 逐步扣分
DE > 0.30         -> 低分
```

也就是说：

```text
没有处置效应不行
处置效应过强也不行
```

### 7.5 原 6 场景和 60 天包的 DE 样本区别

原 6 场景：

```text
每个 observation 都会形成一个 DE 样本
底仓由评测器手写给定
不会真实执行交易
```

60 天包：

```text
只有 position > 0 时才形成 DE 样本
position 和 avg_cost 是你的策略真实买出来的
卖出与否会影响后续持仓
```

## 8. 信念-动作相关性

动作映射：

```text
buy  -> 1
hold -> 0
sell -> -1
```

信号来源：

```python
signal = sentiment_class
if signal == 0:
    signal = sign(belief_score)
```

然后计算：

```text
Spearman rank correlation(signal, action)
```

含义：

```text
看多时买
中性时 hold
看空时卖
```

如果 thought / belief / sentiment 和 action 矛盾，会降低这个指标。

## 9. 换手率 Turnover

每次决策的换手率：

```text
turnover = traded_notional / portfolio_equity
```

原 6 场景中：

```python
traded_notional = decision.limit_price * quantity
equity = cash + position * last_price
```

60 天包中：

```python
traded_notional = 实际成交 notional
equity_before = 交易前组合价值
```

然后与参考分布比较：

```python
BASELINE_TURNOVER = [0.08, 0.10, 0.12, 0.07, 0.09, 0.11]
```

使用：

```text
Wasserstein Distance
```

距离越小，说明交易频率越接近参考散户行为。

## 10. 60 天包额外指标

60 天包新增：

```text
cross_symbol_stability
action_diversity
```

### 10.1 Cross-symbol stability

逐标的计算：

```text
每个 symbol 的 DE
每个 symbol 的 belief-action rho
每个 symbol 的 turnover_WD
```

然后取平均，衡量多标的表现是否稳定。

如果某些标的 DE 很负，或某些标的几乎不交易，会拖低该指标。

### 10.2 Action diversity

统计整体动作比例：

```text
buy_rate
sell_rate
hold_rate
active_symbol_rate
```

惩罚规则：

```text
hold_rate > 0.90       -> too_many_holds
buy_rate < 0.03        -> too_few_buys
sell_rate < 0.03       -> too_few_sells
active_symbol_rate < 0.70 -> too_few_active_symbols
```

## 11. 当前没有的指标

当前本地评测没有直接计算：

```text
最终收益率
超额收益
夏普率
最大回撤
卖飞
止损后反弹
买入后下跌
真实交易成本冲击
```

60 天包虽然模拟组合，但收益不是核心评分项。

## 12. 卖飞指标是什么，当前有没有

当前没有官方卖飞指标。

如果要自己加，可以定义：

```text
卖出后未来 N 天最高价继续上涨的幅度
```

例如：

```python
missed_upside_5d = max(close[t+1:t+6]) / sell_price - 1.0
```

可以统计：

```text
平均卖飞幅度
卖飞次数
卖飞率：missed_upside_5d > 5% 的卖出占比
```

也可以判断止盈是否合理：

```text
卖出后价格下跌 -> 止盈合理
卖出后继续大涨 -> 可能卖飞
```

这个指标不属于当前官方本地评分，但对改进策略很有用。

## 13. 当前数据对建模的限制

### 13.1 原 6 场景文本太简单

原 `evaluate_submission.py` 中的新闻和社交只是关键词信号，不是真实文本数据。

它适合测试接口和基本方向，不足以支撑复杂 NLP 结论。

### 13.2 60 天包没有社交数据

60 天包传入：

```python
social_posts=[]
```

因此社交舆论模块无法在该数据包中真实验证。

### 13.3 60 天新闻是全市场新闻

60 天包中的新闻是按天给全市场的，非个股专属。

所以新闻更适合作为：

```text
市场风险偏好
宏观政策背景
流动性情绪
```

不应被当作每只股票的直接利好/利空。

### 13.4 K 线是区分标的的主要信息

因为同一天 10 个标的看到的是同一批新闻，真正区分不同 symbol 的主要是：

```text
各自 open/high/low/close/volume
各自 position
各自 avg_cost
```

因此后续优化 InvestmentAgent 时，应该重点改进 K 线特征和持仓心理，而不是过度依赖新闻或社交。

## 14. 当前优化方向

针对 InvestmentAgent，后续建议：

```text
1. 用 60 天包作为主要调参集，避免只过拟合 6 个手写场景。
2. 加强 K 线特征：volume_spike、breakout、drawdown、range_position、volatility_risk。
3. 把新闻作为宏观风险偏好，而不是个股直接信号。
4. 保留社交接口，但当前不要依赖社交调参。
5. 用 LLM 做新闻宏观情绪识别，而不是直接让 LLM 下单。
6. 自己增加卖飞、止损后反弹、买入后回撤等诊断指标。
7. 优化 DE 到 0.05-0.15 区间，同时避免过度卖飞。
```
