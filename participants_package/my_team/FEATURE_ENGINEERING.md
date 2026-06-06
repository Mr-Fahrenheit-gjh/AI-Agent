# 技术面与账户状态特征工程说明

本文档说明 `feature_engineering.py` 中已经实现的两类特征：

- 技术面特征：来自官方 `MarketObservation.klines`，即 K 线的 `open / high / low / close / volume`。
- 账户状态特征：来自官方 `MarketObservation.cash / position / avg_cost`，以及可选的内部组合状态字典。

本模块只负责特征构造，不负责买卖决策、LLM 调用或信号聚合。后续可以把这些特征拼接到规则 Agent、LLM prompt 或调参实验中。

对应代码文件：

```text
my_team/feature_engineering.py
```

主要入口函数：

```python
build_market_features(klines)
build_account_features(...)
build_observation_features(observation, ...)
```

---

## 一、技术面特征

技术面特征由 `build_market_features(klines)` 构造。

它兼容两种输入：

```text
官方 KLine 对象
dict 格式 K 线
```

如果 K 线数量不足，所有特征都会使用短历史 fallback，不会中断评测。

### 1. 关注度乘数 `attention_multiplier`

代码位置：

```python
build_market_features()
```

可调参数：

```python
ATTENTION_WINDOW = 20
ATTENTION_MIN = 0.5
ATTENTION_MAX = 3.0
```

公式：

```text
attention_multiplier = V_t / mean(V_{t-W}, ..., V_{t-1})
```

默认：

```text
W = 20
```

其中：

```text
V_t = 当前成交量
mean(V_{t-W}, ..., V_{t-1}) = 过去 W 期平均成交量，不包含当前 K 线
```

为了避免极端成交量导致特征爆炸，最终会裁剪到：

```text
[0.5, 3.0]
```

意义：

```text
衡量成交量是否异常放大，对应行为金融中的有限注意力。
成交量突然放大时，散户更容易被吸引，产生 FOMO、追涨或恐慌交易。
```

短历史处理：

```text
如果没有足够的历史成交量，则返回中性值 attention_multiplier = 1.0
```

---

### 2. 关注度标准化分数 `attention_score`

代码位置：

```python
build_market_features()
```

公式：

```text
attention_score = clip(log(attention_multiplier) / log(3), -1, 1)
```

意义：

```text
把成交量放大倍数映射到 [-1, 1]。

attention_score > 0:
  成交量高于过去均量，关注度上升。

attention_score = 0:
  成交量大致正常。

attention_score < 0:
  成交量低于过去均量，关注度下降。
```

后续可调方向：

```text
如果希望 Agent 对放量更敏感，可以降低 ATTENTION_MAX 或改变 log 映射尺度。
```

---

### 3. 价值锚定位置 `anchor_position`

代码位置：

```python
build_market_features()
```

可调参数：

```python
ANCHOR_WINDOW = 30
```

公式：

```text
High_W = max(high 最近 W 期)
Low_W = min(low 最近 W 期)

anchor_position = (C_t - Low_W) / (High_W - Low_W + eps)
```

默认：

```text
W = 30
```

其中：

```text
C_t = 当前收盘价
High_W = 最近 W 期最高价
Low_W = 最近 W 期最低价
```

意义：

```text
衡量当前价格处于近期高低区间的什么位置。

anchor_position 接近 0:
  当前价格靠近近期低点。

anchor_position 接近 1:
  当前价格靠近近期高点。
```

行为金融含义：

```text
散户常把近期高点和低点当作心理锚。
价格靠近低点时容易觉得“便宜了”；价格靠近高点时容易觉得“贵了”。
```

短历史处理：

```text
不足 30 根 K 线时，用当前可用历史计算。
如果无法计算有效 high/low，则返回中性值 0.5。
```

---

### 4. 价值锚定分数 `anchor_score`

代码位置：

```python
build_market_features()
```

公式：

```text
anchor_score = clip(1 - 2 * anchor_position, -1, 1)
```

意义：

```text
把价格区间位置映射为行为倾向。

anchor_score > 0:
  价格靠近近期低位，散户容易产生“抄底/便宜”的感觉。

anchor_score < 0:
  价格靠近近期高位，散户容易产生“太贵/止盈”的感觉。
```

后续可调方向：

```text
如果想模拟趋势追涨型人格，可以在决策层反向使用或降低该特征权重。
```

---

### 5. 均线特征 `ma_short` 与 `ma_long`

代码位置：

```python
build_market_features()
```

可调参数：

```python
MA_SHORT_WINDOW = 5
MA_LONG_WINDOW = 20
```

公式：

```text
ma_short = mean(close 最近 5 期)
ma_long = mean(close 最近 20 期)
```

意义：

```text
用于描述短期价格趋势和长期价格锚之间的关系。
短均线高于长均线，通常代表短期趋势更强。
```

短历史处理：

```text
不足 5 或 20 根时，用当前可用历史计算。
```

---

### 6. 均线趋势分数 `ma_score`

代码位置：

```python
build_market_features()
```

可调参数：

```python
MA_FULL_SIGNAL = 0.05
```

公式：

```text
ma_score = clip((ma_short / ma_long - 1) / 0.05, -1, 1)
```

意义：

```text
ma_score > 0:
  短均线强于长均线，趋势偏强。

ma_score < 0:
  短均线弱于长均线，趋势偏弱。
```

行为金融含义：

```text
对应散户的趋势外推和代表性启发。
看到近期上涨，容易把标的归类为“好股票”，进而追涨。
```

后续可调方向：

```text
MA_FULL_SIGNAL 越小，均线偏离越容易打满分数；
MA_FULL_SIGNAL 越大，特征更保守。
```

---

### 7. RSI 指标 `rsi`

代码位置：

```python
build_market_features()
_rsi()
```

可调参数：

```python
RSI_WINDOW = 14
RSI_MIN_HISTORY = 6
```

公式：

```text
delta_i = C_i - C_{i-1}
gain_i = max(delta_i, 0)
loss_i = max(-delta_i, 0)

avg_gain = mean(gain 最近 W 期)
avg_loss = mean(loss 最近 W 期)

RS = avg_gain / (avg_loss + eps)
RSI = 100 - 100 / (1 + RS)
```

默认：

```text
W = 14
```

短历史处理：

```text
n >= 15:
  使用 14 周期 RSI。

6 <= n < 15:
  使用 n - 1 周期 RSI。

n < 6:
  返回中性值 RSI = 50。
```

意义：

```text
RSI 高：近期上涨较多，可能超买。
RSI 低：近期下跌较多，可能超卖。
```

---

### 8. RSI 超买超卖分数 `rsi_score`

代码位置：

```python
build_market_features()
```

可调参数：

```python
RSI_SCORE_SCALE = 30.0
```

公式：

```text
rsi_score = clip((50 - RSI) / 30, -1, 1)
```

意义：

```text
rsi_score > 0:
  RSI 低于 50，偏超卖，散户容易想抄底。

rsi_score < 0:
  RSI 高于 50，偏超买，散户容易想止盈或逃顶。
```

行为金融含义：

```text
对应均值回归错觉和处置效应。
盈利状态下看到超买信号，可能增强落袋为安倾向。
亏损状态下看到超卖信号，可能增强“再等等会反弹”的心理。
```

---

### 9. 波动率分数 `volatility_score`

代码位置：

```python
build_market_features()
```

可调参数：

```python
VOLATILITY_WINDOW = 20
VOLATILITY_SCALE = 20.0
```

公式：

```text
r_i = C_i / C_{i-1} - 1
volatility_score = clip(std(r 最近 20 期) * 20, 0, 1)
```

意义：

```text
衡量近期价格不确定性。

volatility_score 越高：
  市场越不稳定，仓位应该更谨慎；
  焦虑型人格可能更容易恐慌；
  激进型人格可能仍然愿意参与。
```

短历史处理：

```text
不足 20 期时使用当前可用收益率。
如果收益率不足 2 个，返回 0。
```

---

### 10. 动量分数 `momentum_score`

代码位置：

```python
build_market_features()
```

可调参数：

```python
MOMENTUM_WINDOW = 5
MOMENTUM_FULL_SIGNAL = 0.10
```

公式：

```text
momentum_score = clip((C_t / C_{t-k} - 1) / 0.10, -1, 1)
```

默认：

```text
k = 5
```

意义：

```text
衡量最近 5 期累计涨跌幅。

momentum_score > 0:
  近期累计上涨。

momentum_score < 0:
  近期累计下跌。
```

与 `ma_score` 的区别：

```text
ma_score 关注短均线相对长均线的位置；
momentum_score 关注最近若干期的直接累计收益。
```

---

### 11. 近期收益冲击 `recent_return_score`

代码位置：

```python
build_market_features()
```

可调参数：

```python
RECENT_RETURN_FULL_SIGNAL = 0.05
```

公式：

```text
recent_return_score = clip((C_t / C_{t-1} - 1) / 0.05, -1, 1)
```

意义：

```text
捕捉最近一根 K 线的价格冲击。

recent_return_score > 0:
  最近一期上涨。

recent_return_score < 0:
  最近一期下跌。
```

行为含义：

```text
短期急涨可能触发 FOMO；
短期急跌可能触发恐慌或抄底冲动。
```

---

### 12. 当前价格与 K 线数量

代码位置：

```python
build_market_features()
```

特征：

```text
current_price = 最新 close
kline_count = 可用 K 线数量
```

意义：

```text
current_price 用于账户状态特征和后续执行约束。
kline_count 用于判断技术指标可信度。
```

---

## 二、账户状态特征

账户状态特征由 `build_account_features(...)` 构造。

主要输入来自官方接口：

```text
cash
symbol
position
avg_cost
current_price
```

可选输入来自我们自己维护的内部组合字典：

```text
latest_prices_by_symbol
positions_by_symbol
avg_cost_by_symbol
```

官方接口每次只传当前 `symbol` 的 `position` 和 `avg_cost`，不会直接传全组合持仓字典。如果后续需要全组合视角，需要在 `TeamSubmission` 或 `InvestmentAgent` 内部自己维护这些字典。

---

### 1. 原始现金 `cash_raw`

代码位置：

```python
build_account_features()
```

公式：

```text
cash_raw = observation.cash
```

意义：

```text
账户当前可用现金。
现金越多，后续买入约束越宽；
现金越少，买入行为应该更受限制。
```

---

### 2. 估算总权益 `estimated_equity`

代码位置：

```python
build_account_features()
_estimated_equity()
```

公式：

如果没有传入组合字典：

```text
estimated_equity = cash + position * current_price
```

如果传入组合字典：

```text
estimated_equity = cash + sum(position_i * latest_price_i)
```

如果某个标的没有最新价格，则 fallback 使用：

```text
当前 symbol 的 current_price
或 avg_cost_by_symbol 中的成本价
```

意义：

```text
估算当前组合总价值，供 cash_ratio、position_weight 等归一化特征使用。
```

---

### 3. 现金比例 `cash_ratio`

代码位置：

```python
build_account_features()
```

公式：

```text
cash_ratio = cash / estimated_equity
```

意义：

```text
衡量组合中现金占比。

cash_ratio 高：
  账户有较强买入能力。

cash_ratio 低：
  账户已经较满仓，继续买入应更谨慎。
```

---

### 4. 可买手数 `buying_power_lots`

代码位置：

```python
build_account_features()
```

可调参数：

```python
LOT_SIZE = 100
```

公式：

```text
buying_power_lots = floor(cash / current_price / 100)
```

意义：

```text
表示当前现金按 100 股一手可以买多少手。
这不是买入决策，只是执行约束特征。
```

---

### 5. 当前标的持仓数量 `position_qty`

代码位置：

```python
build_account_features()
```

公式：

```text
position_qty = observation.position
```

意义：

```text
当前 symbol 的持仓数量。
只有 position_qty > 0 时，浮盈浮亏和处置效应相关特征才有意义。
```

---

### 6. 是否持仓 `has_position`

代码位置：

```python
build_account_features()
```

公式：

```text
has_position = 1 if position_qty > 0 else 0
```

意义：

```text
区分“空仓观察”和“持仓管理”两种心理状态。
```

---

### 7. 当前标的持仓市值 `position_notional`

代码位置：

```python
build_account_features()
```

公式：

```text
position_notional = position_qty * current_price
```

意义：

```text
当前 symbol 的市值敞口。
```

---

### 8. 当前标的仓位权重 `position_weight`

代码位置：

```python
build_account_features()
```

公式：

```text
position_weight = position_notional / estimated_equity
```

意义：

```text
衡量当前标的在组合中的集中度。

position_weight 越高：
  继续加仓越需要谨慎；
  卖出时对组合影响也越大。
```

---

### 9. 平均成本锚 `avg_cost`

代码位置：

```python
build_account_features()
```

公式：

```text
avg_cost = observation.avg_cost
```

意义：

```text
当前标的的平均持仓成本，也是重要心理锚。
```

说明：

```text
60 天评测器中，买入会更新 avg_cost；
卖出时只减少 quantity，不改变剩余持仓 avg_cost；
全卖完后才移除持仓。

这与“卖出不改变心理锚”的行为金融设定一致。
```

---

### 10. 浮盈浮亏收益率 `pnl_return`

代码位置：

```python
build_account_features()
```

公式：

```text
pnl_return = current_price / avg_cost - 1
```

无持仓或 `avg_cost <= 0` 时：

```text
pnl_return = 0
```

意义：

```text
衡量当前价格相对成本锚的浮盈浮亏。

pnl_return > 0:
  处于盈利域。

pnl_return < 0:
  处于亏损域。
```

行为金融含义：

```text
处置效应主要就是围绕这个变量发生：
散户倾向于过早卖出盈利资产，过久持有亏损资产。
```

---

### 11. 成本距离 `cost_distance`

代码位置：

```python
build_account_features()
```

公式：

```text
cost_distance = log(current_price / avg_cost)
```

无持仓或 `avg_cost <= 0` 时：

```text
cost_distance = 0
```

意义：

```text
对数形式的价格-成本距离。
相比普通收益率，它对上涨和下跌更对称，适合后续建模。
```

---

### 12. 盈利/亏损标记 `gain_flag` 与 `loss_flag`

代码位置：

```python
build_account_features()
```

公式：

```text
gain_flag = 1 if pnl_return > 0 else 0
loss_flag = 1 if pnl_return < 0 else 0
```

意义：

```text
用于显式区分盈利域和亏损域。
前景理论和处置效应都强调：人们在盈利和亏损状态下的风险态度不同。
```

---

### 13. 前景理论价值 `prospect_value`

代码位置：

```python
build_account_features()
_prospect_value()
```

可调参数：

```python
PROSPECT_ALPHA = 0.88
PROSPECT_BETA = 0.88
PROSPECT_LOSS_LAMBDA = 2.25
```

公式：

令：

```text
x = pnl_return
```

则：

```text
prospect_value = x^alpha,                    if x >= 0
prospect_value = -lambda * (-x)^beta,         if x < 0
```

默认经典参数：

```text
alpha = 0.88
beta = 0.88
lambda = 2.25
```

意义：

```text
把客观浮盈浮亏转化为主观心理价值。

盈利侧：
  价值函数是凹的，表示盈利带来的边际快乐递减。

亏损侧：
  乘以 lambda = 2.25，表示损失厌恶。
  同样幅度的亏损带来的痛苦，大约是盈利快乐的 2.25 倍。
```

---

### 14. 裁剪后的前景理论价值 `prospect_value_clipped`

代码位置：

```python
build_account_features()
```

公式：

```text
prospect_value_clipped = clip(prospect_value, -1, 1)
```

意义：

```text
提供一个稳定版本，便于后续拼接到规则层或 LLM prompt。
避免极端亏损或极端盈利导致数值过大。
```

---

## 三、总入口特征

`build_observation_features(observation, ...)` 会从官方 `MarketObservation` 中一次性构造：

```text
技术面特征
账户状态特征
```

核心逻辑：

```text
1. 从 observation.klines 构造技术面特征。
2. 从技术面特征中读取 current_price。
3. 使用 observation.cash / observation.symbol / observation.position / observation.avg_cost 构造账户状态特征。
4. 合并为一个 dict 返回。
```

如果没有 K 线：

```text
current_price = 0
技术面特征返回中性值
账户状态特征也不会抛异常
```

---

## 四、参数调优建议

当前所有可调参数都集中在 `feature_engineering.py` 文件顶部。

后续可调整方向：

```text
ATTENTION_WINDOW:
  控制成交量关注度参考周期。

ANCHOR_WINDOW:
  控制历史高低点心理锚周期。

MA_SHORT_WINDOW / MA_LONG_WINDOW:
  控制趋势均线周期。

MA_FULL_SIGNAL:
  控制均线偏离多大时打满趋势分。

RSI_WINDOW:
  控制 RSI 计算周期。

RSI_SCORE_SCALE:
  控制 RSI 分数敏感度。

VOLATILITY_WINDOW / VOLATILITY_SCALE:
  控制波动率特征周期和放大倍数。

MOMENTUM_WINDOW / MOMENTUM_FULL_SIGNAL:
  控制动量周期和满信号阈值。

RECENT_RETURN_FULL_SIGNAL:
  控制单期收益冲击敏感度。

PROSPECT_ALPHA / PROSPECT_BETA / PROSPECT_LOSS_LAMBDA:
  控制前景理论价值函数。
```

目前这些参数先保留为模块常量。后续如果要系统调参，可以再迁移到 `config.yaml`。

---

## 五、使用示例

从官方 observation 直接提取：

```python
from feature_engineering import build_observation_features

features = build_observation_features(observation)
```

只提取技术面：

```python
from feature_engineering import build_market_features

market_features = build_market_features(observation.klines)
```

只提取账户状态：

```python
from feature_engineering import build_account_features

account_features = build_account_features(
    cash=observation.cash,
    symbol=observation.symbol,
    position=observation.position,
    avg_cost=observation.avg_cost,
    current_price=observation.klines[-1].close,
)
```

带内部组合状态字典：

```python
account_features = build_account_features(
    cash=observation.cash,
    symbol=observation.symbol,
    position=observation.position,
    avg_cost=observation.avg_cost,
    current_price=observation.klines[-1].close,
    latest_prices_by_symbol=latest_prices_by_symbol,
    positions_by_symbol=positions_by_symbol,
    avg_cost_by_symbol=avg_cost_by_symbol,
)
```

