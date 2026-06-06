# Belief Score 与 Sentiment Class 设计说明

对应代码文件：

- `my_team/belief_scoring.py`
- `my_team/llm_cognition_prompts.py`
- `my_team/feature_engineering.py`

本文档说明 demo 版 `belief_score` 如何从 LLM 分解项和技术特征中计算出来，以及 `sentiment_class` 如何跟随 `belief_score` 映射为 `-1 / 0 / 1`。

---

## 1. 定义

`belief_score` 表示：

```text
当前人格 Agent 对“这只股票本身”的看涨/看跌认知强度。
```

它不包含：

```text
现金
持仓数量
平均成本
浮盈浮亏
前景理论效用
处置效应
最终交易数量
```

这些因素属于后续 Desire / Utility / Action 层。

---

## 2. Sentiment Class

`sentiment_class` 直接跟随 `belief_score`。

当前 demo 阈值：

```python
SENTIMENT_NEUTRAL_THRESHOLD = 0.10
```

映射规则：

```python
if belief_score > 0.10:
    sentiment_class = 1
elif belief_score < -0.10:
    sentiment_class = -1
else:
    sentiment_class = 0
```

为什么需要中性 0：

```text
0.01 或 -0.03 这种弱信号更像噪声，不应该强行判为看涨或看跌。
```

后续这个阈值可以调：

- 阈值变小：更容易产生正/负情绪标签。
- 阈值变大：更多信号会被归为中性。
- 如果评测中 action 过少，可以降低阈值。
- 如果噪声交易太多，可以提高阈值。

---

## 3. 输入

入口函数：

```python
build_belief_score(cognition, market_features)
```

`cognition` 来自 LLM Belief 层：

```text
macro_direction
macro_strength
micro_direction
micro_strength
technical_bias
attention_bias
uncertainty
```

`market_features` 来自技术特征工程：

```text
ma_score
momentum_score
recent_return_score
rsi_score
anchor_score
volatility_score
```

---

## 4. 技术面 Prior

代码位置：

```python
build_technical_prior(market_features)
```

公式：

```text
technical_prior =
    0.35 * ma_score
  + 0.30 * momentum_score
  + 0.15 * recent_return_score
  + 0.10 * rsi_score
  + 0.10 * anchor_score
  - 0.15 * volatility_score
```

含义：

- `ma_score`：趋势均线信号。
- `momentum_score`：短期动量。
- `recent_return_score`：最新价格冲击。
- `rsi_score`：超买超卖。
- `anchor_score`：近期高低锚。
- `volatility_score`：波动风险，作为扣分项。

这一步不依赖 LLM，保证技术面有稳定基线。

---

## 5. LLM 技术解释融合

LLM 仍可以输出：

```text
technical_bias
```

但它不能完全覆盖规则技术面。

当前融合：

```text
technical_component =
    0.65 * technical_prior
  + 0.35 * llm_technical_bias
```

含义：

```text
规则技术面占主导，LLM 只作为解释性修正。
```

后续可调：

- 如果更信任技术公式，提高 `TECHNICAL_PRIOR_WEIGHT`。
- 如果希望 LLM 更能理解文本与技术冲突，适当提高 `LLM_TECHNICAL_WEIGHT`。

---

## 6. 宏观、微观、注意力组件

微观组件：

```text
micro_component = micro_direction * micro_strength
```

它代表当前 symbol 的直接文本影响，是最重要的文本项。

宏观组件：

```text
macro_component = macro_direction * macro_strength * 0.35
```

宏观消息会被折扣，因为宏观利好不能自动等于当前个股强利好。

注意力组件：

```text
attention_component = attention_bias * 0.30
```

注意力可以放大信念，但不能单独决定信念。

---

## 7. 总分公式

代码位置：

```python
build_belief_score()
```

公式：

```text
raw_score =
    0.45 * micro_component
  + 0.20 * macro_component
  + 0.25 * technical_component
  + 0.10 * attention_component
```

不确定性收缩：

```text
uncertainty_multiplier = 1 - 0.45 * uncertainty

belief_score = clip(raw_score * uncertainty_multiplier, -1, 1)
```

含义：

- 微观个股证据权重最高。
- 宏观证据只做风险偏好背景。
- 技术面提供稳定确认。
- 注意力是放大器，不是主信号。
- 不确定性越高，信念越向 0 收缩。

---

## 8. 当前权重

代码中的主要可调参数：

```python
MICRO_WEIGHT = 0.45
MACRO_WEIGHT = 0.20
TECHNICAL_WEIGHT = 0.25
ATTENTION_WEIGHT = 0.10

MACRO_TO_STOCK_DISCOUNT = 0.35
ATTENTION_TO_BELIEF_SCALE = 0.30
UNCERTAINTY_PENALTY_SCALE = 0.45

TECHNICAL_PRIOR_WEIGHT = 0.65
LLM_TECHNICAL_WEIGHT = 0.35

SENTIMENT_NEUTRAL_THRESHOLD = 0.10
```

这些都是 demo 版本，后续需要通过评测表现和案例回放调整。

---

## 9. 示例

强个股利好：

```text
micro_direction = 1
micro_strength = 0.85
technical_bias = 0.70
attention_bias = 0.30
uncertainty = 0.25

belief_score: 强正
sentiment_class: 1
```

宏观利好但无个股消息：

```text
macro_direction = 1
macro_strength = 0.65
micro_direction = 0
micro_strength = 0.10
technical_bias = 0.30
uncertainty = 0.45

belief_score: 弱正或接近中性
sentiment_class: 0 或 1
```

个股利空：

```text
micro_direction = -1
micro_strength = 0.85
technical_bias = -0.65
attention_bias = -0.50
uncertainty = 0.70

belief_score: 明显负
sentiment_class: -1
```

---

## 10. 后续优化方向

这个模块后续一定要调整。

重点优化方向：

1. 用官方 6 个固定场景回放检查分数方向。
2. 用 60 天数据检查 `belief_score` 分布是否过于集中或过于极端。
3. 调整 `SENTIMENT_NEUTRAL_THRESHOLD`，控制中性 0 的比例。
4. 调整 `MICRO_WEIGHT / TECHNICAL_WEIGHT`，决定文本和技术谁更主导。
5. 调整 `MACRO_TO_STOCK_DISCOUNT`，防止宏观新闻过度影响个股。
6. 调整 `UNCERTAINTY_PENALTY_SCALE`，控制高不确定性时的保守程度。
7. 后续可加入人格对权重的轻微调节，例如 trend 更重技术面，aggressive 更重 attention。
8. 与 Desire / Utility 层联调，观察最终 DE 分布和换手率。

