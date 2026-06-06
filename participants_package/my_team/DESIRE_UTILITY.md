# Desire / Utility 层设计说明

对应代码文件：

- `my_team/desire_utility.py`
- `my_team/llm_desire_prompts.py`
- `my_team/belief_scoring.py`
- `my_team/feature_engineering.py`

本文档说明 Desire / Utility 层如何把 `belief_score` 和账户状态转换成 `buy_desire / sell_desire / hold_desire`，以及 LLM 如何以“有量纲的枚举修正”参与人格化心理调整。

---

## 1. 重要边界

当前设计是 demo 版本，后续必须通过实验调整。

这个层级决定：

```text
我怎么看这只股票之后，在我当前账户状态下，我有多想买、卖、持有？
```

它不直接负责：

```text
最终 action
quantity
limit_price
官方 belief_score
官方 sentiment_class
```

这些属于后续 Action 层。

---

## 2. 总体结构

当前分两步：

```text
1. 规则效用主干
   belief_score + account_features + market_features
   -> rule_buy_desire / rule_sell_desire / rule_hold_desire

2. LLM 人格心理修正
   personality + belief + rule_desires + account_features
   -> ordinal adjustments
   -> final_buy_desire / final_sell_desire / final_hold_desire
```

关键原则：

```text
规则效用函数提供稳定量纲。
LLM 只做小幅、枚举式、人格化心理修正。
```

---

## 3. 为什么不用纯人格参数

之前讨论过给不同人格硬设：

```text
risk_appetite
loss_aversion
disposition_strength
turnover_tendency
```

这个方法有问题：

1. 数值很难校准。
2. 容易制造“伪精确”。
3. 同一人格在不同场景下反应不一定固定。
4. 情绪、后悔、恐慌、止盈冲动这类主观反应更适合由 LLM 做条件式判断。

所以现在采用：

```text
规则主干：稳定、可解释、可控。
LLM 修正：人格化、条件式、但幅度受限。
```

---

## 4. 规则效用主干

代码位置：

```python
build_rule_desires()
```

输入：

```text
belief_score
account_features
cognition
market_features
utility_params
```

默认规则参数：

```python
BASE_LOSS_AVERSION = 2.25
BASE_RISK_APPETITE = 0.60
BASE_DISPOSITION_STRENGTH = 0.55
PROSPECT_ALPHA = 0.88
PROSPECT_BETA = 0.88
```

注意：

```text
这里使用的是中性规则底座，不是人格硬编码。
人格差异主要交给 LLM desire adjustment。
```

---

## 5. 买入欲望

公式：

```text
buy_desire =
    bullish
  * risk_appetite
  * cash_ratio
  * (1 - position_weight)
  * (1 - 0.5 * uncertainty)
```

含义：

- 越看涨，越想买。
- 现金越多，越能买。
- 当前仓位越重，新增买入欲望越低。
- 不确定性越高，买入欲望越低。

---

## 6. 卖出欲望

卖出由三部分组成：

```text
bearish_sell_pressure
profit_taking_pressure
loss_realization_resistance
```

### bearish_sell_pressure

```text
bearish_sell_pressure =
    has_position
  * bearish
  * (0.7 + 0.3 * min(loss_aversion / 2.25, 1.5))
```

股票越看跌，越推动卖出。

### profit_taking_pressure

```text
profit_taking_pressure =
    has_position
  * disposition_strength
  * gain^0.88
  * (0.6 + 0.4 * position_weight)
  * (1 + 0.4 * overheat_score)
```

含义：

- 盈利越大，越想落袋为安。
- 仓位越重，止盈压力越强。
- RSI 过热、价格接近高位锚时，止盈压力更强。

### loss_realization_resistance

```text
loss_realization_resistance =
    has_position
  * loss_aversion
  * loss^0.88
  * (1 - bearish)
```

含义：

- 亏损越大，越不愿实现亏损。
- 但如果 `belief_score` 非常负，`bearish` 高，会削弱死扛压力。
- 这保留了“普通亏损死扛，强利空崩溃止损”的行为。

### sell_desire

```text
sell_desire =
    bearish_sell_pressure
  + profit_taking_pressure
  - 0.5 * loss_realization_resistance
```

---

## 7. 持有欲望

公式：

```text
hold_desire =
    0.25
  + 0.35 * uncertainty
  + 0.25 * loss_realization_resistance
  + 0.15 * (1 - abs(belief_score))
  - 0.20 * max(buy_desire, sell_desire)
```

含义：

- 看不懂时更想持有/观望。
- 亏损不愿实现时更想持有。
- 信念弱时更想持有。
- 买卖欲望强时，持有欲望下降。

---

## 8. LLM 心理修正层

代码位置：

```python
llm_desire_prompts.py
```

入口函数：

```python
build_desire_system_prompt(personality)
build_desire_user_prompt(...)
parse_desire_adjustment(raw)
normalize_desire_adjustment(data)
```

它可以看到：

```text
personality
belief
rule_desires
account_features
market_context
```

它不允许输出：

```text
action
quantity
limit_price
自由小数 desire
```

---

## 9. 为什么用枚举而不是小数

LLM 直接输出：

```json
{"sell_tilt": 0.713}
```

没有可靠量纲，后续很难调。

因此改成：

```json
{
  "buy_adjustment": "slightly_down",
  "sell_adjustment": "moderately_up",
  "hold_adjustment": "unchanged",
  "profit_taking_pressure": "high",
  "loss_hold_pressure": "none",
  "confidence": "medium"
}
```

Python 再统一映射成小幅数值。

---

## 10. 枚举映射表

代码位置：

```python
ADJUSTMENT_DELTAS
PRESSURE_BONUSES
CONFIDENCE_SCALES
```

调整幅度：

```text
strongly_down    -> -0.20
moderately_down  -> -0.12
slightly_down    -> -0.06
unchanged        ->  0.00
slightly_up      ->  0.06
moderately_up    ->  0.12
strongly_up      ->  0.20
```

压力 bonus：

```text
none    -> 0.00
low     -> 0.03
medium  -> 0.06
high    -> 0.10
extreme -> 0.14
```

置信度缩放：

```text
low    -> 0.50
medium -> 1.00
high   -> 1.00
```

注意：`high` 不放大到 1.5，避免 LLM 破坏规则主干。

---

## 11. LLM 修正融合

代码位置：

```python
apply_desire_adjustment(rule_desires, adjustment)
```

公式：

```text
buy_desire =
    rule_buy_desire
  + mapped_buy_adjustment

sell_desire =
    rule_sell_desire
  + mapped_sell_adjustment
  + mapped_profit_taking_pressure

hold_desire =
    rule_hold_desire
  + mapped_hold_adjustment
  + mapped_loss_hold_pressure
```

全部裁剪到 `[0, 1]`。

---

## 12. Few-Shot 设计

`llm_desire_prompts.py` 中包含 4 个示例：

1. 看涨但大幅盈利：焦虑型提高卖出调整和止盈压力。
2. 亏损但利空不强：价值型提高持有，降低卖出。
3. 亏损且明确利空：允许利空压过死扛。
4. 空仓且强看涨：激进型提高买入，但仍保持有界。

这些例子强调：

```text
LLM 解释心理转折，不直接决定交易。
```

---

## 13. 后续实验重点

这个模块必须后续做实验。

重点检查：

1. 官方 6 场景下，desire 是否符合直觉。
2. 60 天数据中，最终 action 是否过度买卖。
3. `profit_taking_pressure` 是否能提高 DE。
4. `loss_realization_resistance` 是否会导致过度死扛。
5. 强利空时 `(1 - bearish)` 是否足以触发 capitulation。
6. LLM 修正是否经常覆盖规则主干。
7. `confidence=low` 是否有效降低 LLM 干预。
8. `ADJUSTMENT_DELTAS` 的最大幅度 `0.20` 是否过大。
9. `PRESSURE_BONUSES` 是否导致卖出或持有过强。
10. 后续和 Action 层联调后，再看 DE、换手率、action diversity。

---

## 14. 推荐调参顺序

优先调：

```text
SENTIMENT_NEUTRAL_THRESHOLD
BASE_DISPOSITION_STRENGTH
LOSS_RESISTANCE_SELL_OFFSET
ADJUSTMENT_DELTAS
PRESSURE_BONUSES
```

再调：

```text
BASE_RISK_APPETITE
HOLD_UNCERTAINTY_WEIGHT
CAPITULATION_BEARISH_SCALE
OVERHEAT_PROFIT_TAKING_SCALE
```

不要一开始就大幅调整所有参数，否则很难判断是哪一项影响了 DE 和换手率。

