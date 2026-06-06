# LLM Belief 层 Prompt 与 Few-Shot 设计说明

对应代码文件：

- `my_team/llm_cognition_prompts.py`
- `my_team/feature_engineering.py`

本文档说明当前 LLM Belief 层的边界、输入输出、人格注入、Few-Shot 标尺，以及后续 Desire / Utility / Action 层的衔接方式。

---

## 1. 当前结论

Belief 层必须去账户状态化。

它只回答：

```text
我怎么看当前这只股票？
```

它不回答：

```text
我现在该不该买？
我赚了 30% 要不要卖？
我现金够不够？
我要下多少单？
```

这些问题属于后续 Desire / Utility / Action 层。

这样拆分的原因是：股票本身的认知和账户状态下的交易冲动必须分开，才能解释真实散户中的“转折关系”。

例子：

```text
Belief signal:
  这只股票新闻和趋势仍然偏正，LLM 输出强微观利好和正技术偏向。

Desire:
  但我已经盈利 30%，RSI 过热，处置效应和前景理论效用让我想落袋为安。

Action:
  最终 sell。

Final thought:
  我仍然觉得它有利好和趋势，但盈利已经很大，价格也偏高，所以先锁定收益。
```

---

## 2. 四层架构

后续完整流程建议是：

```text
1. Belief / Sentiment 层
   输入：K 线、技术指标、新闻、社交舆论、人格提示词
   输出：macro/micro/technical/attention/uncertainty 分解

1b. Belief Scoring 层
   输入：LLM 分解项 + 技术特征
   输出：belief_score, sentiment_class

2. Desire / Utility 层
   输入：belief_score + 账户状态 + 持仓盈亏 + 前景理论效用
   输出：buy_desire, sell_desire, hold_desire, disposition_pressure

3. Action 层
   输入：desire + 现金/仓位/手数/限价约束
   输出：action, quantity, limit_price, official_belief_score, official_sentiment_class

4. Final Thought 层
   输入：belief + desire + action
   输出：完整 thought
```

当前文件 `llm_cognition_prompts.py` 只实现第 1 层。

---

## 3. Belief 层输入

入口函数：

```python
build_cognition_user_prompt(observation, features, personality)
```

标准输入结构：

```json
{
  "observation_meta": {
    "agent_id": "...",
    "symbol": "...",
    "tick": 1,
    "stock_name": "..."
  },
  "personality": {
    "name": "trend"
  },
  "market_features": {
    "open_price": 100.0,
    "high_price": 113.0,
    "low_price": 99.0,
    "close_price": 112.0,
    "volume_raw": 10600.0,
    "current_price": 112.0,
    "attention_score": 0.12,
    "anchor_score": -0.82,
    "ma_score": 0.45,
    "rsi": 100.0,
    "rsi_score": -1.0,
    "volatility_score": 0.12,
    "momentum_score": 0.90,
    "recent_return_score": 0.36
  },
  "news": ["earnings upgrade, strong growth, breakout buying"],
  "social_posts": [{"text": "bullish breakout buy", "influence": 8}]
}
```

明确排除：

```text
cash
position
avg_cost
position_weight
pnl_return
prospect_value
gain_flag / loss_flag
```

这些账户状态特征由 `feature_engineering.py` 继续保留，但不进入 Belief 层 prompt。它们应在 Desire / Utility 层使用。

---

## 4. Belief 层输出

输出 schema：

```json
{
  "scope": "macro|sector|stock_specific|mixed|irrelevant",
  "macro_direction": -1,
  "macro_strength": 0.0,
  "micro_direction": 1,
  "micro_strength": 0.7,
  "technical_direction": 1,
  "technical_strength": 0.6,
  "technical_bias": 0.5,
  "attention_bias": 0.2,
  "uncertainty": 0.3,
  "retail_emotion": "fomo",
  "belief_reason": "..."
}
```

字段解释：

- `scope`：文本影响范围。
- `macro_direction/macro_strength`：宏观方向和强度。
- `micro_direction/micro_strength`：当前 symbol 的直接文本方向和强度。
- `technical_direction/technical_strength`：技术面方向和强度。
- `technical_bias`：技术面连续偏向。
- `attention_bias`：社交热度、成交异动、注意力引发的偏差。
- `uncertainty`：证据不确定性。
- `retail_emotion`：散户情绪标签。
- `belief_reason`：一句第一人称股票认知，不解释账户交易行为。

注意：最终 `belief_score` 和 `sentiment_class` 不由 LLM 直接输出，而由 `belief_scoring.py` 根据上述分解项计算。这样可以避免 LLM 随意给连续小数。

---

## 5. 连续分数标尺

连续强度不能随便打小数，必须有证据锚。

建议标尺：

```text
0.0：无关或没有信息
0.2：弱相关、泛泛而谈、传闻不清
0.5：明确相关，但有混杂或间接影响
0.8：直接相关、方向清楚、文本强烈
1.0：极端直接、重大冲击、几乎无歧义
```

例子：

```text
央行降息：
  scope = macro
  macro_direction = 1
  macro_strength = 0.65
  micro_direction = 0
  micro_strength = 0.1

公司被下调评级 + lawsuit risk：
  scope = stock_specific
  micro_direction = -1
  micro_strength = 0.85

社交热度高但无个股新闻：
  attention_bias = 0.55
  micro_strength = 0.2
  belief_score 接近中性

强利好 + 突破趋势：
  micro_direction = 1
  micro_strength = 0.8
  technical_direction = 1
  technical_strength = 0.75
  belief_score 较高
```

Few-Shot 的核心作用就是给这些连续分数建立锚点。

---

## 6. 人格注入

人格注入由三部分组成：

```text
PERSONALITY_PROMPTS
PERSONALITY_TRAIT_REFERENCES
FEW_SHOT_EXAMPLES
```

### PERSONALITY_PROMPTS

描述四种人格的自然语言风格：

- `trend`：更重视趋势、动量、均线、近期收益。
- `value`：更重视价格锚和是否偏贵/偏便宜。
- `aggressive`：更容易响应强注意力、强利好、强趋势。
- `anxious`：更敏感于负面新闻、波动和不确定性。

### PERSONALITY_TRAIT_REFERENCES

保留 traits 作为定性参考，而不是精确数值：

```text
risk_appetite: low / medium / high
loss_aversion: low / medium / high
herding: low / medium / high
overconfidence: low / medium / high
disposition_effect: low / medium / high
turnover_tendency: low / medium / high
```

重要边界：

```text
人格可以影响解读强度，但不能扭曲明确事实。
```

例如：

- `aggressive` 可以把强利好读得更积极。
- `anxious` 可以在高不确定性下更谨慎。
- 但 `fraud rumor / downgrade / lawsuit risk` 不能被读成利好。

---

## 7. Few-Shot 设计

代码位置：

```python
FEW_SHOT_EXAMPLES
```

当前 Few-Shot 已改成账户状态无关版本。

包含：

1. 宏观宽松不是强个股利好。
2. 个股利空压过超卖诱惑。
3. “take profit” 类文本在 Belief 层不等于账户止盈，只能影响股票层谨慎程度。
4. 社交热度不能自动变成强信念。
5. 官方 `A_gain_bull`：只提取利好和趋势，不读取盈利持仓。
6. 官方 `A_loss_bear`：只提取利空和弱趋势，不读取亏损持仓。
7. 官方 `A_cash_bull`：只提取利好和趋势，不读取现金。
8. 官方 `A_cash_bear`：只提取利空和弱趋势，不读取空仓。
9. 官方 `A_neutral`：震荡和中性文本。
10. 官方 `A_gain_pressure`：估值过热和趋势强之间的股票层张力，不读取账户盈利。

---

## 8. 与官方字段的关系

官方 `AgentDecision` 只有：

```text
belief_score
sentiment_class
```

当前建议保留两个层次：

```text
belief_score:
  Belief Scoring 层对股票本身的连续认知，由 belief_scoring.py 计算。

official_belief_score:
  Action 层给官方接口的最终行为信念分，最好与 action 同方向。
```

原因：

```text
股票本身看涨，不等于最终一定买入。
股票本身看涨，但盈利过高，也可能因为处置效应卖出。
```

所以后续 Action 层可以：

```text
belief_score = +0.65
desire_sell = high because pnl_return = +30%
action = sell
official_belief_score = negative
official_sentiment_class = -1
```

而 final thought 负责解释这个转折。

---

## 9. 后续模块

下一步建议设计：

```text
desire_utility.py
```

它接收：

```text
belief_score
sentiment_class
account_features
personality
```

输出：

```text
buy_desire
sell_desire
hold_desire
profit_taking_pressure
loss_realization_resistance
```

再下一步是：

```text
action_policy.py
```

负责：

```text
action
quantity
limit_price
official_belief_score
official_sentiment_class
```

最后是：

```text
final_thought.py
```

负责把 Belief、Desire、Action 合成一句完整交易 thought。

---

## 10. 当前约束规则

解析函数：

```python
parse_cognition_response(raw)
normalize_cognition(data)
```

会做：

```text
macro_strength / micro_strength / technical_strength / uncertainty 裁剪到 [0, 1]
technical_bias / attention_bias 裁剪到 [-1, 1]
缺字段时回退中性值
```

---

## 11. 后续优化方向

1. 优化连续分数标尺，让每个分数区间都有更多例子。
2. 按人格做专属 Few-Shot，强化“同样事实，不同解读强度”。
3. 继续优化 `belief_score` 的可解释公式：

```text
text_component
technical_component
attention_component
uncertainty_penalty
personality_tilt
```

4. 做 LLM 输出和规则公式的双轨校验，防止 LLM 随意打分。
5. 在 Desire 层接入账户状态和前景理论效用函数。
6. 在 Action 层调节 DE 分布。
7. 在 Final Thought 层明确写出“股票看法”和“最终交易原因”的转折。
