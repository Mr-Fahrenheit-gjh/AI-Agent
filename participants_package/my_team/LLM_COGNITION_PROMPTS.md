# LLM 认知层 Prompt 与 Few-Shot 设计说明

对应代码文件：

- `my_team/llm_cognition_prompts.py`
- `my_team/feature_engineering.py`

本文档说明 LLM 认知层的设计思路、标准化输入输出、人格注入方式、Few-Shot 示例、约束规则、当前特殊处理，以及后续优化方向。

---

## 1. 核心定位

当前 LLM 不作为最终交易执行层，而是作为“认知参数层”。

它负责读取：

- 新闻文本
- 社交文本
- 技术面特征
- 账户/持仓状态特征
- 当前标的信息
- 当前人格名称

它输出：

- 宏观认知
- 当前标的微观认知
- 技术面解释偏向
- 账户心理偏向
- 注意力偏差
- 不确定性
- 建议动作
- 小幅信念修正
- 一句散户内心独白 `thought`

最终是否买入、卖出、持有，以及交易数量、限价、现金约束、持仓约束，仍应由规则层处理。

这样设计的原因是：LLM 擅长解释文本和情境，但不适合作为无约束交易执行器。把它放在中间认知层，可以利用它的语义理解能力，同时保留交易层的可控性。

---

## 2. 为什么去掉 traits

之前标准输入里曾经保留：

```json
{
  "personality": {
    "name": "trend",
    "traits": {
      "risk_appetite": 0.6,
      "loss_aversion": 1.35
    }
  }
}
```

现在已改为：

```json
{
  "personality": {
    "name": "trend"
  }
}
```

原因：

1. 这些数值 traits 没有经过严格校准，放进 LLM 输入会让模型误以为它们是可靠参数。
2. 官方接口没有强制要求把人格参数传给 LLM。
3. 当前阶段人格差异更适合通过 system prompt 的自然语言描述注入。
4. 规则层如果后续需要数值参数，可以单独在规则层维护，不应混进 LLM 标准输入。

当前特殊处理：

- `build_cognition_user_prompt(observation, features, personality)` 只输出人格名称。
- 如果误传入旧式 dict，例如 `{"name": "trend", "risk_appetite": 0.6}`，代码只抽取 `name`，不会把其他字段写入 JSON。
- Few-Shot 示例也全部去掉了 traits。

---

## 3. 标准化输入

入口函数：

```python
build_cognition_user_prompt(observation, features, personality)
```

生成的 user prompt 是一段 JSON，结构如下：

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
    "current_price": 112.0,
    "kline_count": 7,
    "attention_score": 0.12,
    "anchor_score": -0.82,
    "ma_score": 0.45,
    "rsi": 100.0,
    "rsi_score": -1.0,
    "volatility_score": 0.12,
    "momentum_score": 0.9,
    "recent_return_score": 0.36
  },
  "account_features": {
    "cash_ratio": 0.78,
    "buying_power_lots": 714,
    "has_position": 1,
    "position_weight": 0.22,
    "avg_cost": 96.0,
    "pnl_return": 0.167,
    "prospect_value_clipped": 0.21
  },
  "news": ["..."],
  "social_posts": [{"text": "...", "influence": 8}]
}
```

### 输入特征怎么理解

`market_features` 来自 `feature_engineering.py`：

- `open_price/high_price/low_price/close_price/volume_raw`：最新一根 K 线的原始 OHLCV 快照，保留未加工行情。
- `window_high/window_low/avg_volume`：派生指标使用的近期高低锚和成交量均值，方便解释锚定和关注度。
- `attention_score`：成交异动和注意力偏差。
- `anchor_score`：当前价格在近期高低锚中的位置，低位偏正，高位偏负。
- `ma_score`：短长均线趋势。
- `rsi_score`：超买超卖认知，超卖偏正，超买偏负。
- `volatility_score`：波动风险。
- `momentum_score`：近期动量。
- `recent_return_score`：最近一根 K 线收益。

`account_features` 也来自 `feature_engineering.py`：

- `cash_ratio`：现金占估算权益比例。
- `buying_power_lots`：按 100 股一手估算的可买手数。
- `has_position`：是否持仓。
- `position_weight`：当前标的仓位权重。
- `avg_cost`：持仓成本锚。
- `pnl_return`：浮盈浮亏比例。
- `prospect_value_clipped`：前景理论心理价值，亏损侧放大。

LLM 不重新计算这些指标，只解释它们。

---

## 4. 微观分数的映射

`micro_direction` 和 `micro_strength` 不是全局情绪分。

它们只表示当前输入里的 `observation_meta.symbol` 这一个标的的直接信息影响：

```text
micro_direction = 当前 symbol 的微观方向
micro_strength = 当前 symbol 的直接影响强度
```

例子：

- 央行降息：可以是 `scope=macro`，`macro_direction=1`，但 `micro_direction` 不应自动变成 1。
- 某标的被下调评级：应是 `scope=stock_specific`，`micro_direction=-1`。
- 社交热度很高但没有个股新闻：`attention_bias` 可以为正，但 `micro_strength` 不应过高。

这个约束已经写入 `BASE_COGNITION_SYSTEM_PROMPT`：

```text
The micro fields are always for the current input symbol only.
Do not collapse all texts into one global sentiment score.
Keep macro_direction and current-symbol micro_direction separate.
```

后续如果同时处理多只股票，应该为每个 symbol 分别调用一次认知层，或把输出结构改成：

```json
{
  "by_symbol": {
    "STOCK_001": {
      "micro_direction": 1,
      "micro_strength": 0.7
    }
  }
}
```

当前版本仍是单 observation、单 symbol 的结构。

---

## 5. 系统提示词

系统提示词由四部分拼接：

```text
BASE_COGNITION_SYSTEM_PROMPT
PERSONALITY_PROMPTS[personality]
PERSONALITY_TRAIT_REFERENCES[personality]
COGNITION_OUTPUT_SCHEMA_TEXT
FEW_SHOT_EXAMPLES
```

入口函数：

```python
build_cognition_system_prompt(personality, include_few_shots=True)
```

### 基础约束

`BASE_COGNITION_SYSTEM_PROMPT` 规定：

- LLM 是 financial cognition module。
- LLM 不是 execution/trading engine。
- 必须区分宏观信息和当前标的微观信息。
- 不重新计算技术指标。
- 不重新计算账户盈亏。
- 不输出长 CoT。
- `thought` 只输出一句第一人称散户内心独白。
- 输出必须是合法 JSON。
- 不提供精确数值 traits，人格来自 system prompt overlay 和定性 traits 参考。

---

## 6. 人格注入

当前人格注入只通过 system prompt overlay 完成。

代码位置：

```python
PERSONALITY_PROMPTS
PERSONALITY_TRAIT_REFERENCES
```

已有四种人格：

### trend

趋势型散户：

- 更重视动量、均线、成交热度、近期收益。
- 容易外推最近走势。
- 遇到 RSI 过热或高波动时需要提示谨慎。

### value

价值/锚定型散户：

- 更重视价格锚、成本锚、是否“便宜/贵”。
- 对短期社交热度不那么敏感。
- 不轻易追高。

### aggressive

激进型散户：

- 更容易被强注意力、强利好、强动量推动。
- 对波动容忍更高。
- 仍需考虑现金、仓位、不确定性。

### anxious

焦虑型散户：

- 对负面新闻、波动、亏损更敏感。
- 持有盈利时更容易想止盈。
- 持有亏损时不愿承认亏损，但强利空下会更恐慌。

### 为什么不在 user prompt 中放详细人格参数

因为目前数值 traits 缺少校准依据。把它们作为精确数值交给 LLM，会制造一种“精确但未验证”的错觉。现在只在 user JSON 中保留人格名称，主要用于调试和留痕；真正行为倾向由 system prompt overlay、定性 traits 参考和 Few-Shot 共同塑造。

### 定性 traits 参考

现在代码中新增了：

```python
PERSONALITY_TRAIT_REFERENCES
```

它保留了 traits 作为人格画像维度，但不使用精确数值，而是使用方向性描述：

```text
risk_appetite: low / medium / high
loss_aversion: low / medium / high
herding: low / medium / high
overconfidence: low / medium / high
disposition_effect: low / medium / high
turnover_tendency: low / medium / high
```

这样做的含义是：

1. traits 不再是“参数表”，而是人格提示词的一部分。
2. 它们只给 LLM 一个大致性格方向，不要求模型按精确数值计算。
3. Few-Shot 继续用具体场景教模型这些人格应该如何表现。
4. user JSON 仍不携带 traits，避免把未校准值当成观测输入。

这是一种折中方案：保留行为金融人格维度，但避免伪精确。

---

## 7. Few-Shot 示例

代码位置：

```python
FEW_SHOT_EXAMPLES
```

当前包含 10 个例子。

### 4 个通用例子

1. `macro easing is not direct stock-specific news`
   - 教模型识别宏观利好。
   - 宏观方向为正，但不能强行变成当前标的强利好。

2. `stock-specific bad news overwhelms oversold temptation`
   - 教模型在个股利空很强时，不被低位/超卖特征误导。

3. `profit and overbought state create profit-taking pressure`
   - 教模型理解盈利持仓、过热价格、处置效应带来的止盈压力。

4. `social heat alone should not become strong conviction`
   - 教模型区分社交热度和真实个股认知。

### 官方 6 场景例子

1. `A_gain_bull`
   - 盈利持仓 + 看涨新闻/社交。
   - 微观偏正，技术偏正，但账户止盈压力略负。

2. `A_loss_bear`
   - 亏损持仓 + 利空。
   - 微观偏负，技术偏负，账户心理偏负。

3. `A_cash_bull`
   - 空仓 + 看涨信息。
   - 微观偏正，现金能力支持参与。

4. `A_cash_bear`
   - 空仓 + 看跌信息。
   - 微观偏负，建议动作倾向 hold。

5. `A_neutral`
   - 震荡 + 中性新闻。
   - 不确定性高，建议 hold。

6. `A_gain_pressure`
   - 大幅盈利 + 止盈压力。
   - 账户心理偏负，体现处置效应。

Few-Shot 的输入使用特征工程后的 JSON，而不是原始 K 线长列表。这样可以降低 token 成本，也能让模型更稳定地学习“怎么解释特征”。

---

## 8. 标准化输出

输出 schema：

```json
{
  "scope": "macro|sector|stock_specific|mixed|irrelevant",
  "macro_direction": -1,
  "macro_strength": 0.0,
  "micro_direction": 0,
  "micro_strength": 0.0,
  "technical_bias": 0.0,
  "account_bias": 0.0,
  "attention_bias": 0.0,
  "uncertainty": 0.0,
  "retail_emotion": "neutral",
  "suggested_action": "hold",
  "belief_adjustment": 0.0,
  "thought": "..."
}
```

字段含义：

- `scope`：文本影响范围。
- `macro_direction`：宏观方向，-1 利空，0 中性，1 利好。
- `macro_strength`：宏观强度。
- `micro_direction`：当前 symbol 的微观方向。
- `micro_strength`：当前 symbol 的微观强度。
- `technical_bias`：LLM 对技术面特征的综合解释。
- `account_bias`：LLM 对账户/持仓心理状态的解释。
- `attention_bias`：新闻/社交/成交异动带来的注意力偏差。
- `uncertainty`：不确定性。
- `retail_emotion`：散户情绪标签。
- `suggested_action`：参考动作，不直接下单。
- `belief_adjustment`：给规则层使用的小幅信念修正。
- `thought`：一句第一人称散户内心独白。

---

## 9. 输出解析与约束

代码位置：

```python
parse_cognition_response(raw)
normalize_cognition(data)
```

约束规则：

```text
belief_adjustment: [-0.15, 0.15]
macro_strength: [0, 1]
micro_strength: [0, 1]
uncertainty: [0, 1]
technical_bias: [-1, 1]
account_bias: [-1, 1]
attention_bias: [-1, 1]
suggested_action: buy/sell/hold
scope: macro/sector/stock_specific/mixed/irrelevant
```

好处：

1. 防止 LLM 输出极端值直接污染规则层。
2. 缺字段时能回退到中性值。
3. 解析 markdown 包裹 JSON 或额外文本时更稳。
4. 输出结构固定，便于日志记录和调试。

弊端：

1. 裁剪会损失极端语义强度。
2. `belief_adjustment` 范围太窄时，LLM 影响可能不足。
3. `suggested_action` 被弱化后，LLM 的交易判断不能直接体现。
4. 情绪标签是离散枚举，可能表达不了复杂心理。

后续是否调整：

- 如果规则层太保守，可以把 `belief_adjustment` 放宽到 `[-0.2, 0.2]`。
- 如果 LLM 误导过强，应维持或收紧当前范围。
- 如果 thought 经常太长，可在解析层截断到固定字符数。

---

## 10. CoT 与 thought

当前策略：

```text
Think internally, but do not reveal step-by-step reasoning.
```

也就是允许模型内部思考，但不输出完整推理链。

`thought` 只保留一句短的第一人称散户心理：

```text
The rally is tempting, but my gain is already large and the price feels stretched.
```

这样做的好处：

1. 避免长 CoT 占 token。
2. 降低输出格式错误概率。
3. 更符合最终 `AgentDecision.thought` 的简洁风格。
4. 方便赛后解释“行为拟人性”。

后续优化方向：

- 为不同人格设计不同 thought 语气。
- 限制 thought 字数，例如 20 到 35 个英文词。
- 增加 `thought_style`，区分谨慎、贪婪、恐慌、后悔。
- 让 thought 更明确地引用账户锚或文本事件，但仍不暴露长推理链。

---

## 11. Token 与温度

建议配置：

```yaml
temperature: 0.1
max_tokens: 384
```

解释：

- `temperature=0.1`：让 JSON 输出更稳定。
- `max_tokens=384`：限制输出长度，防止模型展开废话。
- Few-Shot 会增加输入 token，不受 `max_tokens` 限制。

如果成本或速度有压力：

1. 保留 4 个通用例子。
2. 官方 6 场景中优先保留 `A_loss_bear` 和 `A_gain_pressure`。
3. 后续可以做动态 Few-Shot 检索，只注入当前场景最相似的 2 到 4 个例子。

---

## 12. config 里的 personality_mix 是什么

`config.example.yaml` 中目前有类似配置：

```yaml
agents:
  personality_mix:
    trend: 0.30
    value: 0.30
    aggressive: 0.25
    anxious: 0.15
  default_personality: "trend"
```

需要注意：

- `default_personality` 是当前更直接的默认人格配置。
- `personality_mix` 表示未来可以用于生成异质性 Agent 的人格分布。
- 当前 `my_team/submission.py` 尚未真正按 `personality_mix` 给不同 agent 分配人格。
- 因此它目前更像“后续实验目标”，不是已经生效的核心逻辑。

后续可优化：

1. 在 `submission.py` 中按 `personality_mix` 为不同 agent 分配人格。
2. 固定随机种子，保证评测可复现。
3. 针对不同市场阶段动态调整人格分布。
4. 把 personality 分布和实际交易风格统计对齐，验证它是否真的产生异质性。

---

## 13. 后续优化方向

### 输出特征结构

当前输出是单 symbol 扁平结构。后续可以扩展：

```json
{
  "by_symbol": {
    "STOCK_001": {
      "micro_direction": 1,
      "micro_strength": 0.7,
      "attention_bias": 0.4
    }
  }
}
```

适用于一次 prompt 处理多个标的的情况。但当前官方 observation 是单 symbol，先保持简单。

### 系统提示词

可调整位置：

```python
BASE_COGNITION_SYSTEM_PROMPT
```

优化方向：

- 强化宏观/微观分离。
- 强化“不要重新计算技术指标”。
- 强化“suggested_action 只是参考”。
- 加入更多金融文本歧义处理规则。

### 人格注入提示词

可调整位置：

```python
PERSONALITY_PROMPTS
PERSONALITY_TRAIT_REFERENCES
```

优化方向：

- 让四种人格和特征响应关系更明确。
- 调整 `PERSONALITY_TRAIT_REFERENCES` 中 high/medium/low 的方向性描述。
- 把 traits 维度和 Few-Shot 场景绑定，例如 anxious 在亏损利空场景中更恐慌，value 在高位锚定场景中更谨慎。
- 给每种人格设计不同 thought 风格。
- 后续如果有理论依据，再把定性 traits 校准成规则层参数，或者在 prompt 中使用区间化描述，而不是直接喂精确点值。

### Few-Shot

可调整位置：

```python
FEW_SHOT_EXAMPLES
```

优化方向：

- 增加 sector 新闻样例。
- 增加谣言/反转/文本冲突样例。
- 按人格做专属 few-shot。
- 做动态 few-shot 检索，减少输入 token。

### 温度和输出长度

可调整位置：

```yaml
temperature: 0.1
max_tokens: 384
```

优化方向：

- JSON 不稳定时降低温度。
- thought 被截断时提高 `max_tokens` 到 512。
- 输出废话变多时降低 `max_tokens` 或在 prompt 中进一步约束 thought。

### 约束规则

可调整位置：

```python
normalize_cognition(data)
```

优化方向：

- 调整 `belief_adjustment` 裁剪范围。
- 对 `thought` 增加长度限制。
- 对 `scope=mixed` 增加更细的规则解释。
- 对 `suggested_action` 增加和分数字段的一致性检查。

---

## 14. 当前结论

当前认知层的原则是：

1. LLM 只做认知，不直接下单。
2. 技术面和账户状态先由规则特征工程标准化。
3. 宏观/当前标的微观分开输出。
4. 人格通过 system prompt overlay、定性 traits 参考和 Few-Shot 共同注入，user JSON 不携带精确 traits。
5. Few-Shot 覆盖通用场景和官方 6 场景。
6. 输出必须经过解析、补默认值和裁剪后才能进入后续规则层。
