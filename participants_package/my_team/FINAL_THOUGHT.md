# Final Thought 层设计说明

对应代码文件：

- `my_team/final_thought.py`
- `my_team/belief_scoring.py`
- `my_team/desire_utility.py`
- `my_team/llm_desire_prompts.py`

本文档说明 Final Thought 层如何把 Belief、Desire、Action 串成一句真实的散户交易心理独白。

---

## 1. 作用

Final Thought 层运行在最后：

```text
Feature Engineering
-> Belief Prompt
-> Belief Scoring
-> Desire / Utility
-> Action
-> Final Thought
```

它不改变：

```text
action
quantity
limit_price
belief_score
sentiment_class
desire scores
```

它只负责解释：

```text
我怎么看这只股票？
我为什么最后这么交易？
如果这两者有冲突，中间的心理转折是什么？
```

---

## 2. 为什么需要这一层

很多真实散户行为不是简单的：

```text
看涨 -> 买
看跌 -> 卖
```

而是：

```text
看涨 -> 但盈利很多 -> 卖出止盈
看跌 -> 但亏损且利空不强 -> 继续持有
中性 -> 不确定性高 -> 观望
```

所以 thought 不能只解释 action，也要解释 `belief` 到 `action` 的转折。

例子：

```text
I still read the stock as bullish, but my gain is large and the price feels stretched, so I choose to lock in part of the profit.
```

这句话同时包含：

```text
Belief: still bullish
Desire: gain large, price stretched
Action: lock in profit
```

---

## 3. 输入

入口函数：

```python
build_thought_user_prompt(...)
```

输入结构：

```json
{
  "personality": {"name": "anxious"},
  "belief": {
    "belief_score": 0.62,
    "sentiment_class": 1,
    "belief_reason": "The stock still has bullish evidence but looks stretched.",
    "uncertainty": 0.30
  },
  "desire": {
    "buy_desire": 0.08,
    "sell_desire": 0.41,
    "hold_desire": 0.37,
    "dominant_desire": "sell",
    "profit_taking_pressure": 0.48,
    "loss_realization_resistance": 0.0,
    "desire_reason": "The large gain and stretched price make locking in profit more tempting."
  },
  "action": {
    "action": "sell",
    "quantity": 100,
    "limit_price": 117.3
  },
  "account_features": {
    "has_position": 1,
    "position_weight": 0.34,
    "pnl_return": 0.30,
    "gain_flag": 1,
    "loss_flag": 0
  },
  "market_context": {
    "anchor_score": -0.90,
    "rsi_score": -1.0,
    "volatility_score": 0.22
  }
}
```

---

## 4. 输出

输出 schema：

```json
{
  "thought": "I still read the stock as bullish, but my gain is large and the price feels stretched, so I choose to lock in part of the profit."
}
```

要求：

```text
一句话
第一人称
像散户内心独白
不暴露长 CoT
不编造输入中没有的事实
```

---

## 5. 规则兜底

代码位置：

```python
build_rule_thought(...)
```

用途：

```text
LLM 不可用
LLM 输出格式错误
需要稳定测试
```

规则兜底会拼接：

```text
belief_reason
desire_reason
action bridge
market overheat hint
```

典型桥接：

```text
sell + belief_score > 0 + 盈利:
  but my gain is meaningful, so I want to lock in part of the profit

hold + belief_score < 0 + 亏损:
  but the signal is not decisive enough for me to realize the loss

buy + belief_score > 0:
  and the desire to participate is strong enough for me to buy

hold + uncertainty high:
  so I prefer to wait rather than force a trade
```

---

## 6. LLM Prompt

代码位置：

```python
build_thought_system_prompt(personality)
build_thought_user_prompt(...)
parse_thought_response(raw)
```

系统提示词核心约束：

```text
You are the Final Thought layer.
Do not change action, quantity, price, belief, or desires.
Explain both stock-level belief and final action.
If belief and action diverge, explain the psychological bridge.
Return only JSON.
```

人格风格：

- `trend`：更容易提 momentum、breakout、trend。
- `value`：更容易提 price anchor、expensive/cheap、patience。
- `aggressive`：更容易提 participation、confidence、risk-taking。
- `anxious`：更容易提 caution、locking gains、avoiding pain、uncertainty。

---

## 7. Few-Shot 示例

当前包含 5 个例子：

1. 看涨但卖出止盈。
2. 看跌但因为亏损不愿实现而持有。
3. 强看涨且有现金，所以买入。
4. 强看跌且亏损，但仍卖出降低风险。
5. 中性且高不确定性，所以持有。

这些例子覆盖了最重要的心理转折。

---

## 8. 与官方 thought 的关系

官方 `AgentDecision.thought` 是最终提交字段。

理想情况下，它应该表达：

```text
belief view
desire pressure
final action
```

但不能太长。当前代码限制：

```python
MAX_THOUGHT_CHARS = 280
```

后续可以调短，例如 180 或 220，以适应评测或日志可读性。

---

## 9. 后续优化方向

1. 用官方 6 场景回放 thought，看是否解释了心理转折。
2. 控制 thought 长度，避免过度啰嗦。
3. 为不同人格设计更强但不夸张的语言风格。
4. 增加“卖出但仍看涨”“持有但看跌”的更多 few-shot。
5. 后续接入 `InvestmentAgent.decide()` 时，优先用规则兜底，LLM 失败不影响决策。
6. 确保 thought 不泄露长推理链，只输出结果性心理独白。

