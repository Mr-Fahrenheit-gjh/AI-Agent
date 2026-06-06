# my_team 投资 Agent 说明

本目录是比赛提交包 `my_team`。当前实现保留了官方示例接口，不修改 `submission_interface`，入口仍然是：

```python
submission.py:create_submission(config)
```

当前版本有两套投资决策流程：

1. **Legacy 规则版**：稳定、可复现、默认启用，也是当前本地固定六场景的高分版本。
2. **LLM-BDI Demo 版**：从官方 Agent 代码出发，在 `InvestmentAgent.decide()` 内部接入 Belief -> Desire -> Action -> Final Thought 分层流程。旧版规则没有删除，只作为 fallback 保留。

## 1. 官方接口边界

官方评测只关心这些入口和返回结构：

```text
reset(seed, config)
decide(observation) -> AgentDecision
match_orders(orders, last_prices, tick) -> MatchResult
```

因此本项目遵守以下原则：

- 不改 `submission_interface/api.py` 中的官方数据结构。
- 不改官方传入的 `MarketObservation` 字段含义。
- 不改 `AgentDecision` 的输出字段。
- 新增模块只在 `my_team/` 内部使用。
- 旧规则引擎保留，LLM 异常或空响应时可回退。

## 2. 运行方式

默认不启用 LLM：

```powershell
cd AI-Agent/participants_package
python -m submission_interface.validator my_team
python evaluate_submission.py my_team
```

启用 LLM：

```powershell
python evaluate_submission.py my_team --use-llm --llm-config-path config.yaml
```

60 天新测试集：

```powershell
python new_data\market_60d_package\evaluate_60d_submission.py my_team --use-llm --llm-config-path config.yaml
```

`config.yaml` 中的 OpenAI 兼容接口配置由 `llm_helper.py` 读取，API key 通过环境变量传入，例如：

```yaml
llm:
  provider: "deepseek"
  base_url: "https://api.deepseek.com"
  model: "deepseek-v4-flash"
  temperature: 0.1
  max_tokens: 384
  api_key_env: "API_KEY"
```

## 3. LLM 开关

`--use-llm` 只表示允许创建 LLM client。当前为了控制速度，默认只启用 **Belief 认知层 LLM**。

可选开关：

```yaml
llm_desire_enabled: true
llm_thought_enabled: true
```

含义：

- `llm_desire_enabled`: 是否让 LLM 读取账户状态，输出人格化 Desire 修正标签。
- `llm_thought_enabled`: 是否让 LLM 在最终动作确定后生成 `thought`。
- 两者默认关闭，因为 60 天新测试集调用次数更多，三层 LLM 全开会非常慢。

## 4. 当前模块结构

```text
my_team/
  submission.py
  investment_agent.py
  exchange.py
  feature_engineering.py
  belief_scoring.py
  desire_utility.py
  llm_cognition_prompts.py
  llm_desire_prompts.py
  final_thought.py
```

核心职责：

- `feature_engineering.py`: 技术面和账户状态特征工程。
- `llm_cognition_prompts.py`: Belief 层系统提示词、人格提示词、few-shot、标准化输入输出。
- `belief_scoring.py`: 把 LLM 认知 JSON 和技术特征映射为 `belief_score` 与 `sentiment_class`。
- `desire_utility.py`: 结合 `belief_score` 和账户状态，计算 buy/sell/hold 三个 desire。
- `llm_desire_prompts.py`: 可选的 LLM 人格化 Desire 修正层，输出离散标签，不直接输出动作。
- `final_thought.py`: 最终 `thought` 生成。默认规则生成，可选 LLM 改写。
- `investment_agent.py`: 统一调度新旧两套流程。

## 5. Legacy 规则版

旧版规则引擎位于 `investment_agent.py` 的 legacy region 中，核心方法是：

```python
_legacy_rule_decide(symbol)
```

它保留了原来的散户行为模拟逻辑：

- 新闻/社交词典情绪。
- K 线动量、近期收益、估值锚。
- 盈利时更容易止盈。
- 亏损时更不愿意实现亏损。
- 输出合法 `Decision`。

如果没有启用 LLM，`decide()` 会直接调用 legacy。

如果启用 LLM 但发生异常，例如接口空响应、网络错误、解析失败导致流程无法继续，`decide()` 会回退 legacy，并在 `thought` 中追加 fallback 标记。

## 6. LLM-BDI Demo 流程

启用 LLM 后，`InvestmentAgent.decide(symbol)` 的内部流程是：

```text
MarketObservation
  -> feature_engineering.build_market_features()
  -> feature_engineering.build_account_features()
  -> LLM Belief cognition
  -> belief_scoring.build_belief_score()
  -> desire_utility.build_rule_desires()
  -> optional LLM desire adjustment
  -> action/quantity/limit_price rule constraints
  -> final_thought rule or optional LLM
  -> Decision
```

### 6.1 Belief 层

Belief 层只读：

- 当前标的元信息。
- 技术面特征。
- 新闻文本。
- 社交文本。
- 人格名称和人格提示词。

Belief 层不读账户状态，这是刻意设计的：  
`belief_score` 应表示“我对这只股票本身看涨还是看跌”，不应该混入“我现在赚了多少、亏了多少、有没有现金”。

输出 JSON 包含：

```json
{
  "scope": "macro|sector|stock_specific|mixed|irrelevant",
  "macro_direction": -1,
  "macro_strength": 0.0,
  "micro_direction": 0,
  "micro_strength": 0.0,
  "technical_direction": 0,
  "technical_strength": 0.0,
  "technical_bias": 0.0,
  "attention_bias": 0.0,
  "uncertainty": 0.0,
  "retail_emotion": "neutral",
  "belief_reason": "..."
}
```

### 6.2 Belief Score 与 Sentiment Class

`belief_scoring.py` 负责把 LLM 输出和技术特征合成：

```text
belief_score in [-1, 1]
sentiment_class in {-1, 0, 1}
```

当前 demo 中：

- `belief_score > 0.10` -> `sentiment_class = 1`
- `belief_score < -0.10` -> `sentiment_class = -1`
- 其他 -> `sentiment_class = 0`

也就是说，`sentiment_class` 跟随 `belief_score` 的方向，但保留中性区间。这个阈值后续需要通过评测实验继续调。

### 6.3 Desire 层

Desire 层才注入账户状态：

- 现金比例。
- 当前标的持仓数量。
- 平均成本心理锚。
- 浮盈浮亏。
- 前景理论价值函数。
- 持仓权重。

这层维护三个分数：

```text
buy_desire
sell_desire
hold_desire
```

最终动作取决于三者竞争，而不是直接等于 `belief_score`。例如：

- `belief_score > 0` 表示仍看涨。
- 但如果账户已大幅盈利，`sell_desire` 可能因为处置效应超过 `buy_desire`。
- 这时可以出现“看涨但卖出止盈”的真实散户转折。

### 6.4 Final Thought

最终 `thought` 应解释完整心理链条：

```text
我怎么看这只股票 -> 我的账户状态如何 -> buy/sell/hold 哪个欲望胜出 -> 为什么最终采取这个动作
```

默认使用 `final_thought.py` 的规则模板生成。  
如果 `llm_thought_enabled: true`，才让 LLM 在动作已确定后改写为更自然的散户内心独白。

## 7. 特征工程

技术面特征来自 `build_market_features(klines)`：

- OHLCV 当前快照：`open/high/low/close/volume/current_price`
- 注意力乘数：成交量相对均量。
- 锚定位置：价格在近期高低点区间的位置。
- MA 趋势：MA5/MA20。
- RSI 超买超卖。
- 波动率。
- 动量。
- 近期收益。

账户状态特征来自 `build_account_features(...)`：

- `cash_raw`
- `estimated_equity`
- `cash_ratio`
- `buying_power_lots`
- `position_qty`
- `avg_cost`
- `pnl_return`
- `cost_distance`
- `prospect_value`
- `position_weight`

详细公式见：

- `FEATURE_ENGINEERING.md`
- `BELIEF_SCORING.md`
- `DESIRE_UTILITY.md`
- `FINAL_THOUGHT.md`

## 8. 实测记录

当前本地测试结果：

```text
python -m submission_interface.validator my_team
通过

python evaluate_submission.py my_team
97.0 / 100
```

LLM-BDI demo 测试记录：

- 官方六场景可以跑通。
- DeepSeek v4flash 在部分场景会返回空文本。
- 三层 LLM 全开在 60 天新测试集上过慢，30 天切片曾超过 15 分钟未完成。
- 只开 Belief LLM 的 30 天切片可完成，但初版过于保守，出现全 hold，得分约 `27.1 / 100`。

因此当前结论是：

- Legacy 规则版仍是稳定主线。
- LLM-BDI 已完成接口和链路 demo。
- 新测试集要取得有效提升，需要继续做调用频率、空响应 fallback、动作阈值和 prompt 稳定性实验。

## 9. 已知问题

1. **LLM 空响应**
   DeepSeek v4flash 在长 prompt 或压力较高时可能返回空文本。当前处理是回退 legacy，避免空响应造成全 hold。

2. **调用成本高**
   每个 observation 调一次 LLM，60 天新测试集会非常慢。三层 LLM 全开不适合直接评测。

3. **Belief 到 Action 偏保守**
   初版 Desire 阈值偏谨慎，容易 hold。后续需要围绕 turnover、DE、action diversity 调参。

4. **人格参数仍是 demo**
   目前人格主要通过自然语言 prompt 注入，少量参数用于规则层。人格参数不是最终校准值，后续应通过实验调优。

5. **新闻/社交情绪连续化仍需校准**
   Few-shot 提供了参考量纲，但宏观/微观强度、uncertainty 和 attention bias 的输出稳定性仍需用新测试集回放验证。

## 10. 后续优化方向

优先级建议：

1. **让新测试集先快起来**
   增加 LLM 缓存，或只在有新闻/社交文本、技术异动较强时调用 LLM。

2. **优化 fallback**
   区分“LLM 空响应”“LLM 有效中性判断”“解析失败”，不要把接口问题当作市场中性。

3. **调 Desire 阈值**
   用 60 天新测试集观察 buy/sell/hold 分布，调整 `desire_utility.py` 和 `_action_from_desires()`。

4. **校准 sentiment 阈值**
   当前 `0.10` 是 demo 阈值。后续可试 `0.07/0.12/0.15`。

5. **完善 Final Thought**
   让 `thought` 明确区分 belief 与 action：例如“仍看涨但盈利太大所以止盈”“看跌但亏损域里不愿实现亏损”。

6. **人格分布实验**
   `config.yaml` 中的 `agents.personality_mix` 是后续多 Agent 人格分布调参入口。当前主要使用 `default_personality`，人格分布还没有作为核心实验变量。

## 11. 当前推荐策略

短期提交或跑分：

- 使用默认 legacy 规则版。
- 不开 LLM，确保稳定和速度。

研究和 demo：

- 开 `--use-llm`，默认只跑 Belief 层。
- 先不要打开 `llm_desire_enabled` 和 `llm_thought_enabled`。
- 观察 LLM 输出日志或 JSON 报告，再逐步调 prompt 和阈值。
