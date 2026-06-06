# my_team

任务二（交易所 + 监管）实现。本地评测：**任务二 100/100**（撮合 100 / F1 100 / 延迟 100）。

## 文件

- `submission.py` — 评测入口，不改签名
- `my_exchange.py` — `MyRegulator` 替换 baseline `RegulatoryAgent`，复用 baseline `ExchangeAgent` + `LimitOrderBook`
- `detectors.py` — 三个规则检测器

## 设计原则

- **撮合层 100% 复用 baseline**：`LimitOrderBook` 价格-时间优先、partial fill 已正确，不动 → 撮合 30 分锁死
- **监管层全规则**：所有检测同 tick 出 alert，latency=0
- **LLM 不进热路径**：`Alert.thought` 用字符串模板填充实际价格/数量/entity，$1 LLM 预算留作隐藏测试 fallback

## 三个 Detector 的逻辑

### WashDetector
- **R1 `on_pre_submit`**：新单与盘口前 5 档某挂单成交且 `entity_id` 相同 → 撮合前 `block_order_and_freeze_entity`
- **R2 `on_trades`**：成交后兜底，买卖双方 entity 相同仍发 `wash_trading`；同时维护 `(buyer, seller)` 无序对的 8-tick 滑动计数，同对手成交 ≥4 次发 `wash_trading_ring`

### SpoofDetector
- **A `on_pre_submit`**（**baseline 没有**）：大单（qty ≥ 同侧 5 档均深 ×3，或深度为空时 ≥500 绝对量）**且** 远离对手价 ≥100bps → pre-trade `block_order_and_flag_entity`
- **B `on_cancel`**：age ≤3 tick **且** 大单 **且** 同 entity 在窗口内累计撤单 ≥2 次 → `intervene_cancel_and_throttle`

### PumpDumpDetector（**baseline 没有**）
每 symbol 维护 5-tick 滑动 trade 窗口，三条 AND 信号全中即发 `pump_and_dump`：
1. 窗口内价格涨幅 ≥3%
2. 单一卖方 entity 占该窗口卖出量 ≥50%
3. 不同买方 entity 数 ≥3

## 与 baseline 的差异表

| 检测项 | Baseline | 我们 |
|---|---|---|
| Wash 自成交拦截 | ✅ | ✅ 一致 |
| Wash 对倒环 | 有序对 | 无序对（A→B 与 B→A 合并） |
| Spoof pre-submit | ❌ | ✅ 新增（解决本地 spoof 单不撤的问题） |
| Spoof on-cancel 阈值 | qty ≥ 4× 均深 | 放松到 3×，深度为空用绝对量兜底 |
| PumpDump | ❌ | ✅ 新增（三条 AND 信号） |
| LLM `thought` | 静态字符串 | 字符串模板，填实际数值 |

## 本地评测

```bash
# 从 participants_package/ 运行
"C:/Users/someo/AppData/Local/Programs/Python/Python313/python.exe" evaluate_submission.py my_team --json my_team/report.json
```
