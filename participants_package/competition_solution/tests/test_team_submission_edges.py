import unittest

from submission_interface.api import KLine, MarketObservation, OrderRequest
from submission_interface.validator import load_submission
from my_team.investment_agent import InvestmentAgent


def make_klines(closes, symbol="SIM"):
    rows = []
    for idx, close in enumerate(closes, start=1):
        prev = closes[idx - 2] if idx > 1 else close
        rows.append(
            KLine(
                symbol=symbol,
                timestamp=f"2023-03-{idx:02d}",
                open=prev,
                high=max(prev, close) * 1.01,
                low=min(prev, close) * 0.99,
                close=close,
                volume=10_000,
            )
        )
    return rows


class TeamSubmissionEdgeTests(unittest.TestCase):
    def setUp(self):
        self.submission = load_submission("my_team", {"agent_count": 8})
        self.submission.reset(seed=7, config={"agent_count": 8})

    def test_cash_too_low_for_lot_holds_with_neutral_signal(self):
        observation = MarketObservation(
            agent_id="low_cash",
            tick=1,
            symbol="SIM",
            klines=make_klines([100, 103, 106, 109, 112, 115]),
            news=["growth upgrade breakout buy"],
            social_posts=[{"text": "bullish breakout buy", "influence": 8}],
            cash=50.0,
            position=0,
            avg_cost=0.0,
        )
        decision = self.submission.decide(observation)
        self.assertEqual(decision.action, "hold")
        self.assertEqual(decision.quantity, 0)
        self.assertEqual(decision.sentiment_class, 0)

    def test_decisions_are_deterministic_after_reset_with_same_seed(self):
        observation = MarketObservation(
            agent_id="deterministic",
            tick=1,
            symbol="SIM",
            klines=make_klines([95, 96, 98, 101, 104, 108]),
            news=["growth upgrade breakout buy"],
            social_posts=[{"text": "bullish breakout buy", "influence": 6}],
            cash=120_000,
            position=0,
            avg_cost=0.0,
        )
        first = self.submission.decide(observation).to_dict()
        self.submission.reset(seed=7, config={"agent_count": 8})
        second = self.submission.decide(observation).to_dict()
        self.assertEqual(first, second)

    def test_decision_outputs_remain_valid_across_varied_scenarios(self):
        scenarios = [
            ("cash_bull", [90, 93, 96, 100, 105, 111], ["upgrade growth"], [{"text": "buy breakout", "influence": 5}], 150_000, 0, 0.0),
            ("cash_bear", [111, 108, 104, 99, 96, 92], ["fraud risk"], [{"text": "panic sell", "influence": 5}], 150_000, 0, 0.0),
            ("gain_bull", [100, 103, 106, 110, 114, 120], ["growth beat"], [{"text": "bull buy", "influence": 4}], 50_000, 200, 95.0),
            ("loss_bear", [120, 116, 110, 104, 98, 90], ["downgrade risk"], [{"text": "avoid sell", "influence": 7}], 50_000, 200, 112.0),
            ("neutral", [100, 101, 100, 101, 100, 101], ["mixed guidance"], [{"text": "wait and see", "influence": 2}], 80_000, 100, 100.0),
        ]
        for agent_id, closes, news, social, cash, position, avg_cost in scenarios:
            with self.subTest(agent_id=agent_id):
                decision = self.submission.decide(
                    MarketObservation(
                        agent_id=agent_id,
                        tick=1,
                        symbol="SIM",
                        klines=make_klines(closes),
                        news=news,
                        social_posts=social,
                        cash=cash,
                        position=position,
                        avg_cost=avg_cost,
                    )
                )
                self.assertIn(decision.action, {"buy", "sell", "hold"})
                self.assertGreater(decision.limit_price, 0)
                self.assertGreaterEqual(decision.quantity, 0)
                self.assertGreaterEqual(decision.belief_score, -1.0)
                self.assertLessEqual(decision.belief_score, 1.0)
                self.assertIn(decision.sentiment_class, {-1, 0, 1})
                if decision.action == "sell":
                    self.assertLessEqual(decision.quantity, position)
                if decision.action == "buy":
                    self.assertLessEqual(decision.limit_price * decision.quantity, cash * 1.01)

    def test_failing_llm_client_falls_back_to_rule_decision(self):
        def failing_llm(system_prompt, user_prompt):
            raise RuntimeError("temporary llm failure")

        agent = InvestmentAgent("llm_fallback", personality="trend", cash=120_000, seed=11, llm_client=failing_llm)
        agent.ingest_market(
            "SIM",
            [
                {"open": 100, "high": 102, "low": 99, "close": close, "volume": 10_000}
                for close in [100, 102, 104, 106, 109, 112]
            ],
        )
        agent.ingest_news("SIM", ["growth upgrade breakout buy"])
        agent.ingest_social("SIM", [{"text": "bullish breakout buy", "influence": 6}])
        decision = agent.decide("SIM")
        self.assertIn(decision.action, {"buy", "sell", "hold"})
        self.assertGreater(decision.limit_price, 0)
        self.assertTrue(decision.thought)

    def test_empty_bearish_position_does_not_short(self):
        observation = MarketObservation(
            agent_id="empty_bear",
            tick=1,
            symbol="SIM",
            klines=make_klines([110, 107, 104, 100, 97, 93]),
            news=["fraud downgrade panic risk"],
            social_posts=[{"text": "panic sell avoid", "influence": 9}],
            cash=120_000,
            position=0,
            avg_cost=0.0,
        )
        decision = self.submission.decide(observation)
        self.assertEqual(decision.action, "hold")
        self.assertEqual(decision.quantity, 0)
        self.assertEqual(decision.sentiment_class, 0)

    def test_near_touch_large_order_is_not_spoofing(self):
        result = self.submission.match_orders(
            [
                OrderRequest("S1", "seller", "SIM", "sell", 100.0, 500, 1, "S1E"),
                OrderRequest("B1", "buyer", "SIM", "buy", 99.5, 2_000, 2, "B1E"),
            ],
            {"SIM": 100.0},
            tick=2,
        )
        self.assertFalse(any("spoof" in alert.alert_type for alert in result.alerts))

    def test_invalid_orders_are_rejected_without_crashing(self):
        result = self.submission.match_orders(
            [
                OrderRequest("BAD_SIDE", "agent", "SIM", "cancel", 100.0, 10, 1, "E1"),
                OrderRequest("BAD_QTY", "agent", "SIM", "buy", 100.0, 0, 1, "E1"),
                OrderRequest("BAD_PRICE", "agent", "SIM", "sell", -1.0, 10, 1, "E1"),
            ],
            {"SIM": 100.0},
            tick=1,
        )
        self.assertEqual(set(result.rejected_order_ids), {"BAD_SIDE", "BAD_QTY", "BAD_PRICE"})
        self.assertEqual(result.accepted_order_ids, [])
        self.assertEqual(result.trades, [])

    def test_rising_prices_without_concentrated_seller_are_not_pump_dump(self):
        alerts = []
        for idx in range(5):
            result = self.submission.match_orders(
                [
                    OrderRequest(f"S{idx}", f"seller_{idx}", "SIM", "sell", 100.0 + idx, 10, idx + 1, f"S{idx}"),
                    OrderRequest(f"B{idx}", f"buyer_{idx}", "SIM", "buy", 110.0, 10, idx + 1, f"B{idx}"),
                ],
                {"SIM": 100.0 + idx},
                tick=idx + 1,
            )
            alerts.extend(result.alerts)
        self.assertFalse(any("pump" in alert.alert_type or "dump" in alert.alert_type for alert in alerts))

    def test_reset_clears_order_book_and_surveillance_state(self):
        self.submission.match_orders(
            [OrderRequest("W1", "acct_a", "SIM", "sell", 10.0, 100, 1, "SAME_OWNER")],
            {"SIM": 10.0},
            tick=1,
        )
        self.submission.reset(seed=7, config={"agent_count": 8})
        result = self.submission.match_orders(
            [OrderRequest("W2", "acct_b", "SIM", "buy", 10.0, 100, 2, "SAME_OWNER")],
            {"SIM": 10.0},
            tick=2,
        )
        self.assertFalse(any("wash" in alert.alert_type for alert in result.alerts))
        self.assertEqual(result.trades, [])

    def test_multi_symbol_surveillance_windows_are_isolated(self):
        alerts = []
        for idx in range(3):
            result = self.submission.match_orders(
                [
                    OrderRequest(f"AS{idx}", "seller_a", "AAA", "sell", 100.0 + idx, 10, idx + 1, "SA"),
                    OrderRequest(f"AB{idx}", f"buyer_a_{idx}", "AAA", "buy", 150.0, 10, idx + 1, f"BA{idx}"),
                    OrderRequest(f"BS{idx}", "seller_b", "BBB", "sell", 200.0 + idx, 10, idx + 1, "SB"),
                    OrderRequest(f"BB{idx}", f"buyer_b_{idx}", "BBB", "buy", 250.0, 10, idx + 1, f"BB{idx}"),
                ],
                {"AAA": 100.0 + idx, "BBB": 200.0 + idx},
                tick=idx + 1,
            )
            alerts.extend(result.alerts)
        self.assertFalse(any("pump" in alert.alert_type or "dump" in alert.alert_type for alert in alerts))

    def test_repeated_normal_near_touch_orders_do_not_emit_alerts(self):
        alerts = []
        for idx in range(6):
            result = self.submission.match_orders(
                [
                    OrderRequest(f"S{idx}", f"seller_{idx}", "SIM", "sell", 100.0, 100, idx + 1, f"SE{idx}"),
                    OrderRequest(f"B{idx}", f"buyer_{idx}", "SIM", "buy", 100.0, 100, idx + 1, f"BE{idx}"),
                ],
                {"SIM": 100.0},
                tick=idx + 1,
            )
            alerts.extend(result.alerts)
        self.assertEqual(alerts, [])

    def test_submit_stage_spoofing_and_pump_dump_are_detected(self):
        self.submission.match_orders(
            [
                OrderRequest("D1", "maker1", "SIM", "sell", 100.0, 50, 1, "M1"),
                OrderRequest("D2", "maker2", "SIM", "sell", 101.0, 50, 1, "M2"),
            ],
            {"SIM": 100.0},
            tick=1,
        )
        spoof = self.submission.match_orders(
            [OrderRequest("SP1", "spoofer", "SIM", "buy", 80.0, 1_000, 2, "SPOOF")],
            {"SIM": 100.0},
            tick=2,
        )
        self.assertTrue(any("spoof" in alert.alert_type for alert in spoof.alerts))

        self.submission.reset(seed=7, config={"agent_count": 8})
        detected = False
        for idx in range(8):
            tick = idx + 1
            result = self.submission.match_orders(
                [
                    OrderRequest(f"P{idx}S", "dump_seller", "SIM", "sell", 100.0 + idx, 10, tick, "DUMP"),
                    OrderRequest(f"P{idx}B", f"buyer_{idx % 4}", "SIM", "buy", 150.0, 10, tick, f"B{idx % 4}"),
                ],
                {"SIM": 100.0 + idx},
                tick=tick,
            )
            detected = detected or any(
                "pump" in alert.alert_type or "dump" in alert.alert_type
                for alert in result.alerts
            )
        self.assertTrue(detected)


if __name__ == "__main__":
    unittest.main()
