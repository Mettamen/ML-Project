"""Tests for time leakage, target alignment, and real NIFTY data integration."""
from io import StringIO
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from model import clean_company, evaluate, latest_forecast, load_csv, make_features


class ForecastTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))
        self.data = pd.DataFrame({
            "Date": pd.bdate_range("2020-01-01", periods=len(close)),
            "Close": close, "Symbol": "TEST",
        })

    def test_future_prices_never_change_past_features(self):
        original, columns = make_features(self.data, 5, False)
        changed = self.data.copy()
        changed.loc[500:, "Close"] *= 1.7
        after, _ = make_features(changed, 5, False)
        cutoff = self.data.loc[500, "Date"]
        pd.testing.assert_frame_equal(
            original.loc[original.Date < cutoff, columns],
            after.loc[after.Date < cutoff, columns],
        )

    def test_horizons_targets_and_training_dates(self):
        for horizon in (1, 5, 10):
            with self.subTest(horizon=horizon):
                frame, columns = make_features(self.data, horizon, False)
                known = frame.dropna(subset=["Target"])
                expected = np.log(self.data.Close.shift(-horizon) / self.data.Close)
                pd.testing.assert_series_equal(
                    known.set_index("Date")["Target"],
                    pd.Series(expected.to_numpy(), index=self.data.Date).loc[known.Date],
                    check_names=False,
                )
                result, metrics = evaluate(frame, columns, window=200)
                self.assertTrue((result["Training labels through"] <= result["Origin date"]).all())
                self.assertTrue((result["Target date"] > result["Origin date"]).all())
                self.assertTrue(np.isfinite(result[["Bayesian", "Lower", "Upper"]]).all().all())
                self.assertTrue((result.Lower <= result.Bayesian).all())
                self.assertTrue((result.Upper >= result.Bayesian).all())
                forecast = latest_forecast(frame, columns, window=200)
                self.assertEqual(forecast["origin"], self.data.Date.max())
                self.assertGreater(forecast["lower"], 0)

    def test_future_change_cannot_change_earlier_test_predictions(self):
        frame, columns = make_features(self.data, 10, False)
        first, _ = evaluate(frame, columns, window=200)
        cutoff = self.data.Date.iloc[550]
        changed = self.data.copy()
        changed.loc[changed.Date >= cutoff, "Close"] *= 1.5
        later_frame, later_columns = make_features(changed, 10, False)
        second, _ = evaluate(later_frame, later_columns, window=200)
        pd.testing.assert_frame_equal(
            first.loc[first["Origin date"] < cutoff, ["Bayesian", "Lower", "Upper"]],
            second.loc[second["Origin date"] < cutoff, ["Bayesian", "Lower", "Upper"]],
        )

    def test_bad_schema_has_a_readable_error(self):
        with self.assertRaisesRegex(ValueError, "Date and Close"):
            load_csv(StringIO("Name,Price\nTCS,123\n"))

    def test_company_cleanup(self):
        broken = pd.concat([self.data, self.data.tail(1)], ignore_index=True)
        broken.loc[0, "Close"] = -1
        cleaned, removed = clean_company(broken, "TEST")
        self.assertEqual(removed, 2)
        self.assertTrue(cleaned.Date.is_unique)
        self.assertTrue(cleaned.Date.is_monotonic_increasing)

    def test_real_dataset(self):
        path = Path(__file__).resolve().parent / "data" / "NIFTY50_all.csv"
        if not path.exists():
            self.skipTest("Download dataset to run the real-data integration test.")
        raw = load_csv(path)
        for symbol in ("TCS", "INFY", "RELIANCE"):
            company, _ = clean_company(raw, symbol)
            company = company.loc[company.Date >= company.Date.max() - pd.DateOffset(years=5)].reset_index(drop=True)
            for horizon in (1, 5, 10):
                with self.subTest(symbol=symbol, horizon=horizon):
                    frame, columns = make_features(company, horizon)
                    result, metrics = evaluate(frame, columns)
                    self.assertFalse(result.empty)
                    self.assertTrue(np.isfinite(metrics["MAE (INR)"]).all())
                    self.assertTrue((result["Training labels through"] <= result["Origin date"]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
