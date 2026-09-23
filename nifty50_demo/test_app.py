"""Smoke test the UI on real data. Run: python -m unittest test_app -v"""
from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest


class AppSmokeTest(unittest.TestCase):
    def test_company_horizon_and_chart_controls(self):
        app = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=90).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.tabs), 3)
        horizon = next(widget for widget in app.selectbox if widget.label.startswith("Forecast horizon"))
        horizon.set_value(5).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        company = next(widget for widget in app.selectbox if widget.label == "Company")
        company.set_value("INFY").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        app.radio[0].set_value("Candlestick").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertTrue(len(app.metric) >= 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
