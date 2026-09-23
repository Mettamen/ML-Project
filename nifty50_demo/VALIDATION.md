# Validation on 23 September 2026

## Original input file

- Source: https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data
- File: `data/NIFTY50_all.csv`
- Rows: 235,192; historical company symbols: 65.
- Date range: 2000-01-03 to 2021-04-30.
- SHA-256: `fecbdf09ebc0fc9ea35d42cc570017ba1bf5d181f3f12d58106bd68e2d881d3d`

## Default TCS experiment

Five years of history, one-session horizon, volume/range enabled, latest 756
eligible training examples, final chronological 20% evaluated, refits every
20 forecast origins, 95% model-based interval.

There are 238 test forecasts with target dates 2020-05-20 through 2021-04-30.

| Measure | Bayesian regression | Last-price baseline |
|---|---:|---:|
| MAE (INR) | 32.116 | 31.950 |
| RMSE (INR) | 43.751 | 43.605 |
| MAPE (%) | 1.207 | 1.200 |
| R² | 0.989 | 0.989 |
| Direction match (%) | 49.160 | 0.420 |
| Observed 95% interval coverage (%) | 99.580 | N/A |

The model does not beat the baseline on MAE in this experiment. Its high R² is
not evidence of profitable forecasting. The baseline always predicts unchanged,
so its direction score only counts exactly unchanged observed closes. The
interval overcovers relative to its nominal level, consistent with conservative
bounds; coverage alone does not establish useful calibration.

## Checks completed

- Six model tests passed, including real-data integration for TCS, INFY, and
  RELIANCE at horizons 1, 5, and 10.
- Future-price changes do not change earlier inputs or earlier test predictions.
- Training-label dates never exceed their prediction origin dates.
- Target alignment, data cleanup, malformed schema, and interval ordering passed.
- One Streamlit UI test passed: initial load, company switch, horizon switch,
  and candlestick control, without app exceptions or error panels.
- Browser inspection confirmed the dashboard, charts, and actual/baseline
  performance display on http://localhost:8502.

These checks establish functional behavior and timing discipline, not market
predictability or clinical/financial suitability.
