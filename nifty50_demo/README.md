# NIFTY-50 Bayesian Stock Forecasting Demo

This is a fresh stock-only project. The complete code is in `app.py` (interface)
and `model.py` (data cleaning, feature creation, training, and evaluation).

## Open the running demo

During the development session: http://localhost:8502

## Run it yourself in VS Code

1. Open this `nifty50_demo` folder in VS Code.
2. Open Terminal > New Terminal.
3. Run these commands one at a time:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

If port 8502 is already in use, use `--server.port 8503` and open the URL printed
by Streamlit. No virtual-environment activation is necessary. If an email prompt
appears, leave it blank and press Enter.

### Use the existing parent project's environment on this computer

From this `nifty50_demo` directory:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

The original CSV has already been downloaded into `data/NIFTY50_all.csv`.
The interface loads it automatically. Alternatively upload it, or a single-company
CSV, using the sidebar. The application works offline once dependencies and data
are downloaded.

## Dataset provenance

Source: Rohan Rao's NIFTY-50 Stock Market Data (2000–2021), hosted on Kaggle:
https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data

The downloaded combined file contains 235,192 rows and 65 historical symbol
values, spanning 2000-01-03 through 2021-04-30. Historic constituents and symbol
changes mean the count can exceed 50. These are individual company prices, not
the NIFTY index. The downloader extracts the original combined file without
changing it. Check the source's license before redistributing data.

## Features

- Company selector and CSV upload.
- One-, five-, or ten-session direct forecasts.
- Selectable history length, training window, and test portion.
- 80%, 90%, or 95% model-based prediction intervals.
- Closing-price/candlestick chart, MA20/MA50, and trading volume.
- Time-ordered walk-forward testing and last-price baseline.
- MAE, RMSE, MAPE, R², direction match, and empirical interval coverage.
- Actual/predicted price chart and error histogram.
- Data-quality summary, company comparison, and coefficient inspection.
- CSV exports of forecasts, evaluation, predictions, and selected data.

## Model explanation you can present

We forecast after a session closes, so that session's closing price, volume,
and high/low range are already available. Features describe recent log returns,
distance from moving averages, volatility, relative volume, and daily range.

For horizon h the target is `log(Close[t+h] / Close[t])`. Bayesian Ridge is a
Bayesian linear regression implementation that fits a distribution over weights
and estimates weight/noise precision hyperparameters from data (empirical Bayes).
Features are standardized using eligible training data only.

The model predicts a Gaussian log-return distribution. Multiplying the origin
price by `exp(predicted_log_return)` produces the median price forecast. Applying
the same transform to `mean +/- z * predictive_std` produces positive prediction
bounds. The displayed forecast is a median, not the lognormal mean.

We evaluate the latest chronological 20% by default and refit every 20 forecast
origins. At each refit, we only include labels whose target date is already known
at that origin's close. For multi-session horizons, unknown overlapping labels
are excluded. Previously observed test outcomes can train later models, as in
real walk-forward use. The scaler is refitted on the same eligible training data.

For the latest forecast we refit on all eligible labels within the chosen window.
Each horizon is predicted directly; we do not recursively feed guessed future
prices into inputs. A horizon of 5 means the fifth subsequent observed trading
session, not five calendar days. The app does not invent exchange dates.

## Accuracy and limitations

No promised stock-market accuracy. MAE/RMSE report price errors and R² is not an
accuracy percentage. Always compare against the baseline: future close equals
the origin close. A high R² alone can coexist with no useful forecasting advantage.

The CSV's Close prices are used as supplied. Stock splits and bonuses can create
large artificial jumps. The app flags moves over 25% but does not silently remove
or claim to adjust them. Default TCS history contains such a jump. Intervals are
conditional on model assumptions, may be too wide or too narrow, and do not
guarantee future coverage. Historical testing is not a claim of profitability.

Changing settings repeatedly after seeing test results uses the holdout as a
validation set. For a final report, freeze settings and collect a new untouched
later test period. Multi-session forecast errors overlap and are correlated.

## Five-minute demonstration

1. Open the app and choose TCS. Point out the dataset's last date in 2021.
2. Show the one-session median price forecast and prediction interval.
3. Switch to candlesticks and inspect moving averages and volume.
4. Open Test results and compare the Bayesian model with the baseline.
5. Change the horizon to five sessions and explain direct forecasting.
6. Explore another company and export the test predictions.

## Understand the code in this order

1. `model.py`: `load_csv` and `clean_company` validate and clean the data.
2. `make_features` creates past-only predictors and future targets.
3. `fit` standardizes the inputs and trains `BayesianRidge`.
4. `predict` converts predicted log returns into prices and bounds.
5. `evaluate` runs the walk-forward experiment and baseline comparison.
6. `latest_forecast` refits for the final available date.
7. `app.py` connects these functions to widgets, charts, and downloads.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest test_model -v
```

For the interface smoke test also run `python -m unittest test_app -v` using the
same virtual environment. Recorded real-data results are in `VALIDATION.md`.

Tests check target alignment, no future influence on earlier features/predictions,
training-label availability, positive intervals, malformed CSVs, cleanup, and
real-data forecasts for TCS, INFY, and RELIANCE at all three horizons.

References:
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html
- https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
