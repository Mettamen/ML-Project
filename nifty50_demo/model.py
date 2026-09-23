"""Bayesian forecasting functions. No Streamlit code, so they can be tested alone.

Timing convention: all features are observed AFTER the origin session closes.
The target is the log price change over the next h observed trading sessions.
"""
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def load_csv(source):
    data = pd.read_csv(source)
    data.columns = [str(name).strip().title() for name in data.columns]
    if data.columns.duplicated().any():
        raise ValueError("Duplicate CSV column names are not supported.")
    if not {"Date", "Close"}.issubset(data.columns):
        raise ValueError("Upload a price CSV containing Date and Close columns.")
    if "Symbol" not in data:
        data["Symbol"] = "UPLOADED STOCK"
    data["Symbol"] = data["Symbol"].fillna("UNKNOWN").astype(str).str.strip()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column in data:
            data[column] = pd.to_numeric(
                data[column].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )
    return data.replace([np.inf, -np.inf], np.nan)


def clean_company(raw, symbol):
    selected = raw.loc[raw["Symbol"] == symbol].copy()
    initial_count = len(selected)
    selected = selected.dropna(subset=["Date", "Close"])
    selected = selected.loc[selected["Close"] > 0]
    selected = selected.sort_values("Date").drop_duplicates("Date", keep="last")
    selected = selected.reset_index(drop=True)
    if len(selected) < 300:
        raise ValueError("Choose a company with at least 300 valid daily closing prices.")
    return selected, initial_count - len(selected)


def make_features(data, horizon=1, extended=True):
    """Only past/current prices enter X. Future prices enter the target only."""
    frame = data[["Date", "Close"]].copy()
    log_price = np.log(frame["Close"])
    returns = log_price.diff()
    columns = []
    for lag in (1, 2, 5, 10, 20):
        name = f"Return over {lag} sessions"
        frame[name] = log_price.diff(lag)
        columns.append(name)
    for window in (5, 20, 50):
        name = f"Distance from MA{window}"
        frame[name] = frame["Close"] / frame["Close"].rolling(window).mean() - 1
        columns.append(name)
    for window in (5, 20):
        name = f"Volatility {window}"
        frame[name] = returns.rolling(window).std()
        columns.append(name)
    if extended and "Volume" in data:
        volume = data["Volume"].where(data["Volume"] >= 0)
        log_volume = np.log1p(volume)
        frame["Relative volume"] = log_volume - log_volume.rolling(20).mean()
        columns.append("Relative volume")
    if extended and {"High", "Low"}.issubset(data.columns):
        frame["Daily range"] = ((data["High"] - data["Low"]) / data["Close"]).where(
            (data["High"] >= data["Low"]) & (data["Low"] > 0)
        )
        columns.append("Daily range")
    frame["Target"] = log_price.shift(-horizon) - log_price
    frame["Target price"] = frame["Close"].shift(-horizon)
    frame["Target date"] = frame["Date"].shift(-horizon)
    # Drop missing inputs, but keep the last horizon rows for a future forecast.
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)
    return frame.reset_index(drop=True), columns


def fit(history, columns):
    scaler = StandardScaler()
    x = scaler.fit_transform(history[columns])
    model = BayesianRidge()
    model.fit(x, history["Target"])
    return scaler, model


def predict(scaler, model, rows, columns, confidence):
    mean, std = model.predict(scaler.transform(rows[columns]), return_std=True)
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    origin = rows["Close"].to_numpy(dtype=float)
    # Transform the Gaussian log-return distribution to positive prices.
    return origin * np.exp(mean), origin * np.exp(mean - z * std), origin * np.exp(mean + z * std)


def score(actual, predicted, origin, lower=None, upper=None):
    actual, predicted, origin = map(np.asarray, (actual, predicted, origin))
    report = {
        "MAE (INR)": mean_absolute_error(actual, predicted),
        "RMSE (INR)": np.sqrt(mean_squared_error(actual, predicted)),
        "MAPE (%)": np.mean(np.abs((actual - predicted) / actual)) * 100,
        "R2": r2_score(actual, predicted),
        "Direction match (%)": np.mean(np.sign(actual - origin) == np.sign(predicted - origin)) * 100,
    }
    if lower is not None:
        report["Interval coverage (%)"] = np.mean((actual >= lower) & (actual <= upper)) * 100
    return report


def evaluate(frame, columns, test_fraction=0.2, window=756, confidence=0.95):
    """Walk forward through the final chronological holdout; refit every 20 origins.

    For h > 1, labels that end AFTER each refit origin are excluded. This purges
    overlapping future labels from training. No scaler is fitted on test data.
    """
    known = frame.dropna(subset=["Target", "Target date", "Target price"])
    test = known.iloc[int(len(known) * (1 - test_fraction)):]
    if len(test) < 30:
        raise ValueError("Not enough test records. Select a longer history.")
    predictions = []
    scaler = model = None
    fitted_through = None
    for step, (_, row) in enumerate(test.iterrows()):
        if step % 20 == 0:
            history = known.loc[
                (known["Date"] < row["Date"]) & (known["Target date"] <= row["Date"])
            ]
            if window:
                history = history.tail(window)
            if len(history) < 120:
                raise ValueError("At least 120 eligible training examples are required.")
            scaler, model = fit(history, columns)
            fitted_through = history["Target date"].max()
        one = row.to_frame().T
        center, lower, upper = predict(scaler, model, one, columns, confidence)
        predictions.append({
            "Origin date": row["Date"], "Target date": row["Target date"],
            "Training labels through": fitted_through,
            "Origin close": row["Close"], "Actual": row["Target price"],
            "Bayesian": center[0], "Baseline": row["Close"],
            "Lower": lower[0], "Upper": upper[0],
        })
    result = pd.DataFrame(predictions)
    metrics = pd.DataFrame({
        "Bayesian regression": score(result["Actual"], result["Bayesian"], result["Origin close"], result["Lower"], result["Upper"]),
        "Last-price baseline": score(result["Actual"], result["Baseline"], result["Origin close"]),
    }).T
    return result, metrics


def latest_forecast(frame, columns, window=756, confidence=0.95):
    latest = frame.tail(1)
    history = frame.dropna(subset=["Target", "Target date"])
    history = history.loc[history["Target date"] <= latest["Date"].iloc[0]]
    if window:
        history = history.tail(window)
    scaler, model = fit(history, columns)
    center, lower, upper = predict(scaler, model, latest, columns, confidence)
    # Coefficients multiply standardized features; effects are on log return.
    weights = pd.DataFrame({"Feature": columns, "Weight": model.coef_})
    return {
        "origin": latest["Date"].iloc[0], "close": float(latest["Close"].iloc[0]),
        "forecast": float(center[0]), "lower": float(lower[0]), "upper": float(upper[0]),
        "training_rows": len(history), "weights": weights,
    }
