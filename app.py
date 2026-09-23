import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


st.set_page_config(
    page_title="Bayesian ML Laboratory",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                linear-gradient(rgba(6, 12, 28, .96), rgba(6, 12, 28, .96)),
                repeating-linear-gradient(0deg, transparent, transparent 34px,
                    rgba(0, 247, 255, .06) 35px),
                repeating-linear-gradient(90deg, transparent, transparent 34px,
                    rgba(255, 0, 170, .05) 35px);
            color: #e8faff;
        }
        h1, h2, h3, p, li, label, .stMarkdown { color: #e8faff !important; }
        .hero {
            background: linear-gradient(100deg, #111a46, #5424a8, #e6007e);
            border: 2px solid #00f7ff;
            padding: 30px;
            border-radius: 18px;
            margin-bottom: 20px;
            box-shadow: 0 0 18px rgba(0,247,255,.7),
                        0 0 35px rgba(255,0,170,.45);
        }
        .hero h1, .hero h3, .hero p { color: white !important; }
        .risk-box {
            padding: 20px; border-radius: 15px; color: white;
            text-align: center; font-size: 20px; font-weight: bold;
            margin: 15px 0; border: 2px solid white;
            box-shadow: 0 0 18px rgba(255,255,255,.25);
        }
        [data-testid="stSidebar"] {
            background: #0b1024; border-right: 2px solid #00f7ff;
        }
        [data-testid="stSidebar"] * { color: #e8faff !important; }
        div[data-testid="stMetric"] {
            background: linear-gradient(135deg, #111a35, #1b1640);
            border: 1px solid #00f7ff; border-radius: 14px;
            padding: 15px; box-shadow: 0 0 12px rgba(0,247,255,.25);
        }
        [data-testid="stMetricValue"] { color: #00f7ff !important; }
        button[data-baseweb="tab"] { color: #b9d8ff !important; font-weight: bold; }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #00f7ff !important; border-bottom-color: #ff00aa !important;
        }
        .stDataFrame { border: 1px solid #8b5cf6; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>⚡ Bayesian Machine Learning Laboratory</h1>
        <h3>Probabilistic diagnosis and stock-price forecasting</h3>
        <p>Two interactive applications of Bayesian reasoning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Educational use only. The medical output is not a diagnosis and the stock "
    "forecast is not financial advice."
)


# -----------------------------------------------------------------------------
# HEART-DISEASE MODULE
# -----------------------------------------------------------------------------
HEART_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
HEART_FEATURES = HEART_COLUMNS[:-1]


@st.cache_data(ttl=86400)
def load_heart_data():
    data = pd.read_csv(
        HEART_DATA_URL,
        header=None,
        names=HEART_COLUMNS,
        na_values="?",
    )
    data["disease"] = (data["num"] > 0).astype(int)
    return data


@st.cache_resource
def train_heart_model():
    data = load_heart_data()
    x_train, x_test, y_train, y_test = train_test_split(
        data[HEART_FEATURES],
        data["disease"],
        test_size=0.25,
        random_state=42,
        stratify=data["disease"],
    )
    imputer = SimpleImputer(strategy="median")
    x_train_ready = imputer.fit_transform(x_train)
    x_test_ready = imputer.transform(x_test)
    model = GaussianNB().fit(x_train_ready, y_train)
    predicted = model.predict(x_test_ready)
    probability = model.predict_proba(x_test_ready)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, predicted),
        "auc": roc_auc_score(y_test, probability),
        "precision": precision_score(y_test, predicted),
        "recall": recall_score(y_test, predicted),
        "matrix": confusion_matrix(y_test, predicted),
    }
    return model, imputer, data, metrics


def feature_evidence(model, patient_values):
    rows = []
    for index, feature in enumerate(HEART_FEATURES):
        value = patient_values[0][index]
        variance_absent = max(model.var_[0][index], 1e-12)
        variance_present = max(model.var_[1][index], 1e-12)
        log_absent = (
            -0.5 * math.log(2 * math.pi * variance_absent)
            - ((value - model.theta_[0][index]) ** 2) / (2 * variance_absent)
        )
        log_present = (
            -0.5 * math.log(2 * math.pi * variance_present)
            - ((value - model.theta_[1][index]) ** 2) / (2 * variance_present)
        )
        score = log_present - log_absent
        rows.append({
            "Feature": feature,
            "Patient value": round(float(value), 2),
            "Evidence": (
                "Supports disease presence" if score > 0
                else "Supports disease absence"
            ),
            "Evidence score": round(score, 2),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# STOCK-FORECASTING MODULE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_stock_data(ticker, start_date, end_date):
    raw = yf.download(
        ticker,
        start=start_date,
        end=end_date + timedelta(days=1),
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError("No market data was returned for this ticker and date range.")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna().rename("Close").to_frame()


def build_lagged_stock_data(stock_data, lag_count):
    supervised = stock_data.copy()
    for lag in range(1, lag_count + 1):
        supervised[f"Lag_{lag}"] = supervised["Close"].shift(lag)
    supervised["Time"] = np.arange(len(supervised))
    return supervised.dropna()


def train_stock_model(stock_data, lag_count):
    prepared = build_lagged_stock_data(stock_data, lag_count)
    if len(prepared) < 60:
        raise ValueError("Use a longer date range so at least 60 usable trading days remain.")

    feature_names = ["Time"] + [f"Lag_{lag}" for lag in range(1, lag_count + 1)]
    split = int(len(prepared) * 0.8)
    train = prepared.iloc[:split]
    test = prepared.iloc[split:]
    model = BayesianRidge()
    model.fit(train[feature_names], train["Close"])
    predicted, std = model.predict(test[feature_names], return_std=True)
    result = test[["Close"]].copy()
    result["Predicted"] = predicted
    result["Lower 95%"] = predicted - 1.96 * std
    result["Upper 95%"] = predicted + 1.96 * std
    metrics = {
        "mae": mean_absolute_error(result["Close"], predicted),
        "rmse": math.sqrt(mean_squared_error(result["Close"], predicted)),
        "r2": r2_score(result["Close"], predicted),
    }
    return model, prepared, feature_names, result, metrics


def forecast_future(model, stock_data, lag_count, feature_names, days):
    history = stock_data["Close"].astype(float).tolist()
    next_time = len(stock_data)
    rows = []
    future_dates = pd.bdate_range(stock_data.index[-1] + pd.Timedelta(days=1), periods=days)
    for forecast_date in future_dates:
        features = [next_time] + [history[-lag] for lag in range(1, lag_count + 1)]
        feature_frame = pd.DataFrame([features], columns=feature_names)
        prediction, std = model.predict(feature_frame, return_std=True)
        price = max(float(prediction[0]), 0.01)
        uncertainty = float(std[0])
        rows.append({
            "Date": forecast_date,
            "Forecast": price,
            "Lower 95%": max(price - 1.96 * uncertainty, 0.01),
            "Upper 95%": price + 1.96 * uncertainty,
        })
        history.append(price)
        next_time += 1
    return pd.DataFrame(rows).set_index("Date")


try:
    heart_model, heart_imputer, heart_data, heart_metrics = train_heart_model()
except Exception as error:
    st.error("The UCI heart-disease dataset could not be loaded.")
    st.exception(error)
    st.stop()


diagnosis_tab, stock_tab, dataset_tab = st.tabs([
    "❤️ Disease Prediction",
    "📈 Stock Forecasting",
    "🧬 Dataset Explorer",
])


with diagnosis_tab:
    st.header("Heart Disease Prediction Using Bayes' Theorem")
    st.sidebar.header("Heart Patient Information")
    age = st.sidebar.slider("Age", 20, 90, 54)
    sex = st.sidebar.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x else "Female")
    cp = st.sidebar.selectbox("Chest Pain Type", [1, 2, 3, 4], index=2)
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 220, 130)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dL)", 100, 600, 246)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes" if x else "No")
    restecg = st.sidebar.selectbox("Resting ECG Result", [0, 1, 2])
    thalach = st.sidebar.slider("Maximum Heart Rate", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise-Induced Angina", [0, 1], format_func=lambda x: "Yes" if x else "No")
    oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.5, 1.4, 0.1)
    slope = st.sidebar.selectbox("ST Segment Slope", [1, 2, 3], index=1)
    ca = st.sidebar.selectbox("Number of Major Vessels", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia Test Result", [3, 6, 7])

    patient = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal,
    ]], columns=HEART_FEATURES)
    patient_ready = heart_imputer.transform(patient)
    probability = heart_model.predict_proba(patient_ready)[0, 1]
    if probability < 0.35:
        risk_level, risk_color = "LOWER ESTIMATED RISK", "#16a34a"
    elif probability < 0.65:
        risk_level, risk_color = "MODERATE ESTIMATED RISK", "#f59e0b"
    else:
        risk_level, risk_color = "HIGHER ESTIMATED RISK", "#dc2626"

    one, two, three = st.columns(3)
    one.metric("Estimated Disease Probability", f"{probability:.1%}")
    two.metric("Dataset Disease Prior", f"{heart_model.class_prior_[1]:.1%}")
    three.metric("Decision Threshold", "50%")
    st.progress(int(probability * 100))
    st.markdown(
        f'<div class="risk-box" style="background:{risk_color};">'
        f'{risk_level}<br>Estimated probability: {probability:.1%}</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Feature Evidence")
    evidence = feature_evidence(heart_model, patient_ready)
    left, right = st.columns(2)
    left.success("Top factors supporting disease presence")
    left.dataframe(evidence.nlargest(3, "Evidence score"), use_container_width=True)
    right.info("Top factors supporting disease absence")
    right.dataframe(evidence.nsmallest(3, "Evidence score"), use_container_width=True)
    with st.expander("View all feature evidence"):
        st.dataframe(evidence.sort_values("Evidence score", ascending=False), use_container_width=True)

    report = patient.copy()
    report["estimated_disease_probability"] = probability
    report["risk_category"] = risk_level
    st.download_button(
        "Download Prediction Report",
        report.to_csv(index=False),
        "bayesian_heart_prediction.csv",
        "text/csv",
    )


with stock_tab:
    st.header("Bayesian Linear Regression for Stock Price Forecasting")
    st.write(
        "Choose a ticker and historical period. Bayesian Ridge regression learns "
        "from time and recent closing-price lags, then estimates future prices with "
        "a 95% uncertainty interval."
    )

    controls_one, controls_two, controls_three, controls_four = st.columns(4)
    ticker = controls_one.text_input("Ticker symbol", "AAPL").strip().upper()
    default_end = date.today()
    default_start = default_end - timedelta(days=3 * 365)
    start_date = controls_two.date_input("Historical start", default_start)
    end_date = controls_three.date_input("Historical end", default_end)
    forecast_days = controls_four.slider("Forecast trading days", 5, 60, 20)
    lag_count = st.slider("Recent closing prices used as predictors", 3, 20, 7)

    if not ticker:
        st.info("Enter a ticker symbol, such as AAPL, MSFT, TCS.NS, or RELIANCE.NS.")
    elif start_date >= end_date:
        st.error("The historical start date must be earlier than the end date.")
    else:
        try:
            with st.spinner(f"Downloading and modelling {ticker}..."):
                stock_data = load_stock_data(ticker, start_date, end_date)
                stock_model, prepared, stock_features, test_result, stock_metrics = (
                    train_stock_model(stock_data, lag_count)
                )
                future = forecast_future(
                    stock_model,
                    stock_data,
                    lag_count,
                    stock_features,
                    forecast_days,
                )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Latest adjusted close", f"{stock_data['Close'].iloc[-1]:,.2f}")
            m2.metric("Test MAE", f"{stock_metrics['mae']:,.2f}")
            m3.metric("Test RMSE", f"{stock_metrics['rmse']:,.2f}")
            m4.metric("Test R²", f"{stock_metrics['r2']:.3f}")

            st.subheader("Historical Closing Price")
            st.line_chart(stock_data["Close"])

            st.subheader("Model Test: Actual vs Predicted")
            st.line_chart(test_result[["Close", "Predicted"]])

            st.subheader(f"Next {forecast_days} Trading Days")
            st.line_chart(future[["Forecast", "Lower 95%", "Upper 95%"]])
            st.dataframe(future.round(2), use_container_width=True)
            st.download_button(
                "Download Stock Forecast",
                future.reset_index().to_csv(index=False),
                f"{ticker}_bayesian_forecast.csv",
                "text/csv",
            )
            st.caption(
                "The interval represents model uncertainty. It does not include all "
                "real-world market risks, news, shocks, or structural changes."
            )
        except Exception as error:
            st.error(f"The forecast could not be created: {error}")


with dataset_tab:
    st.header("UCI Cleveland Heart Disease Dataset")
    d1, d2, d3 = st.columns(3)
    d1.metric("Patient Records", len(heart_data))
    d2.metric("Disease Present", int(heart_data["disease"].sum()))
    d3.metric("Disease Absent", int((heart_data["disease"] == 0).sum()))
    st.bar_chart(
        heart_data["disease"].value_counts().sort_index().rename(
            index={0: "No Disease", 1: "Disease Present"}
        )
    )
    selected_feature = st.selectbox("Compare feature averages", HEART_FEATURES)
    st.bar_chart(
        heart_data.groupby("disease")[selected_feature].mean().rename(
            index={0: "No Disease", 1: "Disease Present"}
        )
    )
    with st.expander("View records and statistics"):
        st.dataframe(heart_data.drop(columns=["num"]), use_container_width=True)
        st.dataframe(heart_data.describe(), use_container_width=True)


