"""Run with: python -m streamlit run app.py

The interface lives here. The forecasting math lives in model.py.
"""
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import clean_company, evaluate, latest_forecast, load_csv, make_features


# 1. Page and reusable chart styling
st.set_page_config(page_title="NIFTY / Bayesian Lab", page_icon="📈", layout="wide")
st.markdown("""
<style>
 .block-container {padding-top: 2rem; max-width: 1500px;}
 .hero {background: linear-gradient(115deg,#111c38,#312354); padding: 30px;
        border: 1px solid #66749e; border-radius: 18px; margin-bottom: 22px;}
 .hero h1 {color:#f4f7ff !important; font-size: 2.3rem; margin: 4px 0;}
 .hero p {color:#c1cbe2 !important; margin-bottom:0;}
 .hero .eyebrow {color:#70efcf !important; letter-spacing:3px; font-size:12px;}
 div[data-testid="stMetric"] {border: 1px solid #63739a; border-radius:12px;
                             padding:15px; min-height:110px;}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="hero"><p class="eyebrow">NIFTY 50 · BAYESIAN REGRESSION</p>
<h1>Understand the forecast.<br>Measure the uncertainty.</h1>
<p>A hands-on stock-price laboratory using the original historical NIFTY dataset.</p></div>
""", unsafe_allow_html=True)

BASE = Path(__file__).resolve().parent
DATASET = BASE / "data" / "NIFTY50_all.csv"
COLORS = {"actual": "#63e8ca", "predicted": "#a69cff", "baseline": "#ffcb78"}


def chart_style(fig, title, ylabel="Price (INR)"):
    fig.update_layout(
        title=title, template="plotly_dark", height=425,
        paper_bgcolor="#10192b", plot_bgcolor="#10192b",
        margin=dict(l=25, r=25, t=60, b=35), hovermode="x unified",
        yaxis_title=ylabel, legend=dict(orientation="h", y=1.12),
    )
    return fig


@st.cache_data(show_spinner=False)
def read_local(path, modification_time):
    return load_csv(path)


@st.cache_data(show_spinner=False)
def read_upload(content):
    return load_csv(BytesIO(content))


@st.cache_data(show_spinner=False)
def run_experiment(data, horizon, extended, test_fraction, window, confidence):
    frame, columns = make_features(data, horizon, extended)
    if len(frame) < 250:
        raise ValueError("Not enough complete feature rows. Use more history or disable volume/range features.")
    result, metrics = evaluate(frame, columns, test_fraction, window, confidence)
    forecast = latest_forecast(frame, columns, window, confidence)
    return result, metrics, forecast, frame, columns


# 2. Load either the bundled original dataset or a user's CSV.
st.sidebar.title("Experiment controls")
uploaded = st.sidebar.file_uploader("Optional: upload your NIFTY CSV", type="csv")
try:
    if uploaded is not None:
        raw = read_upload(uploaded.getvalue())
        source_name = uploaded.name
    elif DATASET.exists():
        raw = read_local(str(DATASET), DATASET.stat().st_mtime_ns)
        source_name = "NIFTY50_all.csv · original Kaggle dataset"
    else:
        st.info("Upload NIFTY50_all.csv in the sidebar, or run: python download_data.py")
        st.link_button("Get the NIFTY-50 dataset", "https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data")
        st.stop()
except Exception as error:
    st.error(f"Unable to load this CSV: {error}")
    st.stop()

symbols = sorted(raw["Symbol"].unique().tolist())
default = symbols.index("TCS") if "TCS" in symbols else 0
symbol = st.sidebar.selectbox("Company", symbols, index=default)
years = st.sidebar.selectbox("History to use", ["5 years", "3 years", "All available"])
horizon = st.sidebar.selectbox("Forecast horizon (trading sessions)", [1, 5, 10])
confidence = st.sidebar.select_slider("Prediction interval", [0.80, 0.90, 0.95], value=0.95, format_func=lambda value: f"{value:.0%}")
test_pct = st.sidebar.slider("Chronological test portion (%)", 15, 30, 20, step=5)
window_name = st.sidebar.selectbox("Training window", ["Latest 756 examples (~3 years)", "Latest 504 examples (~2 years)", "All eligible history"])
window = 756 if "756" in window_name else (504 if "504" in window_name else 0)
extended = st.sidebar.checkbox("Include volume and daily price range", value=True)
st.sidebar.caption("Controls change the experiment. Repeatedly choosing settings from test results turns the test into a validation set.")
st.sidebar.markdown("[Dataset source](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data)")


# 3. Clean one company's history, then run the time-ordered experiment.
try:
    company_data, removed = clean_company(raw, symbol)
    if years != "All available":
        start = company_data["Date"].max() - pd.DateOffset(years=int(years.split()[0]))
        company_data = company_data.loc[company_data["Date"] >= start].reset_index(drop=True)
    if len(company_data) < 300:
        raise ValueError("Use All available history or choose a company with more data.")
    with st.spinner("Testing the Bayesian model through time..."):
        results, metrics, forecast, features, feature_names = run_experiment(
            company_data, horizon, extended, test_pct / 100, window, confidence
        )
except Exception as error:
    st.error(str(error))
    st.stop()

start_date, end_date = company_data["Date"].min(), company_data["Date"].max()
st.subheader(f"{symbol} / {horizon}-session forecast")
st.caption(f"{source_name} · {len(company_data):,} records · {start_date:%d %b %Y} — {end_date:%d %b %Y}")
st.info(f"Historical demo: this forecast begins after {forecast['origin']:%d %b %Y}, not today. Prices are the CSV's unadjusted Close values; the selected company is not the NIFTY index.")
if forecast["origin"] != end_date:
    st.warning("The latest rows have missing model inputs. The forecast uses the last complete input row shown above.")
jump_mask = company_data["Close"].pct_change(fill_method=None).abs() > 0.25
if jump_mask.any():
    st.warning(f"Large price moves detected: {int(jump_mask.sum())} session(s) with changes above 25%. Splits, bonuses, or other events may affect these unadjusted prices. They are retained and may distort forecasts or errors.")

dashboard, backtest, explorer = st.tabs(["Forecast dashboard", "Test results", "Explore the dataset"])


# 4. Forecast dashboard: forecast, interval, chart, and report.
with dashboard:
    a, b, c = st.columns(3)
    a.metric("Last complete closing price", f"₹{forecast['close']:,.2f}")
    change = (forecast["forecast"] / forecast["close"] - 1) * 100
    b.metric("Bayesian median price forecast", f"₹{forecast['forecast']:,.2f}", f"{change:+.2f}% from last close")
    c.metric(f"{confidence:.0%} prediction interval", f"₹{forecast['lower']:,.0f} — ₹{forecast['upper']:,.0f}")
    st.caption(f"Direct prediction {horizon} observed trading sessions ahead. No exchange dates are guessed. The interval is model-based, not guaranteed coverage.")
    left, right = st.columns([2, 1])
    with left:
        kind = st.radio("Chart type", ["Closing price", "Candlestick"], horizontal=True)
    with right:
        display_count = st.selectbox("Sessions to display", [60, 120, 250, 500], index=2)
    plotted = company_data.copy()
    plotted["MA20"] = plotted["Close"].rolling(20).mean()
    plotted["MA50"] = plotted["Close"].rolling(50).mean()
    plotted = plotted.tail(display_count)
    figure = go.Figure()
    if kind == "Candlestick" and {"Open", "High", "Low"}.issubset(plotted.columns):
        figure.add_trace(go.Candlestick(x=plotted["Date"], open=plotted["Open"], high=plotted["High"], low=plotted["Low"], close=plotted["Close"], name="OHLC"))
    else:
        if kind == "Candlestick":
            st.caption("This CSV has no OHLC columns. Showing closing prices instead.")
        figure.add_trace(go.Scatter(x=plotted["Date"], y=plotted["Close"], name="Close", line_color=COLORS["actual"]))
    for column, color in [("MA20", COLORS["baseline"]), ("MA50", COLORS["predicted"])]:
        figure.add_trace(go.Scatter(x=plotted["Date"], y=plotted[column], name=column, line=dict(color=color, width=1.5)))
    figure.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(chart_style(figure, "Price history and moving averages"), width="stretch")
    if "Volume" in plotted:
        volume_chart = go.Figure(go.Bar(x=plotted["Date"], y=plotted["Volume"], marker_color="#6673b1", name="Volume"))
        volume_chart = chart_style(volume_chart, "Trading volume", "Shares")
        volume_chart.update_layout(height=240)
        st.plotly_chart(volume_chart, width="stretch")
    forecast_report = pd.DataFrame([{
        "Symbol": symbol, "Origin date": forecast["origin"], "Horizon sessions": horizon,
        "Origin close": forecast["close"], "Median forecast": forecast["forecast"],
        "Interval level": confidence, "Lower": forecast["lower"], "Upper": forecast["upper"],
        "Training examples": forecast["training_rows"], "Model": "BayesianRidge on log returns",
    }])
    st.download_button("Download forecast CSV", forecast_report.to_csv(index=False), f"{symbol}_forecast.csv", "text/csv")


# 5. Walk-forward holdout performance: always show the simple baseline.
with backtest:
    st.caption(f"{len(results):,} test forecasts · targets from {results['Target date'].min():%d %b %Y} to {results['Target date'].max():%d %b %Y} · model refitted every 20 forecast origins.")
    a, b, c, d = st.columns(4)
    scores = metrics.loc["Bayesian regression"]
    a.metric("MAE", f"₹{scores['MAE (INR)']:,.2f}")
    b.metric("RMSE", f"₹{scores['RMSE (INR)']:,.2f}")
    c.metric("R² — not accuracy", f"{scores['R2']:.3f}")
    d.metric("Observed interval coverage", f"{scores['Interval coverage (%)']:.1f}%")
    st.subheader("Can it beat the last-price baseline?")
    st.dataframe(metrics.round(3), width="stretch")
    gain = metrics.loc["Last-price baseline", "MAE (INR)"] - scores["MAE (INR)"]
    if gain > 0:
        st.success(f"Bayesian regression reduces MAE by ₹{gain:.2f} on this test period.")
    else:
        st.info(f"The last-price baseline has MAE lower by ₹{-gain:.2f} on this period. The demo reports that result without hiding it.")
    st.caption("The baseline predicts no change from the origin close. MAE is average absolute error; RMSE penalizes big misses. Direction match measures up/down/unchanged agreement, not profitability. R² can be negative.")
    if horizon > 1:
        st.caption("Multi-session targets overlap across forecast origins, so errors are correlated; these are not independent trials.")
    show_interval = st.checkbox("Show prediction interval on the test chart", value=True)
    figure = go.Figure()
    if show_interval:
        figure.add_trace(go.Scatter(x=results["Target date"], y=results["Lower"], line_width=0, showlegend=False, hoverinfo="skip"))
        figure.add_trace(go.Scatter(x=results["Target date"], y=results["Upper"], line_width=0, fill="tonexty", fillcolor="rgba(166,156,255,.16)", name=f"{confidence:.0%} interval"))
    for name, color in [("Actual", COLORS["actual"]), ("Bayesian", COLORS["predicted"]), ("Baseline", COLORS["baseline"])]:
        figure.add_trace(go.Scatter(x=results["Target date"], y=results[name], name=name, line=dict(color=color, width=1.8)))
    st.plotly_chart(chart_style(figure, "Predictions against later observed prices"), width="stretch")
    errors = results["Actual"] - results["Bayesian"]
    error_chart = go.Figure(go.Histogram(x=errors, nbinsx=35, marker_color=COLORS["predicted"]))
    error_chart = chart_style(error_chart, "Prediction error distribution", "Forecast count")
    error_chart.update_layout(xaxis_title="Actual minus predicted price (INR)", height=280)
    st.plotly_chart(error_chart, width="stretch")
    first, second = st.columns(2)
    first.download_button("Download test predictions", results.to_csv(index=False), f"{symbol}_test_predictions.csv", "text/csv")
    second.download_button("Download performance report", metrics.to_csv(), f"{symbol}_metrics.csv", "text/csv")
    with st.expander("Inspect test predictions and training dates"):
        st.dataframe(results, width="stretch")
    st.caption("At each refit, training labels must be known by the origin session's close. Past test outcomes may be used for later refits once observed, matching a real walk-forward workflow.")


# 6. Data quality, historical comparison, and transparent feature weights.
with explorer:
    a, b, c = st.columns(3)
    a.metric("Company records in selected history", f"{len(company_data):,}")
    b.metric("Invalid/duplicate rows removed", removed)
    c.metric("Company symbols in uploaded dataset", len(symbols))
    st.caption("Historical datasets can include former constituents or renamed symbols, so the symbol count may exceed 50. Company series are modelled separately.")
    st.dataframe(company_data, width="stretch")
    with st.expander("Missing values and summary statistics"):
        st.dataframe(company_data.isna().sum().rename("Missing values").to_frame())
        st.dataframe(company_data.select_dtypes(include="number").describe(), width="stretch")
    st.subheader("Compare price movements")
    chosen = st.multiselect("Up to four companies", symbols, default=[symbol], max_selections=4)
    figure = go.Figure()
    for selected in chosen:
        subset = raw.loc[(raw["Symbol"] == selected) & (raw["Date"] >= start_date) & (raw["Date"] <= end_date), ["Date", "Close"]].dropna()
        subset = subset.loc[subset["Close"] > 0].sort_values("Date").drop_duplicates("Date")
        if len(subset):
            figure.add_trace(go.Scatter(x=subset["Date"], y=subset["Close"] / subset["Close"].iloc[0] * 100, name=selected))
    if chosen:
        st.plotly_chart(chart_style(figure, "Each series starts at 100 at its first available date", "Rebased price"), width="stretch")
        st.caption("Unadjusted price comparison; excludes dividends and is not a total-return comparison. Companies may have different first available dates.")
    with st.expander("Inspect the inputs and learned weights"):
        st.dataframe(features[["Date"] + feature_names].tail(10), width="stretch")
        weights = forecast["weights"].sort_values("Weight")
        figure = go.Figure(go.Bar(x=weights["Weight"], y=weights["Feature"], orientation="h", marker_color=COLORS["predicted"]))
        figure = chart_style(figure, "Standardized feature coefficients", "Feature")
        figure.update_layout(xaxis_title="Coefficient on log return")
        st.plotly_chart(figure, width="stretch")
        st.caption("These are model coefficients, not causal explanations. Correlated features can change their magnitude and sign.")
    st.download_button("Download selected company data", company_data.to_csv(index=False), f"{symbol}_selected_data.csv", "text/csv")

st.caption("Built for learning and reproducible evaluation. Forecasts are not investment recommendations.")
