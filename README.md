# Bayesian Machine Learning Laboratory

This Streamlit project contains two Bayesian machine-learning demonstrations:

1. Heart-disease probability estimation with Gaussian Naive Bayes and the UCI
   Cleveland dataset.
2. Stock-price forecasting with Bayesian Ridge regression, lagged adjusted
   closing prices, test metrics, and 95% uncertainty intervals.

## Run the project

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The demonstrations are for education only. They are not medical or financial
advice.
