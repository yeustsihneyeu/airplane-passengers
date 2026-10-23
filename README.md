# airplane-passengers

Exploratory data analysis (EDA) of the classic **International Airline Passengers** time series:
monthly number of international airline passengers (in thousands), January 1949 – December 1960.

The goal is to understand the structure of the series (trend, seasonality, noise),
find the transformations needed to make it stationary, and choose candidate forecasting models.

## Dataset

- Source: Kaggle, [`andreazzini/international-airline-passengers`](https://www.kaggle.com/datasets/andreazzini/international-airline-passengers)
- 144 monthly observations, one column: `passengers`
- Downloaded automatically in the notebook with `kagglehub` (no manual download needed)

## Project structure

```
airplane-passengers/
├── notebooks/
│   └── eda.ipynb       # full exploratory analysis
├── requirements.txt    # Python dependencies
└── README.md
```

## What the notebook covers

| # | Section | What is done |
|---|---------|--------------|
| 1 | Goal | Questions the analysis answers |
| 2 | Load the data | Download with `kagglehub`, clean, set a monthly `DatetimeIndex` |
| 3 | Data overview | Shape, types, missing values, duplicates, missing months, statistics |
| 4 | Time series plot | Overall shape of the series |
| 5 | Trend | 12-month rolling mean and rolling std, yearly mean/std |
| 6 | Seasonality | Seasonal plot, boxplots by month, seasonal index, yearly totals |
| 7 | Decomposition | Additive vs multiplicative `seasonal_decompose`, residual comparison |
| 8 | Variance and transformations | Spread vs level, log and Box-Cox transformations |
| 9 | Stationarity | ADF and KPSS tests, regular and seasonal differencing |
| 10 | Autocorrelation | ACF / PACF before and after differencing |
| 11 | Conclusions | Findings, candidate models, next steps |

## Key findings

- **Data quality:** no missing values, no missing months, no duplicates, no clear outliers.
- **Trend:** strong upward trend — the 12-month rolling mean grows about 3.8× (≈125k → ≈475k passengers).
- **Seasonality:** stable 12-month cycle. Peak in July–August (~23% above the trend), lowest in November (~20% below).
- **Multiplicative structure:** seasonal swings grow with the level (mean vs std correlation 0.99);
  multiplicative decomposition leaves 2.4% residuals vs 6.1% for additive.
- **Transformation:** a log transformation stabilizes the variance better than Box-Cox (λ ≈ 0.15).
- **Stationarity:** the series becomes stationary after `log` + `diff(12)` + `diff(1)` (ADF p < 0.001, KPSS p ≥ 0.1).
- **Autocorrelation:** after differencing, significant spikes at lag 1 and lag 12 point to a non-seasonal MA(1) and a seasonal MA(1).

## Candidate models

- **SARIMA(0,1,1)(0,1,1,12)** on the log series — the classic "airline model" (main candidate)
- SARIMA(1,1,0)(1,1,0,12) — alternative
- Holt-Winters with multiplicative seasonality

Forecasts made on the log scale must be converted back with `np.exp()`.

## Next steps

- Time-based split: last 12–24 months as the test set
- Baseline: seasonal naive (value from the same month last year)
- Compare models by AIC and test error (MAPE, RMSE)
- Check model residuals for white noise (Ljung-Box test)

## Getting started

Requires Python 3.9+.

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Open the notebook (in VS Code, or install Jupyter)
pip install jupyter
jupyter notebook notebooks/eda.ipynb
```

The dataset is downloaded on the first run and cached by `kagglehub` in `~/.cache/kagglehub`.

## Tech stack

`pandas`, `numpy`, `matplotlib`, `statsmodels`, `scipy`, `kagglehub`, Jupyter
