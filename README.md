# Sustainable Alpha — ESG ETFs (US & EU)

A compact, production-ready portfolio that combines a **reusable Python utility module**, a **research notebook**, and two **concise PDF companions** to perform end-to-end quantitative analysis on **Sustainable / ESG ETFs** across the **U.S.** and **Eurozone**. The project emphasizes clean, reproducible workflows, model diagnostics aligned with stylized facts, and investor-grade risk/performance reporting.

<p align="center">
  <img alt="ESG" src="https://img.shields.io/badge/ESG-Quant%20Finance-informational">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.13%2B-blue">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20&%20tools-brightgreen">
</p>

---

## 📁 Repository layout

├── sustainable_alpha_esg_etfs_analysis.ipynb # Research notebook (US & EU ESG ETFs)
├── utils.py # Reusable analytics & plotting utilities
├── Time_series.pdf # Theory note: stochastic & time-series foundations
└── stylized_facts_of_time_series.pdf # Stylized facts companion (empirical properties)


---

## ✨ What you get

- **Clean data pipeline** from Yahoo Finance with robust gap handling, winsorization, and return computation.
- **Comprehensive metrics** per ETF: annualized return/volatility, Sharpe, Sortino, drawdowns, Calmar, VaR/CVaR (hist. & parametric), beta/alpha vs benchmark, hit ratio, omega, skew/kurtosis, etc.
- **Benchmark-aware tracking**: tracking difference & tracking error vs S&P 500 (US) and Euro Stoxx 50 (EU).
- **Model diagnostics** against stylized facts: Normal vs Student-t fits, GARCH(1,1) with Normal/t innovations, QQ plots, ACFs of returns/|returns|, conditional volatility plots.
- **Tail density tools**: spline-smoothed PDFs and KDE overlays for visual tail assessment.
- **Presentation-ready plots** and auto-generated **plain-English commentary** summarizing insights.

---

## 📓 The notebook at a glance

**File:** `sustainable_alpha_esg_etfs_analysis.ipynb`

- **Universe (examples):**  
  US — `ESGU`, `IGEB`, `SMMD`, `VTC`  
  EU — `UIMR.DE`, `IUSG`, `ESEG.MI`, `EXW1.DE`, `XEIN.DE`
- **Benchmarks:** `^GSPC` (S&P 500) and `^STOXX50E` (Euro Stoxx 50).
- **Default configuration:** start date `2018-01-01`, `rf_annual = 2%`, log-returns.
- **Workflow:** load → clean → compute metrics → benchmark alignment → tracking stats → stylized-facts diagnostics (Gaussian, Student-t, GARCH) → conclusions.

Outputs include **summary tables**, **rolling analytics**, **distributional fits**, **GARCH diagnostics**, and **region-level commentary**.

---

## 🧰 The `utils.py` toolbox (highlights)

- **Data & metrics**
  - `download_analyze_with_metrics(...)` → returns a tidy `panel` of prices/returns, a per-ticker `summary` table, `extras` (drawdowns, rolling stats, benchmark panel), and an **auto commentary** string.
  - `compute_tracking_metrics(panel, extras, ...)` → tracking difference (TD) & tracking error (TE) vs the benchmark.
- **Visualization**
  - `plot_prices(...)` (normalized or raw) and `plot_returns(..., kind="line"|"cum"|"bar"|"hist")` with optional rolling-vol overlay.
- **Distribution & volatility models**
  - `fit_normal(...)`, `fit_tstudent(...)` (+ overlays via `plot_hist_with_pdf_normal_t(...)`).
  - `fit_garch(..., dist="normal"|"t")` and diagnostics: `plot_hist_stdres_garch(...)`.
  - One-shot diagnostics: `gaussian_qq_acf_diagnostics(...)`, `student_t_qq_acf_diagnostics(...)`, `garch_diagnostics(...)`.
- **Tail density tools**
  - `spline_density_from_hist(...)` + `plot_hist_with_spline(...)`.
  - `kde_density(...)` + `plot_hist_with_kde(...)`.

All functions are documented with clear arguments and return values for easy reuse.

---

## 📚 Companion notes (PDFs)

This repository includes two concise references that connect theory to implementation:

1. **`Time_series.pdf` — Stochastic & Time-Series Foundations**  
   A master’s-level primer covering:
   - Returns (simple vs log), scaling, and aggregation.  
   - Stationarity & ergodicity (why they matter for estimation).  
   - ARCH/GARCH fundamentals: specification, stationarity conditions, persistence, unconditional variance, half-life.  
   - Innovation choices (Gaussian vs Student-t) and their impact on tail risk.  
   - Diagnostics: ACF/Ljung–Box, ARCH tests, QQ-plots.  
   - Risk metrics reflected in code: VaR, CVaR, drawdowns, Sharpe/Sortino, CAPM alpha/beta.

2. **`stylized_facts_of_time_series.pdf` — Stylized Facts of Asset Returns**  
   A practical overview of the empirical properties that real-world return series exhibit, including:
   - Weak linear autocorrelation in returns vs **volatility clustering** in |returns| and squared returns.  
   - **Heavy tails**, tail-risk relevance for VaR/CVaR, and aggregation effects.  
   - **Leverage effect** (asymmetric volatility response).  
   - Implications for model choice, diagnostics, and portfolio risk measurement.

### Suggested reading flow

- **First:** skim `stylized_facts_of_time_series.pdf` to internalize the empirical constraints.  
- **Then:** study `Time_series.pdf` to see how the models/metrics in `utils.py` operationalize those facts.  
- **Finally:** open the notebook and replicate the full, benchmark-aware pipeline on ESG ETFs.

---

## 🚀 Quickstart

> Requires Python **3.10+** (tested on 3.13) and the packages below.

```python
# 1) Install
# pip install numpy pandas yfinance matplotlib scipy statsmodels arch

# 2) Import tools
from utils import (
    download_analyze_with_metrics, compute_tracking_metrics,
    plot_prices, plot_returns,
    gaussian_qq_acf_diagnostics, student_t_qq_acf_diagnostics, garch_diagnostics
)

# 3) Define universe and benchmark
tickers_us   = ["ESGU", "IGEB", "SMMD", "VTC"]
tickers_eu   = ["UIMR.DE", "IUSG", "ESEG.MI", "EXW1.DE", "XEIN.DE"]
benchmark_us = "^GSPC"       # S&P 500
benchmark_eu = "^STOXX50E"   # Euro Stoxx 50

# 4) Run end-to-end analysis (US example)
panel_us, summary_us, extras_us, commentary_us = download_analyze_with_metrics(
    tickers=tickers_us,
    start_date="2018-01-01",
    benchmark=benchmark_us,
    rf_annual=0.02,
    return_method="log",
    make_commentary=True,
    benchmark_name_for_comment="S&P 500",
    return_units_for_comment="daily",
)
print(commentary_us)  # investor-grade, auto-generated insights

# 5) Tracking vs benchmark
te_df = compute_tracking_metrics(panel_us, extras_us)
print(te_df)

# 6) Diagnostics for a single ETF (e.g., ESGU)
gaussian_qq_acf_diagnostics(panel_us, ticker="ESGU")
student_t_qq_acf_diagnostics(panel_us, ticker="ESGU")
garch_diagnostics(panel_us, ticker="ESGU", dist="t")

🧪 Methodological notes

Stylized facts checked: (i) negligible linear autocorrelation of returns; (ii) volatility clustering; (iii) heavy tails; (iv) non-Gaussian standardized residuals under homoskedastic models; (v) improved fit under t-innovations and GARCH with persistent volatility.

Risk measures: historical & parametric VaR/CVaR at configurable confidence (default 95%), drawdowns & time-in-drawdown, hit ratio, and omega.

Benchmarking: CAPM beta/alpha & correlation vs regional benchmark; TD/TE on aligned trading days.

Rolling analytics: rolling annualized volatility and Sharpe over configurable windows (e.g., 63/126/252 trading days).

📦 Requirements

Core: numpy, pandas, yfinance, matplotlib, scipy

Models/diagnostics: arch (GARCH), statsmodels (ACF/diagnostics)

Python: 3.10+ (developed on 3.13)

Install: pip install numpy pandas yfinance matplotlib scipy statsmodels arch

🔁 Reproducibility & data

Data source: Yahoo Finance via yfinance (free, unaudited; subject to revisions/outages).

Determinism: All computations are deterministic for a fixed dataset and configuration.

Dates: Default start date 2018-01-01. Adjust to your analysis horizon.

Note: Metrics are sensitive to data-cleaning choices (e.g., winsorization, forward-fills). The helpers expose these knobs for transparency.

📈 How to extend

Add new ETFs or regions by editing the ticker lists.

Compare ESG vs non-ESG peers using the same workflow.

Swap benchmarks (e.g., MSCI Europe) or add FX overlays for cross-currency views.

Persist results (CSV/Parquet) or automate reporting (e.g., nbconvert, GitHub Actions).

🤝 How to use this as a portfolio

Keep the repo lean (one utils.py, one flagship notebook, two crisp PDFs).

Pin results with a fixed end date and commit figures (optional) for recruiter-friendly browsing.

Add a short “Key Findings” section to the notebook output if targeting non-quant stakeholders.

📝 License & attribution

This repository is provided for research and educational purposes. No investment advice is given.
Data is © respective providers. You are responsible for complying with Yahoo Finance’s terms of use.

If you use this work, please reference Sustainable Alpha — ESG ETFs (US & EU) and include a link to the repository.

🙋 Author

Guillem Borràs — MSc Quantitative Finance · Physicist · Quantum Computing (IBM).