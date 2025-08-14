###############################
# @author: Guillem Borràs     #
# MSc. Quantitative Finance   #
# Physicist                   #
# Quantum Computing IBM       #
###############################

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
from scipy import stats
from scipy.stats import norm as _norm
from scipy.stats import t as _t
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
from scipy.interpolate import UnivariateSpline
try:
    from arch.univariate import ConstantMean, GARCH, Normal, StudentsT
    _HAS_ARCH = True
except Exception:
    _HAS_ARCH = False
try:
    from scipy.stats import norm
except Exception:
    norm = None
try:
    from statsmodels.graphics.tsaplots import plot_acf
except Exception as _e:
    plot_acf = None
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


##########################################    
#NEW: AUTHOMATIC FINANCIAL COMMENTARY!   #
##########################################
# --- 1) Commentary generator (as before) ---
def generate_financial_commentary(summary_df: pd.DataFrame,
                                   benchmark_name: str = "S&P 500",
                                   return_units: str = "daily") -> str:
    """
    Build an automatic commentary from a summary metrics DataFrame.
    Expects (when available): ['ticker','beta','ann_return','ann_vol','sharpe','skew',
                               'kurt_excess','hist_var','hist_cvar','corr_bench'].
    """
    lines = []
    lines.append(f"Analysis of the {return_units} returns for the assets in relation to {benchmark_name}:")

    # 1) Beta & correlation
    if {"beta", "corr_bench"}.issubset(summary_df.columns):
        avg_beta = summary_df['beta'].mean(skipna=True)
        avg_corr = summary_df['corr_bench'].mean(skipna=True)
        level = "strong" if (pd.notna(avg_corr) and avg_corr > 0.7) else "moderate"
        lines.append(
            f"- The assets show an average beta of {avg_beta:.2f} and an average correlation "
            f"of {avg_corr:.2f} with {benchmark_name}, indicating {level} market comovement."
        )

    # 2) Return & volatility
    if {"ann_return","ann_vol"}.issubset(summary_df.columns):
        avg_ret = summary_df['ann_return'].mean(skipna=True)
        avg_vol = summary_df['ann_vol'].mean(skipna=True)
        lines.append(
            f"- The average annualized return across assets is {avg_ret:.2%}, "
            f"with an average annualized volatility of {avg_vol:.2%}."
        )

    # 3) Sharpe
    if "sharpe" in summary_df.columns:
        avg_sharpe = summary_df['sharpe'].mean(skipna=True)
        perf = "attractive" if pd.notna(avg_shre:=avg_sharpe) and avg_shre > 1 else "modest"
        lines.append(f"- The average Sharpe ratio is {avg_sharpe:.2f}, suggesting {perf} risk-adjusted performance.")

    # 4) Skew & kurtosis
    if {"skew","kurt_excess"}.issubset(summary_df.columns):
        avg_skew = summary_df['skew'].mean(skipna=True)
        avg_kurt = summary_df['kurt_excess'].mean(skipna=True)
        skew_desc = "slight right-tail bias" if (pd.notna(avg_skew) and avg_skew > 0) else \
                    "slight left-tail bias" if (pd.notna(avg_skew) and avg_skew < 0) else "near symmetry"
        tail_desc = "fat tails" if (pd.notna(avg_kurt) and avg_kurt > 0) else "thin tails"
        lines.append(
            f"- The return distribution shows {skew_desc} on average and {tail_desc} "
            f"(avg. excess kurtosis = {avg_kurt:.2f}), implying a departure from normality and potential tail risk."
        )

    # 5) VaR & CVaR
    if {"hist_var","hist_cvar"}.issubset(summary_df.columns):
        avg_var = summary_df['hist_var'].mean(skipna=True)
        avg_cvar = summary_df['hist_cvar'].mean(skipna=True)
        lines.append(
            f"- At 95% confidence, the average historical VaR is {avg_var:.2%} and the Conditional VaR is {avg_cvar:.2%}, "
            "highlighting the scale of potential losses under adverse conditions."
        )

    # 6) Wrap-up
    lines.append("- Overall, the assets exhibit characteristics typical of equities: high volatility relative to mean returns, "
                 "significant market sensitivity, and non-negligible tail risk.")
    return "\n".join(lines)

def save_commentary_markdown(commentary: str, path: str = "commentary.md", title: str = "Analysis Commentary"):
    """
    Save the generated commentary to a Markdown file.
    """
    if not isinstance(commentary, str) or not commentary.strip():
        raise ValueError("Empty commentary.")
    md = f"# {title}\n\n{commentary}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    return path

#####################################
#LOADING DATA FROM YAHOO FINANCE    #         
#####################################
# ------------------ Cleaning Utilities ------------------

def _clean_prices(
    px: pd.DataFrame,
    *,
    min_non_na_ratio: float = 0.85,
    max_ffill: int = 3,
    drop_constant: bool = True,
    nonpositive_to_nan: bool = True,
) -> pd.DataFrame:
    """
    Clean close/adj-close price panel (rows: dates, cols: tickers).

    Steps:
      - sort by date and drop duplicate index
      - coerce to numeric
      - set non-positive prices to NaN (optional)
      - forward-fill small gaps (limit=max_ffill) + backfill head (limit=1)
      - drop columns with too many NaNs (coverage < min_non_na_ratio)
      - drop constant series (optional)
      - drop all-NaN rows
    """
    if isinstance(px, pd.Series):
        px = px.to_frame()

    # order & deduplicate
    px = px.copy()
    px = px[~px.index.duplicated(keep="last")]
    px = px.sort_index()

    # numeric
    px = px.apply(pd.to_numeric, errors="coerce")

    # non-positive -> NaN (prices should be > 0)
    if nonpositive_to_nan:
        px = px.mask(px <= 0)

    # fill small gaps (keep gaps large as NaN)
    px = px.ffill(limit=max_ffill).bfill(limit=1)

    # remove columns with low coverage
    coverage = px.notna().mean(axis=0)
    keep_cols = coverage[coverage >= min_non_na_ratio].index
    px = px[keep_cols]

    # drop constant series
    if drop_constant and px.shape[1] > 0:
        nunique = px.nunique(dropna=True)
        const_cols = nunique[nunique <= 1].index
        px = px.drop(columns=const_cols, errors="ignore")

    # drop rows fully NaN
    px = px.dropna(how="all")

    return px


def _compute_returns(
    px: pd.DataFrame,
    *,
    method: str = "simple",          # "simple" or "log"
    winsorize: tuple | None = None,  # e.g., (0.01, 0.99) to clip per-column
) -> pd.DataFrame:
    """Compute returns from clean price panel, with optional per-column winsorization."""
    if method not in {"simple", "log"}:
        raise ValueError("method must be 'simple' or 'log'.")

    if method == "simple":
        rets = px.pct_change()
    else:
        rets = np.log(px / px.shift(1))

    if winsorize:
        lo, hi = winsorize
        def _clip(s: pd.Series):
            ql, qh = s.quantile([lo, hi])
            return s.clip(lower=ql, upper=qh)
        rets = rets.apply(_clip, axis=0)

    return rets


# ------------------ yfinance helper ------------------

def _extract_close(df, tickers):
    """Robustly extract Close (or Adj Close) from yfinance output for multi/single index."""
    if isinstance(df, pd.Series):
        name = tickers if isinstance(tickers, str) else tickers[0]
        return df.to_frame(name=name)

    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if "Close" in lvl0: 
            out = df.xs("Close", axis=1, level=0)
        elif "Adj Close" in lvl0:
            out = df.xs("Adj Close", axis=1, level=0)
        else:
            raise ValueError(f"'Close'/'Adj Close' not found. Levels: {list(pd.unique(lvl0))}")
    else:
        if "Close" in df.columns:
            out = df["Close"]
            if isinstance(out, pd.Series):
                name = tickers if isinstance(tickers, str) else tickers[0]
                out = out.to_frame(name=name)
        else:
            out = df

    if isinstance(out, pd.Series):
        name = tickers if isinstance(tickers, str) else tickers[0]
        out = out.to_frame(name=name)

    return out


# ------------------ Main Function (now calls cleaner automatically) ------------------

def download_analyze_with_metrics(
    tickers,
    start_date,
    benchmark=None,
    rf_annual=0.0,
    periods_per_year=252,
    auto_adjust=True,
    rolling_windows=(63, 126, 252),
    var_level=0.95,
    *,
    # cleaning knobs:
    min_non_na_ratio: float = 0.85,
    max_ffill: int = 3,
    drop_constant: bool = True,
    nonpositive_to_nan: bool = True,
    winsorize_returns: tuple | None = None,   # e.g. (0.01, 0.99)
    return_method: str = "log",            # "simple" or "log"
    make_commentary: bool = True,
    benchmark_name_for_comment: str = "S&P 500",
    return_units_for_comment: str = "daily"
):
    """
    Returns:
      panel   : tidy DataFrame [date, ticker, price, return]
      summary : per-ticker metrics (CAGR, ann_return, ann_vol, Sharpe, Sortino, Calmar, MDD, VaR/CVaR, Omega, skew, kurt, beta/alpha/corr)
      extras  : dict with drawdowns, correlation_matrix, rolling metrics

    Automatically cleans prices via `_clean_prices` and computes returns via `_compute_returns`.
    """
    # --- 1) Download raw prices
    px_raw = yf.download(tickers, start=start_date, progress=False, auto_adjust=auto_adjust)

    # --- 2) Extract close and CLEAN
    px = _extract_close(px_raw, tickers)
    px = _clean_prices(
        px,
        min_non_na_ratio=min_non_na_ratio,
        max_ffill=max_ffill,
        drop_constant=drop_constant,
        nonpositive_to_nan=nonpositive_to_nan,
    )

    # Track actual tickers kept (after cleaning)
    kept_tickers = list(px.columns)

    # --- 3) Returns (winsorization optional)
    rets = _compute_returns(px, method=return_method, winsorize=winsorize_returns)

    # --- 4) Tidy panel
    panel = px.stack(dropna=False).rename("price").to_frame()
    panel["return"] = rets.stack(dropna=False)
    panel.index.set_names(["date", "ticker"], inplace=True)
    panel = panel.reset_index().sort_values(["ticker", "date"])

        # --- 5) Benchmark (optional)
    bm = None
    bm_panel = None      # NEW
    bm_label = None      # NEW
    if benchmark:
        bm_raw = yf.download(benchmark, start=start_date, progress=False, auto_adjust=auto_adjust)
        bm_px = _extract_close(bm_raw, benchmark).iloc[:, 0]              # prices (Series)
        bm_px = _clean_prices(bm_px.to_frame(), max_ffill=max_ffill).iloc[:, 0]

        bm_px_aligned = bm_px.reindex(px.index)

        bm = _compute_returns(bm_px_aligned.to_frame(), method=return_method).iloc[:, 0]
        bm = bm.rename("bm_return") 

        
        bm_label = str(benchmark)   
        bm_panel = pd.DataFrame({
            "date": bm_px_aligned.index,
            "ticker": bm_label,
            "price": bm_px_aligned.values,
            "return": bm.reindex(px.index).values
        })


    # --- 6) 
    metrics = []
    rf_daily = (1 + rf_annual) ** (1 / periods_per_year) - 1

    def _drawdown_series(returns: pd.Series):
        cum = (1.0 + returns.fillna(0)).cumprod()
        peak = cum.cummax()
        dd = cum / peak - 1.0
        return cum, dd

    def _hist_var_cvar(returns: pd.Series, level=0.95):
        r = returns.dropna()
        if r.empty:
            return np.nan, np.nan
        q = r.quantile(1 - level)
        var_pos = -q
        cvar_pos = -r[r <= q].mean() if (r <= q).any() else np.nan
        return float(var_pos), float(cvar_pos)

    def _parametric_var_cvar(mu, sigma, level=0.95):
        if not norm or np.isnan(mu) or np.isnan(sigma):
            return np.nan, np.nan
        z = norm.ppf(1 - level)
        phi = norm.pdf(z)
        var_pos = -(mu + sigma * z)
        cvar_pos = (sigma * phi / (1 - level)) - mu
        return float(var_pos), float(cvar_pos)

    for t in kept_tickers:
        r = rets[t].dropna()
        if r.empty:
            metrics.append({"ticker": t, "n_obs": 0, **{k: np.nan for k in [
                "start","end","mean","ann_return","ann_vol","sharpe","sortino",
                "max_drawdown","calmar","hist_var","hist_cvar","param_var","param_cvar",
                "omega_0","skew","kurt_excess","beta","alpha_annual","corr_bench",
                "hit_ratio","avg_gain","avg_loss","time_in_drawdown_pct"
            ]}})
            continue

        mu_d, sd_d = r.mean(), r.std()
        ann_return = mu_d * periods_per_year
        ann_vol = sd_d * np.sqrt(periods_per_year)

        cum, dd = _drawdown_series(r)
        n_days = r.shape[0]
        cagr = cum.iloc[-1] ** (periods_per_year / n_days) - 1 if n_days > 0 else np.nan
        max_dd = dd.min() if not dd.empty else np.nan
        time_in_dd = (dd < 0).mean() * 100 if not dd.empty else np.nan
        calmar = (cagr / abs(max_dd)) if (pd.notna(cagr) and pd.notna(max_dd) and max_dd != 0) else np.nan

        downside = r[r < 0]
        downside_dev_ann = downside.std() * np.sqrt(periods_per_year) if not downside.empty else np.nan
        sharpe = ((ann_return - rf_annual) / ann_vol) if ann_vol and not np.isnan(ann_vol) and ann_vol != 0 else np.nan
        sortino = ((ann_return - rf_annual) / downside_dev_ann) if downside_dev_ann and not np.isnan(downside_dev_ann) and downside_dev_ann != 0 else np.nan

        hist_var, hist_cvar = _hist_var_cvar(r, level=var_level)
        param_var, param_cvar = _parametric_var_cvar(mu_d, sd_d, level=var_level)

        gains = (r[r > 0]).sum()
        losses = (-r[r < 0]).sum()
        omega_0 = (gains / losses) if losses and losses != 0 else np.nan

        skew = r.skew()
        kurt_excess = r.kurt()

        hit_ratio = (r > 0).mean()
        avg_gain = r[r > 0].mean() if (r > 0).any() else np.nan
        avg_loss = r[r < 0].mean() if (r < 0).any() else np.nan

        beta = alpha_annual = corr_b = np.nan
        if bm is not None:
            rb = bm.reindex_like(r)
            both = pd.concat([r, rb], axis=1, join="inner").dropna()
            if not both.empty and both.iloc[:, 1].var() != 0:
                cov = both.iloc[:, 0].cov(both.iloc[:, 1])
                var_b = both.iloc[:, 1].var()
                beta = cov / var_b if var_b != 0 else np.nan
                ann_ret_b = both.iloc[:, 1].mean() * periods_per_year
                alpha_annual = (ann_return - rf_annual) - beta * (ann_ret_b - rf_annual)
                corr_b = both.iloc[:, 0].corr(both.iloc[:, 1])

        metrics.append({
            "ticker": t,
            "start": r.index.min(),
            "end": r.index.max(),
            "n_obs": n_days,
            "mean": r.mean(),
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": float(max_dd) if pd.notna(max_dd) else np.nan,
            "calmar": calmar,
            "hist_var": hist_var,
            "hist_cvar": hist_cvar,
            "param_var": param_var,
            "param_cvar": param_cvar,
            "omega_0": omega_0,
            "skew": skew,
            "kurt_excess": kurt_excess,
            "beta": beta,
            "alpha_annual": alpha_annual,
            "corr_bench": corr_b,
            "hit_ratio": hit_ratio,
            "avg_gain": avg_gain,
            "avg_loss": avg_loss,
            "time_in_drawdown_pct": time_in_dd
        })

    summary = pd.DataFrame(metrics).sort_values("ticker").reset_index(drop=True)

    # --- 7) Extras
    def _drawdown_df(series: pd.Series):
        cum = (1.0 + series.fillna(0)).cumprod()
        dd = cum / cum.cummax() - 1.0
        return pd.DataFrame({"date": cum.index, "equity": cum.values, "drawdown": dd.values})

    drawdowns = {t: _drawdown_df(rets[t].dropna()) for t in kept_tickers if rets[t].notna().any()}
    corr_mat = rets.corr(min_periods=30)

    rolling = {}
    for w in rolling_windows:
        vol = rets.rolling(w).std() * np.sqrt(periods_per_year)
        if rf_annual == 0:
            sharpe_roll = (rets.rolling(w).mean() * periods_per_year) / (vol.replace(0, np.nan))
        else:
            rf_d = rf_annual / periods_per_year
            sharpe_roll = ((rets.rolling(w).mean() - rf_d) * periods_per_year) / (vol.replace(0, np.nan))
        rolling[w] = {"vol": vol, "sharpe": sharpe_roll}

    extras = {"drawdowns": drawdowns, "correlation_matrix": corr_mat, "rolling": rolling}
    if bm_panel is not None:
        extras["benchmark_panel"] = bm_panel
        extras["benchmark_label"] = bm_label

    # --- 8) Build tidy panel with metrics (optional join)
    panel = panel.merge(
        summary[["ticker", "ann_return", "ann_vol"]],
        on="ticker", how="left"
    )
    commentary = None
    if make_commentary:
        try:
            commentary = generate_financial_commentary(
                summary_df=summary,
                benchmark_name=benchmark_name_for_comment if benchmark else "the benchmark",
                return_units=return_units_for_comment
            )
        except Exception as e:
            commentary = f"(Commentary generation failed: {e})"

    return panel, summary, extras, commentary

# -----------------------------
# Helpers: detect & reshape
# -----------------------------
def _to_wide_prices(data: pd.DataFrame) -> pd.DataFrame:
    """Return wide (index=date, cols=tickers) price panel from wide or tidy input."""
    if {"date", "ticker", "price"}.issubset(data.columns):
        df = (data[["date", "ticker", "price"]]
              .dropna(subset=["price"])
              .pivot(index="date", columns="ticker", values="price")
              .sort_index())
        return df
    # assume wide already (index is datetime, columns tickers)
    if not isinstance(data.index, pd.DatetimeIndex):
        # try infer datetime
        data = data.copy()
        data.index = pd.to_datetime(data.index)
    return data.sort_index()

def _to_wide_returns(data: pd.DataFrame) -> pd.DataFrame:
    """Return wide (index=date, cols=tickers) returns panel from wide/tidy returns or prices."""
    if {"date", "ticker", "return"}.issubset(data.columns):
        df = (data[["date", "ticker", "return"]]
              .pivot(index="date", columns="ticker", values="return")
              .sort_index())
        return df
    # if it's prices, compute returns
    if {"date", "ticker", "price"}.issubset(data.columns) or isinstance(data.index, pd.DatetimeIndex):
        px = _to_wide_prices(data)
        return px.pct_change()
    # else assume it's already returns wide
    if not isinstance(data.index, pd.DatetimeIndex):
        data = data.copy()
        data.index = pd.to_datetime(data.index)
    return data.sort_index()

# -----------------------------
# 1) Price plot
# -----------------------------
def plot_prices(data: pd.DataFrame,
                normalize: bool = True,
                logy: bool = False,
                annotate_last: bool = True,
                title: str | None = None,
                start_date: str | None = None,
                end_date: str | None = None):
    """
    Plot multi-asset price panel with optional date filtering.

    start_date, end_date : str or None
        Filter the plotted data to this date range (YYYY-MM-DD).
    """
    df = _to_wide_prices(data).apply(pd.to_numeric, errors="coerce").ffill().dropna(how="all")

    # --- Date filtering ---
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    if df.empty:
        raise ValueError("No valid price data in the selected date range.")

    if normalize:
        first_valid = df.apply(pd.Series.first_valid_index)
        start_idx = max([i for i in first_valid if pd.notna(i)])
        df = df.loc[start_idx:].div(df.loc[start_idx]).mul(100.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in df.columns:
        ax.plot(df.index, df[col], linewidth=1.8, label=col)

    if logy and not normalize:
        ax.set_yscale("log")

    start, end = df.index.min().date(), df.index.max().date()
    base = "Comparative price evolution"
    if normalize:
        base += " (first common date = 100)"
    ax.set_title(title or f"{base}\n{start} — {end}", pad=12)

    ax.set_xlabel("Date")
    ax.set_ylabel("Index level" if normalize else "Price")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(3, 6, 9, 12)))
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.18)

    if not normalize and ax.get_yscale() != "log":
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))

    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="Ticker")

    if annotate_last:
        last_vals = df.iloc[-1]
        for col in df.columns:
            ax.annotate(f"{last_vals[col]:,.2f}",
                        xy=(df.index[-1], df[col].iloc[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

def compute_tracking_metrics(
    panel: pd.DataFrame,
    extras: dict,
    tickers: list | None = None,
    periods_per_year: int = 252,
    min_overlap: int = 30,
) -> pd.DataFrame:
    """
    Compute Tracking Error (TE) and Tracking Difference (TD) vs the benchmark provided
    by download_analyze_with_metrics (via extras["benchmark_panel"]).

    Parameters
    ----------
    panel : tidy DataFrame with columns ['date','ticker','price','return', ...]
    extras : dict from download_analyze_with_metrics; must contain 'benchmark_panel'
    tickers : list or None
        If None, use all tickers in panel (excluding the benchmark label, if present).
    periods_per_year : int
        Annualization factor (252 for daily).
    min_overlap : int
        Minimum overlapping days required to compute metrics.

    Returns
    -------
    pd.DataFrame with one row per ticker:
        ['ticker','n_overlap','start','end',
         'td_daily','td_annual','te_daily','te_annual']
    Notes
    -----
    TD (tracking difference) is the mean of active returns (ETF - benchmark).
    TE (tracking error) is the standard deviation of active returns.
    Annualization uses:
        td_annual   = td_daily * periods_per_year
        te_annual   = te_daily * sqrt(periods_per_year)
    """
    if not isinstance(extras, dict) or "benchmark_panel" not in extras:
        raise ValueError("extras must contain 'benchmark_panel' with columns ['date','ticker','price','return'].")

    bm_panel = extras["benchmark_panel"]
    bm_series = (
        bm_panel[["date", "return"]]
        .dropna()
        .assign(date=pd.to_datetime(bm_panel["date"]))
        .set_index("date")["return"]
        .astype(float)
        .rename("bm")
    )

    # Default: all tickers in panel except the benchmark label (if included there)
    if tickers is None:
        all_tickers = sorted(pd.unique(panel["ticker"]))
        bench_label = extras.get("benchmark_label")
        tickers = [t for t in all_tickers if (bench_label is None or t != bench_label)]

    rows = []
    for t in tickers:
        s = (
            panel.loc[panel["ticker"] == t, ["date", "return"]]
            .dropna()
            .assign(date=pd.to_datetime(panel.loc[panel["ticker"] == t, "date"]))
            .set_index("date")["return"]
            .astype(float)
            .rename("asset")
        )
        df = pd.concat([s, bm_series], axis=1, join="inner").dropna()
        if df.shape[0] < min_overlap:
            rows.append({
                "ticker": t, "n_overlap": df.shape[0],
                "start": df.index.min(), "end": df.index.max(),
                "td_daily": np.nan, "td_annual": np.nan,
                "te_daily": np.nan, "te_annual": np.nan
            })
            continue

        active = df["asset"] - df["bm"]
        td_daily = float(active.mean())
        te_daily = float(active.std(ddof=0))  # MLE std, consistent with many index factsheets
        td_annual = td_daily * periods_per_year
        te_annual = te_daily * np.sqrt(periods_per_year)

        var_b = float(df["bm"].var(ddof=0))
        rows.append({
            "ticker": t,
            "n_overlap": df.shape[0],
            "start": df.index.min(),
            "end": df.index.max(),
            "td_daily": td_daily,
            "td_annual": td_annual,
            "te_daily": te_daily,
            "te_annual": te_annual
        })

    out = pd.DataFrame(rows)
    # Nice ordering
    cols = ["ticker","n_overlap","start","end","td_daily","td_annual","te_daily","te_annual"]
    return out[cols].sort_values("te_annual")


# -----------------------------
# 1) Returns plot
# -----------------------------
def plot_returns(data: pd.DataFrame,
                 kind: str = "line",
                 periods_per_year: int = 252,
                 rolling_vol_window: int | None = 126,
                 start_date: str | None = None,
                 end_date: str | None = None):
    """
    Plot returns with optional date filtering.
    """
    rets = _to_wide_returns(data).apply(pd.to_numeric, errors="coerce")

    # --- Date filtering ---
    if start_date:
        rets = rets[rets.index >= pd.to_datetime(start_date)]
    if end_date:
        rets = rets[rets.index <= pd.to_datetime(end_date)]

    rets = rets.sort_index().dropna(how="all")
    if rets.empty:
        raise ValueError("No valid return data in the selected date range.")

    fig, ax = plt.subplots(figsize=(12, 6))

    if kind == "line":
        for col in rets.columns:
            ax.plot(rets.index, rets[col], linewidth=1.1, label=col)
        ax.set_ylabel("Return")
        base = "Return time series"
        ax.grid(True, alpha=0.3)

        if rolling_vol_window:
            ax2 = ax.twinx()
            vol = rets.rolling(rolling_vol_window).std() * np.sqrt(periods_per_year)
            for col in vol.columns:
                ax2.plot(vol.index, vol[col], linewidth=0.9, linestyle="--", alpha=0.7)
            ax2.set_ylabel(f"Rolling vol ({rolling_vol_window}) – annualized")
            ax2.tick_params(axis='y', labelsize=9)
        ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(3, 6, 9, 12)))
    
    elif kind == "cum":
        cum = (1.0 + rets.fillna(0)).cumprod().mul(100.0)
        for col in cum.columns:
            ax.plot(cum.index, cum[col], linewidth=1.6, label=col)
        ax.set_ylabel("Cumulative index (start=100)")
        base = "Cumulative returns (rebased to 100)"
        ax.grid(True, alpha=0.35)

        if rolling_vol_window:
            ax2 = ax.twinx()
            vol = rets.rolling(rolling_vol_window).std() * np.sqrt(periods_per_year)
            for col in vol.columns:
                ax2.plot(vol.index, vol[col], linewidth=0.9, linestyle="--", alpha=0.7)
            ax2.set_ylabel(f"Rolling vol ({rolling_vol_window}) – annualized")
            ax2.tick_params(axis='y', labelsize=9)
        ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(3, 6, 9, 12)))

    elif kind == "bar":
        mret = rets.resample("M").apply(lambda x: (1 + x).prod() - 1)
        for col in mret.columns:
            ax.bar(mret.index, mret[col], width=20, alpha=0.6, label=col)
        ax.set_ylabel("Monthly return")
        base = "Monthly aggregated returns"
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_xlabel("Date")
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(3, 6, 9, 12)))
        


    elif kind == "hist":
        for col in rets.columns:
            ax.hist(rets[col].dropna().values, bins=60, alpha=0.4, label=col)
        mu = rets.mean().mean()
        sd = rets.std().mean()

        ax.set_ylabel("Frequency")
        ax.set_xlabel("Return")
        ax.grid(True, alpha=0.2)

        # Centrar la vista en la media (opcional)
        x_range = ax.get_xlim()
        span = max(abs(x_range[1] - mu), abs(mu - x_range[0])) / 2
        ax.set_xlim(mu - span, mu + span)

        # En histogramas, no queremos formatear eje X como fechas
        base = "Histogram of returns"

        # Evitamos formateadores de fechas para el histograma
        ax.xaxis.set_major_locator(plt.MaxNLocator(20))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2%}"))

    else:
        plt.close(fig)
        raise ValueError("kind must be one of: 'line', 'bar', 'hist', 'cum'.")


    start, end = rets.index.min().date(), rets.index.max().date()

    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="Ticker")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

# -------------------- metrics helper --------------------
def _ll_aic_bic(loglik: float, n_params: int, n_obs: int) -> dict:
    """
    Compute log-likelihood, AIC, and BIC for model comparison.
    """
    aic = 2 * n_params - 2 * loglik
    bic = n_params * np.log(n_obs) - 2 * loglik
    return {"loglik": float(loglik), "aic": float(aic), "bic": float(bic)}


# -------------------- Normal fit --------------------
def fit_normal(returns: pd.Series) -> dict:
    """
    Gaussian MLE with closed-form estimates:
        mu = sample mean
        sigma = sample std with ddof=0 (MLE)
    Returns a dict with parameters and information criteria.
    """
    r = pd.Series(returns).dropna().values
    if r.size < 3:
        raise ValueError("Too few observations to fit a Normal model.")
    mu = float(r.mean())
    sigma = float(r.std(ddof=0))
    if sigma <= 0:
        raise ValueError("Zero variance; cannot fit a Normal model.")

    ll = _norm.logpdf(r, loc=mu, scale=sigma).sum()
    out = {"model": "normal", "mu": mu, "sigma": sigma}
    out.update(_ll_aic_bic(ll, n_params=2, n_obs=r.size))
    return out


# -------------------- Student-t fit --------------------
def fit_tstudent(returns: pd.Series, nu_lower: float = 2.01) -> dict:
    """
    Student-t MLE: estimates df (nu), location (mu), and scale (sigma).
    We enforce nu > nu_lower (default 2.01) so that variance is finite.

    Optimization is done on transformed parameters to keep constraints:
        nu = nu_lower + exp(nu_raw) > nu_lower
        sigma = exp(log_sigma) > 0
    """
    r = pd.Series(returns).dropna().values
    if r.size < 5:
        raise ValueError("Too few observations to fit a Student-t model.")

    def nll(params):
        # params = [nu_raw, mu, log_sigma]
        nu = nu_lower + np.exp(params[0])
        mu = params[1]
        sigma = np.exp(params[2])
        return -_t.logpdf(r, df=nu, loc=mu, scale=sigma).sum()

    # Initialize from Normal MLE
    mu0 = r.mean()
    sigma0 = r.std(ddof=0)
    x0 = np.array([np.log(5.0 - nu_lower), mu0, np.log(sigma0 if sigma0 > 0 else 1e-6)])

    res = minimize(nll, x0=x0, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"Student-t fit failed to converge: {res.message}")

    nu = nu_lower + np.exp(res.x[0])
    mu = res.x[1]
    sigma = np.exp(res.x[2])

    ll = -res.fun
    out = {"model": "t", "nu": float(nu), "mu": float(mu), "sigma": float(sigma)}
    out.update(_ll_aic_bic(ll, n_params=3, n_obs=r.size))
    return out


# -------------------- GARCH(1,1) fit --------------------
def fit_garch(
    returns: pd.Series,
    dist: str = "normal",        # "normal" or "t"
    scale_100: bool = True,      # multiply by 100 for numerical stability (common in 'arch')
    mean: str = "constant"       # "constant" or "zero" (here we keep ConstantMean for both)
) -> dict:
    """
    Fit a GARCH(1,1) to returns and return model parameters + diagnostics.

    Returned keys:
        - model: "garch(1,1)-normal" or "garch(1,1)-t"
        - mu, omega, alpha1, beta1, (nu if t)
        - loglik, aic, bic, n_obs, converged
        - uncond_var (if alpha+beta<1), else NaN
    """
    if not _HAS_ARCH:
        raise ImportError("Package 'arch' not available. Install with: pip install arch")

    r = pd.Series(returns).dropna()
    if r.size < 50:
        raise ValueError("Too few observations for GARCH (need ~50+).")

    # Scale by 100 if requested (common practice with 'arch')
    scale = 100.0 if scale_100 else 1.0
    y = r.values * scale

    am = ConstantMean(y)  # if you want zero-mean, set am = ZeroMean(y)
    am.volatility = GARCH(p=1, q=1)

    if dist.lower() == "normal":
        am.distribution = Normal()
    elif dist.lower() in {"t", "student", "students_t"}:
        am.distribution = StudentsT()
    else:
        raise ValueError("dist must be 'normal' or 't'.")

    res = am.fit(disp="off")
    params = res.params.to_dict()

    mapped = {
        "model": f"garch(1,1)-{dist.lower()}",
        # 'mu' is on the scaled space; rescale back to original returns
        "mu": float(params.get("mu", 0.0)) / scale,
        "omega": float(params.get("omega")),
        "alpha1": float(params.get("alpha[1]", params.get("alpha1", np.nan))),
        "beta1": float(params.get("beta[1]", params.get("beta1", np.nan))),
    }
    if dist.lower() in {"t", "student", "students_t"}:
        mapped["nu"] = float(params.get("nu", np.nan))

    out = dict(mapped)
    out.update({
        "loglik": float(res.loglikelihood),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "n_obs": int(res.nobs),
        "converged": bool(res.convergence_flag == 0)
    })

    # Unconditional variance: omega / (1 - alpha - beta) when stationary
    alpha = mapped["alpha1"]
    beta = mapped["beta1"]
    if np.isfinite(alpha) and np.isfinite(beta) and (alpha + beta) < 0.9999:
        out["uncond_var"] = float(mapped["omega"] / max(1e-12, (1.0 - alpha - beta)))
    else:
        out["uncond_var"] = np.nan

    return out


# -------------------- plotting helpers --------------------
def plot_hist_with_pdf_normal_t(returns: pd.Series,
                                fit_normal_dict: dict,
                                fit_t_dict: dict,
                                bins: int = 60,
                                title: str = "Histogram with Normal & Student-t fits"):
    """
    Plot a histogram of returns with fitted Normal and Student-t PDFs overlaid.
    This is appropriate for i.i.d. models.
    """
    r = pd.Series(returns).dropna().values
    if r.size == 0:
        raise ValueError("No valid returns to plot.")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(r, bins=bins, density=True, alpha=0.35, label="Returns")

    xs = np.linspace(np.percentile(r, 0.1), np.percentile(r, 99.9), 1000)

    # Normal PDF
    muN, sdN = fit_normal_dict["mu"], fit_normal_dict["sigma"]
    ax.plot(xs, _norm.pdf(xs, loc=muN, scale=sdN), linewidth=2.0, label="Normal fit")

    # Student-t PDF
    nuT, muT, sdT = fit_t_dict["nu"], fit_t_dict["mu"], fit_t_dict["sigma"]
    ax.plot(xs, _t.pdf(xs, df=nuT, loc=muT, scale=sdT), linewidth=2.0,
            label=f"Student-t fit (ν≈{nuT:.1f})")

    ax.set_title(title)
    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.show()


def plot_hist_stdres_garch(returns: pd.Series,
                           dist: str,
                           bins: int = 60,
                           title: str | None = None,
                           scale_100: bool = True):
    """
    For GARCH models, the right diagnostic is to compare the histogram of
    standardized residuals with the assumed innovation distribution (Normal or t).
    We refit a GARCH(1,1) with the requested innovation and then plot.

    NOTE: This function re-fits the model to compute standardized residuals to
    keep the interface simple. If you already have a fitted result, you could
    pass the residuals directly instead.
    """
    if not _HAS_ARCH:
        raise ImportError("Package 'arch' not available. Install with: pip install arch")

    r = pd.Series(returns).dropna()
    if r.empty:
        raise ValueError("No valid returns to plot standardized residuals.")

    scale = 100.0 if scale_100 else 1.0
    y = r.values * scale

    am = ConstantMean(y)
    am.volatility = GARCH(p=1, q=1)
    if dist.lower() == "normal":
        am.distribution = Normal()
    else:
        am.distribution = StudentsT()
    res = am.fit(disp="off")

    z = pd.Series(res.std_resid).dropna().values

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(z, bins=bins, density=True, alpha=0.35, label="Standardized residuals")

    xs = np.linspace(np.percentile(z, 0.1), np.percentile(z, 99.9), 1000)
    if dist.lower() == "normal":
        ax.plot(xs, _norm.pdf(xs, loc=0, scale=1), linewidth=2.0, label="N(0,1)")
    else:
        # Use fitted 'nu' if available; fallback to quick fit on z
        try:
            nu = float(res.params.get("nu", np.nan))
        except Exception:
            nu = np.nan
        if not np.isfinite(nu):
            nu = max(2.01, _t.fit(z, floc=0, fscale=1)[0])
        ax.plot(xs, _t.pdf(xs, df=nu, loc=0, scale=1), linewidth=2.0, label=f"$t$($v$ = {nu:.1f})")

    ax.set_title(title or f"GARCH standardized residuals vs innovation PDF ({dist})")
    ax.set_xlabel("Standardized residual")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.show()


# -------------------- orchestration --------------------
def fit_return_models(
    returns: pd.Series,
    run_normal: bool = True,
    run_t: bool = True,
    run_garch_normal: bool = True,
    run_garch_t: bool = True
) -> dict:
    """
    Fit Normal, Student-t, GARCH-Normal, and GARCH-t to a returns series.

    Returns a dict with each model’s parameter dict and a 'comparison'
    DataFrame ranked by BIC (lower is better).
    """
    results = {}
    if run_normal:
        results["normal"] = fit_normal(returns)
    if run_t:
        results["t"] = fit_tstudent(returns)
    if run_garch_normal:
        results["garch_normal"] = fit_garch(returns, dist="normal")
    if run_garch_t:
        results["garch_t"] = fit_garch(returns, dist="t")

    comp = []
    for k, v in results.items():
        comp.append({"model": k, "aic": v["aic"], "bic": v["bic"], "loglik": v["loglik"]})
    results["comparison"] = pd.DataFrame(comp).sort_values("bic").reset_index(drop=True)
    return results




def fit_models_from_panel(
    panel: pd.DataFrame,
    ticker: str | None = None,
    run_normal: bool = True,
    run_t: bool = True,
    run_garch_normal: bool = True,
    run_garch_t: bool = True
) -> dict:
    """
    Adapters to run distribution and GARCH fits directly on the 'panel' returned
    by `download_analyze_with_metrics` function.

    Convenience wrapper:
    - If `ticker` is provided: fits on that ticker's return series.
    - If `ticker` is None: fits each ticker found in `panel['ticker']` and
      returns a dict-of-dicts plus a comparison table per ticker.

    Returns
    -------
    If ticker is not None:
        dict with keys: {"normal", "t", "garch_normal", "garch_t", "comparison"}
    Else:
        dict mapping each ticker -> same dict as above.
    """
    required_cols = {"ticker", "date", "return"}
    if not required_cols.issubset(set(panel.columns)):
        missing = required_cols - set(panel.columns)
        raise ValueError(f"`panel` must contain columns {required_cols}. Missing: {missing}")

    def _fit_one(_series: pd.Series) -> dict:
        _series = pd.Series(_series).dropna()
        return fit_return_models(
            returns=_series,
            run_normal=run_normal,
            run_t=run_t,
            run_garch_normal=run_garch_normal,
            run_garch_t=run_garch_t
        )

    if ticker is not None:
        s = (panel.loc[panel["ticker"] == ticker]
                   .sort_values("date")["return"]
                   .astype(float)
                   .dropna())
        if s.empty:
            raise ValueError(f"No returns found for ticker '{ticker}'.")
        return _fit_one(s)

    # Fit all tickers
    results = {}
    for t, sub in panel.groupby("ticker", sort=True):
        s = sub.sort_values("date")["return"].astype(float).dropna()
        if s.empty:
            continue
        try:
            results[t] = _fit_one(s)
        except Exception as e:
            results[t] = {"error": str(e)}
    return results

#---------------splines-----------------#

def spline_density_from_hist(returns: pd.Series | np.ndarray,
                             bins: int = 60,
                             k: int = 3,
                             s: float | None = None,
                             x_points: int = 1000):
    """
    Fit a smoothing spline to the histogram density of returns.
    Ensures non-negativity and re-normalizes to integrate to 1.

    Parameters
    ----------
    returns : array-like
        Return series.
    bins : int
        Number of histogram bins (density=True).
    k : int
        Spline degree (1..5). Use 3 for cubic.
    s : float or None
        Smoothing factor; higher s = smoother. If None, UnivariateSpline chooses.
    x_points : int
        Number of evaluation points for the spline PDF.

    Returns
    -------
    xs : np.ndarray
        Grid for plotting the spline PDF.
    pdf_spline : np.ndarray
        Spline-smoothed density, nonnegative, integrates to 1.
    spline_fun : callable
        The (non-clipped, non-renormalized) spline function before post-processing.
    (bin_centers, hist_density) : tuple of arrays (for diagnostics)
    """
    r = np.asarray(pd.Series(returns).dropna().values)
    # Histogram as density
    counts, bin_edges = np.histogram(r, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    hist_density = counts

    # Fit smoothing spline on (centers, density)
    spline = UnivariateSpline(bin_centers, hist_density, k=k, s=s)

    # Evaluate on a fine grid
    xs = np.linspace(bin_edges.min(), bin_edges.max(), x_points)
    pdf = spline(xs)

    # Enforce non-negativity and renormalize to integrate to 1
    pdf = np.clip(pdf, 0.0, np.inf)
    area = np.trapz(pdf, xs)
    if area > 0:
        pdf /= area

    return xs, pdf, spline, (bin_centers, hist_density)

# --- Example usage + plot
def plot_hist_with_spline(returns, bins=60, k=3, s=None, title="Histogram + spline PDF"):
    xs, pdf_spline, spline_fun, (centers, dens) = spline_density_from_hist(
        returns, bins=bins, k=k, s=s, x_points=1200
    )

    plt.figure(figsize=(9,5))
    plt.hist(returns, bins=bins, density=True, alpha=0.35, label="Returns (hist)")
    plt.plot(xs, pdf_spline, linewidth=2, label=f"Spline PDF (k={k}, s={s})")
    # optional: show the raw histogram points used for fitting
    plt.scatter(centers, dens, s=10, alpha=0.7, label="Hist density points")
    plt.title(title)
    plt.xlabel("Return"); plt.ylabel("Density")
    plt.grid(True, alpha=0.25); plt.legend()
    plt.show()

#---------------KDE-----------------#

def kde_density(returns, x_points=1000, bw_method="scott"):
    r = np.asarray(pd.Series(returns).dropna().values)
    kde = gaussian_kde(r, bw_method=bw_method)  # 'scott' or 'silverman' or float
    xs = np.linspace(np.percentile(r, 0.5), np.percentile(r, 99.5), x_points)
    pdf = kde(xs)
    # Already nonnegative and integrates ~1 over R; no need to renormalize on [xs]
    return xs, pdf, kde

def plot_hist_with_kde(returns, bins=60, bw_method="scott", title="Histogram + KDE"):
    xs, pdf_kde, _ = kde_density(returns, x_points=1200, bw_method=bw_method)
    plt.figure(figsize=(9,5))
    plt.hist(returns, bins=bins, density=True, alpha=0.35, label="Returns (hist)")
    plt.plot(xs, pdf_kde, linewidth=2, label=f"KDE (bw={bw_method})")
    plt.title(title)
    plt.xlabel("Return"); plt.ylabel("Density")
    plt.grid(True, alpha=0.25); plt.legend()
    plt.show()

#---------------Advanced Plot Gaussian Model------------------------#

def _extract_returns_from_input(panel_or_tuple_or_df, ticker: str | None = None) -> pd.Series:
    """
    Accepts:
      1) Full 4-tuple from `download_analyze_with_metrics` -> uses element [0] (panel)
      2) The tidy `panel` DataFrame with ['date','ticker','return']
      3) A wide returns DataFrame (DatetimeIndex rows, tickers columns)
      4) A 1D Series of returns
    Returns a pd.Series of returns (name=ticker if available).
    """
    # Case 1: user passed the full tuple
    if isinstance(panel_or_tuple_or_df, tuple) and len(panel_or_tuple_or_df) >= 1:
        panel_or_tuple_or_df = panel_or_tuple_or_df[0]

    # Case 2: tidy panel
    if isinstance(panel_or_tuple_or_df, pd.DataFrame) and {"date","ticker","return"}.issubset(panel_or_tuple_or_df.columns):
        panel = panel_or_tuple_or_df
        if ticker is None:
            uniq = panel["ticker"].dropna().unique()
            if len(uniq) != 1:
                raise ValueError("Multiple tickers in panel; please pass `ticker=` explicitly.")
            ticker = str(uniq[0])
        s = (panel.loc[panel["ticker"] == ticker]
                    .sort_values("date")["return"].astype(float).dropna())
        s.name = ticker
        s.index = pd.to_datetime(panel.loc[panel["ticker"] == ticker, "date"])[: len(s)]
        return s

    # Case 3: wide returns
    if isinstance(panel_or_tuple_or_df, pd.DataFrame) and isinstance(panel_or_tuple_or_df.index, pd.DatetimeIndex):
        df = panel_or_tuple_or_df.sort_index()
        if ticker is None:
            if df.shape[1] != 1:
                raise ValueError("Wide DataFrame has multiple columns; pass `ticker=` or a single-column DataFrame.")
            ticker = df.columns[0]
        s = df[ticker].dropna().astype(float)
        s.name = ticker
        return s

    # Case 4: already a Series
    if isinstance(panel_or_tuple_or_df, pd.Series):
        s = pd.Series(panel_or_tuple_or_df).dropna().astype(float)
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            pass
        return s

    # As a convenience: if a tidy *price* panel slipped in, convert via utils helper
    if isinstance(panel_or_tuple_or_df, pd.DataFrame) and ("price" in panel_or_tuple_or_df.columns):
        wide_rets = _to_wide_returns(panel_or_tuple_or_df)
        return _extract_returns_from_input(wide_rets, ticker=ticker)

    raise TypeError("Unsupported input; pass the tuple or panel from download_analyze_with_metrics, a wide returns DataFrame, or a Series.")


def gaussian_qq_acf_diagnostics(
    data,
    ticker: str | None = None,
    *,
    bins: int = 60,
    fit_mean_var: bool = True,
    acf_lags: int = 30,
    figsize_hist_qq=(16, 6),
    figsize_acf=(16, 6),
):
    """
    Produce Gaussian diagnostics akin to `tests_gaussian_white_noise`:
      - Figure 1: Histogram + Normal PDF (with fitted mu/sigma or N(0,1)) and QQ-plot.
      - Figure 2: ACF of returns and ACF of absolute returns.

    Parameters
    ----------
    data : tuple | DataFrame | Series
        Output of `download_analyze_with_metrics` (tuple or its `panel`),
        a wide returns DataFrame, or a 1D Series of returns.
    ticker : str, optional
        Needed when multiple tickers are present.
    bins : int
        Histogram bins.
    fit_mean_var : bool
        If True, estimate mu and sigma from the sample; else use N(0,1).
    acf_lags : int
        Number of lags for ACF plots.
    figsize_hist_qq : tuple
        Size of the (histogram, QQ) figure.
    figsize_acf : tuple
        Size of the (ACF, |.| ACF) figure.

    Returns
    -------
    (fig1, fig2)
        Matplotlib figures (hist+QQ, ACFs). Axes can be accessed via fig.axes.
    """
    r = _extract_returns_from_input(data, ticker=ticker)
    if r.empty:
        raise ValueError("No returns to analyze.")
    noise = r.values.astype(float)

    # Fit or set Normal parameters
    if fit_mean_var:
        mu = float(np.mean(noise))
        sigma = float(np.std(noise, ddof=0))
    else:
        mu, sigma = 0.0, 1.0

    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("Non-positive or invalid sigma; cannot run Gaussian diagnostics.")

    # --- Figure 1: Histogram+PDF and QQ-plot ---
    fig1, axs = plt.subplots(1, 2, figsize=figsize_hist_qq, sharex=False)

    # Histogram + PDF
    axs[0].hist(noise, bins=bins, density=True, alpha=0.35, label="Returns")
    xs = np.linspace(np.percentile(noise, 0.1), np.percentile(noise, 99.9), 800)
    axs[0].plot(xs, stats.norm.pdf(xs, loc=mu, scale=sigma), linewidth=2.0,
                label=f"Normal PDF (mu={mu:.4g}, sigma={sigma:.4g})")
    axs[0].set_title("Histogram with Normal PDF")
    axs[0].set_xlabel("Return"); axs[0].set_ylabel("Density")
    axs[0].grid(True, alpha=0.25); axs[0].legend(frameon=False)

    # QQ-plot (probplot) against Normal(mu, sigma)
    stats.probplot(noise, sparams=(mu, sigma), dist='norm', plot=axs[1])
    axs[1].set_title("Normal QQ-plot")
    axs[1].grid(True, alpha=0.25)

    fig1.tight_layout()

    # --- Figure 2: ACFs (returns and absolute returns) ---
    fig2 = None
    if plot_acf is not None:
        fig2, axs2 = plt.subplots(1, 2, figsize=figsize_acf, sharex=True, sharey=True)
        plot_acf(noise, lags=acf_lags, ax=axs2[0])
        plot_acf(np.abs(noise), lags=acf_lags, ax=axs2[1])
        axs2[0].set_title('ACF of returns.')
        axs2[0].set_xlabel(r'$\tau$'); axs2[0].set_ylabel(r'$\rho(\tau)$')
        axs2[1].set_title('ACF of absolute returns.')
        axs2[1].set_xlabel(r'$\tau$'); axs2[1].set_ylabel(r'$\rho(\tau)$')
        fig2.tight_layout()
    else:
        print("statsmodels is not available; skipping ACF plots. Install via `pip install statsmodels`.")

    return fig1, fig2

#---------------------Advanced Plot t-Student Model---------------------#
def student_t_qq_acf_diagnostics(
    data,
    ticker: str | None = None,
    *,
    bins: int = 60,
    acf_lags: int = 30,
    figsize_hist_qq=(16, 6),
    figsize_acf=(16, 6),
):
    r = _extract_returns_from_input(data, ticker=ticker)
    if r.empty:
        raise ValueError("No returns to analyze.")
    noise = r.values.astype(float)

    # Fit Student-t parameters
    fit = fit_tstudent(r)
    nu, mu, sigma = fit["nu"], fit["mu"], fit["sigma"]

    fig1, axs = plt.subplots(1, 2, figsize=figsize_hist_qq)

    axs[0].hist(noise, bins=bins, density=True, alpha=0.35, label="Returns")
    xs = np.linspace(np.percentile(noise, 0.1), np.percentile(noise, 99.9), 800)
    axs[0].plot(xs, stats.t.pdf(xs, df=nu, loc=mu, scale=sigma), linewidth=2.0,
                label=f"t PDF (nu={nu:.2f}, mu={mu:.4g}, sigma={sigma:.4g})")
    axs[0].set_title("Histogram with Student-t PDF")
    axs[0].set_xlabel("Return"); axs[0].set_ylabel("Density")
    axs[0].grid(True, alpha=0.25); axs[0].legend(frameon=False)

    stats.probplot(noise, sparams=(nu, mu, sigma), dist='t', plot=axs[1])
    axs[1].set_title("Student-t QQ-plot")
    axs[1].grid(True, alpha=0.25)
    fig1.tight_layout()

    fig2 = None
    if plot_acf is not None:
        fig2, axs2 = plt.subplots(1, 2, figsize=figsize_acf, sharex=True, sharey=True)
        plot_acf(noise, lags=acf_lags, ax=axs2[0])
        plot_acf(np.abs(noise), lags=acf_lags, ax=axs2[1])
        axs2[0].set_title('ACF of returns')
        axs2[1].set_title('ACF of absolute returns')
        fig2.tight_layout()
    else:
        print("statsmodels not available; skipping ACF plots.")

    return fig1, fig2

def garch_diagnostics(
    data,
    ticker: str | None = None,
    *,
    dist: str = "t",                   # 't' or 'normal'
    acf_lags: int = 30,
    bins: int = 60,
    periods_per_year: int = 252,
    figsize_resid_qq=(16, 6),
    figsize_acf=(16, 6),
    figsize_vol=(16, 4),
):
    """
    Fit a GARCH(1,1) and produce diagnostics analogous to the Gaussian/Student‑t helpers.

    Parameters
    ----------
    data : tuple | DataFrame | Series
        Output from `download_analyze_with_metrics` (tuple or `panel`),
        a wide returns DataFrame, or a 1D returns Series.
    ticker : str, optional
        Required when `data` contains multiple tickers.
    dist : {'t','normal'}
        Innovation distribution for the GARCH model.
    acf_lags : int
        Number of lags for ACF plots.
    bins : int
        Histogram bins for standardized residuals.
    periods_per_year : int
        Used to annualize conditional volatility in the third figure.

    Returns
    -------
    (fig1, fig2, fig3, res)
        Matplotlib figures for (resid hist+QQ), (ACFs), (cond. vol) and the fitted ARCH result object.
    """
    if not _HAS_ARCH:
        raise ImportError("Package 'arch' is required. Install with: pip install arch")

    r = _extract_returns_from_input(data, ticker=ticker)
    if r.empty:
        raise ValueError("No returns to analyze.")

    # Fit GARCH(1,1) with ConstantMean on scaled returns (common for arch)
    scale = 100.0
    y = (r.values * scale).astype(float)

    am = ConstantMean(y)
    am.volatility = GARCH(p=1, q=1)
    if dist.lower() in {"t", "student", "students_t"}:
        am.distribution = StudentsT()
        dist_key = "t"
    elif dist.lower() in {"n", "norm", "normal"}:
        am.distribution = Normal()
        dist_key = "normal"
    else:
        raise ValueError("dist must be 't' or 'normal'.")

    res = am.fit(disp="off")

    # Standardized residuals and conditional volatility (back to original scale)
    z = pd.Series(res.std_resid, index=r.index).dropna()           # standardized resid
    cond_vol = pd.Series(res.conditional_volatility / scale, index=r.index).dropna()
    cond_vol_ann = cond_vol * np.sqrt(periods_per_year)

    # ----- Figure 1: histogram + innovation PDF, and QQ-plot -----
    fig1, axs = plt.subplots(1, 2, figsize=figsize_resid_qq)
    axs[0].hist(z.values, bins=bins, density=True, alpha=0.35, label="Std. residuals")

    xs = np.linspace(np.percentile(z, 0.1), np.percentile(z, 99.9), 1000)
    if dist_key == "normal":
        axs[0].plot(xs, stats.norm.pdf(xs, loc=0, scale=1), linewidth=2.0, label="N(0,1)")
        stats.probplot(z.values, dist=stats.norm, plot=axs[1])
    else:
        # Use fitted nu from result if available; otherwise estimate from z
        try:
            nu = float(res.params.get("nu", np.nan))
        except Exception:
            nu = np.nan
        if not np.isfinite(nu):
            # Fallback quick fit for df given z ~ t(0,1)
            nu = max(2.01, stats.t.fit(z.values, floc=0, fscale=1)[0])
        axs[0].plot(xs, stats.t.pdf(xs, df=nu, loc=0, scale=1), linewidth=2.0, label=f"t(df={nu:.1f})")
        stats.probplot(z.values, sparams=(nu,), dist=stats.t, plot=axs[1])

    axs[0].set_title("Std. residuals: histogram vs innovation PDF")
    axs[0].set_xlabel("$z_t$"); axs[0].set_ylabel("Density")
    axs[0].grid(True, alpha=0.25); axs[0].legend(frameon=False)

    axs[1].set_title(f"QQ-plot vs {('Normal' if dist_key=='normal' else 'Student-t')} innovations")
    axs[1].grid(True, alpha=0.25)
    fig1.tight_layout()

    # ----- Figure 2: ACFs -----
    fig2 = None
    if plot_acf is not None:
        fig2, axs2 = plt.subplots(1, 2, figsize=figsize_acf, sharex=True, sharey=True)
        plot_acf(z.values, lags=acf_lags, ax=axs2[0])
        plot_acf((z.values ** 2), lags=acf_lags, ax=axs2[1])
        axs2[0].set_title("ACF of standardized residuals $z_t$")
        axs2[1].set_title("ACF of squared residuals $z_t^2$")
        for ax in axs2:
            ax.set_xlabel("lag"); ax.set_ylabel("corr"); ax.grid(True, alpha=0.2)
        fig2.tight_layout()
    else:
        print("statsmodels not available; skipping ACF plots.")

    # ----- Figure 3: Conditional volatility (annualized) -----
    if dist_key == "normal":
        fig3, ax3 = plt.subplots(1, 1, figsize=figsize_vol)
        ax3.plot(cond_vol_ann.index, cond_vol_ann.values, linewidth=1.4)
        ax3.set_title("GARCH(1,1) conditional volatility (annualized)")
        ax3.set_xlabel("Date"); ax3.set_ylabel("Volatility (ann.)")
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        return fig1, fig2, fig3, res
    
    else:
        return fig1, fig2, res