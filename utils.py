###############################
# @author: Guillem Borràs     #
# MSc. Quantitative Finance   #
# Physicist                   #
# Quantum Computing IBM       #
###############################

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
# Optional SciPy for parametric VaR/CVaR
try:
    from scipy.stats import norm
except Exception:
    norm = None
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
    if benchmark:
        bm_raw = yf.download(benchmark, start=start_date, progress=False, auto_adjust=auto_adjust)
        bm_px = _extract_close(bm_raw, benchmark).iloc[:, 0]
        bm_px = _clean_prices(bm_px.to_frame(), max_ffill=max_ffill).iloc[:, 0]
        bm = _compute_returns(bm_px.to_frame(), method=return_method).iloc[:, 0]
        bm = bm.reindex(px.index).rename("bm_return")

    # --- 6) Metrics per ticker (igual que antes; recorta a kept_tickers)
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
        span = max(abs(x_range[1] - mu), abs(mu - x_range[0]))
        ax.set_xlim(mu - span, mu + span)

        # En histogramas, no queremos formatear eje X como fechas
        base = "Histogram of returns"

        # Evitamos formateadores de fechas para el histograma
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.2%}"))

    else:
        plt.close(fig)
        raise ValueError("kind must be one of: 'line', 'bar', 'hist', 'cum'.")


    start, end = rets.index.min().date(), rets.index.max().date()

    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, title="Ticker")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()