
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from tqdm import tqdm

import colorednoise as cn
from scipy.signal import butter, sosfilt


NOISE_COLORS: Dict[str, Optional[float]] = {
    "white":  0.0,
    "pink":   1.0,
    "red":    2.0,
    "blue":  -1.0,
    "violet": -2.0,
    "grey":   None,
}

SIGMA_GRID = [0.0] + np.logspace(-4, 0, 15).tolist()


def generate_noise(n: int, color: str, seed: int = 42) -> np.ndarray:

    rng = np.random.default_rng(seed)

    beta = NOISE_COLORS[color]
    if beta is not None:
        raw = cn.powerlaw_psd_gaussian(beta, n, random_state=rng)
    else:
        # grey: band-pass filtered white noise
        raw = rng.standard_normal(n)
        sos = butter(4, [0.05, 0.85], btype="band", output="sos")
        raw = sosfilt(sos, raw)

    std = raw.std()
    if std > 1e-10:
        raw = (raw - raw.mean()) / std
    return raw


def make_noisy_market(
    market_da: xr.DataArray,
    sigma: float,
    color: str,
    test_start: str = "2019-01-01",
    seed: int = 42,
) -> xr.DataArray:
    noisy = market_da.copy(deep=True)
    times = pd.DatetimeIndex(noisy.coords["time"].values)
    test_mask = times >= pd.Timestamp(test_start)
    n_test = int(test_mask.sum())
    test_idx = np.where(test_mask)[0]

    assets = noisy.coords["asset"].values
    n_assets = len(assets)
    fields_all = list(noisy.coords["field"].values)

    data = noisy.values
    dims = list(noisy.dims)
    f_ax = dims.index("field")
    t_ax = dims.index("time")
    a_ax = dims.index("asset")

    close_like = [f for f in ["close", "high", "low"] if f in fields_all]
    close_like_idxs = [fields_all.index(f) for f in close_like]
    open_idx = fields_all.index("open") if "open" in fields_all else None

    for i in range(n_assets):
        noise = generate_noise(n_test, color, seed=seed + i)
        multiplier = 1.0 + sigma * noise

        for fi in close_like_idxs:
            slc = [slice(None)] * len(dims)
            slc[f_ax] = fi
            slc[a_ax] = i
            slc[t_ax] = test_idx
            data[tuple(slc)] *= multiplier

        # open_t uses noise_{t-1}
        if open_idx is not None:
            multiplier_shifted = np.ones(n_test)
            multiplier_shifted[1:] = multiplier[:-1]
            slc = [slice(None)] * len(dims)
            slc[f_ax] = open_idx
            slc[a_ax] = i
            slc[t_ax] = test_idx
            data[tuple(slc)] *= multiplier_shifted

    return noisy


def _replace_features_in_df(
    noisy_features_da: xr.DataArray,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    fields_available = [str(f) for f in noisy_features_da.coords["field"].values]
    feat_data = {}
    for field in fields_available:
        if field in feature_cols:
            s = noisy_features_da.sel(field=field).to_pandas()
            s.index = pd.to_datetime(s.index)
            feat_data[field] = s

    noisy_df = df.copy()
    dates = pd.to_datetime(noisy_df["date"])
    assets = noisy_df["asset"].values

    for col in feature_cols:
        if col not in feat_data:
            continue
        fd = feat_data[col]
        new_vals = np.full(len(noisy_df), np.nan)
        for asset_name in np.unique(assets):
            if asset_name not in fd.columns:
                continue
            mask = assets == asset_name
            asset_dates = dates[mask]
            col_s = fd[asset_name]
            idxs = col_s.index.searchsorted(asset_dates.values, side="right") - 1
            valid = idxs >= 0
            raw = col_s.values[np.clip(idxs, 0, len(col_s) - 1)]
            vals = np.where(valid, raw, np.nan)
            vals[~valid] = np.nan
            new_vals[mask] = vals
        noisy_df[col] = new_vals

    return noisy_df


def _recompute_noisy_target(
    noisy_market_da: xr.DataArray,
    noisy_df: pd.DataFrame,
    horizon_days: int = 30,
) -> None:

    noisy_close_df = noisy_market_da.sel(field="close").to_pandas()
    noisy_close_df.index = pd.to_datetime(noisy_close_df.index)

    has_divs = "divs" in noisy_market_da.coords["field"].values
    if has_divs:
        divs_df = noisy_market_da.sel(field="divs").to_pandas().fillna(0)
        divs_df.index = pd.to_datetime(divs_df.index)
    else:
        divs_df = None

    dates = pd.to_datetime(noisy_df["date"])
    assets = noisy_df["asset"].values
    noisy_targets = np.full(len(noisy_df), np.nan)
    rebal_dates = sorted(pd.to_datetime(noisy_df["date"].unique()))

    mkt_ret_cache = {}
    for rd in rebal_dates:
        future_idx = noisy_close_df.index.searchsorted(
            rd + pd.Timedelta(days=horizon_days))
        if future_idx >= len(noisy_close_df):
            continue
        idx_now = noisy_close_df.index.searchsorted(rd, side="right") - 1
        if idx_now < 0:
            continue
        prices_now = noisy_close_df.iloc[idx_now]
        prices_future = noisy_close_df.iloc[future_idx]

        valid = (prices_now > 0) & prices_now.notna() & prices_future.notna()
        if divs_df is not None:
            now_date = noisy_close_df.index[idx_now]
            future_date = noisy_close_df.index[future_idx]
            d_sum = divs_df.loc[(divs_df.index > now_date) &
                                (divs_df.index <= future_date)].sum().fillna(0)
            rets = (prices_future[valid] + d_sum[valid]) / prices_now[valid] - 1
        else:
            rets = prices_future[valid] / prices_now[valid] - 1
        mkt_ret_cache[rd] = float(rets.mean()) if len(rets) > 10 else np.nan

    for i in range(len(noisy_df)):
        asset_name = assets[i]
        rd = pd.Timestamp(dates.iloc[i])

        if rd not in mkt_ret_cache or np.isnan(mkt_ret_cache[rd]):
            continue
        if asset_name not in noisy_close_df.columns:
            continue

        close_s = noisy_close_df[asset_name].dropna()
        if len(close_s) == 0:
            continue

        idx_now = close_s.index.searchsorted(rd, side="right") - 1
        if idx_now < 0:
            continue
        price_now = close_s.iloc[idx_now]
        if price_now <= 0 or pd.isna(price_now):
            continue

        target_date = rd + pd.Timedelta(days=horizon_days)
        idx_future = close_s.index.searchsorted(target_date)
        if idx_future >= len(close_s):
            continue
        price_future = close_s.iloc[idx_future]
        if pd.isna(price_future):
            continue

        div_sum = 0.0
        if divs_df is not None and asset_name in divs_df.columns:
            date_now = close_s.index[idx_now]
            date_future = close_s.index[idx_future]
            d_slice = divs_df.loc[(divs_df.index > date_now) &
                                   (divs_df.index <= date_future), asset_name]
            div_sum = d_slice.sum()
            if pd.isna(div_sum):
                div_sum = 0.0

        stock_ret = (price_future + div_sum) / price_now - 1
        noisy_targets[i] = stock_ret - mkt_ret_cache[rd]

    original_target = noisy_df["target"].copy()
    noisy_df["target"] = noisy_targets
    still_nan = np.isnan(noisy_targets)
    if still_nan.any():
        noisy_df.loc[still_nan, "target"] = original_target[still_nan]


def recompute_noisy_test_features(
    noisy_market_da: xr.DataArray,
    fund_da: xr.DataArray,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    compute_features_fn,
    horizon_days: int = 30,
) -> pd.DataFrame:

    noisy_features_da = compute_features_fn(noisy_market_da, fund_da)
    noisy_df = _replace_features_in_df(noisy_features_da, test_df, feature_cols)
    _recompute_noisy_target(noisy_market_da, noisy_df, horizon_days)
    return noisy_df


def sortino_from_equity(equity: pd.Series, risk_free: float = 0.0,
                        periods_per_year: float = 252) -> float:
    """Compute annualized Sortino ratio from daily equity series."""
    if len(equity) < 2:
        return 0.0
    returns = equity.pct_change().dropna()
    excess = returns - risk_free / periods_per_year
    mean_excess = excess.mean() * periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() < 1e-12:
        return 0.0
    downside_std = downside.std() * np.sqrt(periods_per_year)
    return float(mean_excess / downside_std)


def custom_backtest(
    test_df: pd.DataFrame,
    pred_col: str,
    market_da: xr.DataArray,
    top_n: int = 20,
    slippage_bps: float = 10.0,
) -> Dict[str, any]:

    test_df = test_df.dropna(subset=[pred_col]).copy()
    test_df["date"] = pd.to_datetime(test_df["date"])

    close_da = market_da.sel(field="close")
    close_df = close_da.to_pandas()
    close_df.index = pd.to_datetime(close_df.index)

    has_divs = "divs" in market_da.coords["field"].values
    if has_divs:
        divs_df = market_da.sel(field="divs").to_pandas().fillna(0)
        divs_df.index = pd.to_datetime(divs_df.index)
    else:
        divs_df = None

    months = sorted(test_df["date"].unique())
    if not months:
        return {"equity": pd.Series(dtype=float), "metrics": {}}

    slip_rate = slippage_bps / 10_000.0

    portfolio_value = 1.0
    daily_equity = {}
    prev_assets_set: set = set()

    for i, month in enumerate(months):
        month = pd.Timestamp(month)
        month_df = test_df[test_df["date"] == month]
        actual_n = min(top_n, len(month_df))
        if actual_n < 1:
            continue
        top = month_df.nlargest(actual_n, pred_col)
        top_assets = list(top["asset"].values)

        if i + 1 < len(months):
            next_month = pd.Timestamp(months[i + 1])
            mask = (close_df.index >= month) & (close_df.index < next_month)
        else:
            mask = close_df.index >= month
        period_prices = close_df.loc[mask, top_assets]
        if len(period_prices) < 2:
            continue

        start_prices = period_prices.iloc[0]
        valid_mask = start_prices.notna() & (start_prices > 0)
        if valid_mask.sum() == 0:
            continue
        valid_assets = valid_mask[valid_mask].index.tolist()
        cur_assets_set = set(valid_assets)

        if prev_assets_set:
            stayed = len(cur_assets_set & prev_assets_set)
            max_n = max(len(cur_assets_set), len(prev_assets_set))
            turnover_frac = 1.0 - stayed / max_n if max_n > 0 else 0.0
        else:
            turnover_frac = 1.0
        slip_cost = slip_rate * turnover_frac
        prev_assets_set = cur_assets_set

        invested = portfolio_value * (1 - slip_cost)

        if divs_df is not None:
            try:
                period_divs = divs_df.loc[mask, valid_assets].fillna(0)
            except KeyError:
                period_divs = None
        else:
            period_divs = None

        for d_idx in range(len(period_prices)):
            t = period_prices.index[d_idx]
            day_prices = period_prices.iloc[d_idx][valid_assets]
            if period_divs is not None and d_idx > 0:
                cum_divs = period_divs.iloc[1:d_idx + 1].sum()
            else:
                cum_divs = 0.0
            avg_ratio = ((day_prices + cum_divs) / start_prices[valid_assets]).mean()
            daily_equity[t] = invested * avg_ratio

        end_prices = period_prices.iloc[-1][valid_assets]
        if period_divs is not None and len(period_prices) > 1:
            total_divs = period_divs.iloc[1:].sum()
        else:
            total_divs = 0.0
        end_ratio = ((end_prices + total_divs) / start_prices[valid_assets]).mean()
        portfolio_value = invested * end_ratio

    if not daily_equity:
        return {"equity": pd.Series(dtype=float), "metrics": {}}

    equity = pd.Series(daily_equity).sort_index()

    all_daily_rets = equity.pct_change().dropna()

    total_days = (equity.index[-1] - equity.index[0]).days
    years = total_days / 365.25 if total_days > 0 else 1.0
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1

    if years > 0 and (1 + total_ret) > 0:
        ann_ret = (1 + total_ret) ** (1 / years) - 1
    else:
        ann_ret = -1.0 if total_ret < -1 else 0.0

    if len(all_daily_rets) > 1:
        daily_mean = all_daily_rets.mean()
        daily_std = all_daily_rets.std()
        ann_vol = daily_std * np.sqrt(252)
        sharpe = (daily_mean * 252) / ann_vol if ann_vol > 1e-10 else 0.0
        mar_daily = 0.0
        excess = all_daily_rets - mar_daily
        downside_sq = np.minimum(0.0, excess) ** 2
        downside_dev_daily = np.sqrt(downside_sq.mean())
        
        if downside_dev_daily > 1e-10:
            sortino = (excess.mean() * 252) / (downside_dev_daily * np.sqrt(252))
        else:
            sortino = 0.0

    else:
        ann_vol = 0.0
        sharpe = 0.0
        sortino = 0.0

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    metrics = {
        "total_return": total_ret,
        "annual_return": ann_ret,
        "volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
    }

    return {"equity": equity, "metrics": metrics}


def _momentum_backtest(
    test_df: pd.DataFrame,
    market_da: xr.DataArray,
    top_n: int = 20,
    slippage_bps: float = 10.0,
    mom_col: str = "mom9m",
) -> Dict[str, any]:
    """Momentum strategy: rank by trailing momentum, top-N selection."""
    df = test_df.dropna(subset=[mom_col]).copy()
    return custom_backtest(df, mom_col, market_da, top_n=top_n,
                           slippage_bps=slippage_bps)


def _ew_backtest(
    test_df: pd.DataFrame,
    market_da: xr.DataArray,
    slippage_bps: float = 10.0,
) -> Dict[str, any]:
    """Equal-weight (all liquid assets) backtest."""
    df = test_df.copy()
    df["_ew_pred"] = 1.0
    n_assets = df.groupby("date")["asset"].transform("count")
    max_n = int(n_assets.max())
    return custom_backtest(df, "_ew_pred", market_da, top_n=max_n,
                           slippage_bps=slippage_bps)


def run_custom_backtest_all(
    test_result: pd.DataFrame,
    market_da: xr.DataArray,
    models_info: List[Tuple[str, str]],
    top_n_grid: List[int] = None,
    slippage_bps: float = 10.0,
    include_benchmarks: bool = True,
    mom_col: str = "mom9m",
) -> Tuple[Dict[str, pd.Series], pd.DataFrame]:
    """Run custom backtest for all models, Momentum and EW benchmarks.

    Returns (equity_curves dict, metrics DataFrame).
    """
    if top_n_grid is None:
        top_n_grid = [5, 10, 20, 30, 50, 100, 150, 200]

    equities = {}
    rows = []

    for model_name, pred_col in models_info:
        for top_n in top_n_grid:
            key = f"{model_name} top-{top_n}"
            result = custom_backtest(test_result, pred_col, market_da,
                                     top_n=top_n, slippage_bps=slippage_bps)
            if len(result["equity"]) > 0:
                equities[key] = result["equity"]
                m = result["metrics"]
                rows.append({
                    "Strategy": key,
                    "Model": model_name,
                    "Top-N": top_n,
                    "Total Return": m["total_return"],
                    "Annual Return": m["annual_return"],
                    "Volatility": m["volatility"],
                    "Sharpe": m["sharpe"],
                    "Sortino": m["sortino"],
                    "Max DD": m["max_drawdown"],
                })

    if include_benchmarks:
        if mom_col in test_result.columns:
            for top_n in top_n_grid:
                key = f"Momentum top-{top_n}"
                result = _momentum_backtest(test_result, market_da,
                                            top_n=top_n,
                                            slippage_bps=slippage_bps,
                                            mom_col=mom_col)
                if len(result["equity"]) > 0:
                    equities[key] = result["equity"]
                    m = result["metrics"]
                    rows.append({
                        "Strategy": key,
                        "Model": "Momentum",
                        "Top-N": top_n,
                        "Total Return": m["total_return"],
                        "Annual Return": m["annual_return"],
                        "Volatility": m["volatility"],
                        "Sharpe": m["sharpe"],
                        "Sortino": m["sortino"],
                        "Max DD": m["max_drawdown"],
                    })

        ew_result = _ew_backtest(test_result, market_da,
                                  slippage_bps=slippage_bps)
        if len(ew_result["equity"]) > 0:
            equities["EW (all)"] = ew_result["equity"]
            m = ew_result["metrics"]
            rows.append({
                "Strategy": "EW (all)",
                "Model": "EW",
                "Top-N": -1,
                "Total Return": m["total_return"],
                "Annual Return": m["annual_return"],
                "Volatility": m["volatility"],
                "Sharpe": m["sharpe"],
                "Sortino": m["sortino"],
                "Max DD": m["max_drawdown"],
            })

    metrics_df = pd.DataFrame(rows)
    return equities, metrics_df


def plot_custom_backtest(equities: Dict[str, pd.Series],
                         metrics_df: pd.DataFrame,
                         top_n_grid: List[int] = None):

    if top_n_grid is None:
        top_n_grid = [5, 10, 20, 30, 50, 100, 150, 200]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    colors = MODEL_COLORS

    models_with_topn = metrics_df[metrics_df["Top-N"] > 0]["Model"].unique()
    for model_name in models_with_topn:
        model_df = metrics_df[metrics_df["Model"] == model_name]
        if len(model_df) == 0:
            continue
        best_row = model_df.loc[model_df["Sharpe"].idxmax()]
        key = best_row["Strategy"]
        if key in equities:
            eq = equities[key]
            ls = "--" if model_name in ("Momentum",) else "-"
            axes[0].plot(eq.index, eq.values,
                         label=f"{key} (SR={best_row['Sharpe']:.2f})",
                         color=colors.get(model_name, "grey"), lw=1.5,
                         ls=ls)

    if "EW (all)" in equities:
        ew_row = metrics_df[metrics_df["Model"] == "EW"]
        sr = ew_row["Sharpe"].iloc[0] if len(ew_row) else 0
        eq = equities["EW (all)"]
        axes[0].plot(eq.index, eq.values,
                     label=f"EW (all) (SR={sr:.2f})",
                     color=colors.get("EW", "black"), lw=2, ls=":")

    axes[0].axhline(1.0, color="grey", ls=":", alpha=0.5)
    axes[0].set_title("Custom Backtest: Best top-N per model")
    axes[0].set_ylabel("Equity")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for model_name in models_with_topn:
        model_df = metrics_df[metrics_df["Model"] == model_name]
        sharpes = []
        for top_n in top_n_grid:
            row = model_df[model_df["Top-N"] == top_n]
            sharpes.append(row["Sharpe"].iloc[0] if len(row) > 0 else np.nan)
        ls = "--" if model_name == "Momentum" else "-"
        marker = "s" if model_name == "Momentum" else "o"
        axes[1].plot(top_n_grid, sharpes, marker=marker, ls=ls,
                     label=model_name,
                     color=colors.get(model_name, "grey"))

    if "EW" in metrics_df["Model"].values:
        ew_sharpe = metrics_df[metrics_df["Model"] == "EW"]["Sharpe"].iloc[0]
        axes[1].axhline(ew_sharpe, color=colors.get("EW", "black"),
                         ls=":", lw=1.5, label=f"EW (SR={ew_sharpe:.2f})")

    axes[1].set_xlabel("Top-N")
    axes[1].set_ylabel("Sharpe Ratio")
    axes[1].set_title("Custom Backtest: Sharpe vs Top-N")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()



def _compute_ic_from_test(test_df: pd.DataFrame, pred_col: str) -> float:
    """Compute mean monthly Spearman IC for robustness."""
    from scipy.stats import spearmanr
    df = test_df.dropna(subset=[pred_col, "target"]).copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    ics = []
    for _, group in df.groupby("month"):
        if len(group) < 5:
            continue
        ic, _ = spearmanr(group[pred_col], group["target"])
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0


def run_robustness_grid(
    models_dict: Dict[str, Tuple],
    market_da: xr.DataArray,
    fund_da: xr.DataArray,
    test_df: pd.DataFrame,
    scaler,
    feature_cols: List[str],
    compute_features_fn,
    predictions_to_daily_weights_fn,
    top_n_per_model: Dict[str, int],
    sigma_grid: List[float] = None,
    colors: List[str] = None,
    test_start: str = "2019-01-01",
    seed: int = 42,
    ew_weights_fn=None,
    fixed_top_n: int = 30,
) -> pd.DataFrame:
    import qnt.stats as qnstats

    if sigma_grid is None:
        sigma_grid = SIGMA_GRID
    if colors is None:
        colors = list(NOISE_COLORS.keys())

    _clean = lambda x: x.replace([np.inf, -np.inf], np.nan).fillna(0)

    all_feature_cols = list(set(feature_cols))
    for _, (_, feat_cols, *_) in models_dict.items():
        for c in feat_cols:
            if c not in all_feature_cols:
                all_feature_cols.append(c)

    results = []

    def _extract_metrics(stat, test_data=None, pred_col=None):
        sp = stat.to_pandas()
        
        def _get(field):
            if isinstance(sp, pd.DataFrame) and field in sp.columns:
                return float(sp[field].iloc[-1])
            try:
                return float(stat.sel(field=field).to_pandas().iloc[-1])
            except Exception:
                return np.nan
        
        sharpe = _get("sharpe_ratio")
        ann_ret = _get("mean_return")
        
        try:
            if isinstance(sp, pd.DataFrame) and "equity" in sp.columns:
                eq = sp["equity"]
            else:
                eq = stat.sel(field="equity").to_pandas()
                if isinstance(eq, pd.DataFrame):
                    eq = eq.iloc[:, 0]
            sortino = sortino_from_equity(eq)
        except Exception:
            sortino = np.nan
        
        ic = _compute_ic_from_test(test_data, pred_col) if test_data is not None and pred_col else np.nan
        
        return sharpe, sortino, ann_ret, ic

    for color in colors:
        for sigma in tqdm(sigma_grid, desc=f"Noise: {color}"):
            noisy_mkt = make_noisy_market(market_da, sigma, color,
                                          test_start=test_start, seed=seed)
            noisy_test = recompute_noisy_test_features(
                noisy_mkt, fund_da, test_df, all_feature_cols, compute_features_fn)

            for model_name, model_info in models_dict.items():
                model, feat_cols, is_seq, hist_df, model_scaler = model_info

                noisy_model_df = noisy_test.copy()
                noisy_model_df[feat_cols] = model_scaler.transform(
                    _clean(noisy_model_df[feat_cols]))

                if is_seq:
                    preds = model.predict(noisy_model_df, feat_cols, history_df=hist_df)
                else:
                    preds = model.predict(noisy_model_df, feat_cols)

                noisy_model_df["pred"] = preds

                best_n = top_n_per_model.get(model_name, 50)
                for top_n, tn_mode in [(best_n, "best"), (fixed_top_n, "fixed")]:
                    w = predictions_to_daily_weights_fn(
                        noisy_model_df, "pred", market_da, top_n=top_n)
                    if w is None:
                        results.append({
                            "model": model_name, "color": color, "sigma": sigma,
                            "top_n_mode": tn_mode, "top_n": top_n,
                            "sharpe": np.nan, "sortino": np.nan,
                            "annual_return": np.nan, "ic": np.nan,
                        })
                        continue

                    test_market = market_da.sel(time=w.time)
                    stat = qnstats.calc_stat(test_market, w, slippage_factor=0.05)
                    sharpe, sortino, ann_ret, ic = _extract_metrics(
                        stat, noisy_model_df, "pred")

                    results.append({
                        "model": model_name, "color": color, "sigma": sigma,
                        "top_n_mode": tn_mode, "top_n": top_n,
                        "sharpe": sharpe, "sortino": sortino,
                        "annual_return": ann_ret, "ic": ic,
                    })

            if ew_weights_fn is not None:
                w_ew = ew_weights_fn(noisy_test, market_da)
                if w_ew is not None:
                    test_market = market_da.sel(time=w_ew.time)
                    stat = qnstats.calc_stat(test_market, w_ew, slippage_factor=0.05)
                    sharpe, sortino, ann_ret, _ = _extract_metrics(stat)
                    for tn_mode in ["best", "fixed"]:
                        results.append({
                            "model": "EW", "color": color, "sigma": sigma,
                            "top_n_mode": tn_mode, "top_n": 0,
                            "sharpe": sharpe, "sortino": sortino,
                            "annual_return": ann_ret, "ic": np.nan,
                        })

            mom_col = "mom9m" if "mom9m" in noisy_test.columns else None
            if mom_col is None:
                for c in noisy_test.columns:
                    if "mom" in c.lower() and "9" in c:
                        mom_col = c
                        break
            if mom_col is not None:
                noisy_test_mom = noisy_test.copy()
                noisy_test_mom["pred_mom"] = noisy_test_mom[mom_col]
                
                best_n_mom = top_n_per_model.get("Momentum", fixed_top_n)
                for top_n, tn_mode in [(best_n_mom, "best"), (fixed_top_n, "fixed")]:
                    w_mom = predictions_to_daily_weights_fn(
                        noisy_test_mom, "pred_mom", market_da, top_n=top_n)
                    if w_mom is not None:
                        test_market = market_da.sel(time=w_mom.time)
                        stat = qnstats.calc_stat(test_market, w_mom, slippage_factor=0.05)
                        sharpe, sortino, ann_ret, ic = _extract_metrics(
                            stat, noisy_test_mom, "pred_mom")
                        results.append({
                            "model": "Momentum", "color": color, "sigma": sigma,
                            "top_n_mode": tn_mode, "top_n": top_n,
                            "sharpe": sharpe, "sortino": sortino,
                            "annual_return": ann_ret, "ic": ic,
                        })

    return pd.DataFrame(results)



MODEL_COLORS = {
    "RF": "blue", "XGBoost": "green", "LSTM": "red", "LSTM+MLP": "cyan",
    "LinReg": "grey", "LSTM+XGB": "magenta", "LSTM-Ret": "deeppink",
    "LSTM-TS": "orange", "GRU": "purple", "GRU-Ret": "darkviolet",
    "Transformer": "brown", "EW": "black", "Momentum": "darkgoldenrod",
}

NOISE_LINESTYLES = {
    "white": "-", "pink": "--", "red": "-.", "blue": ":", "violet": "-", "grey": "--",
}

METRIC_NAMES = {
    "sharpe": "Sharpe Ratio", "sortino": "Sortino Ratio",
    "annual_return": "Annual Return", "ic": "IC",
}


def plot_robustness_model_vs_noises(
    results_df: pd.DataFrame,
    top_n_mode: str = "best",
    metrics: List[str] = None,
    baseline_df: Optional[pd.DataFrame] = None,
):

    if metrics is None:
        metrics = ["sharpe", "sortino", "annual_return", "ic"]
    
    df = results_df[results_df["top_n_mode"] == top_n_mode]
    models = sorted(df["model"].unique())
    noise_colors_list = sorted(df["color"].unique())
    mode_label = "Best top-N" if top_n_mode == "best" else "Fixed top-N"
    
    for model_name in models:
        model_df = df[df["model"] == model_name]
        
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            for nc in noise_colors_list:
                sub = model_df[model_df["color"] == nc].sort_values("sigma")
                if len(sub) == 0:
                    continue
                ax.plot(sub["sigma"], sub[metric], "o-",
                        label=nc.capitalize(), markersize=3,
                        ls=NOISE_LINESTYLES.get(nc, "-"))
            
            if baseline_df is not None and model_name in baseline_df.index and metric in baseline_df.columns:
                ax.axhline(baseline_df.loc[model_name, metric],
                           color="black", ls=":", alpha=0.6, label="baseline")
            
            ax.set_xlabel("σ (noise intensity)")
            ax.set_ylabel(METRIC_NAMES.get(metric, metric))
            ax.set_title(f"{model_name} — {METRIC_NAMES.get(metric, metric)} ({mode_label})")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.show()


def plot_robustness_noise_vs_models(
    results_df: pd.DataFrame,
    top_n_mode: str = "best",
    metrics: List[str] = None,
    baseline_df: Optional[pd.DataFrame] = None,
):

    if metrics is None:
        metrics = ["sharpe", "sortino", "annual_return", "ic"]
    
    df = results_df[results_df["top_n_mode"] == top_n_mode]
    models = sorted(df["model"].unique())
    noise_colors_list = sorted(df["color"].unique())
    mode_label = "Best top-N" if top_n_mode == "best" else "Fixed top-N"
    
    for noise_color in noise_colors_list:
        noise_df = df[df["color"] == noise_color]
        
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 5))
            
            for model_name in models:
                sub = noise_df[noise_df["model"] == model_name].sort_values("sigma")
                if len(sub) == 0:
                    continue
                ax.plot(sub["sigma"], sub[metric], "o-",
                        label=model_name, markersize=3,
                        color=MODEL_COLORS.get(model_name, "grey"))
            
            ax.set_xlabel("σ (noise intensity)")
            ax.set_ylabel(METRIC_NAMES.get(metric, metric))
            ax.set_title(f"{noise_color.capitalize()} noise — {METRIC_NAMES.get(metric, metric)} ({mode_label})")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            plt.tight_layout()
            plt.show()


def plot_robustness_all(
    results_df: pd.DataFrame,
    baseline_df: Optional[pd.DataFrame] = None,
    metrics: List[str] = None,
):

    if metrics is None:
        metrics = ["sharpe", "sortino", "annual_return", "ic"]
    
    for mode in ["best", "fixed"]:
        mode_label = "Best top-N" if mode == "best" else "Fixed top-30"
        print(f"\n{'='*60}")
        print(f"  ROBUSTNESS PLOTS: {mode_label}")
        print(f"{'='*60}\n")
        
        print(f"--- Per Model (each model: noise colors on one plot) [{mode_label}] ---")
        plot_robustness_model_vs_noises(
            results_df, top_n_mode=mode, metrics=metrics, baseline_df=baseline_df)
        
        print(f"\n--- Per Noise Color (each noise: models on one plot) [{mode_label}] ---")
        plot_robustness_noise_vs_models(
            results_df, top_n_mode=mode, metrics=metrics, baseline_df=baseline_df)



def _compute_monthly_ics(test_df: pd.DataFrame, pred_col: str,
                         target_col: str = "target") -> List[float]:
    from scipy.stats import spearmanr
    df = test_df.dropna(subset=[pred_col, target_col]).copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    ics = []
    for _, group in df.groupby("month"):
        if len(group) < 5:
            continue
        ic, _ = spearmanr(group[pred_col], group[target_col])
        if not np.isnan(ic):
            ics.append(ic)
    return ics


def _compute_ic(test_df: pd.DataFrame, pred_col: str,
                target_col: str = "target") -> float:
    ics = _compute_monthly_ics(test_df, pred_col, target_col)
    return float(np.mean(ics)) if ics else 0.0


def _compute_icir(test_df: pd.DataFrame, pred_col: str,
                  target_col: str = "target") -> float:
    ics = _compute_monthly_ics(test_df, pred_col, target_col)
    if len(ics) < 2:
        return 0.0
    return float(np.mean(ics) / (np.std(ics) + 1e-8))


def _compute_monthly_pearson(test_df: pd.DataFrame, col_a: str,
                             col_b: str) -> List[float]:
    """Return list of per-month Pearson correlations between col_a and col_b."""
    from scipy.stats import pearsonr
    df = test_df.dropna(subset=[col_a, col_b]).copy()
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    rs = []
    for _, group in df.groupby("month"):
        if len(group) < 5:
            continue
        r, _ = pearsonr(group[col_a], group[col_b])
        if not np.isnan(r):
            rs.append(r)
    return rs


def _compute_pearson(test_df: pd.DataFrame, col_a: str,
                     col_b: str) -> float:
    rs = _compute_monthly_pearson(test_df, col_a, col_b)
    return float(np.mean(rs)) if rs else np.nan


def _compute_pred_corr(clean_df: pd.DataFrame, noisy_df: pd.DataFrame,
                       clean_pred_col: str,
                       noisy_pred_col: str = "pred") -> float:

    from scipy.stats import spearmanr
    clean = clean_df.copy()
    noisy = noisy_df.copy()
    clean["date"] = pd.to_datetime(clean["date"])
    noisy["date"] = pd.to_datetime(noisy["date"])
    months = sorted(set(clean["date"].unique()) & set(noisy["date"].unique()))
    rs = []
    for month in months:
        cl = clean[clean["date"] == month].set_index("asset")
        ns = noisy[noisy["date"] == month].set_index("asset")
        common = cl.index.intersection(ns.index)
        if len(common) < 5:
            continue
        r, _ = spearmanr(cl.loc[common, clean_pred_col],
                         ns.loc[common, noisy_pred_col])
        if not np.isnan(r):
            rs.append(r)
    return float(np.mean(rs)) if rs else np.nan



def compute_topn_stability(
    clean_df: pd.DataFrame,
    noisy_df: pd.DataFrame,
    clean_pred_col: str,
    n_values: List[int] = None,
    noisy_pred_col: str = "pred",
) -> Dict[int, float]:

    if n_values is None:
        n_values = [10, 30, 50, 100]

    clean = clean_df.dropna(subset=[clean_pred_col]).copy()
    noisy = noisy_df.dropna(subset=[noisy_pred_col]).copy()
    clean["date"] = pd.to_datetime(clean["date"])
    noisy["date"] = pd.to_datetime(noisy["date"])

    months = sorted(set(clean["date"].unique()) & set(noisy["date"].unique()))
    result = {n: [] for n in n_values}

    for month in months:
        cl = clean[clean["date"] == month]
        ns = noisy[noisy["date"] == month]
        for n in n_values:
            if len(cl) < n or len(ns) < n:
                continue
            top_clean = set(cl.nlargest(n, clean_pred_col)["asset"].values)
            top_noisy = set(ns.nlargest(n, noisy_pred_col)["asset"].values)
            overlap = len(top_clean & top_noisy) / n
            result[n].append(overlap)

    return {n: float(np.mean(v)) if v else np.nan for n, v in result.items()}



_DISPLAY_TO_KEY = {
    "LinReg": "linreg", "RF": "rf", "XGBoost": "xgb",
    "LSTM": "lstm", "LSTM+MLP": "lstm_mlp", "LSTM+XGB": "lstm_xgb",
    "LSTM-Ret": "lstm_ret", "LSTM-TS": "lstm_ts",
    "GRU": "gru", "GRU-TS": "gru_ts",
    "GRU-Ret": "gru_ret",
    "Transformer": "transformer", "Transformer-TS": "transformer_ts",
}


def _find_clean_pred_col(display_name: str, columns) -> Optional[str]:
    """Find prediction column in clean test_result for a given display name."""
    key = _DISPLAY_TO_KEY.get(display_name)
    if key is not None:
        col = f"pred_{key}"
        if col in columns:
            return col
    if display_name == "Momentum":
        if "pred_momentum" in columns:
            return "pred_momentum"
    return None



def run_robustness_tables(
    models_dict: Dict[str, Tuple],
    market_da: xr.DataArray,
    fund_da: xr.DataArray,
    test_df: pd.DataFrame,
    raw_history_df: pd.DataFrame,
    scaler,
    feature_cols: List[str],
    compute_features_fn,
    custom_backtest_fn,
    top_n_fixed: int,
    top_n_val: Dict[str, int],
    top_n_test: Dict[str, int],
    sigma_grid: List[float] = None,
    colors: List[str] = None,
    test_start: str = "2019-01-01",
    seed: int = 42,
    clean_test_result: Optional[pd.DataFrame] = None,
    n_stability_values: List[int] = None,
    ew_weights_fn=None,
    mom_col: str = "mom9m",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if sigma_grid is None:
        sigma_grid = SIGMA_GRID
    if colors is None:
        colors = list(NOISE_COLORS.keys())
    if n_stability_values is None:
        n_stability_values = [10, 30, 50, 100]

    _clean_fn = lambda x: x.replace([np.inf, -np.inf], np.nan).fillna(0)

    all_feature_cols = list(set(feature_cols))
    for _, (_, feat_cols, *_) in models_dict.items():
        for c in feat_cols:
            if c not in all_feature_cols:
                all_feature_cols.append(c)
    if mom_col not in all_feature_cols:
        all_feature_cols.append(mom_col)

    model_clean_rows: List[dict] = []
    model_noisy_rows: List[dict] = []
    portfolio_clean_rows: List[dict] = []
    portfolio_noisy_rows: List[dict] = []

    def _portfolio_row(model, color, sigma, tn_mode, top_n, m):
        return {"model": model, "color": color, "sigma": sigma,
                "top_n_mode": tn_mode, "top_n": top_n,
                "sharpe": m.get("sharpe", np.nan),
                "sortino": m.get("sortino", np.nan),
                "annual_return": m.get("annual_return", np.nan),
                "max_drawdown": m.get("max_drawdown", np.nan)}

    def _stability_dict(model_name, noisy_model_df):
        stab = {}
        if clean_test_result is not None:
            cpred = _find_clean_pred_col(model_name, clean_test_result.columns)
            if cpred is not None:
                sv = compute_topn_stability(
                    clean_test_result, noisy_model_df,
                    cpred, n_stability_values, noisy_pred_col="pred")
                for nv in n_stability_values:
                    stab[f"stability_{nv}"] = sv.get(nv, np.nan)
                return stab
        for nv in n_stability_values:
            stab[f"stability_{nv}"] = np.nan
        return stab

    for color in colors:
        for sigma in tqdm(sigma_grid, desc=f"Noise: {color}"):
            noisy_mkt = make_noisy_market(market_da, sigma, color,
                                          test_start=test_start, seed=seed)

            noisy_features_da = compute_features_fn(noisy_mkt, fund_da)

            noisy_test = _replace_features_in_df(
                noisy_features_da, test_df, all_feature_cols)
            _recompute_noisy_target(noisy_mkt, noisy_test)
            noisy_test["clean_target"] = test_df["target"].values

            noisy_history = _replace_features_in_df(
                noisy_features_da, raw_history_df, all_feature_cols)
            _recompute_noisy_target(noisy_mkt, noisy_history)

            scaled_hist_cache: Dict[Any, pd.DataFrame] = {}

            for model_name, model_info in models_dict.items():
                model, feat_cols, is_seq, _, model_scaler = model_info

                noisy_model_df = noisy_test.copy()
                noisy_model_df[feat_cols] = model_scaler.transform(
                    _clean_fn(noisy_model_df[feat_cols]))

                if is_seq:
                    cache_key = (id(model_scaler), tuple(sorted(feat_cols)))
                    if cache_key not in scaled_hist_cache:
                        h = noisy_history.copy()
                        h[feat_cols] = model_scaler.transform(
                            _clean_fn(h[feat_cols]))
                        scaled_hist_cache[cache_key] = h
                    preds = model.predict(
                        noisy_model_df, feat_cols,
                        history_df=scaled_hist_cache[cache_key])
                else:
                    preds = model.predict(noisy_model_df, feat_cols)
                noisy_model_df["pred"] = preds

                val_n = top_n_val.get(model_name, top_n_fixed)
                test_n = top_n_test.get(model_name, top_n_fixed)
                for top_n, tn_mode in [
                    (top_n_fixed, "fixed"),
                    (val_n, "val_best"),
                    (test_n, "test_best"),
                ]:
                    bt_c = custom_backtest_fn(
                        noisy_model_df, "pred", market_da, top_n=top_n)
                    portfolio_clean_rows.append(_portfolio_row(
                        model_name, color, sigma, tn_mode, top_n,
                        bt_c.get("metrics", {})))

                    bt_n = custom_backtest_fn(
                        noisy_model_df, "pred", noisy_mkt, top_n=top_n)
                    portfolio_noisy_rows.append(_portfolio_row(
                        model_name, color, sigma, tn_mode, top_n,
                        bt_n.get("metrics", {})))

                stab = _stability_dict(model_name, noisy_model_df)

                pearson_c = _compute_pearson(noisy_model_df, "pred",
                                            "clean_target")
                pearson_n = _compute_pearson(noisy_model_df, "pred",
                                            "target")

                pred_corr = np.nan
                if clean_test_result is not None:
                    cpred = _find_clean_pred_col(model_name,
                                                clean_test_result.columns)
                    if cpred is not None:
                        pred_corr = _compute_pred_corr(
                            clean_test_result, noisy_model_df, cpred, "pred")

                ic_c = _compute_ic(noisy_model_df, "pred",
                                   target_col="clean_target")
                icir_c = _compute_icir(noisy_model_df, "pred",
                                       target_col="clean_target")
                model_clean_rows.append({
                    "model": model_name, "color": color, "sigma": sigma,
                    "ic": ic_c, "icir": icir_c,
                    "pearson": pearson_c, "pred_corr": pred_corr,
                    **stab})

                ic_n = _compute_ic(noisy_model_df, "pred",
                                   target_col="target")
                icir_n = _compute_icir(noisy_model_df, "pred",
                                       target_col="target")
                model_noisy_rows.append({
                    "model": model_name, "color": color, "sigma": sigma,
                    "ic": ic_n, "icir": icir_n,
                    "pearson": pearson_n, "pred_corr": pred_corr,
                    **stab})

            if ew_weights_fn is not None:
                n_all = len(noisy_test["asset"].unique())
                noisy_test_ew = noisy_test.copy()
                noisy_test_ew["pred_ew"] = 1.0
                for tn_mode in ["fixed", "val_best", "test_best"]:
                    bt_c = custom_backtest_fn(
                        noisy_test_ew, "pred_ew", market_da, top_n=n_all)
                    portfolio_clean_rows.append(_portfolio_row(
                        "EW", color, sigma, tn_mode, 0,
                        bt_c.get("metrics", {})))
                    bt_n = custom_backtest_fn(
                        noisy_test_ew, "pred_ew", noisy_mkt, top_n=n_all)
                    portfolio_noisy_rows.append(_portfolio_row(
                        "EW", color, sigma, tn_mode, 0,
                        bt_n.get("metrics", {})))

            actual_mom_col = None
            if mom_col in noisy_test.columns:
                actual_mom_col = mom_col
            else:
                for c in noisy_test.columns:
                    if "mom" in c.lower() and "9" in c:
                        actual_mom_col = c
                        break

            if actual_mom_col is not None:
                noisy_test_mom = noisy_test.copy()
                noisy_test_mom["pred_mom"] = noisy_test_mom[actual_mom_col]

                val_n_mom = top_n_val.get("Momentum", top_n_fixed)
                test_n_mom = top_n_test.get("Momentum", top_n_fixed)
                for top_n, tn_mode in [
                    (top_n_fixed, "fixed"),
                    (val_n_mom, "val_best"),
                    (test_n_mom, "test_best"),
                ]:
                    bt_c = custom_backtest_fn(
                        noisy_test_mom, "pred_mom", market_da, top_n=top_n)
                    portfolio_clean_rows.append(_portfolio_row(
                        "Momentum", color, sigma, tn_mode, top_n,
                        bt_c.get("metrics", {})))
                    bt_n = custom_backtest_fn(
                        noisy_test_mom, "pred_mom", noisy_mkt, top_n=top_n)
                    portfolio_noisy_rows.append(_portfolio_row(
                        "Momentum", color, sigma, tn_mode, top_n,
                        bt_n.get("metrics", {})))

                stab_mom = {}
                pred_corr_mom = np.nan
                if (clean_test_result is not None
                        and "pred_momentum" in clean_test_result.columns):
                    sv = compute_topn_stability(
                        clean_test_result, noisy_test_mom,
                        "pred_momentum", n_stability_values,
                        noisy_pred_col="pred_mom")
                    for nv in n_stability_values:
                        stab_mom[f"stability_{nv}"] = sv.get(nv, np.nan)
                    pred_corr_mom = _compute_pred_corr(
                        clean_test_result, noisy_test_mom,
                        "pred_momentum", "pred_mom")
                else:
                    for nv in n_stability_values:
                        stab_mom[f"stability_{nv}"] = np.nan

                pearson_c_mom = _compute_pearson(noisy_test_mom, "pred_mom",
                                                "clean_target")
                pearson_n_mom = _compute_pearson(noisy_test_mom, "pred_mom",
                                                "target")

                ic_c = _compute_ic(noisy_test_mom, "pred_mom",
                                   target_col="clean_target")
                icir_c = _compute_icir(noisy_test_mom, "pred_mom",
                                       target_col="clean_target")
                model_clean_rows.append({
                    "model": "Momentum", "color": color, "sigma": sigma,
                    "ic": ic_c, "icir": icir_c,
                    "pearson": pearson_c_mom, "pred_corr": pred_corr_mom,
                    **stab_mom})

                ic_n = _compute_ic(noisy_test_mom, "pred_mom",
                                   target_col="target")
                icir_n = _compute_icir(noisy_test_mom, "pred_mom",
                                       target_col="target")
                model_noisy_rows.append({
                    "model": "Momentum", "color": color, "sigma": sigma,
                    "ic": ic_n, "icir": icir_n,
                    "pearson": pearson_n_mom, "pred_corr": pred_corr_mom,
                    **stab_mom})

    return (pd.DataFrame(model_clean_rows),
            pd.DataFrame(model_noisy_rows),
            pd.DataFrame(portfolio_clean_rows),
            pd.DataFrame(portfolio_noisy_rows))
