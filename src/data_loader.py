"""
data_loader.py — загрузка SPX данных, feature engineering (26 фичей),
построение ежемесячного датасета с excess return target.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
import qnt.ta as qnta
from tqdm import tqdm

MARKET_FIELDS = frozenset(
    ["open", "close", "low", "high", "vol", "divs", "split_cumprod", "is_liquid"]
)

FUNDAMENTAL_FEATURES = frozenset([
    "agr", "bm", "dy", "nincr", "roeq", "sp", "mvel1", "ep", "lv",
])

TS_FEATURES = [
    "beta", "betasq", "chmom", "dolvol", "idiovol", "ill",
    "maxret", "mom1m", "mom6m", "mom9m", "mom12m", "mom36m", "retvol",
    "str", "high52w", "skew", "abnormal_turn",
    "rsi_14", "macd_hist", "atr_14", "sma20_ratio", "sma50_ratio",
    "quarter", "is_january", "vol_regime",
]


def load_spx_data(base: Path):
    """SPX market + fundamentals."""
    market_da = xr.open_dataarray(base / "exports_spx" / "spx_market.nc")
    fund_da = xr.open_dataarray(base / "exports_spx" / "spx_fundamentals.nc")
    return market_da, fund_da


def _safe_div(a, b, fill=0.0):
    result = a / b
    return result.where(np.isfinite(result), fill)


def compute_paper_features(market_da: xr.DataArray,
                           fund_da: xr.DataArray) -> xr.DataArray:
    close = market_da.sel(field="close")
    high = market_da.sel(field="high")
    low = market_da.sel(field="low")
    volume = market_da.sel(field="vol")
    divs = market_da.sel(field="divs")

    f = lambda name: fund_da.sel(field=name)

    daily_ret = close / close.shift(time=1) - 1
    market_ret = daily_ret.mean(dim="asset")

    feats = {}

    assets = f("assets")
    feats["agr"] = assets / assets.shift(time=252) - 1

    cov_rm = (daily_ret * market_ret).rolling(time=60).mean() - \
             daily_ret.rolling(time=60).mean() * market_ret.rolling(time=60).mean()
    var_m = market_ret.rolling(time=60).var()
    feats["beta"] = _safe_div(cov_rm, var_m)
    feats["betasq"] = feats["beta"] ** 2

    shares = f("shares")
    market_cap = close * shares
    feats["bm"] = _safe_div(f("equity"), market_cap)

    mom6m = close / close.shift(time=126) - 1
    feats["chmom"] = mom6m - mom6m.shift(time=21)

    dollar_vol = close * volume
    feats["dolvol"] = np.log(dollar_vol.rolling(time=21).mean().where(
        dollar_vol.rolling(time=21).mean() > 0, 1.0))

    cum_divs_12m = divs.rolling(time=252).sum()
    feats["dy"] = _safe_div(cum_divs_12m, close)

    residuals = daily_ret - feats["beta"] * market_ret
    feats["idiovol"] = residuals.rolling(time=60).std()

    abs_ret_over_dvol = _safe_div(abs(daily_ret), dollar_vol)
    feats["ill"] = abs_ret_over_dvol.rolling(time=21).mean()

    feats["maxret"] = daily_ret.rolling(time=21).max()

    feats["mom1m"] = close / close.shift(time=21) - 1
    feats["mom6m"] = mom6m
    feats["mom9m"] = close / close.shift(time=189) - 1
    feats["mom12m"] = close / close.shift(time=252) - 1
    feats["mom36m"] = close / close.shift(time=756) - 1

    feats["mvel1"] = np.log(market_cap.where(market_cap > 0, 1.0))

    eps = f("eps")
    eps_diff = eps - eps.shift(time=63)
    feats["nincr"] = (eps_diff > 0).astype(float).rolling(time=252).sum()

    feats["retvol"] = daily_ret.rolling(time=21).std()

    feats["roeq"] = f("roe")

    feats["sp"] = _safe_div(f("total_revenue"), market_cap)

    feats["ep"] = _safe_div(eps, close)

    feats["lv"] = _safe_div(assets - f("equity"), market_cap)

    feats["str"] = close / close.shift(time=5) - 1

    feats["high52w"] = close / close.rolling(time=252).max()

    # skew = (E[X³] - 3μσ² - μ³) / σ³
    _m1 = daily_ret.rolling(time=60).mean()
    _m2 = (daily_ret ** 2).rolling(time=60).mean()
    _m3 = (daily_ret ** 3).rolling(time=60).mean()
    _var = _m2 - _m1 ** 2
    feats["skew"] = _safe_div(_m3 - 3 * _m1 * _var - _m1 ** 3, _var ** 1.5)

    turnover = _safe_div(volume, shares)
    turn_avg = turnover.rolling(time=252).mean()
    feats["abnormal_turn"] = _safe_div(turnover.rolling(time=21).mean(), turn_avg)

    feats["rsi_14"] = qnta.rsi(close, 14) / 100.0
    _, _, macd_hist = qnta.macd(close, 12, 26, 9)
    feats["macd_hist"] = _safe_div(macd_hist, close)
    feats["atr_14"] = _safe_div(qnta.atr(high, low, close, 14), close)
    sma20 = qnta.sma(close, 20)
    sma50 = qnta.sma(close, 50)
    feats["sma20_ratio"] = _safe_div(close - sma20, sma20)
    feats["sma50_ratio"] = _safe_div(close - sma50, sma50)

    times = pd.DatetimeIndex(close.coords["time"].values)
    ones = xr.ones_like(close)

    quarter_vals = np.array([t.quarter for t in times], dtype=np.float32) / 4.0
    feats["quarter"] = ones * xr.DataArray(
        quarter_vals, dims=["time"], coords={"time": close.coords["time"]})

    jan_vals = np.array([1.0 if t.month == 1 else 0.0 for t in times], dtype=np.float32)
    feats["is_january"] = ones * xr.DataArray(
        jan_vals, dims=["time"], coords={"time": close.coords["time"]})

    market_vol_60 = daily_ret.std(dim="asset").rolling(time=60).mean()
    _vol_vals = market_vol_60.values.copy()
    _vol_min = np.minimum.accumulate(np.where(np.isnan(_vol_vals), np.inf, _vol_vals))
    _vol_max = np.maximum.accumulate(np.where(np.isnan(_vol_vals), -np.inf, _vol_vals))
    _vol_range = _vol_max - _vol_min
    _safe_range = np.where(_vol_range > 1e-10, _vol_range, 1.0)
    _vol_norm = np.where(_vol_range > 1e-10,
                         (_vol_vals - _vol_min) / _safe_range, 0.0)
    _vol_norm = np.clip(_vol_norm, 0, 1)
    vol_norm = xr.DataArray(_vol_norm, dims=["time"],
                            coords={"time": market_vol_60.coords["time"]})
    feats["vol_regime"] = ones * vol_norm

    for k in feats:
        feats[k] = feats[k].where(np.isfinite(feats[k]))

    ds = xr.Dataset({k: v.drop_vars("field", errors="ignore") for k, v in feats.items()})
    return ds.to_array(dim="field")


def extract_report_dates(
    fund_series: pd.DataFrame,
    gap_days: int = 21,
) -> pd.DatetimeIndex:
    """Из forward-filled ряда извлекает даты реальных отчётов."""
    fund_series = fund_series.copy()
    fund_series.index = pd.to_datetime(fund_series.index)
    diff = fund_series.diff()
    changed = (diff.fillna(0).abs() > 1e-10).any(axis=1)
    changed.iloc[0] = fund_series.iloc[0].notna().any()
    change_dates = fund_series.index[changed]
    if len(change_dates) == 0:
        return pd.DatetimeIndex([])

    clusters = [[change_dates[0]]]
    for d in change_dates[1:]:
        if (d - clusters[-1][-1]).days <= gap_days:
            clusters[-1].append(d)
        else:
            clusters.append([d])

    return pd.DatetimeIndex([c[-1] for c in clusters])


def _clean_fund_fields(fund_da: xr.DataArray) -> list:
    return [f for f in fund_da.coords["field"].values.astype(str)
            if f not in MARKET_FIELDS]


def build_monthly_dataset(
    market_da: xr.DataArray,
    features_da: xr.DataArray,
    fund_da: xr.DataArray,
    horizon_days: int = 30,
    use_is_liquid: bool = True,
    require_all_features: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ежемесячный датасет с excess total return target (с учётом дивидендов)."""
    close_da = market_da.sel(field="close")
    is_liquid_da = (
        market_da.sel(field="is_liquid")
        if use_is_liquid and "is_liquid" in market_da.coords["field"].values
        else None
    )

    all_times = pd.DatetimeIndex(market_da.coords["time"].values)
    assets = list(features_da.coords["asset"].values.astype(str))
    fields = [str(f) for f in features_da.coords["field"].values]
    n_features = len(fields)

    rebal_dates = pd.date_range(all_times.min(), all_times.max(), freq="MS")
    print(f"Total assets: {len(assets)}, features: {n_features}")

    close_df = close_da.to_pandas()
    close_df.index = pd.to_datetime(close_df.index)

    has_divs = "divs" in market_da.coords["field"].values
    if has_divs:
        divs_da = market_da.sel(field="divs")
        divs_df = divs_da.to_pandas().fillna(0)
        divs_df.index = pd.to_datetime(divs_df.index)
    else:
        divs_df = None

    market_ret_cache = {}
    for rd in tqdm(rebal_dates, desc="Market returns"):
        future_idx = close_df.index.searchsorted(rd + pd.Timedelta(days=horizon_days))
        if future_idx >= len(close_df):
            continue
        now_prices = close_df.loc[rd] if rd in close_df.index else close_df.iloc[
            close_df.index.searchsorted(rd) - 1] if close_df.index.searchsorted(rd) > 0 else None
        future_prices = close_df.iloc[future_idx]
        if now_prices is None:
            continue
        valid = (now_prices > 0) & now_prices.notna() & future_prices.notna()
        if is_liquid_da is not None:
            try:
                liq = is_liquid_da.sel(time=rd, method="ffill").to_pandas()
                valid = valid & (liq > 0.5)
            except Exception:
                pass
        if divs_df is not None:
            future_date = close_df.index[future_idx]
            d_sum = divs_df.loc[(divs_df.index > now_prices.name) &
                                (divs_df.index <= future_date)].sum().fillna(0)
            rets = ((future_prices[valid] + d_sum[valid]) / now_prices[valid] - 1)
        else:
            rets = (future_prices[valid] / now_prices[valid] - 1)
        market_ret_cache[rd] = float(rets.mean()) if len(rets) > 10 else np.nan

    feat_data = {}
    for field in fields:
        s = features_da.sel(field=field).to_pandas()
        s.index = pd.to_datetime(s.index)
        feat_data[field] = s

    rows = []
    for asset in tqdm(assets, desc="Building monthly dataset"):
        close_s = close_df[asset].dropna() if asset in close_df.columns else pd.Series(dtype=float)
        if len(close_s) < 60:
            continue

        is_liq_s = None
        if is_liquid_da is not None:
            try:
                is_liq_s = is_liquid_da.sel(asset=asset).to_pandas()
                if isinstance(is_liq_s, pd.DataFrame):
                    is_liq_s = is_liq_s.squeeze()
                is_liq_s.index = pd.to_datetime(is_liq_s.index)
            except Exception:
                is_liq_s = None

        for rd in rebal_dates:
            if rd not in market_ret_cache or np.isnan(market_ret_cache[rd]):
                continue

            if is_liq_s is not None:
                try:
                    liq = is_liq_s.asof(rd)
                    if pd.isna(liq) or float(liq) < 0.5:
                        continue
                except Exception:
                    continue

            # closest trading day on or before rd
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

            div_sum_asset = 0.0
            if divs_df is not None and asset in divs_df.columns:
                date_now = close_s.index[idx_now]
                date_future = close_s.index[idx_future]
                d_slice = divs_df.loc[(divs_df.index > date_now) &
                                       (divs_df.index <= date_future), asset]
                div_sum_asset = d_slice.sum()
                if pd.isna(div_sum_asset):
                    div_sum_asset = 0.0
            stock_ret = (price_future + div_sum_asset) / price_now - 1
            excess_ret = stock_ret - market_ret_cache[rd]

            # point-in-time features as of rd (no leakage)
            row = {"asset": asset, "date": rd, "target": excess_ret}
            n_valid_features = 0
            for field in fields:
                fd = feat_data[field]
                if asset not in fd.columns:
                    row[field] = np.nan
                    continue
                col = fd[asset]
                idx = col.index.searchsorted(rd, side="right") - 1
                if idx >= 0:
                    val = col.iloc[idx]
                    row[field] = float(val) if not pd.isna(val) else np.nan
                    if not pd.isna(val):
                        n_valid_features += 1
                else:
                    row[field] = np.nan

            if require_all_features:
                if n_valid_features == n_features:
                    rows.append(row)
            else:
                if n_valid_features > 0:
                    rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) > 0:
        stats = df.groupby("date").agg(
            n_assets=("asset", "nunique"),
            n_samples=("asset", "count"),
        ).reset_index()
        stats["date"] = pd.to_datetime(stats["date"])
        
        print(f"\nDataset: {len(df)} samples, {len(df['asset'].unique())} unique assets, "
              f"{len(df['date'].unique())} months")
        print(f"Assets per month: min={stats['n_assets'].min()}, "
              f"max={stats['n_assets'].max()}, mean={stats['n_assets'].mean():.1f}")
    else:
        stats = pd.DataFrame(columns=["date", "n_assets", "n_samples"])
        print("Dataset is empty!")
    
    return df, stats
