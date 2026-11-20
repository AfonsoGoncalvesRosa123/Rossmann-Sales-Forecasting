"""
Rossmann store sales: 42-day-ahead demand forecasting with time-based backtests.

Usage:
    python forecast_rossmann.py <data_dir> <results_dir> [--fast]

<data_dir> must contain train.csv and store.csv (Kaggle "Rossmann Store Sales").

Design
------
* Task: at an "origin" day t (last day whose sales are known), forecast the daily
  sales of every store for the next H = 42 days (direct multi-horizon model).
* Only features available at the origin are used: lagged / rolling sales up to t,
  last-year sales (target day - 364 days, always <= t because H < 364), and
  calendar / promotion / holiday information of the target day (known in advance).
* Evaluation is a rolling-origin backtest: the model is trained only on targets
  up to the origin and tested on the following 42 days. Two origins are used.
* Metrics are computed on days when the store was open (Sales > 0).
* Models: seasonal naive (last week, last year), optional ETS (needs statsmodels),
  and one global gradient boosting model on log1p(Sales).
"""
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

H = 42
RNG = np.random.default_rng(42)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# --------------------------------------------------------------------------- data
def load(data_dir):
    data_dir = Path(data_dir)
    tr = pd.read_csv(data_dir / "train.csv", dtype={"StateHoliday": str}, parse_dates=["Date"])
    st = pd.read_csv(data_dir / "store.csv")
    tr["StateHoliday"] = tr["StateHoliday"].fillna("0").astype(str).map(
        {"0": 0, "a": 1, "b": 2, "c": 3}).fillna(0).astype(int)
    return tr, st


class Panel:
    """Store x day matrices (NaN where unknown)."""

    def __init__(self, tr, st):
        self.dates = pd.date_range(tr["Date"].min(), tr["Date"].max(), freq="D")
        self.stores = np.sort(tr["Store"].unique())
        ns, nd = len(self.stores), len(self.dates)
        si = np.searchsorted(self.stores, tr["Store"].values)
        di = (tr["Date"] - self.dates[0]).dt.days.values

        def mat(col, fill=np.nan, valid=None):
            m = np.full((ns, nd), fill, dtype=np.float32)
            v = tr[col].values.astype(np.float32)
            if valid is not None:
                v = np.where(valid, v, np.nan)
            m[si, di] = v
            return m

        valid = (tr["Open"].values == 1) & (tr["Sales"].values > 0)
        self.S = mat("Sales", valid=valid)              # sales on valid open days, NaN otherwise
        self.P = mat("Promo", fill=0)
        self.SCH = mat("SchoolHoliday", fill=0)
        self.SH = mat("StateHoliday", fill=0)
        self.nd, self.ns = nd, ns
        self.dow = self.dates.dayofweek.values
        self.dom = self.dates.day.values
        self.month = self.dates.month.values
        self.woy = self.dates.isocalendar().week.values.astype(int)
        s = st.set_index("Store").reindex(self.stores)
        self.store_feats = pd.DataFrame({
            "store_type": s["StoreType"].astype("category").cat.codes.values,
            "assortment": s["Assortment"].astype("category").cat.codes.values,
            "log_comp_dist": np.log1p(s["CompetitionDistance"].values.astype(float)),
            "promo2": s["Promo2"].values,
        })


def nanmean(a, axis=1):
    cnt = np.sum(~np.isnan(a), axis=axis)
    tot = np.nansum(a, axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


FEATURES = ["h", "dow", "dom", "month", "woy", "promo", "school_hol", "state_hol",
            "m7", "m28", "m91", "trend", "dow_mean8", "promo_mean", "nopromo_mean", "promo_lift",
            "ly1", "ly7", "ly_growth", "last_week_same_dow",
            "store_type", "assortment", "log_comp_dist", "promo2"]


def build_origin(p, t, max_target=None):
    """Feature table for origin t, one row per (store, target day) with target in t+1..t+H."""
    last = p.nd - 1 if max_target is None else min(max_target, p.nd - 1)
    hs = np.arange(1, H + 1)
    hs = hs[t + hs <= last]
    if len(hs) == 0:
        return None
    idx = t + hs                                           # target day indices
    S = p.S
    win = lambda k: S[:, max(0, t - k + 1): t + 1]
    m7, m28, m91 = nanmean(win(7)), nanmean(win(28)), nanmean(win(91))
    # same-weekday mean over the last 8 weeks
    dm = np.full((p.ns, 7), np.nan, dtype=np.float32)
    lo = max(0, t - 55)
    for k in range(7):
        cols = np.arange(lo, t + 1)[p.dow[lo:t + 1] == k]
        dm[:, k] = nanmean(S[:, cols])
    # promo / non-promo level over the last 91 days
    lo91 = max(0, t - 90)
    s91, p91 = S[:, lo91:t + 1], p.P[:, lo91:t + 1]
    promo_m = nanmean(np.where(p91 == 1, s91, np.nan))
    nopromo_m = nanmean(np.where(p91 == 0, s91, np.nan))
    # growth vs same window one year earlier
    if t - 364 - 27 >= 0:
        m28_ly = nanmean(S[:, t - 364 - 27: t - 364 + 1])
    else:
        m28_ly = np.full(p.ns, np.nan, dtype=np.float32)

    dow_t = p.dow[idx]
    ly_idx = idx - 364
    ly1 = np.full((p.ns, len(hs)), np.nan, dtype=np.float32)
    ly7 = np.full((p.ns, len(hs)), np.nan, dtype=np.float32)
    ok = ly_idx - 3 >= 0
    for j in np.where(ok)[0]:
        ly1[:, j] = S[:, ly_idx[j]]
        ly7[:, j] = nanmean(S[:, ly_idx[j] - 3: ly_idx[j] + 4])
    # last observed value on the same weekday (weekly seasonal naive)
    back = idx - 7 * np.ceil(hs / 7).astype(int)
    lastwk = S[:, back]

    def tile(v):
        return np.repeat(v[:, None], len(hs), axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        cols = {
            "h": np.tile(hs, (p.ns, 1)),
            "dow": np.tile(dow_t, (p.ns, 1)),
            "dom": np.tile(p.dom[idx], (p.ns, 1)),
            "month": np.tile(p.month[idx], (p.ns, 1)),
            "woy": np.tile(p.woy[idx], (p.ns, 1)),
            "promo": p.P[:, idx],
            "school_hol": p.SCH[:, idx],
            "state_hol": p.SH[:, idx],
            "m7": tile(m7), "m28": tile(m28), "m91": tile(m91),
            "trend": tile(m28 / m91),
            "dow_mean8": dm[:, dow_t],
            "promo_mean": tile(promo_m), "nopromo_mean": tile(nopromo_m),
            "promo_lift": tile(promo_m / nopromo_m),
            "ly1": ly1, "ly7": ly7, "ly_growth": tile(m28 / m28_ly),
            "last_week_same_dow": lastwk,
            "store_type": tile(p.store_feats["store_type"].values),
            "assortment": tile(p.store_feats["assortment"].values),
            "log_comp_dist": tile(p.store_feats["log_comp_dist"].values),
            "promo2": tile(p.store_feats["promo2"].values),
        }
    df = pd.DataFrame({k: v.ravel().astype(np.float32) for k, v in cols.items()})
    df["store"] = np.repeat(p.stores, len(hs))
    df["date"] = p.dates[np.tile(idx, p.ns)]
    df["y"] = S[:, idx].ravel()
    df["origin"] = t
    return df


def train_frame(p, last_origin, stride, store_frac=1.0):
    """Training rows: origins every `stride` days, all targets known by `last_origin`."""
    frames = []
    for t in range(91, last_origin, stride):
        d = build_origin(p, t, max_target=last_origin)
        if d is not None:
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["y"].notna()]
    if store_frac < 1:
        keep = RNG.choice(p.stores, int(len(p.stores) * store_frac), replace=False)
        df = df[df["store"].isin(keep)]
    return df


# ------------------------------------------------------------------------ metrics
def metrics(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    m = ~np.isnan(pred) & ~np.isnan(y)
    y, pred = y[m], pred[m]
    return {
        "WAPE": np.abs(y - pred).sum() / y.sum(),
        "RMSE": float(np.sqrt(np.mean((y - pred) ** 2))),
        "RMSPE": float(np.sqrt(np.mean(((y - pred) / y) ** 2))),
        "n": int(m.sum()),
    }


# -------------------------------------------------------------------------- ETS
def ets_forecast(p, t, stores_sample):
    """Optional baseline: damped additive Holt-Winters (weekly season) on log sales."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    out = {}
    lo = max(0, t - 364)
    idx = np.arange(t + 1, min(t + H, p.nd - 1) + 1)
    for s in stores_sample:
        i = int(np.searchsorted(p.stores, s))
        ser = pd.Series(p.S[i, lo:t + 1], index=p.dates[lo:t + 1])
        # closed days -> value one week earlier, so the weekly pattern is kept
        ser = ser.groupby(ser.index.dayofweek).transform(lambda x: x.ffill().bfill())
        # weekdays with no sales at all in the window (e.g. stores closed every Sunday)
        # -> series mean; those days are never scored, this only keeps the fit NaN-free
        ser = ser.fillna(ser.mean())
        if ser.isna().any():
            continue
        ser = np.log1p(ser.astype(float))
        try:
            fit = ExponentialSmoothing(ser.values, trend="add", damped_trend=True,
                                       seasonal="add", seasonal_periods=7).fit()
            f = np.expm1(fit.forecast(len(idx)))
            out[s] = pd.Series(f, index=p.dates[idx])
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------- the run
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fast = "--fast" in sys.argv
    if len(args) != 2:
        sys.exit(__doc__)
    data_dir, out = args
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tr, st = load(data_dir)
    p = Panel(tr, st)
    print(f"{p.ns} stores, {p.nd} days: {p.dates[0].date()} to {p.dates[-1].date()}")

    stride = 28 if fast else 14
    frac = 0.5 if fast else 1.0
    origins = {"A (test = last 42 days)": p.nd - 1 - H,
               "B (test = previous 42 days)": p.nd - 1 - 2 * H}

    rows, by_h, all_pred = [], [], []
    last_model, last_test = None, None
    try:
        import statsmodels  # noqa: F401
        have_ets = True
    except ImportError:
        have_ets = False
        print("statsmodels not installed: skipping the ETS baseline (pip install statsmodels).")

    for name, t in origins.items():
        print(f"\nOrigin {name}: last known day {p.dates[t].date()}")
        train = train_frame(p, t, stride, frac)
        test = build_origin(p, t, max_target=t + H)
        test = test[test["y"].notna()].reset_index(drop=True)
        print(f"  training rows {len(train):,}, test rows {len(test):,}")

        model = HistGradientBoostingRegressor(
            max_iter=150 if fast else 400, learning_rate=0.1, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0, random_state=42)
        model.fit(train[FEATURES], np.log1p(train["y"]))
        preds = {"Gradient boosting": np.expm1(model.predict(test[FEATURES]))}
        naive_w = test["last_week_same_dow"].fillna(test["dow_mean8"])
        preds["Seasonal naive (last week)"] = naive_w.values
        preds["Seasonal naive (last year)"] = test["ly1"].fillna(naive_w).values

        if have_ets:
            sample = RNG.choice(p.stores, 60, replace=False)
            f = ets_forecast(p, t, sample)
            e = test[test["store"].isin(list(f))].copy()
            e["pred"] = [f[s].get(d, np.nan) for s, d in zip(e["store"], e["date"])]
            sub = test["store"].isin(list(f))
            for k, v in preds.items():
                r = metrics(test.loc[sub, "y"], np.asarray(v)[sub.values])
                rows.append({"origin": name, "model": k + " [ETS sample]", **r})
            rows.append({"origin": name, "model": "ETS Holt-Winters [ETS sample]",
                         **metrics(e["y"], e["pred"])})

        for k, v in preds.items():
            rows.append({"origin": name, "model": k, **metrics(test["y"], v)})
        test = test.assign(pred=preds["Gradient boosting"], origin_name=name)
        all_pred.append(test[["origin_name", "store", "date", "h", "y", "pred"]])
        for lo, hi in [(1, 14), (15, 28), (29, 42)]:
            m = test["h"].between(lo, hi)
            for k, v in preds.items():
                by_h.append({"origin": name, "model": k, "horizon_days": f"{lo}-{hi}",
                             **metrics(test.loc[m, "y"], np.asarray(v)[m.values])})
        print(pd.DataFrame([r for r in rows if r["origin"] == name and "[ETS" not in r["model"]])
              [["model", "WAPE", "RMSE", "RMSPE"]].round(4).to_string(index=False))
        last_model, last_test, last_name = model, test, name
        last_train_origin = t

    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(by_h).to_csv(out / "metrics_by_horizon.csv", index=False)
    pred_all = pd.concat(all_pred)
    pred_all.to_csv(out / "predictions.csv", index=False)

    # ---------- figures
    main_res = res[~res["model"].str.contains("ETS sample")]
    piv = main_res.pivot(index="model", columns="origin", values="WAPE")
    ax = piv.plot.barh(figsize=(8, 4))
    ax.set_xlabel("WAPE (lower is better)")
    ax.set_title("Forecast error by model and backtest origin")
    plt.tight_layout(); plt.savefig(out / "model_comparison.png", dpi=150); plt.close()

    h = pd.DataFrame(by_h)
    h = h[h["origin"] == list(origins)[0]].pivot(index="horizon_days", columns="model", values="WAPE")
    ax = h.plot.bar(figsize=(8, 4), rot=0)
    ax.set_ylabel("WAPE"); ax.set_title("Error by forecast horizon (origin A)")
    plt.tight_layout(); plt.savefig(out / "error_by_horizon.png", dpi=150); plt.close()

    a = pred_all[pred_all["origin_name"] == list(origins)[0]]
    tot = a.groupby("date")[["y", "pred"]].sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(tot.index, tot["y"], label="Actual"); ax.plot(tot.index, tot["pred"], label="Forecast")
    ax.set_title("Total sales of open stores, 42-day forecast (origin A)")
    ax.legend(); fig.autofmt_xdate(); plt.tight_layout()
    plt.savefig(out / "total_forecast.png", dpi=150); plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, s in zip(axes, RNG.choice(p.stores, 3, replace=False)):
        d = a[a["store"] == s].sort_values("date")
        i = int(np.searchsorted(p.stores, s))
        t = origins[list(origins)[0]]
        hist = pd.Series(p.S[i, t - 56:t + 1], index=p.dates[t - 56:t + 1]).dropna()
        ax.plot(hist.index, hist.values, color="gray", label="History")
        ax.plot(d["date"], d["y"], label="Actual"); ax.plot(d["date"], d["pred"], label="Forecast")
        ax.set_title(f"Store {s}", fontsize=10)
    axes[0].legend(ncol=3); fig.autofmt_xdate(); plt.tight_layout()
    plt.savefig(out / "store_examples.png", dpi=150); plt.close()

    samp = last_test.sample(min(15000, len(last_test)), random_state=42)
    pi = permutation_importance(last_model, samp[FEATURES], np.log1p(samp["y"]),
                                n_repeats=3, random_state=42, scoring="neg_root_mean_squared_error")
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
    imp.to_csv(out / "permutation_importance.csv", header=["rmse_increase_log"])
    ax = imp.tail(12).plot.barh(figsize=(7, 5))
    ax.set_xlabel("Increase in RMSE (log scale) when shuffled")
    ax.set_title("Permutation importance (origin B model, its test window)")
    plt.tight_layout(); plt.savefig(out / "permutation_importance.png", dpi=150); plt.close()

    print("\nSummary (WAPE):")
    print(piv.round(4).to_string())
    print(f"\nDone in {time.time() - t0:.0f}s. Files written to {out}/")


if __name__ == "__main__":
    main()
