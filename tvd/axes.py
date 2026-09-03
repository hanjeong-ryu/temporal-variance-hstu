import numpy as np
import pandas as pd

from .data import DAY


def user_summary(df):
    g = df.groupby("userId")["timestamp"]
    s = pd.DataFrame({"n": g.size(), "tmin": g.min(), "tmax": g.max()})
    s["span"] = s.tmax - s.tmin
    ok = s.span > 0
    s["lam"] = np.nan
    s.loc[ok, "lam"] = (s.loc[ok, "n"] - 1) / s.loc[ok, "span"]
    s["tau"] = 1.0 / s["lam"]      # span/(n-1)
    return s


def axis1(s, B=300, rng=None):
    # 활동률 이질성. 주지표는 between(k=1) 이고 이건 방향 확인용
    rng = rng or np.random.default_rng(0)
    lam = s["lam"].dropna().values
    loglam = np.log(lam)

    out = {
        "n_users_used":  len(loglam),
        "Var(log lam)":  float(np.var(loglam, ddof=1)),
        "SD(log lam)":   float(np.std(loglam, ddof=1)),
        "CV(lam)":       float(np.std(lam, ddof=1) / np.mean(lam)),
        "IQR(log lam)":  float(np.subtract(*np.percentile(loglam, [75, 25]))),
    }
    boot = [np.var(loglam[rng.integers(0, len(loglam), len(loglam))], ddof=1)
            for _ in range(B)]
    out["Var(log lam) 95%CI"] = tuple(np.percentile(boot, [2.5, 97.5]).round(4))

    tau_d = s["tau"].dropna() / DAY
    out["median tau (days)"] = float(tau_d.median())
    out["SD tau (days)"] = float(tau_d.std(ddof=1))
    return out


def axis2(s):
    d = s["span"] / DAY
    return {
        "span_median_days": float(d.median()),
        "span_mean_days":   float(d.mean()),
        "span_p10_days":    float(d.quantile(0.10)),
        "span_p90_days":    float(d.quantile(0.90)),
        "Var(log span)":    float(np.var(np.log(d[d > 0]), ddof=1)),
        "zero_span_users":  int((s["span"] == 0).sum()),
    }


def axis3(df):
    # 간격이 0 에 몰려 있으면 rho_k 의 k 의존성이 기록 방식 탓일 수 있음
    dt = df.groupby("userId")["timestamp"].diff().dropna()
    return {
        "adjacent_pairs":  int(len(dt)),
        "ratio_dt_eq_0":   float((dt == 0).mean()),
        "ratio_dt_lt_60s": float((dt < 60).mean()),
        "median_dt_sec":   float(dt.median()),
    }
