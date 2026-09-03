# Var(T|K=k) = E_u[Var(T|u,k)] + Var_u(E[T|u,k])
#              within            between
# rho_k = between / (within + between)

import numpy as np
import pandas as pd

from .data import KS


def lag_pairs(df, ks=KS, cohort="fixed", max_users=20000,
              pairs_per_user=200, disjoint=False, seed=0):
    rng = np.random.default_rng(seed)
    kmax = max(ks)
    cnt = df.groupby("userId").size()

    # kmax+1 이 아니라 kmax+2. 상호작용이 kmax+1 개면 k=kmax 에서 쌍이 하나뿐이라
    # var 가 NaN 이 되고 큰 k 에서만 코호트가 줄어든다
    need = kmax + 2 if cohort == "fixed" else 2
    eligible = np.asarray(cnt[cnt >= need].index)
    if len(eligible) > max_users:
        eligible = rng.choice(eligible, max_users, replace=False)

    us, kk, tt = [], [], []
    sub = df[df.userId.isin(eligible)]
    for u, ts in sub.groupby("userId")["timestamp"]:
        ts = ts.values
        n = len(ts)
        for k in ks:
            if n <= k + 1:
                continue
            idx = np.arange(k, n, k) if disjoint else np.arange(k, n)
            if len(idx) > pairs_per_user:
                idx = rng.choice(idx, pairs_per_user, replace=False)
            us.append(np.full(len(idx), u))
            kk.append(np.full(len(idx), k))
            tt.append((ts[idx] - ts[idx - k]).astype("float64"))

    return pd.DataFrame({"userId": np.concatenate(us),
                         "K": np.concatenate(kk),
                         "T": np.concatenate(tt)})


def decompose(df, ks=KS, **kw):
    P = lag_pairs(df, ks=ks, **kw)

    rows = []
    for k, gk in P.groupby("K"):
        g = gk.groupby("userId")["T"]
        m, v, nu = g.mean(), g.var(ddof=1), g.size()
        ok = v.notna()
        m, v, nu = m[ok], v[ok], nu[ok]

        within = float(v.mean())
        raw = float(m.var(ddof=1))     # Var(mu_u) + E[sigma^2/n]
        bias = float((v / nu).mean())
        between = raw - bias           # 이질성이 거의 없으면 음수도 나옴. 클리핑 안 함

        rows.append({
            "K": int(k),
            "within": within,
            "between": between,
            "rho_k": between / (within + between),
            "within_over_k": within / k,
            "between_over_k2": between / k ** 2,
            "bias_frac": bias / raw if raw > 0 else np.nan,
            "n_users": int(len(m)),
        })

    dec = pd.DataFrame(rows).sort_values("K").reset_index(drop=True)
    # ks 첫 값을 1 로 놓은 것. 이론이 맞으면 둘 다 평평해야 함
    dec["within_norm"] = dec["within_over_k"] / dec["within_over_k"].iloc[0]
    dec["between_norm"] = dec["between_over_k2"] / dec["between_over_k2"].iloc[0]
    return dec
