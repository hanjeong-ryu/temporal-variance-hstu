import numpy as np

from .axes import axis1, axis2, axis3, user_summary
from .data import DAY, KS, preprocess, truncate_tail
from .decomp import decompose

COLS = ["K", "within", "between", "rho_k",
        "within_norm", "between_norm", "bias_frac", "n_users"]

FMT = {"within": "{:.3e}".format, "between": "{:.3e}".format,
       "rho_k": "{:.4f}".format, "within_norm": "{:.3f}".format,
       "between_norm": "{:.3f}".format, "bias_frac": "{:.4f}".format}


def _p(d):
    for k, v in d.items():
        print(f"   {k:<24} {v:,.6f}" if isinstance(v, float) else f"   {k:<24} {v}")


def run(name, df_raw, maxlen=None, cohort="fixed", ks=KS, verbose=True, **kw):
    df = preprocess(df_raw)
    if maxlen:
        df = truncate_tail(df, maxlen)

    s = user_summary(df)
    dec = decompose(df, ks=ks, cohort=cohort, **kw)
    var_tau = float(dec["between"].iloc[0])

    if verbose:
        print("=" * 72)
        print(f"  {name}   [{'최근 %d개' % maxlen if maxlen else '전체 이력'}]"
              f"   cohort={cohort}")
        print("=" * 72)
        print(f"[기초] users={df.userId.nunique():,}  items={df.movieId.nunique():,}  "
              f"interactions={len(df):,}  avg_len={len(df)/df.userId.nunique():.2f}")

        print("\n[축 1] 활동률 이질성 (부차 지표)");    _p(axis1(s))
        print("\n[축 2] 관측 기간");                   _p(axis2(s))
        print("\n[축 3] 타임스탬프 해상도 (경쟁 설명)"); _p(axis3(df))

        print("\n[축 4] Var(T|K=k) 분해   (단위: 초^2)")
        print(dec[COLS].to_string(index=False, formatters=FMT))
        print(f"\n   >> Var(tau) = between(k=1) = {var_tau:.3e} sec^2 "
              f"→ SD(tau) = {np.sqrt(var_tau)/DAY:.2f} days")
        print()

    return {"summary": s, "decomp": dec, "var_tau": var_tau}
