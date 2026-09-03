# 데이터 없이 도는 확인용. 10초.
#  (1) 이질성이 있을 때만 rho_k 가 k 를 따라 오르는지
#  (2) 겹치는 쌍을 쓰면 참값이 0 인데도 rho_k 가 오르는지
# (2)는 between 보정식이 쌍 사이 독립을 가정하기 때문. 겹치면 보정이 모자람

import pandas as pd

from tvd import decompose, make_synthetic_log, preprocess

KS = (1, 2, 5, 10, 20)


def rho(sd, disjoint):
    df = preprocess(make_synthetic_log(sd_log_lambda=sd, seed=0))
    dec = decompose(df, ks=KS, cohort="fixed", disjoint=disjoint, seed=0)
    return dec.set_index("K")["rho_k"]


print("[1] 이질성이 rho_k 를 만드는가")
print("-" * 52)
df = preprocess(make_synthetic_log(sd_log_lambda=1.0, seed=0))
dec = decompose(df, ks=KS, cohort="fixed", disjoint=True, seed=0)
print(dec[["K", "within_norm", "between_norm", "rho_k"]].to_string(
    index=False, formatters={"within_norm": "{:.3f}".format,
                             "between_norm": "{:.3f}".format,
                             "rho_k": "{:.4f}".format}))
print("\n지수분포로 만들었으니 증분이 iid. within_norm, between_norm 이 둘 다 1 근처면")
print("within ~ k, between ~ k^2 라는 뜻")

print("\n[2] 겹치는 쌍의 과소보정 (rho_k)")
print("-" * 52)
tab = pd.DataFrame({"동질/겹침": rho(0.0, False), "동질/분리": rho(0.0, True),
                    "이질/겹침": rho(1.0, False), "이질/분리": rho(1.0, True)})
print(tab.to_string(float_format="{:.4f}".format))
print("\n동질 로그의 참값은 rho_k = 0. 겹치면 k 가 커질수록 0 에서 멀어지고")
print("분리하면 0 근처에 남는다")
