import pandas as pd

DAY = 86400.0
MAXLEN = 200      # HSTU / FuXi-beta 계열 최대 시퀀스 길이
KS = (1, 2, 5, 10, 20, 50)


def load_ml1m(path="data/ml-1m/ratings.dat"):
    df = pd.read_csv(path, sep="::", engine="python", header=None,
                     names=["userId", "movieId", "rating", "timestamp"])
    return df[["userId", "movieId", "timestamp"]].astype(
        {"userId": "int32", "movieId": "int32", "timestamp": "int64"})


def load_ml20m(path="data/ml-20m/ratings.csv"):
    return pd.read_csv(path, usecols=["userId", "movieId", "timestamp"],
                       dtype={"userId": "int32", "movieId": "int32",
                              "timestamp": "int64"})


def preprocess(df, min_inter=5, min_item=5):
    # generative-recommenders 의 preprocess_rating 과 같은 방식.
    # 한 번만 걸러서 엄밀한 5-core 는 아님. 참조 구현과 맞추려고 그대로 둠.
    item_cnt = df.movieId.map(df.movieId.value_counts())
    user_cnt = df.userId.map(df.userId.value_counts())
    df = df[(item_cnt >= min_item) & (user_cnt >= min_inter)]

    vc = df.userId.value_counts()
    df = df[df.userId.isin(vc[vc >= min_inter].index)]
    return df.sort_values(["userId", "timestamp"],
                          kind="mergesort").reset_index(drop=True)


def truncate_tail(df, n=MAXLEN):
    # 모델이 보는 창과 맞춤. 정렬된 df 를 넣을 것
    return df.groupby("userId", group_keys=False).tail(n).reset_index(drop=True)
