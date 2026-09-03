# Temporal Variance Decomposition

순차 추천 로그에서 lag $k$ 만큼 떨어진 두 상호작용 사이의 소요 시간 $T$ 를 보고,
그 분산을 사용자 내(within) 와 사용자 간(between) 으로 나눈다.

$$\mathrm{Var}(T \mid K=k) = \underbrace{E_u[\mathrm{Var}(T \mid u,k)]}_{\text{within}} + \underbrace{\mathrm{Var}_u(E[T \mid u,k])}_{\text{between}}$$

평균정상성 아래 between 항은 $k^2 \mathrm{Var}(\tau)$ 로 닫힌다. 여기서
$\rho_k = \text{between} / (\text{within} + \text{between})$ 는 사용자별 시간
정규화로 제거할 수 있는 불확실성의 비율이다.

TiSASRec 계열은 사용자별 시간 정규화가 성능을 올린다는 것을 보였다. 이 코드는
그 정규화가 제거하는 게 정확히 무엇이고 각 거리에서 얼마인지를, 모델을 학습하지
않고 로그만으로 잰다.

## 실행

```
pip install -r requirements.txt
python toy_example.py
```

`toy_example.py` 는 데이터가 필요 없다. 합성 로그를 만들어서 도는지만 확인한다.

MovieLens 로 재현하려면 데이터를 먼저 받는다.

```
mkdir -p data && cd data
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
wget https://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-1m.zip && unzip ml-20m.zip && cd ..

python run_measure.py --dataset ml1m
python run_measure.py --dataset ml20m
```

결과는 `results/decomp_*.csv` 로 떨어진다. ML-20M 은 CPU 로 5~10분.

## 옵션

```
--dataset {ml1m,ml20m}
--maxlen N                 사용자별 최근 N개만 사용 (기본 200, 0 이면 전체)
--cohort {fixed,varying}   모든 k 에 같은 사용자를 쓸지
--disjoint                 lag 쌍을 겹치지 않게 추출
--max-users N              사용자 표본 상한 (기본 20000)
--pairs-per-user N         사용자당 쌍 상한 (기본 200)
--seed N
```

## 재는 것

| 축 | 대상 | 역할 |
|---|---|---|
| 1 | 활동률 이질성 `Var(log λ)` | 부차 지표. 방향 확인용 |
| 2 | 관측 기간 | 검열이 분해에 미치는 영향 |
| 3 | 타임스탬프 해상도 | 경쟁 설명 배제 |
| 4 | `Var(T\|K=k)` 분해 | 주지표. `Var(τ) = between(k=1)`, `ρ_k` |

## 걸리기 쉬운 곳

**코호트를 고정하지 않으면** k 가 커질수록 활동적인 사용자만 남아서 $\rho_k$ 증가가
composition effect 와 섞인다. 기본값은 `--cohort fixed` 이고, `varying` 으로
대조군을 볼 수 있다.

고정 조건은 상호작용 수 $\ge k_{\max}+2$ 다. $k_{\max}+1$ 로 잡으면 그 사용자는
$k=k_{\max}$ 에서 쌍이 하나뿐이라 분산이 NaN 이 되어 조용히 빠진다. 출력의
`n_users` 열이 k 마다 같은지 확인하는 게 이 조건이 지켜졌는지 보는 방법이다.

**between 은 위로 치우친다.** 표본 평균의 분산에 $E[\sigma^2/n]$ 이 섞여 있어서
빼줘야 한다. 보정량의 비중은 출력의 `bias_frac` 열로 본다. 이질성이 거의 없으면
보정 후 between 이 음수가 될 수 있는데, 클리핑하지 않는다. 음수 자체가
"잴 만한 이질성이 없다" 는 신호다.

**겹치는 lag 쌍을 쓰면 보정이 부족하다.** 보정식이 쌍 사이 독립을 가정하는데
겹치는 쌍은 상관이 있다. 참값이 $\rho_k = 0$ 인 합성 로그로 재보면 이렇게 나온다.

| $k$ | 겹침 | 분리 |
|----:|------:|------:|
| 1  | 0.0004 | 0.0004 |
| 5  | 0.0372 | 0.0036 |
| 20 | 0.1869 | 0.0099 |

`toy_example.py` 에서 재현된다. 같은 이유로 within 은 반대 방향으로 치우친다.
겹치는 쌍의 표본분산은 $\mathrm{Var}(T|u,k)$ 를 과소추정하고, 그것도 $\rho_k$ 를
올리는 쪽으로 작용한다. 실제 데이터 결과를 보고할 때 `--disjoint` 를 같이
돌려보는 게 안전하다.

## 구성

```
tvd/
  data.py        상수, 로드, 5-core 필터, 최근 N개 절단
  axes.py        축 1~3
  decomp.py      축 4. lag 쌍 추출과 within/between 분해
  synthetic.py   합성 로그
  report.py      실행과 출력
run_measure.py   MovieLens 측정
toy_example.py   데이터 없이 도는 예제
```

`tvd/` 아래 전부 합쳐 250줄 정도.

전처리는 참조 구현(HSTU generative-recommenders)과 맞추느라 아이템·사용자를
한 번만 거른다. 엄밀한 5-core 가 아니라서 `min_inter` 를 바꿔가며 결과가
얼마나 흔들리는지 확인하는 게 좋다.

## 참고

- Zhai et al., Actions Speak Louder than Words (HSTU), ICML 2024
- Li et al., Time Interval Aware Self-Attention (TiSASRec), WSDM 2020
- Laird & Ware, Random-Effects Models for Longitudinal Data, Biometrics 1982

분해 항등식 자체는 random-slope 모형의 주변분산, 갱신이론의 표준 결과와 동치다.
여기서 하는 건 그걸 순차 추천 로그에 적용해서 $\rho_k$ 를 진단 지표로 정의하고
실제로 재는 것이다.
