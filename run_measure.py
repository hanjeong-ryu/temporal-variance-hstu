# python run_measure.py --dataset ml1m
# python run_measure.py --dataset ml20m --disjoint
# python run_measure.py --dataset ml1m --maxlen 0     (전체 이력)

import argparse
import os

from tvd import MAXLEN, load_ml1m, load_ml20m, run

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["ml1m", "ml20m"], default="ml1m")
ap.add_argument("--path", default=None)
ap.add_argument("--maxlen", type=int, default=MAXLEN)
ap.add_argument("--cohort", choices=["fixed", "varying"], default="fixed")
ap.add_argument("--disjoint", action="store_true")
ap.add_argument("--max-users", type=int, default=20000)
ap.add_argument("--pairs-per-user", type=int, default=200)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--outdir", default="results")
args = ap.parse_args()

loader = load_ml1m if args.dataset == "ml1m" else load_ml20m
df = loader(args.path) if args.path else loader()

res = run(args.dataset.upper(), df,
          maxlen=args.maxlen or None,
          cohort=args.cohort,
          disjoint=args.disjoint,
          max_users=args.max_users,
          pairs_per_user=args.pairs_per_user,
          seed=args.seed)

os.makedirs(args.outdir, exist_ok=True)
tag = f"{args.dataset}_{args.cohort}_maxlen{args.maxlen}"
if args.disjoint:
    tag += "_disjoint"
out = os.path.join(args.outdir, f"decomp_{tag}.csv")
res["decomp"].to_csv(out, index=False)
print("saved ->", out)
