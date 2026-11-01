import pandas as pd
import config as C

d = pd.read_csv(C.RESULTS / "metrics_long.csv")
print("rows", len(d), "runs", d["run"].nunique())
sub = d[d["run"].str.startswith(("pathmnist_n1500", "pathmnist_n3000"))]
g = sub.groupby("run").agg(points=("epoch", "size"), M_max=("M", "max"),
                           M_last=("M", "last"), last_ep=("epoch", "last"),
                           sim_max=("sim_max", "max"), n_fid=("fid", "count"))
print(g.to_string())
for run in ("pathmnist_n1500_s1", "pathmnist_n3000_s1", "pathmnist_n1500_s2", "pathmnist_n3000_s2"):
    tail = sub[sub["run"] == run].tail(3)[["epoch", "M", "sim_max"]]
    print(run, "\n", tail.to_string(index=False))