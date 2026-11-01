import json
j = json.load(open("results/law_bootstrap.json", encoding="utf-8"))
print("model", j["model"], "n_boot", j["n_boot"], "eps", j["eps"])
print("point", {k: round(v, 4) for k, v in j["point"]["params"].items()} if isinstance(j["point"]["params"], dict) else [round(v,4) for v in j["point"]["params"]])
print("r2_log", round(j["point"]["r2_log"], 4))
print("ci:", {k: (round(v["point"],4), round(v["lo"],4), round(v["hi"],4)) for k, v in j["ci"].items()})
