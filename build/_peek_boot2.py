import json
j = json.load(open("results/law_bootstrap.json", encoding="utf-8"))
for k, v in j["ci"].items():
    print(f'{k:12s} {v["point"]:10.4f}  [{v["lo"]:.4f}, {v["hi"]:.4f}]')
print("r2_log", round(j["r2_log_ci"]["point"],4), round(j["r2_log_ci"]["lo"],4), round(j["r2_log_ci"]["hi"],4))
