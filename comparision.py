import json
from scipy.stats import pearsonr

with open("self_eval_confidence_scores.json", "r", encoding="utf-8") as f:
    self_eval_scores = json.load(f)

with open("fact_verification_results.json", "r", encoding="utf-8") as f:
    verification_results = json.load(f)

rows = []
for name, verifs in verification_results.items():
    if not verifs:
        continue
    true_count = sum(1 for v in verifs if v["verdict"] == "TRUE")
    false_count = sum(1 for v in verifs if v["verdict"] == "FALSE")
    judged = true_count + false_count
    false_rate = false_count / judged if judged > 0 else None

    eval_entry = self_eval_scores.get(name)
    if not eval_entry or false_rate is None:
        continue

    rows.append({
        "name": name,
        "false_rate": false_rate,
        "avg_true_probability": eval_entry["avg_true_probability"]
    })

print(f"{'Name':<30} {'FALSE-rate':>10} {'Self-eval avg':>14}")
for r in sorted(rows, key=lambda x: -x["false_rate"]):
    print(f"{r['name']:<30} {r['false_rate']:>10.3f} {r['avg_true_probability']:>14.4f}")

false_rates = [r["false_rate"] for r in rows]
confidences = [r["avg_true_probability"] for r in rows]

r_val, p_val = pearsonr(false_rates, confidences)
print(f"\nPearson correlation: FALSE-rate vs self-eval true_probability: r={r_val:.3f}, p={p_val:.3f}")
print("(Expectation: NEGATIVE correlation - higher self-eval confidence should mean lower FALSE-rate)")