import json
from scipy.stats import pearsonr

with open("nli_consensus_contradiction_scores.json", "r", encoding="utf-8") as f:
    consistency_scores = json.load(f)

with open("fact_verification_results.json", "r", encoding="utf-8") as f:
    verification_results = json.load(f)

rows = []
for name, verifs in verification_results.items():
    if not verifs:
        continue
    true_count = sum(1 for v in verifs if v["verdict"] == "TRUE")
    false_count = sum(1 for v in verifs if v["verdict"] == "FALSE")
    unknown_count = sum(1 for v in verifs if v["verdict"] == "UNKNOWN")
    judged = true_count + false_count
    false_rate = false_count / judged if judged > 0 else None

    consistency = consistency_scores.get(name)
    if not consistency or false_rate is None:
        continue

    rows.append({
        "name": name,
        "false_rate": false_rate,
        "unknown_rate": unknown_count / len(verifs),
        "consensus_rate": consistency["consensus_rate"],
        "contradiction_rate": consistency["contradiction_rate"],
    })

# Print the table
print(f"{'Name':<30} {'FALSE-rate':>10} {'UNKNOWN-rate':>13} {'Consensus':>10} {'Contradiction':>13}")
for r in sorted(rows, key=lambda x: -x["false_rate"]):
    print(f"{r['name']:<30} {r['false_rate']:>10.3f} {r['unknown_rate']:>13.3f} "
          f"{r['consensus_rate']:>10.3f} {r['contradiction_rate']:>13.3f}")

# Correlations
false_rates = [r["false_rate"] for r in rows]
consensus_rates = [r["consensus_rate"] for r in rows]
contradiction_rates = [r["contradiction_rate"] for r in rows]

r1, p1 = pearsonr(false_rates, consensus_rates)
r2, p2 = pearsonr(false_rates, contradiction_rates)

print(f"\nPearson correlation: FALSE-rate vs consensus_rate:    r={r1:.3f}, p={p1:.3f}")
print(f"Pearson correlation: FALSE-rate vs contradiction_rate: r={r2:.3f}, p={p2:.3f}")
print(f"\n(Expectation: consensus should correlate NEGATIVELY with false_rate;")
print(f" contradiction should correlate POSITIVELY with false_rate)")