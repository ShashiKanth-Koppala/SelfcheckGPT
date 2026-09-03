import json
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# Load extracted facts
with open("extracted_facts.json", "r", encoding="utf-8") as f:
    extracted_facts = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, good enough for this

SIMILARITY_THRESHOLD = 0.65  # facts above this cosine similarity are treated as "the same claim"

def clean_fact_list(sample_facts):
    """Handle the _raw_unparsed fallback entries and empty samples gracefully."""
    if isinstance(sample_facts, dict) and "_raw_unparsed" in sample_facts:
        return []  # skip malformed samples for scoring; fix these by hand later if you want them included
    if not isinstance(sample_facts, list):
        return []
    return [f for f in sample_facts if isinstance(f, str) and f.strip()]

def score_person(name, samples):
    """
    samples: list of fact-lists, one per generated biography sample.
    Returns a list of {fact, support_count, total_samples, consistency_score, matched_samples}
    """
    # Flatten all facts with their sample index
    all_facts = []
    for sample_idx, sample_facts in enumerate(samples):
        cleaned = clean_fact_list(sample_facts)
        for fact in cleaned:
            all_facts.append({"text": fact, "sample_idx": sample_idx})

    if not all_facts:
        return []

    texts = [f["text"] for f in all_facts]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    n = len(all_facts)
    similarity_matrix = embeddings @ embeddings.T  # cosine similarity since normalized

    assigned = [False] * n
    clusters = []

    for i in range(n):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if similarity_matrix[i][j] >= SIMILARITY_THRESHOLD:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    total_samples = len(samples)
    results = []
    for cluster in clusters:
        sample_indices = sorted(set(all_facts[idx]["sample_idx"] for idx in cluster))
        representative_text = all_facts[cluster[0]]["text"]  # first occurrence as the representative phrasing
        consistency_score = len(sample_indices) / total_samples
        results.append({
            "fact": representative_text,
            "matched_samples": sample_indices,
            "support_count": len(sample_indices),
            "total_samples": total_samples,
            "consistency_score": round(consistency_score, 2)
        })

    # Sort by consistency descending, so most-agreed facts appear first
    results.sort(key=lambda x: -x["consistency_score"])
    return results

all_scores = {}
person_level_summary = {}

for name, samples in extracted_facts.items():
    print(f"Scoring {name}...")
    scored_facts = score_person(name, samples)
    all_scores[name] = scored_facts

    if scored_facts:
        avg_consistency = np.mean([f["consistency_score"] for f in scored_facts])
        low_consistency_count = sum(1 for f in scored_facts if f["consistency_score"] < 0.5)
        person_level_summary[name] = {
            "avg_fact_consistency": round(float(avg_consistency), 3),
            "total_unique_facts": len(scored_facts),
            "low_consistency_facts": low_consistency_count
        }
    else:
        person_level_summary[name] = {"avg_fact_consistency": None, "total_unique_facts": 0, "low_consistency_facts": 0}

# Save detailed per-fact scores
with open("fact_consistency_scores_65.json", "w", encoding="utf-8") as f:
    json.dump(all_scores, f, ensure_ascii=False, indent=2)

# Save person-level summary, sorted by lowest average consistency first (most likely hallucination-prone)
sorted_summary = dict(sorted(person_level_summary.items(), key=lambda x: (x[1]["avg_fact_consistency"] is None, x[1]["avg_fact_consistency"] or 0)))
with open("person_consistency_summary_65.json", "w", encoding="utf-8") as f:
    json.dump(sorted_summary, f, ensure_ascii=False, indent=2)

print("\nDone. Saved fact_consistency_scores_65.json and person_consistency_summary_65.json")
print("\nPerson-level summary (lowest consistency first):")
for name, summary in sorted_summary.items():
    print(f"  {name}: avg_consistency={summary['avg_fact_consistency']}, unique_facts={summary['total_unique_facts']}, low_consistency={summary['low_consistency_facts']}")