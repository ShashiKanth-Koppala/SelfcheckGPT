import json
import numpy as np
from sentence_transformers import SentenceTransformer

with open("extracted_facts.json", "r", encoding="utf-8") as f:
    extracted_facts = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

SAME_CLAIM_THRESHOLD = 0.65       # facts this similar = the same claim (as before)
CONTRADICTION_LOW = 0.45          # facts in this similarity band are "same topic, different value"
CONTRADICTION_HIGH = 0.64
MAX_SAMPLE_OVERLAP_FOR_CONTRADICTION = 0.3  # if two claims share too many samples, they're not a contradiction


def clean_fact_list(sample_facts):
    if isinstance(sample_facts, dict) and "_raw_unparsed" in sample_facts:
        return []
    if not isinstance(sample_facts, list):
        return []
    return [f for f in sample_facts if isinstance(f, str) and f.strip()]


def cluster_facts(all_facts, embeddings, threshold):
    n = len(all_facts)
    similarity_matrix = embeddings @ embeddings.T
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
            if similarity_matrix[i][j] >= threshold:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)
    return clusters


def score_person(name, samples):
    all_facts = []
    for sample_idx, sample_facts in enumerate(samples):
        for fact in clean_fact_list(sample_facts):
            all_facts.append({"text": fact, "sample_idx": sample_idx})

    if not all_facts:
        return None

    texts = [f["text"] for f in all_facts]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    clusters = cluster_facts(all_facts, embeddings, SAME_CLAIM_THRESHOLD)
    total_samples = len(samples)

    cluster_info = []
    for cluster in clusters:
        sample_indices = sorted(set(all_facts[idx]["sample_idx"] for idx in cluster))
        centroid = embeddings[cluster].mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        cluster_info.append({
            "text": all_facts[cluster[0]]["text"],
            "samples": set(sample_indices),
            "support_count": len(sample_indices),
            "centroid": centroid
        })

    # Split into singleton (unconfirmed, mentioned once) vs multi-support (confirmed by 2+ samples) facts
    singleton_facts = [c for c in cluster_info if c["support_count"] == 1]
    multi_support_facts = [c for c in cluster_info if c["support_count"] >= 2]

    # Consensus rate: how strongly the confirmed facts are agreed upon
    if multi_support_facts:
        consensus_rate = np.mean([c["support_count"] / total_samples for c in multi_support_facts])
    else:
        consensus_rate = None  # no fact was ever repeated - can't assess consensus

    # Contradiction detection: look for pairs of multi-support facts that are
    # topically similar (same "slot") but phrased differently AND appear in
    # largely non-overlapping samples - suggesting the model gave different
    # answers to the same underlying question across samples.
    contradiction_pairs = []
    for i in range(len(multi_support_facts)):
        for j in range(i + 1, len(multi_support_facts)):
            a, b = multi_support_facts[i], multi_support_facts[j]
            sim = float(np.dot(a["centroid"], b["centroid"]))
            if CONTRADICTION_LOW <= sim < CONTRADICTION_HIGH:
                overlap = len(a["samples"] & b["samples"]) / min(len(a["samples"]), len(b["samples"]))
                if overlap <= MAX_SAMPLE_OVERLAP_FOR_CONTRADICTION:
                    contradiction_pairs.append({
                        "fact_a": a["text"],
                        "fact_b": b["text"],
                        "similarity": round(sim, 3)
                    })

    contradiction_rate = len(contradiction_pairs) / len(multi_support_facts) if multi_support_facts else None

    return {
        "consensus_rate": round(float(consensus_rate), 3) if consensus_rate is not None else None,
        "contradiction_rate": round(float(contradiction_rate), 3) if contradiction_rate is not None else None,
        "multi_support_fact_count": len(multi_support_facts),
        "singleton_fact_count": len(singleton_facts),
        "contradiction_pairs": contradiction_pairs
    }


all_results = {}
for name, samples in extracted_facts.items():
    print(f"Scoring {name}...")
    all_results[name] = score_person(name, samples)

with open("consensus_contradiction_scores.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

# Print a sorted summary: highest contradiction rate first (most likely to contain hallucinations)
print("\nSorted by contradiction rate (highest first):")
sortable = [(name, r) for name, r in all_results.items() if r and r["contradiction_rate"] is not None]
sortable.sort(key=lambda x: -x[1]["contradiction_rate"])
for name, r in sortable:
    print(f"  {name}: consensus={r['consensus_rate']}, contradiction={r['contradiction_rate']}, "
          f"multi_support={r['multi_support_fact_count']}, singletons={r['singleton_fact_count']}")