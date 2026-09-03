import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from itertools import combinations

with open("extracted_facts.json", "r", encoding="utf-8") as f:
    extracted_facts = json.load(f)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# NLI model: given (premise, hypothesis), predicts entailment / neutral / contradiction
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"  # lighter than roberta-large-mnli, runs fine on CPU/Mac
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME)
nli_model.eval()

# This model's label order (check config.id2label to confirm for your specific checkpoint)
# cross-encoder/nli-deberta-v3-base uses: 0=contradiction, 1=entailment, 2=neutral
LABELS = ["contradiction", "entailment", "neutral"]

SAME_CLAIM_THRESHOLD = 0.65
CANDIDATE_PAIR_THRESHOLD = 0.35  # loose topical-relatedness gate before bothering with NLI at all


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


def nli_predict(premise, hypothesis):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    label_idx = int(torch.argmax(probs))
    return LABELS[label_idx], float(probs[label_idx])


def score_person(name, samples):
    all_facts = []
    for sample_idx, sample_facts in enumerate(samples):
        for fact in clean_fact_list(sample_facts):
            all_facts.append({"text": fact, "sample_idx": sample_idx})

    if not all_facts:
        return None

    texts = [f["text"] for f in all_facts]
    embeddings = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

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

    multi_support_facts = [c for c in cluster_info if c["support_count"] >= 2]
    singleton_facts = [c for c in cluster_info if c["support_count"] == 1]

    if multi_support_facts:
        consensus_rate = np.mean([c["support_count"] / total_samples for c in multi_support_facts])
    else:
        consensus_rate = None

    # Candidate pairs: loosely topically related (share SOME similarity, not near-duplicate)
    # AND don't co-occur much across samples (so it's plausible they're competing claims)
    candidate_pairs = []
    for a, b in combinations(multi_support_facts, 2):
        sim = float(np.dot(a["centroid"], b["centroid"]))
        if sim < CANDIDATE_PAIR_THRESHOLD:
            continue  # too unrelated to even bother checking
        overlap = len(a["samples"] & b["samples"]) / min(len(a["samples"]), len(b["samples"]))
        if overlap > 0.3:
            continue  # co-occur too often to be competing claims
        candidate_pairs.append((a, b, sim))

    contradictions = []
    for a, b, sim in candidate_pairs:
        label_ab, conf_ab = nli_predict(a["text"], b["text"])
        label_ba, conf_ba = nli_predict(b["text"], a["text"])
        # Treat as a real contradiction only if NLI flags it in at least one direction
        # with reasonable confidence - contradiction is often symmetric but not always
        # detected symmetrically by the model, so checking both directions catches more.
        if label_ab == "contradiction" or label_ba == "contradiction":
            contradictions.append({
                "fact_a": a["text"],
                "fact_b": b["text"],
                "embedding_similarity": round(sim, 3),
                "nli_a_to_b": {"label": label_ab, "confidence": round(conf_ab, 3)},
                "nli_b_to_a": {"label": label_ba, "confidence": round(conf_ba, 3)}
            })

    contradiction_rate = len(contradictions) / len(multi_support_facts) if multi_support_facts else None

    return {
        "consensus_rate": round(float(consensus_rate), 3) if consensus_rate is not None else None,
        "contradiction_rate": round(float(contradiction_rate), 3) if contradiction_rate is not None else None,
        "multi_support_fact_count": len(multi_support_facts),
        "singleton_fact_count": len(singleton_facts),
        "candidate_pairs_checked": len(candidate_pairs),
        "contradictions": contradictions
    }


all_results = {}
for name, samples in extracted_facts.items():
    print(f"Scoring {name}...")
    all_results[name] = score_person(name, samples)

with open("nli_consensus_contradiction_scores.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print("\nSorted by contradiction rate (highest first):")
sortable = [(n, r) for n, r in all_results.items() if r and r["contradiction_rate"] is not None]
sortable.sort(key=lambda x: -x[1]["contradiction_rate"])
for name, r in sortable:
    print(f"  {name}: consensus={r['consensus_rate']}, contradiction={r['contradiction_rate']}, "
          f"checked={r['candidate_pairs_checked']}")