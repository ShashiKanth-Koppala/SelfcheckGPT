import json
import re
import datetime
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_community.llms import Ollama
from datasets import Dataset

with open("celebrities_data.json", "r", encoding="utf-8") as f:
    wiki_data = json.load(f)

with open("generated_biographies.json", "r", encoding="utf-8") as f:
    generated_bios = json.load(f)

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "of", "to", "for", "and", "or", "was",
    "is", "with", "from", "her", "his", "their", "as", "by", "she", "he",
    "born", "earned", "received", "awarded"
}

# Reduced subset: your two key contrast cases plus a spread of the rest
SUBSET_PEOPLE = [
    "Chien-Jen Chen",           # highest FALSE-rate, lowest consensus - the clean hit
    "Luis Walter Alvarez",      # high FALSE-rate but clean consensus - the blind spot
    "Ada Yonath",                # highest consensus, still meaningfully wrong - second blind spot
    "Har Gobind Khorana",        # real contradiction detected
    "Barbara McClintock",        # cleanest, most reliable case
    "Srinivasa Ramanujan",       # lowest FALSE-rate
    "Vera Rubin",
    "Rachel Carson",
]
SAMPLES_PER_PERSON = 2


def get_keywords(text):
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return set(w for w in words if w not in STOPWORDS and len(w) > 2)


def find_relevant_paragraphs(bio_text, paragraphs, top_k=3):
    bio_keywords = get_keywords(bio_text)
    scored = []
    for para in paragraphs:
        overlap = len(bio_keywords & get_keywords(para))
        if overlap > 0:
            scored.append((overlap, para))
    scored.sort(key=lambda x: -x[0])
    top = [p for _, p in scored[:top_k]]
    return top if top else paragraphs[:top_k]


# Point RAGAS at your local Ollama model - same model used everywhere else in this project
ollama_llm = Ollama(model="llama3", base_url="http://localhost:11434", timeout=60)
ragas_llm = LangchainLLMWrapper(ollama_llm)

results_per_person = {}

print(f"Running RAGAS faithfulness on {len(SUBSET_PEOPLE)} people, "
      f"{SAMPLES_PER_PERSON} samples each ({len(SUBSET_PEOPLE) * SAMPLES_PER_PERSON} total evaluations)\n")

for name in SUBSET_PEOPLE:
    bios = generated_bios.get(name, [])[:SAMPLES_PER_PERSON]
    paragraphs = wiki_data.get(name, {}).get("paragraphs", [])

    if not paragraphs or not bios:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Skipping {name} - missing data")
        continue

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting {name} ({len(bios)} samples)...")

    rows = []
    for bio in bios:
        if bio is None:
            continue
        contexts = find_relevant_paragraphs(bio, paragraphs, top_k=3)
        rows.append({
            "question": f"Tell me about {name}",
            "answer": bio,
            "contexts": contexts,
        })

    if not rows:
        continue

    dataset = Dataset.from_list(rows)

    try:
        scored = evaluate(dataset, metrics=[faithfulness], llm=ragas_llm)
        scores_df = scored.to_pandas()
        per_sample_scores = scores_df["faithfulness"].tolist()
        avg_score = sum(per_sample_scores) / len(per_sample_scores)
        results_per_person[name] = {
            "per_sample_faithfulness": per_sample_scores,
            "avg_faithfulness": avg_score
        }
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Done {name}: avg={avg_score:.3f}")
    except Exception as e:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Failed on {name}: {e}")
        results_per_person[name] = {"error": str(e)}

with open("ragas_faithfulness_scores_subset.json", "w", encoding="utf-8") as f:
    json.dump(results_per_person, f, ensure_ascii=False, indent=2)

print("\nDone. Saved ragas_faithfulness_scores_subset.json")
print("\nSummary (lowest faithfulness first):")
valid = [(n, r["avg_faithfulness"]) for n, r in results_per_person.items() if "avg_faithfulness" in r]
valid.sort(key=lambda x: x[1])
for name, score in valid:
    print(f"  {name}: {score:.3f}")