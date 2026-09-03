import json
import re
import time
import requests

with open("celebrities_data.json", "r", encoding="utf-8") as f:
    wiki_data = json.load(f)

with open("extracted_facts.json", "r", encoding="utf-8") as f:
    extracted_facts = json.load(f)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:latest"

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "of", "to", "for", "and", "or", "was",
    "is", "with", "from", "her", "his", "their", "as", "by", "she", "he",
    "born", "earned", "received", "awarded"  # common but low-signal in this dataset
}


def clean_fact_list(sample_facts):
    if isinstance(sample_facts, dict) and "_raw_unparsed" in sample_facts:
        return []
    if not isinstance(sample_facts, list):
        return []
    return [f for f in sample_facts if isinstance(f, str) and f.strip()]


def get_keywords(text):
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return set(w for w in words if w not in STOPWORDS and len(w) > 2)


def find_relevant_paragraphs(fact_text, paragraphs, top_k=2):
    fact_keywords = get_keywords(fact_text)
    if not fact_keywords:
        return paragraphs[:top_k]  # fallback: just take the first couple

    scored = []
    for para in paragraphs:
        para_keywords = get_keywords(para)
        overlap = len(fact_keywords & para_keywords)
        if overlap > 0:
            scored.append((overlap, para))

    scored.sort(key=lambda x: -x[0])
    top_paragraphs = [p for _, p in scored[:top_k]]

    # If nothing matched at all, fall back to the intro paragraph(s) rather than nothing
    if not top_paragraphs:
        top_paragraphs = paragraphs[:top_k]

    return top_paragraphs


VERIFY_PROMPT_TEMPLATE = """You are checking a factual claim against a reference text. Answer using ONLY the information in the reference text below - do not use any outside knowledge.

Reference text:
\"\"\"
{reference}
\"\"\"

Claim: "{claim}"

Does the reference text support this claim? Respond with exactly one word: TRUE (the reference confirms this claim), FALSE (the reference contradicts this claim), or UNKNOWN (the reference does not mention this or doesn't provide enough information to judge).

Answer:"""


def verify_fact(claim, reference_text):
    prompt = VERIFY_PROMPT_TEMPLATE.format(reference=reference_text, claim=claim)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    raw = response.json()["response"].strip().upper()

    if "TRUE" in raw:
        return "TRUE"
    elif "FALSE" in raw:
        return "FALSE"
    else:
        return "UNKNOWN"


def get_multi_support_facts_from_raw(sample_fact_lists, min_support=2):
    """
    Re-derive which facts are 'multi-support' using simple exact-ish grouping
    isn't reliable across paraphrases, so instead we just verify EVERY unique
    fact string here and let you cross-reference with your consensus scoring
    output afterward using the fact text as the join key.
    """
    seen = {}
    for sample_facts in sample_fact_lists:
        for fact in clean_fact_list(sample_facts):
            seen[fact] = seen.get(fact, 0) + 1
    return list(seen.keys())  # verify all unique fact strings; cheap enough at this scale


verification_results = {}

for name, samples in extracted_facts.items():
    print(f"Verifying facts for {name}...")
    paragraphs = wiki_data.get(name, {}).get("paragraphs", [])
    if not paragraphs:
        print(f"  No Wikipedia paragraphs found for {name}, skipping")
        verification_results[name] = []
        continue

    unique_facts = get_multi_support_facts_from_raw(samples)
    person_results = []

    for fact in unique_facts:
        relevant_paras = find_relevant_paragraphs(fact, paragraphs, top_k=2)
        reference_text = "\n\n".join(relevant_paras)
        try:
            verdict = verify_fact(fact, reference_text)
        except Exception as e:
            print(f"  Failed to verify '{fact}': {e}")
            verdict = "ERROR"

        person_results.append({
            "fact": fact,
            "verdict": verdict,
            "matched_paragraphs": relevant_paras
        })
        time.sleep(0.2)

    verification_results[name] = person_results

with open("fact_verification_results.json", "w", encoding="utf-8") as f:
    json.dump(verification_results, f, ensure_ascii=False, indent=2)

print("\nDone. Saved fact_verification_results.json")

# Quick summary
print("\nVerification summary:")
for name, results in verification_results.items():
    if not results:
        continue
    true_count = sum(1 for r in results if r["verdict"] == "TRUE")
    false_count = sum(1 for r in results if r["verdict"] == "FALSE")
    unknown_count = sum(1 for r in results if r["verdict"] == "UNKNOWN")
    print(f"  {name}: TRUE={true_count}, FALSE={false_count}, UNKNOWN={unknown_count}, total={len(results)}")