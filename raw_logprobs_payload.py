import json
import requests
import math
import time

with open("generated_biographies.json", "r", encoding="utf-8") as f:
    generated_bios = json.load(f)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:latest"

SELF_EVAL_PROMPT_TEMPLATE = """I will show you a short biography and ask whether it is factually accurate. Judge based on whether the specific facts stated (dates, places, achievements, institutions) are correct.

Example 1:
Biography: "Charles Darwin was born in 1809 in Shrewsbury, England. He studied medicine at the University of Edinburgh before turning to natural history, and later developed the theory of evolution by natural selection, publishing On the Origin of Species in 1859."
Question: Is the above biography of Charles Darwin factually accurate?
Answer: True

Example 2:
Biography: "Isaac Newton was born in 1750 in Paris, France. He is best known for inventing the telephone and for his work developing the modern automobile engine in the early 1800s."
Question: Is the above biography of Isaac Newton factually accurate?
Answer: False

Example 3:
Biography: "Marie Curie was born in Warsaw in 1867. She conducted pioneering research on radioactivity and won Nobel Prizes in both Physics and Chemistry. She also briefly served as President of Poland in the 1920s before returning to scientific research."
Question: Is the above biography of Marie Curie factually accurate?
Answer: False

Example 4:
Biography: "Nikola Tesla was born in 1856 in Smiljan, in what is now Croatia. He worked extensively on alternating current (AC) electrical systems and held numerous patents related to electrical engineering, including work on induction motors and wireless power transmission."
Question: Is the above biography of Nikola Tesla factually accurate?
Answer: True

Now judge this one, using the same standard:

Biography: "{biography}"
Question: Is the above biography of {name} factually accurate?
Answer:"""


def get_self_eval_confidence(name, biography):
    prompt = SELF_EVAL_PROMPT_TEMPLATE.format(biography=biography, name=name)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 1},
        "logprobs": True,
        "top_logprobs": 5
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    data = response.json()

    raw_text = data.get("response", "").strip()
    logprobs_data = data.get("logprobs")

    true_probability = None
    if logprobs_data:
        try:
            candidates = logprobs_data[0].get("top_logprobs", [])
            for candidate in candidates:
                token_text = candidate.get("token", "").strip().lower()
                if token_text == "true":
                    true_probability = math.exp(candidate["logprob"])
                    break
            if true_probability is None:
                # "True" didn't appear in top_logprobs at all - meaning the model
                # was so confident in its answer that True fell outside the top 5
                # candidates. Treat this as a very low/near-zero true_probability
                # rather than leaving it as None, since None would break averaging.
                true_probability = 0.0
        except (KeyError, IndexError, TypeError) as e:
            print(f"    Could not parse logprobs for {name}: {e}")

    return {"raw_answer": raw_text, "true_probability": true_probability}


results = {}

for name, bios in generated_bios.items():
    print(f"Scoring {name}...")
    sample_scores = []
    for i, bio in enumerate(bios):
        if bio is None:
            sample_scores.append(None)
            continue
        try:
            result = get_self_eval_confidence(name, bio)
            sample_scores.append(result["true_probability"])
        except Exception as e:
            print(f"  Failed on sample {i+1}: {e}")
            sample_scores.append(None)
        time.sleep(0.3)

    valid_scores = [s for s in sample_scores if s is not None]
    avg_confidence = sum(valid_scores) / len(valid_scores) if valid_scores else None

    results[name] = {
        "per_sample_true_probability": sample_scores,
        "avg_true_probability": avg_confidence
    }
    print(f"  avg_true_probability={avg_confidence:.3f}" if avg_confidence is not None else "  no valid scores")

with open("self_eval_confidence_scores.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nDone. Saved self_eval_confidence_scores.json")
print("\nSorted lowest confidence first (most likely to be flagged as inaccurate):")
sortable = [(n, r["avg_true_probability"]) for n, r in results.items() if r["avg_true_probability"] is not None]
sortable.sort(key=lambda x: x[1])
for name, score in sortable:
    print(f"  {name}: {score:.3f}")