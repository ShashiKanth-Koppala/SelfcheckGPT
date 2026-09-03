import json
import time
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:latest"
# Load the generated biographies
with open("generated_biographies.json", "r", encoding="utf-8") as f:
    generated_bios = json.load(f)

EXTRACTION_PROMPT_TEMPLATE = """Extract every distinct factual claim from the biography below as a JSON list of short, atomic statements.

Rules:
- Each statement must contain exactly ONE fact (a single date, place, institution, achievement, relationship, or award).
- Do not combine multiple facts into one statement (e.g. "Born in 1878 in Vienna" must become two separate statements: "Born in 1878" and "Born in Vienna").
- Do not add any information that is not explicitly present in the text.
- Do not include opinions, adjectives, or narrative commentary (e.g. skip "groundbreaking" or "notable").
- Respond with ONLY a valid JSON list of strings. No preamble, no explanation, no markdown formatting.

Biography:
\"\"\"
{biography}
\"\"\"

JSON list of atomic facts:"""

def extract_facts(biography_text):
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(biography=biography_text)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0  # deterministic extraction, no creativity needed here
        }
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    raw_output = response.json()["response"].strip()

    # Try to parse as JSON directly
    try:
        facts = json.loads(raw_output)
        if isinstance(facts, list):
            return facts
    except json.JSONDecodeError:
        pass

    # Fallback: try to find a JSON array within the text (in case the model
    # added stray text despite instructions)
    start = raw_output.find("[")
    end = raw_output.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            facts = json.loads(raw_output[start:end+1])
            if isinstance(facts, list):
                return facts
        except json.JSONDecodeError:
            pass

    # If all parsing fails, return the raw text wrapped so you can inspect it later
    print("  Warning: could not parse JSON, storing raw output for manual review")
    return {"_raw_unparsed": raw_output}

extracted_facts = {}

for name, bios in generated_bios.items():
    print(f"Extracting facts for {name}...")
    extracted_facts[name] = []
    for i, bio in enumerate(bios):
        if bio is None:
            extracted_facts[name].append([])
            continue
        try:
            facts = extract_facts(bio)
            extracted_facts[name].append(facts)
        except Exception as e:
            print(f"  Failed on sample {i+1} for {name}: {e}")
            extracted_facts[name].append([])
        time.sleep(0.3)

output_filename = "extracted_facts.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(extracted_facts, f, ensure_ascii=False, indent=4)

print(f"Done. Saved to {output_filename}")