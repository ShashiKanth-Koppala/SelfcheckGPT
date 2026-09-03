import json
import time
import requests

# Load the names from your JSON so we generate for the same 19 people
with open("celebrities_data.json", "r", encoding="utf-8") as f:
    scraped_data = json.load(f)

names = list(scraped_data.keys())

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:latest"
N_SAMPLES = 5
TEMPERATURE = 0.8

def generate_biography(name, temperature=TEMPERATURE):
    prompt = (
        f"Write a short biography of {name}, covering their background, "
        f"key achievements, and why they're notable. Keep it to one paragraph."
    )
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"]

generated_bios = {}

for name in names:
    print(f"Generating for {name}...")
    samples = []
    for i in range(N_SAMPLES):
        try:
            bio = generate_biography(name)
            samples.append(bio)
        except Exception as e:
            print(f"  Failed sample {i+1} for {name}: {e}")
            samples.append(None)
        time.sleep(0.5)  # small pause between local calls, mostly to avoid overloading Ollama
    generated_bios[name] = samples

output_filename = "generated_biographies.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(generated_bios, f, ensure_ascii=False, indent=4)

print(f"Done. Saved to {output_filename}")