import json
import time
import requests 

celebrities = [
    "Subrahmanyan Chandrasekhar",
    "Chien-Shiung Wu",
    "Har Gobind Khorana",
    "Katherine Johnson",
    "Norman Borlaug",
    "Vera Rubin",
    "Ada Yonath",
    "Luis Walter Alvarez",
    "Rosalind Franklin",
    "Srinivasa Ramanujan",
    "Emmy Noether",
    "Grace Hopper",
    "Rachel Carson",
    "Wangari Maathai",
    "Abdus Salam",
    "Dorothy Hodgkin",
    "Nikolai Vavilov",
    "Lise Meitner",
    "Chien-Jen Chen",
]

URL = "https://en.wikipedia.org/w/api.php"

# Wikipedia requires a unique User-Agent. Replace with your details if desired.
# Use a standard browser header string to prevent Wikipedia from blocking your script
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
}

# Dictionary to hold the final scraped data
scraped_data = {}

# 3. Loop through each name and fetch data from the API
for name in celebrities:
    # Set query parameters to get clean text extracts of the introduction
    params = {
        "action": "query",
        "format": "json",
        "titles": name,
        "prop": "extracts",
        "explaintext": True,  # Get plain text instead of raw HTML
        "redirects": 1,  # Automatically follow Wikipedia redirects
    }

    try:
        response = requests.get(url=URL, params=params, headers=HEADERS)
        response.raise_for_status()  # Raise an error for bad HTTP statuses
        data = response.json()

        # Parse out the pages from the nested JSON response
        pages = data.get("query", {}).get("pages", {})

        for page_id, page_info in pages.items():
            # Check if the page actually exists (-1 means page not found)
            if page_id != "-1":
                extract_text = page_info.get("extract", "")
                paragraphs = [p.strip() for p in extract_text.split("\n\n") if p.strip()]
                scraped_data[name] = {
                    "title": page_info.get("title"),
                    "summary": page_info.get("extract"),
                }
            else:
                print(f" Wikipedia page not found for '{name}'")
                scraped_data[name] = {"title": name, "summary": "Page not found."}

    except Exception as e:
        print(f" Failed to fetch data for {name}. Error: {e}")
        scraped_data[name] = {"title": name, "summary": f"Error: {e}"}

    # Respectful pause to prevent hitting rate limits
    time.sleep(10)

# 4. Save the compiled dictionary to a JSON file
output_filename = "celebrities_data.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(scraped_data, f, ensure_ascii=False, indent=4)
