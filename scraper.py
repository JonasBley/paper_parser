import requests
import json
# Conceptual SQLite Implementation
import sqlite3


def save_to_database(papers):
    conn = sqlite3.connect('literature_archive.db')
    cursor = conn.cursor()

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS papers
                   (
                       id
                       TEXT
                       PRIMARY
                       KEY,
                       title
                       TEXT,
                       authors
                       TEXT,
                       abstract
                       TEXT,
                       url
                       TEXT,
                       date_published
                       TEXT,
                       tags
                       TEXT
                   )
                   ''')

    for p in papers:
        # Join the list of tags into a comma-separated string for storage
        tag_string = ", ".join(p['tags'])
        cursor.execute('''
                       INSERT
                       OR IGNORE INTO papers (id, title, authors, abstract, url, date_published, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?)
                       ''', (p['link'], p['title'], p['authors'], p['abstract'], p['link'], p['date'], tag_string))

    conn.commit()
    conn.close()
    

# Define the ideal cognitive framework as the system prompt
SYSTEM_PROMPT_JSON = """You are an expert academic screener. Evaluate the following abstract against these seven distinct criteria:
1. Cognitive Frameworks: Empirical STEM education focusing on cognitive models (e.g., Fidelity of Gestalt, Functional Fidelity) or Cognitive Load Theory.
2. Multimedia & Representations: Research grounded in cognitive theories of multimedia learning or the implementation of multiple representations in STEM education.
3. Visual Attention: Studies utilizing eye-tracking methodologies to assess learning, gaze patterns, or visual attention in STEM.
4. Quantum/Modern Curriculum: Curriculum innovation in modern physics, quantum mechanics, or quantum optics education at the secondary/tertiary level.
5. Workforce: Quantum workforce development and competences.
6. Epistemology: Epistemological perspectives on abstract mathematics (e.g., Galois theory).
7. Emerging Tech: Application or evaluation of Artificial Intelligence (AI), Generative AI, or Augmented/Virtual Reality (AR/VR) in physics/STEM education.

Analyze the text and output a valid JSON object where the keys are the criteria names and the values are boolean (true/false) indicating a match.
Example: {"Cognitive Frameworks": false, "Multimedia & Representations": true, "Visual Attention": true, "Quantum/Modern Curriculum": false, "Workforce": false, "Epistemology": false, "Emerging Tech": false}
Output ONLY the JSON object. Do not include markdown formatting, preambles, or explanations."""


def extract_categories_with_llm(abstract):
    """
    Passes the abstract to the local LLM and extracts matching categories.
    Returns a list of matched category strings, or an empty list if none match or an error occurs.
    """
    if not abstract or len(abstract) < 50:
        return []

    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "gemma4:e4b",  # Or llama3, depending on your deployment
        "prompt": abstract,
        "system": SYSTEM_PROMPT_JSON,
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        raw_output = response.json().get('response', '').strip()

        # Sanitize output: Strip markdown code blocks if the model generates them
        if raw_output.startswith('```json'):
            raw_output = raw_output[7:]
        if raw_output.startswith('```'):
            raw_output = raw_output[3:]
        if raw_output.endswith('```'):
            raw_output = raw_output[:-3]

        data = json.loads(raw_output.strip())

        # Isolate and return only the criteria marked as True
        matched_tags = [key for key, value in data.items() if value is True]
        return matched_tags

    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"LLM Processing Error: {e}")
        return []

# In your fetch_arxiv() and fetch_prper_rss() functions, replace:
# if is_relevant(title) or is_relevant(summary):
# With:
# if evaluate_abstract_with_llm(summary):