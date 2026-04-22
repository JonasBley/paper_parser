import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import requests
import json
import sqlite3
import re

# --- Configuration ---
NOW = datetime.now(timezone.utc)
SEVEN_DAYS_AGO = NOW - timedelta(days=7)
DATE_FILTER = SEVEN_DAYS_AGO.strftime("%Y-%m-%d")

CONTACT_EMAIL = "your_email@example.com"

CROSSREF_JOURNALS = {
    "APS PRPER": "2469-9896",
    "American Journal of Physics": "1943-2909",
    "EPJ Quantum Technology": "2196-0763",
    "CBE—Life Sciences Education": "1931-7913"
}

SYSTEM_PROMPT_JSON = """You are an expert academic screener. Evaluate the following abstract against these seven distinct criteria around STEM education:
1. Cognitive Frameworks: Empirical STEM education focusing on cognitive models (e.g., Fidelity of Gestalt, Functional Fidelity) or Cognitive Load Theory.
2. Multimedia & Representations: Research grounded in cognitive theories of multimedia learning or the implementation of multiple representations in STEM education.
3. Visual Attention: Studies utilizing eye-tracking methodologies to assess learning, gaze patterns, or visual attention in STEM education.
4. Quantum/Modern Curriculum: Curriculum innovation in modern physics, quantum mechanics, or quantum optics and quantum computing education at the secondary/tertiary level.
5. Workforce: Quantum workforce development and competences.
6. Emerging Tech: Application or evaluation of Artificial Intelligence (AI), Generative AI, or Augmented/Virtual Reality (AR/VR) in STEM education.

Analyze the text and output a valid JSON object where the keys are the criteria names and the values are boolean (true/false) indicating a match.
Example: {"Cognitive Frameworks": false, "Multimedia & Representations": true, "Visual Attention": true, "Quantum/Modern Curriculum": false, "Workforce": false, "Epistemology": false, "Emerging Tech": false}
Output ONLY the JSON object. Do not include markdown formatting, preambles, or explanations."""


# --- Database & LLM Logic ---

def generate_markdown(papers, file_prefix):
    """Generates a markdown file for a specific subset of papers."""
    if not papers:
        return

    date_str = NOW.strftime("%Y-%m-%d")
    filename = f"{file_prefix}_{date_str}.md"

    content = f"# Literature Digest ({date_str})\n\nFound {len(papers)} papers in this category.\n\n"
    for idx, p in enumerate(papers, 1):
        content += f"## {idx}. {p['title']}\n"
        content += f"**Source:** {p['source']} | **Date:** {p['date']}\n"
        content += f"**Tags:** {', '.join(p['tags'])}\n\n"
        content += f"**Authors:** {p['authors']}\n\n"
        content += f"**Abstract:** {p['abstract']}\n\n"
        content += f"[Read Paper]({p['link']})\n\n"
        content += "---\n\n"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Generated {filename}")


def save_to_database(papers):
    if not papers:
        print("No new relevant papers to save.")
        return

    conn = sqlite3.connect('literature_archive.db')
    cursor = conn.cursor()

    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS papers
                   (
                       id
                       TEXT
                       PRIMARY
                       KEY,
                       source
                       TEXT,
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
        tag_string = ", ".join(p['tags'])
        cursor.execute('''
                       INSERT
                       OR IGNORE INTO papers (id, source, title, authors, abstract, url, date_published, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ''', (p['link'], p['source'], p['title'], p['authors'], p['abstract'], p['link'], p['date'],
                             tag_string))

    conn.commit()
    conn.close()
    print(f"Successfully committed {len(papers)} papers to literature_archive.db")


def extract_categories_with_llm(text_to_evaluate):
    if not text_to_evaluate or len(text_to_evaluate) < 50:
        return []

    url = "http://localhost:1234/v1/chat/completions"
    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_JSON},
            {"role": "user", "content": text_to_evaluate}
        ],
        "temperature": 0.0,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()

        raw_output = response.json()['choices'][0]['message']['content'].strip()

        if raw_output.startswith('```json'):
            raw_output = raw_output[7:]
        if raw_output.startswith('```'):
            raw_output = raw_output[3:]
        if raw_output.endswith('```'):
            raw_output = raw_output[:-3]

        data = json.loads(raw_output.strip())
        return [key for key, value in data.items() if value is True]

    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        print(f"LLM Error: {e}")
        return []


# --- Data Ingestion Logic ---

def clean_html(raw_html):
    if not raw_html:
        return ""
    clean_text = re.sub(re.compile('<.*?>'), '', raw_html)
    return " ".join(clean_text.split())


def fetch_arxiv():
    print("Fetching arXiv (physics.ed-ph, quant-ph)...")
    papers = []
    query = "cat:physics.ed-ph OR cat:quant-ph"
    url = f"[http://export.arxiv.org/api/query?search_query=](http://export.arxiv.org/api/query?search_query=){urllib.parse.quote(query)}&sortBy=submittedDate&sortOrder=descending&max_results=200"

    try:
        with urllib.request.urlopen(url) as response:
            root = ET.fromstring(response.read())
            ns = {'atom': '[http://www.w3.org/2005/Atom](http://www.w3.org/2005/Atom)'}

            for entry in root.findall('atom:entry', ns):
                pub_date_str = entry.find('atom:published', ns).text
                pub_date = datetime.strptime(pub_date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

                if pub_date < SEVEN_DAYS_AGO:
                    continue

                title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
                abstract = clean_html(entry.find('atom:summary', ns).text)

                evaluation_text = f"Title: {title}\nAbstract: {abstract}"
                tags = extract_categories_with_llm(evaluation_text)

                # Explicitly categorize non-educational papers instead of discarding them
                if not tags:
                    tags = ["Technical / Pure Physics"]

                authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
                papers.append({
                    'source': 'arXiv',
                    'title': title,
                    'authors': ", ".join(authors),
                    'link': entry.find("atom:link[@rel='alternate']", ns).attrib['href'],
                    'abstract': abstract,
                    'date': pub_date.strftime("%Y-%m-%d"),
                    'tags': tags
                })
    except Exception as e:
        print(f"arXiv Fetch Error: {e}")
    return papers


def fetch_crossref_api():
    print("Fetching publisher APIs via Crossref...")
    papers = []
    headers = {
        "User-Agent": f"LiteratureScraper/1.0 (mailto:{CONTACT_EMAIL})"
    }

    for source_name, issn in CROSSREF_JOURNALS.items():
        url = f"[https://api.crossref.org/journals/](https://api.crossref.org/journals/){issn}/works?filter=from-pub-date:{DATE_FILTER}&rows=50"

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            items = data.get('message', {}).get('items', [])

            for item in items:
                title_list = item.get('title', [])
                title = title_list[0] if title_list else "Unknown Title"
                link = item.get('URL', '')

                author_list = item.get('author', [])
                authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in author_list]
                author_string = ", ".join(authors) if authors else "Unknown Authors"

                abstract_raw = item.get('abstract', '')
                abstract = clean_html(abstract_raw)

                evaluation_text = f"Title: {title}\nAbstract: {abstract}"
                tags = extract_categories_with_llm(evaluation_text)

                # Retain non-educational papers from publishers as well
                if not tags:
                    tags = ["Technical / Pure Physics"]

                papers.append({
                    'source': source_name,
                    'title': title,
                    'authors': author_string,
                    'link': link,
                    'abstract': abstract or "Abstract not deposited with Crossref.",
                    'date': NOW.strftime("%Y-%m-%d"),
                    'tags': tags
                })

        except requests.exceptions.RequestException as e:
            print(f"Crossref API Error for {source_name}: {e}")

    return papers


# --- Orchestration ---

if __name__ == "__main__":
    print(f"Starting pipeline execution for period since {DATE_FILTER}...")

    arxiv_results = fetch_arxiv()
    crossref_results = fetch_crossref_api()
    all_papers = arxiv_results + crossref_results

    # 1. Save all ingested literature to the unified database
    save_to_database(all_papers)

    # 2. Bifurcate the data structurally
    educational_papers = [p for p in all_papers if "Technical / Pure Physics" not in p['tags']]
    technical_papers = [p for p in all_papers if "Technical / Pure Physics" in p['tags']]

    # 3. Generate two independent markdown digests
    generate_markdown(educational_papers, "digest_education")
    generate_markdown(technical_papers, "digest_technical")

    print("Pipeline execution complete.")