import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import requests
import json
import sqlite3
import re

# --- Configuration ---
# Temporal bound: last 7 days
NOW = datetime.now(timezone.utc)
SEVEN_DAYS_AGO = NOW - timedelta(days=7)
DATE_FILTER = SEVEN_DAYS_AGO.strftime("%Y-%m-%d")

# Crossref Polite Pool Requirement - Replace with your actual email
CONTACT_EMAIL = "your_email@example.com"

# Target Journals by their unique ISSN
CROSSREF_JOURNALS = {
    "APS PRPER": "2469-9896",
    "American Journal of Physics": "1943-2909",
    "EPJ Quantum Technology": "2196-0763",
    "CBE—Life Sciences Education": "1931-7913"  # Added: High quality empirical STEM education
}

SYSTEM_PROMPT_JSON = """You are an expert academic screener. Evaluate the following abstract against these seven distinct criteria:
1. Cognitive Frameworks: Empirical STEM education focusing on cognitive models (e.g., Fidelity of Gestalt, Functional Fidelity) or Cognitive Load Theory.
2. Multimedia & Representations: Research grounded in cognitive theories of multimedia learning or the implementation of multiple representations in STEM education.
3. Visual Attention: Studies utilizing eye-tracking methodologies to assess learning, gaze patterns, or visual attention in STEM.
4. Quantum/Modern Curriculum: Curriculum innovation in modern physics, quantum mechanics, or quantum optics and quantum computing education at the secondary/tertiary level.
5. Workforce: Quantum workforce development and competences.
6. Epistemology: Epistemological perspectives on abstract mathematics (e.g., Galois theory).
7. Emerging Tech: Application or evaluation of Artificial Intelligence (AI), Generative AI, or Augmented/Virtual Reality (AR/VR) in physics/STEM education.

Analyze the text and output a valid JSON object where the keys are the criteria names and the values are boolean (true/false) indicating a match.
Example: {"Cognitive Frameworks": false, "Multimedia & Representations": true, "Visual Attention": true, "Quantum/Modern Curriculum": false, "Workforce": false, "Epistemology": false, "Emerging Tech": false}
Output ONLY the JSON object. Do not include markdown formatting, preambles, or explanations."""


# --- Database & LLM Logic ---

def generate_markdown(papers):
    if not papers:
        return

    date_str = NOW.strftime("%Y-%m-%d")
    filename = f"digest_{date_str}.md"

    content = f"# Literature Digest ({date_str})\n\nFound {len(papers)} relevant papers.\n\n"
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

    # Updated to target LM Studio's default port and endpoint
    url = "http://localhost:1234/v1/chat/completions"

    # Updated to use the OpenAI-compatible message schema
    payload = {
        "model": "local-model",  # LM Studio ignores this and uses whatever you loaded in the UI
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

        # The path to the text response is different in OpenAI's schema
        raw_output = response.json()['choices'][0]['message']['content'].strip()

        # Sanitize markdown artifacts
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
    """Removes HTML and JATS XML tags common in academic metadata."""
    if not raw_html:
        return ""
    # Remove XML/HTML tags
    clean_text = re.sub(re.compile('<.*?>'), '', raw_html)
    # Remove excessive whitespace and newlines
    return " ".join(clean_text.split())


def fetch_arxiv():
    print("Fetching arXiv (physics.ed-ph, quant-ph)...")
    papers = []
    query = "cat:physics.ed-ph OR cat:quant-ph"

    # FIX 1: Cleaned URL string
    url = f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(query)}&sortBy=submittedDate&sortOrder=descending&max_results=200"

    try:
        with urllib.request.urlopen(url) as response:
            root = ET.fromstring(response.read())

            # FIX 2: Cleaned XML namespace string
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            for entry in root.findall('atom:entry', ns):
                pub_date_str = entry.find('atom:published', ns).text
                pub_date = datetime.strptime(pub_date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

                if pub_date < SEVEN_DAYS_AGO:
                    continue

                title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
                abstract = clean_html(entry.find('atom:summary', ns).text)

                # Evaluate Title + Abstract for maximum context
                evaluation_text = f"Title: {title}\nAbstract: {abstract}"
                tags = extract_categories_with_llm(evaluation_text)

                if tags:
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
    """Queries the Crossref REST API for structured metadata of specific journals."""
    papers = []
    headers = {
        "User-Agent": f"LiteratureScraper/1.0 (mailto:{CONTACT_EMAIL})"
    }

    for source_name, issn in CROSSREF_JOURNALS.items():
        print(f"Fetching {source_name} via Crossref...")

        # Querying specific journal ISSN, filtering by publication date, fetching 50 rows
        # url = f"[https://api.crossref.org/journals/](https://api.crossref.org/journals/){issn}/works?filter=from-pub-date:{DATE_FILTER}&rows=50"
        url = f"https://api.crossref.org/journals/{issn}/works?filter=from-pub-date:{DATE_FILTER}&rows=50"

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

            items = data.get('message', {}).get('items', [])

            for item in items:
                # Extract Title
                title_list = item.get('title', [])
                title = title_list[0] if title_list else "Unknown Title"

                # Extract URL (prefer DOI URL, fallback to any available link)
                link = item.get('URL', '')

                # Extract Authors
                author_list = item.get('author', [])
                authors = []
                for a in author_list:
                    given = a.get('given', '')
                    family = a.get('family', '')
                    authors.append(f"{given} {family}".strip())
                author_string = ", ".join(authors) if authors else "Unknown Authors"

                # Extract and clean Abstract (Crossref sometimes encapsulates these in <jats:p> tags)
                abstract_raw = item.get('abstract', '')
                abstract = clean_html(abstract_raw)

                # Fallback: If no abstract is provided by the publisher to Crossref, evaluate the title
                evaluation_text = f"Title: {title}\nAbstract: {abstract}"

                tags = extract_categories_with_llm(evaluation_text)

                if tags:
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
    all_relevant_papers = arxiv_results + crossref_results

    # Save to database AND generate the readable file
    save_to_database(all_relevant_papers)
    generate_markdown(all_relevant_papers)

    print("Pipeline execution complete.")