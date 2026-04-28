import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import requests
import json
import sqlite3
import re
import time
from sentence_transformers import SentenceTransformer, util
import torch

# --- Configuration ---
NOW = datetime.now(timezone.utc)

# 5 Year Backlog (approx 1825 days)
START_DATE = NOW - timedelta(days=1825)
DATE_FILTER = START_DATE.strftime("%Y-%m-%d")

CONTACT_EMAIL = "jonas.bley@uni-leipzig.de"

CROSSREF_JOURNALS = {
    "APS PRPER": "2469-9896",
    "American Journal of Physics": "1943-2909",
    "EPJ Quantum Technology": "2196-0763",
    "European Journal of Physics": "1361-6404",
    "CBE—Life Sciences Education": "1931-7913",
    "International Journal of STEM Education": "2196-7822",
    "Journal of Research in Science Teaching": "1098-2736",
    "Science Education": "1098-237X",
    "International Journal of Science Education": "1464-5289",
    "Journal of Educational Psychology": "0022-0663",
    "Learning and Instruction": "0959-4752",
    "Cognition and Instruction": "1532-690X",
    "Computers & Education": "0360-1315",
    "IEEE Transactions on Education": "0018-9359",
    "Journal of Engineering Education": "1069-4730"
}

SYSTEM_PROMPT_JSON = """You are an expert academic screener. Evaluate the abstract against the following criteria.

CRITICAL GATING CRITERION:
1. Educational Focus: Is this paper primarily focused on education, teaching, learning, student understanding, curriculum development, or pedagogy? (If the paper is purely technical/scientific physics research with no focus on education, this MUST be false).

SPECIFIC SUB-CRITERIA (Only evaluate as true if 'Educational Focus' is also true):
2. Cognitive Frameworks: Empirical STEM education focusing on cognitive models (e.g., Fidelity of Gestalt, Functional Fidelity), spatial reasoning, or Cognitive Load Theory.
3. Multimedia & Representations: Research grounded in cognitive theories of multimedia learning or the implementation of multiple representations in STEM education.
4. STEM educational research methodology: Studies including methods measuring gaze patterns or visual attention with eye-tracking, measurements of spatial reasoning ability, cognitive load measurements.
5. Quantum/Modern Curriculum: Curriculum innovation in modern physics, quantum mechanics, or quantum optics and quantum computing education at the secondary/tertiary level.
6. Workforce: Quantum workforce development and competences.
7. Emerging Tech: Application or evaluation of Artificial Intelligence (AI), Generative AI, or Augmented/Virtual Reality (AR/VR) in physics/STEM education.
8. Climate Change and Sustainability: Research on Climate Change and Sustainability in STEM educaitonal settings.

Output a valid JSON object with the following exact structure:
{
  "reasoning": "Write one sentence explaining if this paper focuses on teaching/learning/students or if it is purely technical physics.",
  "Educational Focus": false,
  "Cognitive Frameworks": false,
  "Multimedia & Representations": false,
  "STEM educational research methodology": false,
  "Quantum/Modern Curriculum": false,
  "Workforce": false,
  "Emerging Tech": false,
  "Climate Change and Sustainability": false
}

Output ONLY the JSON object. Do not include markdown formatting like ```json or explanations outside the JSON."""

# --- Semantic Ranking Setup ---
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

ANCHOR_TEXT = """This research investigates educational frameworks and cognitive processes in advanced STEM education, with a primary focus on quantum physics and emerging quantum technologies. It utilizes novel teaching techniques like augmented/virtual reality, (generative) artificial intelligence, or interactive environments. Central to this work is the empirical analysis of learners' mental models—specifically utilizing the dual-dimension construct of 'Fidelity of Gestalt' and 'Functional Fidelity'—to understand conceptions of phenomena such as quantum entanglement and quantum processes, linear light polarization, and climate change and sustainability. The literature encompasses curriculum innovation, including the integration of two-state qubit systems and reduced Dirac notation at the secondary level, and extends to workforce competence modeling for the quantum industry."""
ANCHOR_VECTOR = embedder.encode(ANCHOR_TEXT, convert_to_tensor=True)


def calculate_relevance(abstract):
    if not abstract:
        return 0.0
    abstract_vector = embedder.encode(abstract, convert_to_tensor=True)
    score = util.cos_sim(ANCHOR_VECTOR, abstract_vector).item()
    return round(score, 3)


# --- Database & LLM Logic ---

def generate_markdown(papers, file_prefix):
    if not papers:
        return

    papers.sort(key=lambda x: x.get('relevance_score', 0.0), reverse=True)

    date_str = NOW.strftime("%Y-%m-%d")
    filename = f"{file_prefix}_{date_str}.md"

    content = f"# Literature Digest ({date_str})\n\nFound {len(papers)} papers in this category.\n\n"
    for idx, p in enumerate(papers, 1):
        content += f"## {idx}. {p['title']}\n"
        content += f"**Source:** {p['source']} | **Date:** {p['date']} | **Relevance Score:** {p.get('relevance_score', 0.0)}\n"
        content += f"**Tags:** {', '.join(p['tags'])}\n\n"
        # --- ADDED REASONING TO MARKDOWN ---
        content += f"**LLM Reasoning:** {p.get('reasoning', 'Filtered out by semantic threshold.')}\n\n"
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

    # --- ADDED REASONING COLUMN TO SCHEMA ---
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            source TEXT,
            title TEXT,
            authors TEXT,
            abstract TEXT,
            url TEXT,
            date_published TEXT,
            tags TEXT,
            relevance_score REAL,
            reasoning TEXT
        )
    ''')

    for p in papers:
        tag_string = ", ".join(p['tags'])
        # --- ADDED REASONING VALUE TO INSERT COMMAND ---
        cursor.execute('''
            INSERT OR IGNORE INTO papers (id, source, title, authors, abstract, url, date_published, tags, relevance_score, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (p['link'], p['source'], p['title'], p['authors'], p['abstract'], p['link'], p['date'],
              tag_string, p.get('relevance_score', 0.0), p.get('reasoning', '')))

    conn.commit()
    conn.close()
    print(f"Successfully committed {len(papers)} papers to literature_archive.db")


def extract_categories_with_llm(text_to_evaluate):
    if not text_to_evaluate or len(text_to_evaluate) < 50:
        return []

    text_to_evaluate = text_to_evaluate[:4000]
    url = "http://localhost:1234/v1/chat/completions"
    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_JSON},
            {"role": "user", "content": text_to_evaluate}
        ],
        "temperature": 0.0,
        "stream": False,

        # --- RECOMMENDED ADDITIONS FOR GEMMA / JSON EXTRACTION ---

        "max_tokens": 500,
        # Safety net. You only need a short JSON object. If the model glitches and starts
        # writing an essay, this cuts it off after 500 tokens, saving your GPU from hanging.

        "response_format": {"type": "json_object"},
        # Feature available in LM Studio/Ollama. This forces the model's output layer to ONLY
        # generate valid JSON strings, practically eliminating parsing errors.

        "top_p": 1.0
        # Standard practice when temperature is 0.0 to ensure the probability distribution
        # isn't artificially clipped before the greedy selection.
    }

    raw_output = ""
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        raw_output = response.json()['choices'][0]['message']['content'].strip()

        match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
        if not match:
            print(f"LLM Error: No JSON structure found. Model output: {raw_output}")
            return []

        # ... (previous API call logic remains the same)
        json_string = match.group(0)

        # Parse the sanitized string
        data = json.loads(json_string)

        # Extract the True boolean values
        matched_tags = [key for key, value in data.items() if value is True and isinstance(value, bool)]

        # --- NEW: Extract the reasoning string ---
        reasoning = data.get("reasoning", "No reasoning provided.")

        # Return both as a tuple
        return matched_tags, reasoning

    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        error_preview = raw_output[:100].replace('\n', ' ') if raw_output else "Empty Response"
        print(f"LLM Parsing Error ({type(e).__name__}): {e} | Model output preview: '{error_preview}'")
        return [], "LLM Parsing Error."


# --- Data Ingestion Logic ---

def clean_html(raw_html):
    if not raw_html:
        return ""
    clean_text = re.sub(re.compile('<.*?>'), '', raw_html)
    return " ".join(clean_text.split())


def fetch_arxiv():
    print("Fetching arXiv with pagination...")
    papers = []
    query = "cat:physics.ed-ph OR cat:quant-ph OR cat:physics.gen-ph"
    protocol = "http" + "://"
    domain = "export.arxiv.org/api/query"

    start = 0
    max_results = 1000  # arXiv maximum per request

    while True:
        print(f" -> Fetching arXiv batch: {start} to {start + max_results}")
        url = f"{protocol}{domain}?search_query={urllib.parse.quote(query)}&sortBy=submittedDate&sortOrder=descending&start={start}&max_results={max_results}"

        try:
            with urllib.request.urlopen(url) as response:
                root = ET.fromstring(response.read())
                ns = {'atom': 'http' + '://' + 'www.w3.org/2005/Atom'}
                entries = root.findall('atom:entry', ns)

                if not entries:
                    break  # Exhausted all results

                oldest_reached = False
                for entry in entries:
                    pub_date_str = entry.find('atom:published', ns).text
                    pub_date = datetime.strptime(pub_date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

                    # Because arXiv sorts descending, if we hit an old paper, the rest are also old
                    if pub_date < START_DATE:
                        oldest_reached = True
                        continue

                    title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
                    abstract = clean_html(entry.find('atom:summary', ns).text)

                    # --- NEW LOGIC: Check official arXiv categories ---
                    categories = [c.attrib['term'] for c in entry.findall('atom:category', ns)]
                    is_explicit_education = 'physics.ed-ph' in categories

                    # 1. Score the paper semantically FIRST
                    score = calculate_relevance(abstract)

                    # 2. Pre-filter logic: If score is very low AND it's not officially an education paper, skip LLM
                    if score < 0.15 and not is_explicit_education:
                        tags = ["Technical / Pure Physics"]
                    else:
                        evaluation_text = f"Title: {title}\nAbstract: {abstract}"
                        tags, reasoning = extract_categories_with_llm(evaluation_text)

                        if is_explicit_education and "Educational Focus" not in tags:
                            tags.append("Educational Focus")

                        if "Educational Focus" not in tags:
                            tags = ["Technical / Pure Physics"]

                    authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
                    author_string = ", ".join(authors) if authors else "Unknown Authors"

                    papers.append({
                        'source': 'arXiv',
                        'title': title,
                        'authors': author_string,
                        'link': entry.find("atom:link[@rel='alternate']", ns).attrib['href'],
                        'abstract': abstract,
                        'date': pub_date_str,
                        'tags': tags,
                        'relevance_score': score,
                        'reasoning': reasoning
                    })

                if oldest_reached:
                    break  # Safely exit the pagination loop

                start += max_results
                time.sleep(3)  # arXiv API compliance rate limit

        except Exception as e:
            print(f"arXiv Fetch Error: {e}")
            break

    return papers


def fetch_crossref_api():
    print("Fetching publisher APIs via Crossref with cursor pagination...")
    papers = []
    headers = {"User-Agent": f"LiteratureScraper/1.0 (mailto:{CONTACT_EMAIL})"}

    for source_name, issn in CROSSREF_JOURNALS.items():
        print(f" -> Fetching {source_name}...")
        protocol = "https" + "://"
        domain = f"api.crossref.org/journals/{issn}/works"

        cursor = "*"  # Crossref's native pagination token
        rows = 1000

        while cursor:
            # We must quote the cursor because it often contains special URL characters
            url = f"{protocol}{domain}?filter=from-pub-date:{DATE_FILTER}&rows={rows}&cursor={urllib.parse.quote(cursor)}"

            try:
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

                items = data.get('message', {}).get('items', [])
                if not items:
                    break  # Journal exhausted

                for item in items:
                    title_list = item.get('title', [])
                    title = title_list[0] if title_list else "Unknown Title"
                    link = item.get('URL', '')

                    author_list = item.get('author', [])
                    authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in author_list]
                    author_string = ", ".join(authors) if authors else "Unknown Authors"

                    abstract_raw = item.get('abstract', '')
                    abstract = clean_html(abstract_raw)

                    issued_parts = item.get('issued', {}).get('date-parts', [[None]])[0]
                    if issued_parts[0] is not None:
                        year = issued_parts[0]
                        month = issued_parts[1] if len(issued_parts) > 1 else 1
                        day = issued_parts[2] if len(issued_parts) > 2 else 1
                        pub_date_str = f"{year}-{month:02d}-{day:02d}"
                    else:
                        pub_date_str = "Unknown Date"

                    # 1. Score the paper semantically FIRST
                    score = calculate_relevance(abstract)

                    # 2. Pre-filter logic to save LLM compute
                    # Example logic for your loops:
                    if score < 0.15:
                        tags = ["Technical / Pure Physics"]
                        reasoning = "Bypassed LLM (Semantic score below 0.15 threshold)."
                    else:
                        evaluation_text = f"Title: {title}\nAbstract: {abstract}"
                        tags, reasoning = extract_categories_with_llm(evaluation_text)

                        if "Educational Focus" not in tags:
                            tags = ["Technical / Pure Physics"]


                    papers.append({
                        'source': source_name,
                        'title': title,
                        'authors': author_string,
                        'link': link,
                        'abstract': abstract,
                        'date': pub_date_str,
                        'tags': tags,
                        'relevance_score': score,
                        'reasoning': reasoning
                    })

                # Retrieve the next cursor for pagination. If it matches the current one, stop.
                next_cursor = data.get('message', {}).get('next-cursor')
                if not next_cursor or next_cursor == cursor:
                    break
                cursor = next_cursor

            except requests.exceptions.RequestException as e:
                print(f"Crossref API Error for {source_name}: {e}")
                break

    return papers


# --- Orchestration ---

if __name__ == "__main__":
    print(f"Starting extended pipeline execution for period since {DATE_FILTER}...")

    arxiv_results = fetch_arxiv()
    crossref_results = fetch_crossref_api()
    all_papers = arxiv_results + crossref_results

    save_to_database(all_papers)

    educational_papers = [p for p in all_papers if "Educational Focus" in p['tags']]
    technical_papers = [p for p in all_papers if "Technical / Pure Physics" in p['tags']]

    generate_markdown(educational_papers, "digest_education")
    generate_markdown(technical_papers, "digest_technical")

    print("Pipeline execution complete.")