import urllib.request
import urllib.parse
import urllib.error
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

# TARGET WINDOW CONTROL (Monthly Chunks):
# 0 = The last 30 days
# 1 = 1 to 3 months ago
# 2 = 3 to 6 months ago
# ...
MONTHS_BACK = 0
CHUNK_SIZE_DAYS = 9

# Calculate exact date boundaries for the specific chunk
END_DATE = NOW - timedelta(days=CHUNK_SIZE_DAYS * MONTHS_BACK)
START_DATE = END_DATE - timedelta(days=CHUNK_SIZE_DAYS)

# Formatting for specific APIs
CROSSREF_FROM = START_DATE.strftime("%Y-%m-%d")
CROSSREF_UNTIL = END_DATE.strftime("%Y-%m-%d")

# arXiv requires exact YYYYMMDDHHMM timestamp formatting
ARXIV_FROM = START_DATE.strftime("%Y%m%d%H%M")
ARXIV_UNTIL = END_DATE.strftime("%Y%m%d%H%M")

CONTACT_EMAIL = "jonas.bley@uni-leipzig.de"

CROSSREF_JOURNALS = {
    # --- Existing Educational Journals ---
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
    "Journal of Engineering Education": "1069-4730",

    # --- NEW: High-Impact & Quantum Tech Journals ---
    "Nature": "1476-4687",
    "Nature Physics": "1745-2481",
    "Nature Communications": "2041-1723",
    "npj Quantum Information": "2056-6387",
    "Science": "1095-9203",
    "Science Advances": "2375-2548",
    "PRX Quantum": "2691-3399",  # Highly recommended addition from the APS family
    "Physical Review Letters": "1079-7114"  # The standard for major physics breakthroughs
}

SYSTEM_PROMPT_JSON = SYSTEM_PROMPT_JSON = """You are an expert academic screener. Evaluate the abstract against the following criteria.

CRITICAL GATING CRITERION:
1. Educational Focus: Is this paper primarily focused on education, teaching, learning, student understanding, curriculum development, or pedagogy?

UNIVERSAL CRITERION (Applies to all papers):
2. Review Paper: Is this a systematic literature review, meta-analysis, or broad survey of a field?

EDUCATIONAL SUB-CRITERIA (Evaluate carefully if 'Educational Focus' is true):
3. Cognitive Frameworks: Empirical STEM education focusing on cognitive models, spatial reasoning, or Cognitive Load Theory.
4. Multimedia & Representations: Research grounded in cognitive theories of multimedia learning or multiple representations.
5. STEM educational research methodology: Studies measuring gaze patterns, spatial reasoning ability, or cognitive load.
6. Quantum/Modern Curriculum: Curriculum innovation in modern physics or quantum mechanics education.
7. Workforce: Quantum workforce development and competences.
8. Emerging Tech: Application of AI, Generative AI, or AR/VR in STEM education.
9. Climate Change and Sustainability: Research on climate change/sustainability in STEM educational settings.

TECHNICAL SUB-CRITERIA (Evaluate carefully if 'Educational Focus' is false):
10. Algorithmic & Theoretical Advances: Theoretical developments in quantum computing, quantum simulation, machine learning, or quantum information/communication theory.
11. Experimental Advances: Real-world physical realization, laboratory experiments, or hardware advances in quantum communication, sensing, computing, or simulation.
12. Quantum Foundations: Research into the fundamental nature of quantum mechanics, entanglement theory, Bell tests, or quantum interpretations.
13. Quantum Materials & Solid State: Research focusing on condensed matter, superconductivity, topological insulators, or physical material properties.
14. Architecture & Error Correction: Systems engineering for quantum computers, including logical qubits, surface codes, cryogenics, and control electronics.
15. Quantum Metrology & Sensing: Application of quantum phenomena for high-precision measurement, such as NV centers, atom interferometry, or quantum radar.
16. Interdisciplinary Applications: Application of quantum models or hardware to external domains like quantum biology, computational chemistry, or financial modeling.
17. Policy, Security & Ethics: Research on post-quantum cryptography, export controls, intellectual property, or the societal impact of quantum technology.
18. Replicability & Meta-Science: Studies focusing on replicating previous high-profile claims, publishing null results, or analyzing publication trends within the discipline.

Output a valid JSON object with the following exact structure:
{
  "reasoning": "Write one sentence explaining your categorization logic. DO NOT use quotation marks, backslashes, or LaTeX symbols in this sentence.",
  "Educational Focus": false,
  "Review Paper": false,
  "Cognitive Frameworks": false,
  "Multimedia & Representations": false,
  "STEM educational research methodology": false,
  "Quantum/Modern Curriculum": false,
  "Workforce": false,
  "Emerging Tech": false,
  "Climate Change and Sustainability": false,
  "Algorithmic & Theoretical Advances": false,
  "Experimental Advances": false,
  "Quantum Foundations": false,
  "Quantum Materials & Solid State": false,
  "Architecture & Error Correction": false,
  "Quantum Metrology & Sensing": false,
  "Interdisciplinary Applications": false,
  "Policy, Security & Ethics": false,
  "Replicability & Meta-Science": false
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

    # Convert datetime to a safe, purely alphanumeric string
    date_str = END_DATE.strftime("%Y-%m-%d")
    filename = f"{file_prefix}_{date_str}.md"

    content = f"# Literature Digest ({date_str})\n\nFound {len(papers)} papers in this category.\n\n"
    for idx, p in enumerate(papers, 1):
        content += f"## {idx}. {p['title']}\n"
        content += f"**Source:** {p['source']} | **Date:** {p['date']} | **Relevance Score:** {p.get('relevance_score', 0.0)}\n"

        # Format tags to stand out if they are empty
        display_tags = ', '.join(p['tags']) if p['tags'] else "Uncategorized Technical Paper"
        content += f"**Tags:** {display_tags}\n\n"

        content += f"**LLM Reasoning:** {p.get('reasoning', 'No reasoning provided.')}\n\n"
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
                       TEXT,
                       relevance_score
                       REAL,
                       reasoning
                       TEXT
                   )
                   ''')

    for p in papers:
        tag_string = ", ".join(p['tags'])
        cursor.execute('''
                       INSERT
                       OR IGNORE INTO papers (id, source, title, authors, abstract, url, date_published, tags, relevance_score, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ''', (p['link'], p['source'], p['title'], p['authors'], p['abstract'], p['link'], p['date'],
                             tag_string, p.get('relevance_score', 0.0), p.get('reasoning', '')))

    conn.commit()
    conn.close()
    print(f"Successfully committed {len(papers)} papers to literature_archive.db")


def extract_categories_with_llm(text_to_evaluate):
    if not text_to_evaluate or len(text_to_evaluate) < 50:
        return [], "Abstract too short for evaluation."

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
        "max_tokens": 1500,
        "top_p": 1.0
    }

    raw_output = ""
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        raw_output = response.json()['choices'][0]['message']['content'].strip()

        match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
        if not match:
            print(f"LLM Error: No JSON structure found. Model output: {raw_output}")
            return [], "LLM returned empty or invalid formatting."

        json_string = match.group(0)

        # Escape any stray backslashes from LaTeX to prevent JSONDecodeError
        json_string = json_string.replace('\\', '\\\\')

        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"Failed to parse sanitized JSON: {e}")
            return [], "LLM produced unparseable JSON despite sanitization."

        matched_tags = [key for key, value in data.items() if value is True and isinstance(value, bool)]
        reasoning = data.get("reasoning", "No reasoning provided.")

        return matched_tags, reasoning

    except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError) as e:
        error_preview = raw_output[:100].replace('\n', ' ') if raw_output else "Empty Response"
        print(f"LLM Parsing Error ({type(e).__name__}): {e} | Model output preview: '{error_preview}'")
        return [], "LLM Parsing or Connection Error."


# --- Data Ingestion Logic ---

def clean_html(raw_html):
    if not raw_html:
        return ""
    clean_text = re.sub(re.compile('<.*?>'), '', raw_html)
    return " ".join(clean_text.split())


def fetch_arxiv():
    print(f"Fetching arXiv from {ARXIV_FROM} to {ARXIV_UNTIL}...")
    papers = []

    # --- Server-Side Date Query ---
    base_categories = "cat:physics.ed-ph OR cat:quant-ph OR cat:physics.gen-ph"
    query = f"({base_categories}) AND submittedDate:[{ARXIV_FROM} TO {ARXIV_UNTIL}]"

    protocol = "http" + "://"
    domain = "export.arxiv.org/api/query"

    start = 0
    max_results = 200  # Keep at 200 for stability

    while True:
        print(f" -> Fetching arXiv batch: {start} to {start + max_results}")
        url = f"{protocol}{domain}?search_query={urllib.parse.quote(query)}&sortBy=submittedDate&sortOrder=descending&start={start}&max_results={max_results}"

        max_retries = 5
        retry_count = 0
        success = False
        exhausted = False  # --- NEW: Flag to track if we've run out of papers ---

        while retry_count < max_retries and not success:
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": f"LiteratureScraper/1.0 (mailto:{CONTACT_EMAIL})"}
                )
                with urllib.request.urlopen(req) as response:
                    root = ET.fromstring(response.read())
                    ns = {'atom': 'http' + '://' + 'www.w3.org/2005/Atom'}
                    entries = root.findall('atom:entry', ns)

                    if not entries:
                        exhausted = True  # Mark the chunk as completely finished
                        success = True  # Treat the API call as successful so it doesn't trigger the error log
                        break  # Break the inner retry loop

                    oldest_reached = False
                    for entry in entries:
                        pub_date_str = entry.find('atom:published', ns).text
                        pub_date = datetime.strptime(pub_date_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

                        if pub_date < START_DATE:
                            oldest_reached = True
                            continue

                        title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
                        abstract = clean_html(entry.find('atom:summary', ns).text)

                        categories = [c.attrib['term'] for c in entry.findall('atom:category', ns)]
                        is_explicit_education = 'physics.ed-ph' in categories

                        authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
                        author_string = ", ".join(authors) if authors else "Unknown Authors"

                        papers.append({
                            'source': 'arXiv',
                            'title': title,
                            'authors': author_string,
                            'link': entry.find("atom:link[@rel='alternate']", ns).attrib['href'],
                            'abstract': abstract,
                            'date': pub_date.strftime("%Y-%m-%d"),
                            'is_explicit_education': is_explicit_education
                        })

                    if oldest_reached:
                        return papers

                    start += max_results
                    time.sleep(4)
                    success = True

            except urllib.error.HTTPError as e:
                if e.code in [429, 500, 502, 503, 504]:
                    retry_count += 1
                    sleep_time = 30 * retry_count
                    print(
                        f"    [!] arXiv Server Error ({e.code}). Pausing for {sleep_time} seconds before retry {retry_count}/{max_retries}...")
                    time.sleep(sleep_time)
                else:
                    print(f"arXiv HTTP Error: {e}")
                    break
            except urllib.error.URLError as e:
                retry_count += 1
                print(
                    f"    [!] Network Error ({e.reason}). Pausing for 30 seconds before retry {retry_count}/{max_retries}...")
                time.sleep(30)
            except Exception as e:
                print(f"arXiv Fetch Error: {e}")
                break

        if exhausted:
            print("Reached the end of the arXiv results for this chunk.")
            break

        if not success:
            print("Failed to fetch arXiv batch after max retries. Moving on to Crossref.")
            break

    return papers


def fetch_crossref_api():
    print("Fetching publisher APIs via Crossref with cursor pagination...")
    raw_papers = []
    headers = {"User-Agent": f"LiteratureScraper/1.0 (mailto:{CONTACT_EMAIL})"}

    for source_name, issn in CROSSREF_JOURNALS.items():
        print(f" -> Fetching {source_name}...")
        protocol = "https" + "://"
        domain = f"api.crossref.org/journals/{issn}/works"

        cursor = "*"
        rows = 1000

        while cursor:
            # --- 'until-pub-date' filter ---
            url = f"{protocol}{domain}?filter=from-pub-date:{CROSSREF_FROM},until-pub-date:{CROSSREF_UNTIL}&rows={rows}&cursor={urllib.parse.quote(cursor)}"

            try:
                response = requests.get(url, headers=headers)
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()

                items = data.get('message', {}).get('items', [])
                if not items:
                    break

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

                    # Only save the raw data, DO NOT call the LLM here
                    raw_papers.append({
                        'source': source_name,
                        'title': title,
                        'authors': author_string,
                        'link': link,
                        'abstract': abstract,
                        'date': pub_date_str,
                        'is_explicit_education': False # Crossref doesn't have arXiv's strict category tags
                    })

                next_cursor = data.get('message', {}).get('next-cursor')
                if not next_cursor or next_cursor == cursor:
                    break
                cursor = next_cursor

            except requests.exceptions.RequestException as e:
                print(f"Crossref API Error for {source_name}: {e}")
                break

    return raw_papers


def process_and_evaluate_papers(raw_papers):
    print(f"\nBeginning semantic and LLM evaluation for {len(raw_papers)} papers...")
    processed_papers = []

    for i, p in enumerate(raw_papers):
        print(f"Evaluating paper {i + 1}/{len(raw_papers)}...")

        # 1. Calculate Semantic Score
        score = calculate_relevance(p['abstract'])

        # 2. Extract Categories via LLM
        evaluation_text = f"Title: {p['title']}\nAbstract: {p['abstract']}"

        if score < 0.15 and not p['is_explicit_education']:
            # Apply base tags for bypassed papers
            tags = ["Technical / Pure Physics", "Bypassed (Low Relevance)"]
            reasoning = "Bypassed LLM (Semantic score below 0.15 threshold and not categorized as education)."
        else:
            tags, reasoning = extract_categories_with_llm(evaluation_text)

            if p['is_explicit_education'] and "Educational Focus" not in tags:
                tags.append("Educational Focus")

            # CORRECTED LOGIC: Append the routing tag rather than overwriting the list.
            # This preserves tags 10-13 from the LLM output.
            if "Educational Focus" not in tags:
                if "Technical / Pure Physics" not in tags:
                    tags.append("Technical / Pure Physics")

            # Fallback for an empty LLM response
            if not tags:
                tags = ["Uncategorized", "Technical / Pure Physics"]

        p['tags'] = tags
        p['relevance_score'] = score
        p['reasoning'] = reasoning
        del p['is_explicit_education']
        processed_papers.append(p)

    return processed_papers

# --- Orchestration ---

if __name__ == "__main__":
    print(f"Starting extended pipeline execution for period from {START_DATE} to {END_DATE}...")

    # Phase 1: Rapid Download
    arxiv_results = fetch_arxiv()
    crossref_results = fetch_crossref_api()
    all_raw_papers = arxiv_results + crossref_results

    # Phase 2: Offline Evaluation
    final_evaluated_papers = process_and_evaluate_papers(all_raw_papers)

    # Phase 3: Save and Generate Markdown
    save_to_database(final_evaluated_papers)

    educational_papers = [p for p in final_evaluated_papers if "Educational Focus" in p['tags']]
    technical_papers = [p for p in final_evaluated_papers if "Technical / Pure Physics" in p['tags']]

    generate_markdown(educational_papers, "digest_education")
    generate_markdown(technical_papers, "digest_technical")

    print("Pipeline execution complete.")