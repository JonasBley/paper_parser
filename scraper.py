import urllib.request
import urllib.parse
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import re
import os

# --- Configuration ---
# Target keywords using regular expressions for flexible matching
TARGET_PATTERNS = [
    r"quantum\s+education",
    r"physics\s+education",
    r"generative\s+ai",
    r"llm\b",
    r"large\s+language\s+model",
    r"stem\s+education"
]

# Temporal bound: strictly 7 days from execution time
NOW = datetime.now(timezone.utc)
SEVEN_DAYS_AGO = NOW - timedelta(days=7)


def is_relevant(text):
    """Evaluates if the text contains any of the target patterns."""
    if not text:
        return False
    text = text.lower()
    return any(re.search(pattern, text) for pattern in TARGET_PATTERNS)


def fetch_arxiv():
    """Queries the arXiv API for recent physics education and quantum papers."""
    print("Fetching data from arXiv...")
    papers = []

    # Query physics.ed-ph and quant-ph categories
    query = "cat:physics.ed-ph OR cat:quant-ph"
    url = f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(query)}&sortBy=submittedDate&sortOrder=descending&max_results=100"

    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
            root = ET.fromstring(data)

            # XML Namespace for arXiv
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            for entry in root.findall('atom:entry', ns):
                published_str = entry.find('atom:published', ns).text
                published_date = datetime.strptime(published_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

                if published_date < SEVEN_DAYS_AGO:
                    continue  # Skip papers older than 7 days

                title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
                summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
                link = entry.find("atom:link[@rel='alternate']", ns).attrib['href']
                authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]

                if is_relevant(title) or is_relevant(summary):
                    papers.append({
                        'source': 'arXiv',
                        'title': title,
                        'authors': ", ".join(authors),
                        'link': link,
                        'abstract': summary,
                        'date': published_date.strftime("%Y-%m-%d")
                    })
    except Exception as e:
        print(f"Error fetching arXiv: {e}")

    return papers


def fetch_prper_rss():
    """Fetches the RSS feed for APS PRPER."""
    print("Fetching data from PRPER RSS...")
    papers = []
    url = "http://feeds.aps.org/rss/recent/prper.xml"

    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
            root = ET.fromstring(data)

            for item in root.findall('.//item'):
                # RSS uses pubDate (e.g., Tue, 14 Apr 2026 04:00:00 EDT)
                # Parsing standard RSS dates can be complex due to timezones;
                # using a simplified heuristic assuming feed contains only recent papers.
                title = item.find('title').text
                link = item.find('link').text
                description = item.find('description').text or ""

                if is_relevant(title) or is_relevant(description):
                    papers.append({
                        'source': 'APS PRPER',
                        'title': title,
                        'authors': 'See link for authors',  # RSS doesn't always split authors cleanly
                        'link': link,
                        'abstract': description.strip()[:500] + "...",  # Truncate long HTML descriptions
                        'date': 'Recent'
                    })
    except Exception as e:
        print(f"Error fetching PRPER: {e}")

    return papers


def generate_markdown(papers):
    """Compiles the extracted data into a Markdown report."""
    date_str = NOW.strftime("%Y-%m-%d")
    filename = f"digest_{date_str}.md"

    if not papers:
        content = f"# Literature Digest ({date_str})\n\nNo new papers matched the criteria this week.\n"
    else:
        content = f"# Literature Digest ({date_str})\n\nFound {len(papers)} relevant papers.\n\n"
        for idx, p in enumerate(papers, 1):
            content += f"## {idx}. {p['title']}\n"
            content += f"**Source:** {p['source']} | **Date:** {p['date']}\n\n"
            content += f"**Authors:** {p['authors']}\n\n"
            content += f"**Abstract:** {p['abstract']}\n\n"
            content += f"[Read Paper]({p['link']})\n\n"
            content += "---\n\n"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"Successfully generated {filename}")


if __name__ == "__main__":
    all_papers = fetch_arxiv() + fetch_prper_rss()
    generate_markdown(all_papers)