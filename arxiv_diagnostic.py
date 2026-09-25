"""Run on the same computer/environment as the literature pipeline."""
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

END = datetime.now(timezone.utc)
START = END - timedelta(days=24)
QUERY = (
    '(cat:physics.ed-ph OR cat:quant-ph OR cat:physics.gen-ph) '
    f'AND submittedDate:[{START:%Y%m%d%H%M} TO {END:%Y%m%d%H%M}]'
)
BASE = 'https://export.arxiv.org/api/query?'
TESTS = [
    ('Minimal query', BASE + 'search_query=all:electron&max_results=1'),
    ('Current pipeline query', BASE + 'search_query=' + urllib.parse.quote(QUERY)
     + '&sortBy=submittedDate&sortOrder=descending&start=0&max_results=200'),
]
HEADERS = {
    'User-Agent': 'LiteratureScraper/1.0 (mailto:jonas.bley@uni-leipzig.de)',
    'Accept': 'application/atom+xml, application/xml;q=0.9, */*;q=0.8',
}
RESPONSE_HEADERS = (
    'Date', 'Server', 'Content-Type', 'Retry-After', 'Via',
    'X-Cache', 'X-Served-By', 'X-Timer',
)


def show_headers(headers):
    for name in RESPONSE_HEADERS:
        if headers.get(name):
            print(f'{name}: {headers[name]}')


def main():
    print('UTC test time:', END.isoformat())
    for index, (label, url) in enumerate(TESTS):
        if index:
            time.sleep(4)
        print(f'\n{label}\nURL: {url}', flush=True)
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=30) as response:
                print('HTTP:', response.status)
                show_headers(response.headers)
                root = ET.fromstring(response.read())
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                if root.tag != '{http://www.w3.org/2005/Atom}feed':
                    print('Unexpected response: not an Atom feed.')
                    continue
                entries = root.findall('atom:entry', ns)
                print('Atom entries:', len(entries))
                for entry in entries:
                    if '/api/errors' in entry.findtext('atom:id', '', ns):
                        print('API error:', entry.findtext('atom:summary', '', ns))
        except urllib.error.HTTPError as error:
            print('HTTP:', error.code)
            show_headers(error.headers)
            print('Body:', error.read(2000).decode('utf-8', errors='replace') or '(empty)')
            error.close()
        except (urllib.error.URLError, TimeoutError, ET.ParseError) as error:
            print(type(error).__name__ + ':', error)


if __name__ == '__main__':
    main()
