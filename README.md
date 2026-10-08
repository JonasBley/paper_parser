# Literature digest tool, version 2

A resumable literature monitor and historical backfill tool for arXiv and the original 23 journals in Crossref. Classifications use LM Studio locally by default; embeddings also run locally. No service calls occur when importing the module or displaying help.

## Setup

Use Python 3.11 or 3.12 in a virtual environment for broad machine-learning package compatibility. Install the included requirements. The first evaluation downloads the configured embedding model; allow sufficient disk space and network access.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:CONTACT_EMAIL = 'your-contact-email'
```

Start the LM Studio local server and load a chat model before running the script. The default OpenAI-compatible endpoint is `http://127.0.0.1:1234/v1/chat/completions`; the default model ID is `qwen/qwen3.8-27b`, which was listed by the local server during setup. If you load another model, set `LLM_MODEL` to the ID returned by LM Studio's `/v1/models` endpoint. Override the endpoint with `LLM_API_URL` or `--llm-url`.

The original contact email is intentionally no longer embedded in the script. Existing `LLM_API_URL`, `LLM_MODEL`, `LLM_READ_TIMEOUT_SECONDS`, and `ARXIV_CA_BUNDLE` environment settings are supported. LM Studio does not require an API key by default. For a hosted endpoint, set `LLM_API_KEY` or `SAIA_API_KEY`; TLS verification remains enabled. Do not commit API keys.

### Use GWDG Academic Cloud (SAIA)

The script can send classification requests to SAIA's OpenAI-compatible Chat AI API instead of LM Studio. Request a SAIA API key through the [KISSKI LLM Service](https://kisski.gwdg.de/en/leistungen/2-02-llm-service) booking page, using the email address associated with your Academic Cloud account. Keep the key private.

In PowerShell, set the cloud endpoint, a model ID that is currently available to your account, and the key before running the script:

```powershell
$env:LLM_API_URL = 'https://chat-ai.academiccloud.de/v1/chat/completions'
$env:LLM_MODEL = 'qwen3-30b-a3b-instruct-2507'
$secureKey = Read-Host 'SAIA API key' -AsSecureString
$env:SAIA_API_KEY = [System.Net.NetworkCredential]::new('', $secureKey).Password
Remove-Variable secureKey

python literature_digest.py --start 2026-09-01 --end 2026-10-01 --enrich-abstracts
```

The model ID above is one listed in the SAIA information provided to this project; model availability can change. Check the [current SAIA model list](https://docs.hpc.gwdg.de/services/ai-services/chat-ai/models/index.html) and use the exact API model ID. `--llm-url` and `--llm-model` can also be supplied on the command line. The API key is read from `SAIA_API_KEY` (or `LLM_API_KEY`) and sent as a bearer token. These environment settings apply to the current PowerShell session.

SAIA's API documentation describes this API for interactive inference and recommends HPC batch inference for very large asynchronous workloads. Check your account's quota and rate-limit response headers (`x-ratelimit-limit-*`, `x-ratelimit-remaining-*`, and `ratelimit-reset`) before planning throughput. This script evaluates papers sequentially; it does not currently parallelize classification requests.

## Retrieve and evaluate a date window

```powershell
python literature_digest.py --start 2026-09-01 --end 2026-10-01 --enrich-abstracts --sample-size 150
```

Dates use UTC, with an inclusive start and exclusive end: this example covers September. Backfill boundaries must be UTC midnight because Crossref publication filtering uses calendar dates. Without dates, the tool selects the last 12 complete UTC days. `--chunk-days 12 --chunk-index 1` selects the preceding 12-day block. These are day blocks, not calendar months.

## Monitor updates

```powershell
python literature_digest.py --mode monitor --enrich-abstracts
```

Crossref monitoring uses index dates rather than publication dates, catching late metadata and changes. Each source has its own persisted watermark, advanced only after complete retrieval. Subsequent runs overlap by three days by default, and reuse unchanged classifications. arXiv monitoring uses submission dates with overlap; it does not guarantee discovery of revisions to old preprints. To inspect old arXiv revisions, rerun an appropriate backfill. The default end is today's UTC midnight; explicitly specify an end timestamp in monitor mode for an intraday run.

Crossref's inclusive index filter can include records exactly at the end; deduplication makes that overlap safe. Published dates are retained separately, with year/month/day precision. Actual per-source monitor windows appear in the JSON summary; the headline window is the requested/default fallback window.

## Checkpoints and resume

```powershell
python literature_digest.py --start 2026-09-01 --end 2026-10-01 --ingest-only
python literature_digest.py --resume RUN_ID_FROM_LOG --enrich-abstracts
```

Raw records commit during retrieval. Each successful evaluation and each embedding batch commits separately. Resume reuses saved records without fetching again, and retries failed evaluations. `--force-evaluate` also repeats valid cached evaluations. Changed title/abstract, source categories, model, endpoint, prompt, profiles, or screening settings invalidate the relevant evaluation cache.

Use the same classification options when resuming to reuse results. Runs with partial retrieval should be fetched again using the same date window or monitoring mode; `--resume` only resumes evaluation, not unfinished source pagination. Source health includes pending sources when interrupted. Exit code 2 indicates partial retrieval or failed classifications; exit code 130 indicates interruption. `needs_review` and deliberate screening skips are visible in reports but are not execution errors.

## Classification and ranking

- Separate education and technical profile families provide independent scores. Titles are included in ranking.
- No semantic gate is enabled by default. This avoids excluding relevant work based on an unvalidated cosine cutoff.
- Missing or short abstracts can receive provisional title-based labels, but always remain `needs_review`. An optional Crossref DOI lookup attempts abstract enrichment; unavailable abstracts remain explicit. This is metadata enrichment, not publisher full-text scraping.
- arXiv education categories are stored as provenance. Disagreement with the LLM causes review rather than forcing a label.
- Education and technical labels may coexist. Education-focused papers appear in the education digest with all their labels. Non-education papers enter the technical digest only when a technical label is supported. Unrelated classified papers appear under `other`.
- Model responses require the exact schema, strict boolean values, nonempty reasoning, and consistent education labels. Invalid JSON is retried without changing backslash escapes. Failures have no inferred subject classification.
- `--json-mode` enables JSON output mode only when your endpoint supports it. Provider support is not assumed.
- Input truncation forces review. Embedding token truncation is reported separately; cosine values retain full precision in JSON and are rounded only for display.

To enable a gate after validation, provide both thresholds:

```powershell
python literature_digest.py --resume RUN_ID --education-threshold 0.15 --technical-threshold 0.15
```

The numbers above demonstrate syntax only; they are **not validated recommendations**. A paper is skipped only when it is below both thresholds, has sufficient abstract evidence, and lacks an arXiv education category. Skipped papers remain archived and appear in their own digest.

## Validate accuracy with human labels

`--sample-size 150` exports a reproducible stratified CSV. It includes missing-abstract records and arXiv education records rather than sampling only high-ranked papers. Fill `human_relevant` and the `human_…` category columns using `true`/`false` or `1`/`0`; leave unreviewed fields blank. Keep IDs unchanged. Label independently of the model when possible. Inspect borderline-score papers separately using the full JSON export; the sample is not automatically stratified by score.

```powershell
python literature_digest.py --labels digests\label_sample_RUN_ID.csv
```

Use the same model, endpoint, input-length and threshold settings as the evaluations being measured. The benchmark matches the current configuration rather than mixing older model results. It reports per-category precision and recall, unclassified cases, relevant papers skipped by the screening gate, and unresolved relevant cases. Category metrics exclude unresolved cases, so inspect their counts alongside accuracy. A stratified sample does not estimate overall literature prevalence. Compare models/prompts using separate databases or saved benchmark reports; validate on a held-out sample before adopting new thresholds.

## Outputs and storage

The default SQLite archive is `literature_archive_v2.db`. It is deliberately separate from the old `literature_archive.db`, preserving your existing archive. Version 1 databases are not migrated automatically. Rerun the needed date windows into version 2, or implement an explicit migration if you need to preserve old classifications.

Each run writes JSON summaries, a full paper/evaluation export, and Markdown digests for education, technical work, review, failures, skips, and other records. Files include both window dates and a unique run ID. Each digest separates new, updated, and unchanged records and keeps full abstracts in expandable details. Blank categories still produce a report, so a missing file does not masquerade as zero results.

Records have structured authors, normalized DOI/arXiv identifiers, provenance, date precision, observation timestamps, and refreshed metadata. DOI links bridge publisher and preprint records when arXiv supplies a DOI. Unlinked preprints are not merged by fuzzy titles; this avoids false merges but can leave duplicates until a DOI bridge arrives. Publisher abstracts are preferred when available. Report change counts describe source metadata changes, not changes to the model's judgment.

Cache provenance records model names, prompt/profile versions, and evaluation timestamps. For strict reproducibility, use a pinned local embedding model snapshot via `--embedding-model PATH` and a provider model version with stable weights; a reused model name alone cannot detect upstream weight changes.

## Operational options

```powershell
python literature_digest.py --sources crossref --journal 'APS PRPER' --journal 'PRX Quantum'
python literature_digest.py --sources arxiv --start 2026-09-01 --end 2026-09-13
python literature_digest.py --help
python -m unittest discover -s tests -v
```

HTTP sessions are reused; transient failures retry with backoff and respect `Retry-After`. arXiv requests, including retries, are spaced by at least 3.1 seconds. Source failures do not discard other sources. JSON reports state source completeness rather than implying a successful run after a retrieval failure.

Tests use local fixtures and fake services: they do not contact your accounts, download models, or assess real-world classification quality. Dependencies and the live provider still require an environment smoke test. Abstract availability and the fixed source selection limit coverage; this is a personal monitoring tool, not a validated systematic-review search protocol.

## API references

- [Crossref incremental retrieval and pagination](https://www.crossref.org/documentation/retrieve-metadata/rest-api/tips-for-using-the-crossref-rest-api/)
- [Crossref filters](https://www.crossref.org/documentation/retrieve-metadata/rest-api/rest-api-filters/)
- [arXiv API manual](https://github.com/arXiv/arxiv-docs/blob/develop/source/help/api/user-manual.md)
