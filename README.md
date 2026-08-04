# Academic Literature Review Agent

Give it a research topic. It decomposes the topic into sub-questions, runs parallel
research agents against ArXiv and the web, extracts structured claims from each paper
(full text where available, abstract as a graceful fallback), cross-references findings
to flag genuine contradictions between papers, and synthesizes a structured literature
review: background, methodology comparison, key findings, open questions, and a
deterministic citation list.

## Architecture

```
Research topic
      |
      v
Orchestrator -- decomposes into 3-5 sub-questions
      |
      v
Research agents (x N, concurrent, semaphore-capped)
   each: ArXiv search + web search -> dedupe (exact key + embedding
   similarity) -> fetch PDF (fallback to abstract) -> LLM claim extraction
      |
      v
Contradiction detector -- per sub-question, flags genuine disagreements
      |
      v
Synthesis agent -- writes background / methodology / findings / open
   questions sections; assembles the reference list deterministically
      |
      v
Structured literature review (JSON), polled via the job API

Memory store (SQLite) sits alongside the research agents and synthesis
stage: paper metadata, embeddings, and cached extraction results persist
across jobs so a second query on an overlapping topic reuses prior work
instead of re-fetching and re-parsing the same PDFs.
```

### Why these design choices

- **Decomposition is its own stage.** Separating "what to research" from "how to
  research it" means decomposition quality can be evaluated and iterated on
  independently of search/extraction quality.
- **Research agents run concurrently but under a semaphore**, not unbounded fan-out.
  `MAX_CONCURRENT_RESEARCH_AGENTS` caps how many sub-questions are in flight at once,
  so 20 sub-questions don't turn into 20 simultaneous ArXiv/PDF/LLM calls.
- **A single shared rate limiter gates all ArXiv calls** across every concurrent agent,
  respecting ArXiv's documented etiquette (no more than one request per ~3 seconds)
  regardless of how much pipeline concurrency is configured.
- **Dedup is two-layered**: exact `paper_key` matching catches the common case (the
  same paper found by both ArXiv and web search), and embedding cosine-similarity
  catches near-duplicates (a web mirror or blog repost with a reworded title).
- **PDF extraction degrades, never crashes.** A paywalled, scanned, or oversized PDF
  falls back to abstract-only extraction with `extraction_failed` / `failure_reason`
  recorded on the finding, so a handful of bad PDFs never take down a whole job.
- **The web search provider is pluggable** (`WebSearchProvider` ABC). Ships with
  DuckDuckGo (no API key required) and Brave Search (drop in `BRAVE_API_KEY`); adding
  Google Scholar via SerpAPI is a ~40-line class following the same interface.
- **The LLM is the only source of narrative text; the reference list is not.**
  `SynthesisAgent._collect_references` builds the citation list directly from what
  was actually fetched and extracted, so citations can never drift from reality even
  if the model hallucinates in prose.
- **Contradiction detection is conservative by prompt design**: papers that differ in
  scope, dataset, or setting are explicitly *not* contradictions -- only claims that
  cannot both be true on the same question are flagged.

## Project layout

```
app/
  main.py                    FastAPI app: submit + poll review jobs
  container.py                Composition root -- wires every service together
  config.py                    Environment-driven settings (pydantic-settings)
  core/
    orchestrator.py            Topic -> sub-questions
    research_agent.py          Search + fetch + extract, per sub-question
    contradiction_detector.py  Cross-references findings per sub-question
    synthesis.py                Writes the final structured review
    pipeline.py                 Wires the above into one job run
    prompts.py                   All LLM system prompts, versioned with the code
  services/
    llm_client.py                Anthropic wrapper: retries + robust JSON parsing
    arxiv_client.py             ArXiv Atom API client, rate-limited
    web_search_client.py       Pluggable web search (DuckDuckGo / Brave)
    pdf_fetcher.py                PDF download + text extraction, fails soft
    embeddings.py                 Local sentence-transformer embeddings for dedup
  db/
    memory_store.py             SQLite: paper cache + extraction cache across jobs
    job_store.py                  In-process job tracking with TTL eviction
  utils/
    logging_config.py             Structured JSON logging
    resilience.py                  Retry/backoff + rate limiter primitives
    rate_limit_middleware.py     Per-IP API rate limiting
tests/                            Unit tests (parsing, JSON robustness, retries,
                                   contradiction filtering) -- no network required
smoke_test.py                     End-to-end pipeline run with all externals mocked
```

## Running it

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY

docker compose up --build
```

Or locally without Docker:

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app --reload
```

### API

```bash
# Submit a review (returns immediately with a job id)
curl -X POST http://localhost:8000/reviews \
  -H "Content-Type: application/json" \
  -d '{"topic": "Retrieval-augmented generation for long-context question answering"}'
# -> {"job_id": "a1b2c3d4e5f6", "status": "pending", "created_at": "..."}

# Poll for status / result
curl http://localhost:8000/reviews/a1b2c3d4e5f6
```

`status` moves through `pending -> decomposing -> researching ->
detecting_contradictions -> synthesizing -> complete` (or `failed`, with `error` set).
Once `complete`, `result` contains the full `LiteratureReview` object.

## Testing

```bash
pytest                 # unit tests: XML parsing, JSON-robustness, retry logic,
                        # contradiction-detector filtering -- all network-free
python smoke_test.py   # full pipeline run, LLM/search/PDF layers mocked
```

## Scaling beyond a single process

This ships as a single FastAPI process with an in-memory job store and background
tasks, which is the right amount of infrastructure for one team running literature
reviews on demand. If this needs to scale out:

- Swap `JobStore` (in `app/db/job_store.py`) for a Redis- or Postgres-backed
  implementation behind the same interface -- nothing else changes.
- Move job execution from `BackgroundTasks` to a real queue (Celery/RQ/Arq) so jobs
  survive a process restart and multiple API replicas can share one worker pool.
- `MemoryStore` (SQLite) can be swapped for Postgres + a vector extension (pgvector)
  if paper volume grows past what a single SQLite file comfortably handles.

## Operational notes

- **Cost/latency**: each paper costs one claim-extraction LLM call; a 5-sub-question
  review pulling 8 papers each can mean ~40 extraction calls plus 1 decomposition
  call, up to 5 contradiction-detection calls, and 1 synthesis call. Tune
  `MAX_PAPERS_PER_SUB_QUESTION` and `MAX_SUB_QUESTIONS` to control this directly.
  The cache in `MemoryStore` means a paper is only ever extracted once, even across
  unrelated future jobs.
- **ArXiv etiquette**: `ARXIV_MIN_REQUEST_INTERVAL_SECONDS` (default 3.1s) is a
  process-wide floor on request spacing, not a per-agent one -- confirmed by the
  smoke test and safe to leave as-is unless ArXiv's own guidance changes.
- **Failure isolation**: a single paper's extraction failure, PDF fetch failure, or
  a single sub-question's contradiction-detection failure never fails the whole job
  -- it's recorded (`extraction_failed`, `agent_results[].errors`) and the pipeline
  continues with what it has.
