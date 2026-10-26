# Lead Intelligence System

Auto-qualifies inbound leads, scores and prioritizes them, and drafts first-touch
outreach for the ones worth pursuing — so the sales team reviews a ranked queue
instead of 1,200 raw rows.

## How to run

```bash
pip install -r requirements.txt

# Mock mode (default) — no API key needed, runs the full pipeline offline
python main.py --input leads.csv --output output_report.json

# Real LLM mode — set provider + key, then run the same way
export LLM_API_KEY="your-groq-or-openai-key"
# also edit config.yaml: llm.provider -> "groq" (or "openai"), llm.model -> your model
python main.py --input leads.csv --output output_report.json

# Process multiple files, or anchor "today" for a historical/demo dataset
python main.py --input data/leads_training.csv data/leads_testing.csv --as-of-date 2024-01-20
```

Output is written to `output_report.json`: a summary block (totals, qualified %,
top rejection reasons), a per-lead array (score, factor breakdown, decision,
reasoning, priority rank, outreach messages), and a `sample_outreach_messages`
list.

## The rubric

Each lead gets a 0–10 score on five weighted factors — **company size fit**
(25%, tapered around a 50–2,000 employee ICP), **industry fit** (20%, mapped
to strong/moderate/weak lists), **source quality** (20%, inbound demo/referral
beats cold outreach), **engagement recency** (20%, days since last touch), and
**data confidence** (15%, penalized per missing field). The weighted blend
gives a `fit_score` (1–10): **≥7 Qualified**, **5–6.9 Flagged for Human
Review**, **<5 Disqualified** — and any lead missing 3+ fields is force-flagged
for review regardless of score, since a score built on that little data isn't
trustworthy. Qualified leads are then ranked into a priority queue by score.

**Why the LLM doesn't compute the score:** asking an LLM to output a 1–10
number directly is inconsistent — the same lead can score differently run to
run, and the sales team can't audit the logic. So the score comes from
deterministic rules in `main.py` (transparent, auditable, cheap), and the LLM
is used for what it's actually good at: writing the plain-language reasoning
for each decision, and drafting 2 personalized outreach message variants per
qualified lead, referencing that lead's real company/industry/size details.
Leads are batched (default 8 per call, one call handles a whole batch) rather
than called one-by-one.

## Robustness

- **Missing/malformed fields** (blank, `NA`, unparseable size or date) never
  crash the pipeline — each gets a neutral-to-penalized factor score, is
  logged in `missing_fields`, and drives the lead toward "Flagged for Human
  Review" rather than a confident wrong answer.
- **LLM failures** (rate limits, timeouts, bad JSON) retry with exponential
  backoff (`max_retries`, capped backoff in `config.yaml`). If a batch still
  fails after retries, those leads are downgraded to "Flagged for Human
  Review" with a note — the run finishes instead of crashing.
- **Mock mode** (`llm.provider: mock`, or simply no API key set) swaps in a
  deterministic local generator so the whole pipeline — scoring, batching,
  report generation — is runnable and testable without any API access.

## Scale and integration testing

- **100+ leads in one run:** `generate_stress_test_data.py` produces synthetic
  leads (clearly separate from the curated `leads.csv` sample) purely to prove
  the pipeline holds up at volume. `stress_test_report.json` is the result of
  running all 104 combined leads (49 real + 55 synthetic) through the full
  pipeline in one call — zero crashes, all three decisions represented.
- **Real (non-mock) LLM code path:** this dev environment has no internet
  access, so the actual API call has not been exercised against a live
  OpenAI/Groq endpoint. To still validate `_call_api`, retry/backoff, and JSON
  parsing rather than leaving them untested, `mock_openai_server.py` is a
  minimal local server that mimics the OpenAI chat completions shape
  (including a forced first-call 429) — pointing the real client at it
  confirmed the full non-mock path works: request built correctly, backoff
  triggered and waited the configured interval, retry succeeded, response
  parsed and attached per lead. **This is not a substitute for testing
  against a real provider** — only you can do that final check, by exporting
  a real `LLM_API_KEY` and setting `llm.provider` in `config.yaml`.

## Known limitations

- The rubric's industry/source lists are hand-curated for this exercise; a
  real deployment would tune them against actual conversion data.
- `--as-of-date` exists because the sample data is dated 2023–2024; without
  it, recency scoring against the real current date makes old demo leads look
  stale. In production, dates are naturally current and this flag is unneeded.
- The mock LLM's message templates are simple placeholders — real LLM mode
  produces materially more varied, natural copy.
- No persistent storage/dedup between runs; each run is stateless.
- `review_min` (5.0) was calibrated against this specific training set's score
  distribution — worth re-checking against a larger/real lead sample before
  relying on it in production.
