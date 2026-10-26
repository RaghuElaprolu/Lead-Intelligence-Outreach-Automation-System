#!/usr/bin/env python3
"""
Lead Intelligence System
-------------------------
Ingests a CSV of leads, scores each one against a weighted rubric, calls an LLM
in batches to write the qualitative reasoning and outreach messages, and writes
a report the sales team can act on.

Design decision worth calling out up front: the 1-10 fit_score is computed by
deterministic rule-based logic (see rubric.py), not by asking the LLM to invent
a number. LLMs are inconsistent scorers — ask the same lead twice and you'll get
two different scores. The LLM's job here is judgment and language: explaining
*why* a lead landed where it did, and writing outreach copy that references real
details. That split is what "chain logic (rubric) + LLM (judgment)" means in
practice.

Usage:
    python main.py --input data/leads.csv --output output_report.json
    python main.py --input data/leads_training.csv data/leads_testing.csv --config config.yaml
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lead_qualifier")

REQUIRED_COLUMNS = ["name", "company", "company_size", "industry", "source", "last_interaction_date"]
MISSING_TOKENS = {"", "na", "n/a", "none", "null", "unknown", "unknown source", "nan"}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class Lead:
    row_id: int
    name: Optional[str]
    company: Optional[str]
    company_size: Optional[int]
    industry: Optional[str]
    source: Optional[str]
    last_interaction_date: Optional[str]
    missing_fields: list = field(default_factory=list)

    # filled in during scoring / LLM steps
    factor_scores: dict = field(default_factory=dict)
    fit_score: float = 0.0
    decision: str = ""
    rejection_reason: Optional[str] = None
    reasoning: str = ""
    priority_rank: Optional[int] = None
    outreach_messages: list = field(default_factory=list)

    def display_name(self) -> str:
        if self.name:
            return self.name
        if self.company:
            return f"(unnamed contact at {self.company})"
        return f"Lead #{self.row_id} (no name or company on file)"


# --------------------------------------------------------------------------- #
# Ingestion + cleaning
# --------------------------------------------------------------------------- #

def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s.lower() in MISSING_TOKENS


def load_leads(paths: list[str]) -> list[Lead]:
    """Read one or more CSVs, normalize messy fields, and track what's missing
    per lead rather than dropping incomplete rows. A lead with gaps is still a
    lead — the pipeline should degrade gracefully, not silently discard data."""
    leads: list[Lead] = []
    row_id = 0

    for path in paths:
        p = Path(path)
        if not p.exists():
            log.warning("Input file not found, skipping: %s", path)
            continue

        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing_cols = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing_cols:
                log.warning("%s is missing expected columns %s — those fields will read as missing", path, missing_cols)

            for raw in reader:
                row_id += 1
                missing = []

                name = raw.get("name")
                name = None if _is_missing(name) else name.strip()
                if name is None:
                    missing.append("name")

                company = raw.get("company")
                company = None if _is_missing(company) else company.strip()
                if company is None:
                    missing.append("company")

                size_raw = raw.get("company_size")
                company_size = None
                if _is_missing(size_raw):
                    missing.append("company_size")
                else:
                    try:
                        company_size = int(float(str(size_raw).strip()))
                        if company_size <= 0:
                            company_size = None
                            missing.append("company_size")
                    except ValueError:
                        missing.append("company_size")

                industry = raw.get("industry")
                industry = None if _is_missing(industry) else industry.strip()
                if industry is None:
                    missing.append("industry")

                source = raw.get("source")
                source = None if _is_missing(source) else source.strip()
                if source is None:
                    missing.append("source")

                date_raw = raw.get("last_interaction_date")
                last_interaction_date = None
                if _is_missing(date_raw):
                    missing.append("last_interaction_date")
                else:
                    try:
                        datetime.strptime(date_raw.strip(), "%Y-%m-%d")
                        last_interaction_date = date_raw.strip()
                    except ValueError:
                        missing.append("last_interaction_date")

                leads.append(Lead(
                    row_id=row_id,
                    name=name,
                    company=company,
                    company_size=company_size,
                    industry=industry,
                    source=source,
                    last_interaction_date=last_interaction_date,
                    missing_fields=missing,
                ))

    log.info("Loaded %d leads from %d file(s)", len(leads), len(paths))
    return leads


# --------------------------------------------------------------------------- #
# Rubric scoring (deterministic — no LLM involved)
# --------------------------------------------------------------------------- #

def _score_company_size(size: Optional[int], cfg: dict) -> float:
    if size is None:
        return 3.0  # unknown size — mild penalty, not disqualifying on its own
    lo, hi = cfg["ideal_min"], cfg["ideal_max"]
    floor, ceiling = cfg["hard_floor"], cfg["hard_ceiling"]
    if size < floor or size > ceiling:
        return 0.5
    if lo <= size <= hi:
        return 10.0
    if size < lo:
        # taper down from ideal_min to hard_floor
        span = max(lo - floor, 1)
        return round(1.0 + 9.0 * (size - floor) / span, 2)
    # size > hi: taper down from ideal_max to hard_ceiling
    span = max(ceiling - hi, 1)
    return round(1.0 + 9.0 * (ceiling - size) / span, 2)


def _score_industry(industry: Optional[str], cfg: dict) -> float:
    if industry is None:
        return 4.0
    if industry in cfg["strong"]:
        return 10.0
    if industry in cfg["moderate"]:
        return 6.5
    if industry in cfg["weak"]:
        return 4.0
    return 5.0  # unmapped industry — neutral, not punished for being novel


def _score_source(source: Optional[str], cfg: dict) -> float:
    if source is None:
        return cfg["unknown_score"]
    if source in cfg["high"]:
        return 9.5
    if source in cfg["medium"]:
        return 7.0
    if source in cfg["low"]:
        return 4.5
    return cfg["unknown_score"]


def _score_recency(date_str: Optional[str], cfg: dict, today: date) -> float:
    if date_str is None:
        return cfg["missing_score"]
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    days = max((today - d).days, 0)
    for band in cfg["bands"]:
        if band["max_days"] is None or days <= band["max_days"]:
            return band["score"]
    return cfg["bands"][-1]["score"]


def _score_data_confidence(missing_fields: list, cfg: dict) -> float:
    penalty = len(missing_fields) * cfg["penalty_per_missing_field"]
    return max(cfg["max_score"] - penalty, 0.0)


def score_lead(lead: Lead, rubric_cfg: dict, today: date) -> None:
    w = rubric_cfg["weights"]
    factors = {
        "company_size_fit": _score_company_size(lead.company_size, rubric_cfg["company_size"]),
        "industry_fit": _score_industry(lead.industry, rubric_cfg["industry_fit"]),
        "source_quality": _score_source(lead.source, rubric_cfg["source_quality"]),
        "engagement_recency": _score_recency(lead.last_interaction_date, rubric_cfg["engagement_recency"], today),
        "data_confidence": _score_data_confidence(lead.missing_fields, rubric_cfg["data_confidence"]),
    }
    lead.factor_scores = {k: round(v, 2) for k, v in factors.items()}
    weighted = sum(factors[k] * w[k] for k in w)
    lead.fit_score = round(weighted, 2)


def decide(lead: Lead, thresholds: dict) -> None:
    force_review = len(lead.missing_fields) >= thresholds["force_review_missing_fields"]
    if force_review:
        lead.decision = "Flagged for Human Review"
        lead.rejection_reason = "Insufficient data to score confidently (%d fields missing)" % len(lead.missing_fields)
        return

    if lead.fit_score >= thresholds["qualified_min"]:
        lead.decision = "Qualified"
    elif lead.fit_score >= thresholds["review_min"]:
        lead.decision = "Flagged for Human Review"
        lead.rejection_reason = "Borderline fit score — too close to call automatically"
    else:
        lead.decision = "Disqualified"
        weakest = min(lead.factor_scores, key=lead.factor_scores.get)
        lead.rejection_reason = f"Low fit score, driven mainly by weak {weakest.replace('_', ' ')}"


def assign_priority(leads: list[Lead]) -> None:
    qualified = [l for l in leads if l.decision == "Qualified"]
    qualified.sort(key=lambda l: l.fit_score, reverse=True)
    for i, lead in enumerate(qualified, start=1):
        lead.priority_rank = i


# --------------------------------------------------------------------------- #
# LLM client — batched calls with retry/backoff, provider-agnostic
# --------------------------------------------------------------------------- #

class LLMError(Exception):
    pass


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat completions endpoint.
    Works with OpenAI, Groq, or any other OpenAI-compatible free-tier API by
    swapping base_url/model in config.yaml. Falls back to a local mock
    generator when provider is "mock" or no API key is set, so the pipeline
    is always runnable end-to-end (e.g. offline dev, CI, or a reviewer without
    a key on hand)."""

    def __init__(self, llm_cfg: dict):
        self.cfg = llm_cfg
        self.provider = llm_cfg.get("provider", "mock")
        self.api_key = os.environ.get(llm_cfg.get("api_key_env", "LLM_API_KEY"), "")
        self.mock = self.provider == "mock" or not self.api_key

        if self.mock:
            log.info("LLM provider set to '%s' with no API key found — running in MOCK mode. "
                     "Set provider + %s in your environment for real calls.",
                     self.provider, llm_cfg.get("api_key_env", "LLM_API_KEY"))

        default_base_urls = {
            "openai": "https://api.openai.com/v1",
            "groq": "https://api.groq.com/openai/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
        }
        self.base_url = llm_cfg.get("base_url") or default_base_urls.get(self.provider, "")

    def batch_generate(self, leads: list[Lead]) -> dict[int, dict]:
        """Send one batch of leads to the LLM and return {row_id: {reasoning, messages}}.
        Retries with exponential backoff on failure; returns an empty dict (never raises)
        so a bad batch degrades to human review instead of crashing the run."""
        if self.mock:
            return {l.row_id: self._mock_response(l) for l in leads}

        prompt = self._build_prompt(leads)
        max_retries = self.cfg.get("max_retries", 4)
        base = self.cfg.get("backoff_base_seconds", 1.5)
        cap = self.cfg.get("backoff_max_seconds", 30)

        for attempt in range(1, max_retries + 1):
            try:
                raw = self._call_api(prompt)
                parsed = self._parse_response(raw, leads)
                return parsed
            except LLMError as e:
                wait = min(base * (2 ** (attempt - 1)), cap) + random.uniform(0, 0.5)
                log.warning("LLM batch call failed (attempt %d/%d): %s — retrying in %.1fs",
                            attempt, max_retries, e, wait)
                time.sleep(wait)
            except Exception as e:  # network errors, timeouts, etc.
                wait = min(base * (2 ** (attempt - 1)), cap) + random.uniform(0, 0.5)
                log.warning("Unexpected error calling LLM (attempt %d/%d): %s — retrying in %.1fs",
                            attempt, max_retries, e, wait)
                time.sleep(wait)

        log.error("LLM batch failed after %d attempts — leads in this batch will be flagged for manual review", max_retries)
        return {}

    def _build_prompt(self, leads: list[Lead]) -> str:
        lead_blocks = []
        for l in leads:
            lead_blocks.append({
                "row_id": l.row_id,
                "name": l.name or "unknown",
                "company": l.company or "unknown company",
                "company_size": l.company_size,
                "industry": l.industry or "unknown",
                "source": l.source or "unknown",
                "last_interaction_date": l.last_interaction_date or "unknown",
                "fit_score": l.fit_score,
                "decision": l.decision,
                "factor_scores": l.factor_scores,
            })
        variants = self.cfg.get("message_variants_per_lead", 2)
        instructions = (
            "You are a B2B sales analyst. For each lead below, you are given its rubric "
            "scores and decision already computed — do not change the score or decision. "
            "Your job: (1) write a 1-2 sentence reasoning explaining the decision in plain "
            "language, referencing the specific factor that mattered most; (2) if decision "
            f"is 'Qualified', write {variants} short, distinct first-outreach messages "
            "(2-3 sentences each) that reference specific real details about the lead "
            "(company, industry, or size) — never generic filler. If not Qualified, return "
            "an empty messages list.\n\n"
            "Respond with ONLY a JSON array, one object per lead, no prose, no markdown "
            "fences, in this exact shape:\n"
            '[{"row_id": 1, "reasoning": "...", "messages": ["...", "..."]}]\n\n'
            f"Leads:\n{json.dumps(lead_blocks, indent=2)}"
        )
        return instructions

    def _call_api(self, prompt: str) -> str:
        import urllib.request
        import urllib.error

        url = f"{self.base_url}/chat/completions"
        body = json.dumps({
            "model": self.cfg["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.get("temperature", 0.4),
        }).encode("utf-8")

        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")

        try:
            with urllib.request.urlopen(req, timeout=self.cfg.get("request_timeout_seconds", 30)) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
                return payload["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise LLMError(f"rate limited (HTTP 429)") from e
            raise LLMError(f"HTTP {e.code}: {e.reason}") from e
        except urllib.error.URLError as e:
            raise LLMError(f"network error: {e.reason}") from e

    def _parse_response(self, raw: str, leads: list[Lead]) -> dict[int, dict]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise LLMError(f"could not parse model output as JSON: {e}") from e

        out = {}
        for item in data:
            rid = item.get("row_id")
            if rid is not None:
                out[rid] = {"reasoning": item.get("reasoning", ""), "messages": item.get("messages", [])}
        return out

    def _mock_response(self, lead: Lead) -> dict:
        """Deterministic, template-based stand-in for the real LLM call. Lets the whole
        pipeline run offline / without a key while still producing lead-specific text."""
        top_factor = max(lead.factor_scores, key=lead.factor_scores.get) if lead.factor_scores else "overall fit"
        bottom_factor = min(lead.factor_scores, key=lead.factor_scores.get) if lead.factor_scores else "overall fit"

        if lead.decision == "Qualified":
            reasoning = (f"Scored {lead.fit_score}/10, led by strong {top_factor.replace('_', ' ')} "
                         f"— a solid match worth prioritizing.")
            company = lead.company or "your company"
            industry = lead.industry or "your space"
            size = lead.company_size
            messages = [
                (f"Hi {lead.name or 'there'}, noticed {company} is scaling in {industry} — "
                 f"we've helped similarly-sized teams cut manual lead review time significantly. "
                 f"Worth a quick chat?"),
                (f"Hi {lead.name or 'there'}, given {company}'s footprint"
                 + (f" (~{size} people)" if size else "")
                 + f" in {industry}, I think there's a strong fit here. Open to a 15-minute intro this week?"),
            ][: self.cfg.get("message_variants_per_lead", 2)]
        else:
            reasoning = (f"Scored {lead.fit_score}/10, held back mainly by weak "
                         f"{bottom_factor.replace('_', ' ')}. {lead.rejection_reason or ''}".strip())
            messages = []

        return {"reasoning": reasoning, "messages": messages}


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #

def build_report(leads: list[Lead], sample_message_count: int) -> dict:
    total = len(leads)
    qualified = [l for l in leads if l.decision == "Qualified"]
    disqualified = [l for l in leads if l.decision == "Disqualified"]
    review = [l for l in leads if l.decision == "Flagged for Human Review"]

    rejection_counts: dict[str, int] = {}
    for l in disqualified + review:
        reason = l.rejection_reason or "unspecified"
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    sample_messages = []
    for l in sorted(qualified, key=lambda x: x.priority_rank or 999):
        for msg in l.outreach_messages:
            sample_messages.append({"lead": l.display_name(), "company": l.company, "message": msg})
        if len(sample_messages) >= sample_message_count:
            break

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": {
            "total_processed": total,
            "qualified": len(qualified),
            "disqualified": len(disqualified),
            "flagged_for_review": len(review),
            "qualified_pct": round(100 * len(qualified) / total, 1) if total else 0.0,
            "top_rejection_reasons": sorted(rejection_counts.items(), key=lambda kv: kv[1], reverse=True),
        },
        "leads": [
            {
                "row_id": l.row_id,
                "name": l.display_name(),
                "company": l.company,
                "industry": l.industry,
                "company_size": l.company_size,
                "source": l.source,
                "fit_score": l.fit_score,
                "factor_scores": l.factor_scores,
                "decision": l.decision,
                "priority_rank": l.priority_rank,
                "reasoning": l.reasoning,
                "rejection_reason": l.rejection_reason,
                "missing_fields": l.missing_fields,
                "outreach_messages": l.outreach_messages,
            }
            for l in sorted(leads, key=lambda x: (x.priority_rank is None, x.priority_rank or 0, -x.fit_score))
        ],
        "sample_outreach_messages": sample_messages[:sample_message_count],
    }


# --------------------------------------------------------------------------- #
# Pipeline orchestration
# --------------------------------------------------------------------------- #

def chunked(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def run_pipeline(input_paths: list[str], config_path: str, output_path: str, as_of: Optional[str] = None) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    leads = load_leads(input_paths)
    if not leads:
        log.error("No leads loaded — check your --input paths")
        sys.exit(1)

    today = datetime.strptime(as_of, "%Y-%m-%d").date() if as_of else date.today()
    for lead in leads:
        score_lead(lead, cfg["rubric"], today)
        decide(lead, cfg["decision_thresholds"])
    assign_priority(leads)

    client = LLMClient(cfg["llm"])
    batch_size = cfg["llm"].get("batch_size", 8)
    n_batches = (len(leads) + batch_size - 1) // batch_size
    log.info("Sending %d leads to LLM in %d batch(es) of up to %d", len(leads), n_batches, batch_size)

    for i, batch in enumerate(chunked(leads, batch_size), start=1):
        log.info("Batch %d/%d (%d leads)...", i, n_batches, len(batch))
        results = client.batch_generate(batch)
        for lead in batch:
            r = results.get(lead.row_id)
            if r:
                lead.reasoning = r.get("reasoning", "")
                lead.outreach_messages = r.get("messages", []) if lead.decision == "Qualified" else []
            else:
                # batch failed even after retries — don't crash the run, downgrade to review
                lead.reasoning = "LLM reasoning unavailable after retries; needs manual review."
                if lead.decision == "Qualified":
                    lead.decision = "Flagged for Human Review"
                    lead.rejection_reason = "LLM call failed — auto-qualification could not be confirmed"

    report = build_report(leads, cfg["output"].get("sample_message_count", 5))

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("Done. %d processed | %d qualified | %d flagged for review | %d disqualified",
              report["summary"]["total_processed"], report["summary"]["qualified"],
              report["summary"]["flagged_for_review"], report["summary"]["disqualified"])
    log.info("Report written to %s", output_path)
    return report


def main():
    parser = argparse.ArgumentParser(description="Lead Intelligence System — auto-qualify and prioritize inbound leads.")
    parser.add_argument("--input", nargs="+", default=["data/leads_training.csv"],
                         help="One or more CSV files to process (default: data/leads_training.csv)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--output", default=None, help="Path to write output_report.json (overrides config)")
    parser.add_argument("--as-of-date", default=None, metavar="YYYY-MM-DD",
                         help="Treat this date as 'today' for recency scoring (default: actual today). "
                              "Useful when scoring a historical/demo dataset.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_preview = yaml.safe_load(f)
    output_path = args.output or cfg_preview["output"]["report_path"]

    run_pipeline(args.input, args.config, output_path, as_of=args.as_of_date)


if __name__ == "__main__":
    main()
