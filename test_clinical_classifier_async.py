#!/usr/bin/env python3
"""
Async batch classifier for clinical-trial filings using a vLLM OpenAI-compatible server.

- Reads .json files from ./docs/ (expects array of filing records)
- Writes results incrementally to ./results/classifier_results.jsonl
- Supports guided (JSON) or plain text output modes
- Processes records concurrently up to CONCURRENT_REQUESTS

Usage:
    python test_clinical_classifier.py
    CONCURRENT_REQUESTS=16 python test_clinical_classifier.py
    python test_clinical_classifier.py --no-guided
"""

import os
import sys
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from instructions import SYSTEM_PROMPT, GuidedGeneration

# ==========================================
# CONFIGURATION
# ==========================================
load_dotenv(override=False)

OPENAI_API_BASE = "https://app-clinical-trial.cloud.aau.dk/v1"
OPENAI_API_KEY  = "token-99898157f592e2bd278a49fa1a4a0ee3"

CONCURRENT_REQUESTS = 1

# Toggle guided generation (structured output) via env or default True
USE_GUIDED_GENERATION = True


TEMPERATURE = 0.0
MAX_TOKENS = 500
TOP_P = 1.0
FREQUENCY_PENALTY = 0.0
PRESENCE_PENALTY = 0.0
STOP_SEQUENCES = None

DOCS_DIR    = Path("docs")
RESULTS_DIR = Path("results")
RESULTS_JSONL = RESULTS_DIR / "results_async.jsonl"

# ==========================================
# HELPERS
# ==========================================

def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSONL.touch(exist_ok=True)

def create_user_message(form_date, form_type, ticker, sic, acc_no, filing_text):
    return f"""FORM_DATE: {form_date}
FORM_TYPE: {form_type}
TICKER: {ticker}
SIC: {sic}
ACC_NO: {acc_no}

TEXT:
{filing_text}
"""

def parse_structured_output(text: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for line in text.strip().splitlines():
        if " -> " in line:
            key, value = line.split(" -> ", 1)
            result[key.strip()] = value.strip()
    return result

# ==========================================
# CLIENT
# ==========================================

def build_async_client() -> AsyncOpenAI:
    print(f"Connecting (async) to: {OPENAI_API_BASE}")
    print(f"Using API key: {OPENAI_API_KEY[:6]}... (hidden)")
    print(f"Guided generation: {'ENABLED' if USE_GUIDED_GENERATION else 'DISABLED'}")
    print(f"Concurrency: {CONCURRENT_REQUESTS}\n")
    return AsyncOpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)

async def pick_model_async(client: AsyncOpenAI) -> str:
    try:
        models = await client.models.list()
        available = [m.id for m in models.data]
        if not available:
            raise RuntimeError("No models available on the server.")
        chosen = os.getenv("OPENAI_MODEL", available[0])
        print(f"Available models: {', '.join(available)}")
        print(f"Using model: {chosen}\n")
        return chosen
    except Exception as e:
        print(f"ERROR: Failed to list models: {e}", file=sys.stderr)
        sys.exit(2)

# ==========================================
# INFERENCE
# ==========================================

async def classify_record_async(client: AsyncOpenAI, model: str, record: dict, guided: bool) -> Dict[str, str]:
    """Async classification for one record."""
    user_message = create_user_message(
        record.get("filling_date", "unknown"),
        record.get("form", "unknown"),
        record.get("ticker", "unknown"),
        record.get("sic", "unknown"),
        record.get("acc_no", "unknown"),
        record.get("text", "")
    )

    if guided:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            frequency_penalty=FREQUENCY_PENALTY,
            presence_penalty=PRESENCE_PENALTY,
            stop=STOP_SEQUENCES,
            extra_body={
                "guided_json": GuidedGeneration.model_json_schema(),
                "guided_decoding_backend": "outlines",
            },
        )
        data = json.loads(resp.choices[0].message.content)
        out = {k: str(v) for k, v in data.items()}
    else:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            frequency_penalty=FREQUENCY_PENALTY,
            presence_penalty=PRESENCE_PENALTY,
            stop=STOP_SEQUENCES,
        )
        raw = resp.choices[0].message.content
        out = parse_structured_output(raw)

    # Normalize & ensure all fields
    return {
        "clinical": out.get("clinical", "unknown"),
        "summary": out.get("summary", ""),
        "event_type": out.get("event_type", "unknown"),
        "phase": out.get("phase", "unknown"),
        "outcome": out.get("outcome", "unknown"),
        "confidence": out.get("confidence", "none"),
        "press_release_date": out.get("press_release_date", "unknown"),
        "evidence_spans": out.get("evidence_spans", ""),
    }

# ==========================================
# CONCURRENT PROCESSING (JSONL OUTPUT)
# ==========================================

async def process_records_concurrently(
    client: AsyncOpenAI,
    model: str,
    records: List[dict],
    guided: bool,
    concurrency: int,
    json_path: Path,
    jsonl_path: Path = RESULTS_JSONL
):
    """Run async inference and append each result to a JSONL file."""
    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()

    async def write_jsonl(record_data: dict):
        async with file_lock:
            with jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record_data, ensure_ascii=False) + "\n")

    async def run_one(idx: int, rec: dict):
        async with semaphore:
            try:
                result = await classify_record_async(client, model, rec, guided)
                error_msg = None
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                result = {
                    "clinical": "unknown",
                    "summary": "",
                    "event_type": "unknown",
                    "phase": "unknown",
                    "outcome": "unknown",
                    "confidence": "none",
                    "press_release_date": "unknown",
                    "evidence_spans": "",
                }

            ticker = rec.get("ticker", "UNKNOWN")
            acc_no = rec.get("acc_no", "unknown")
            form_date = rec.get("filling_date", "unknown")
            sample_id = f"{ticker}_{acc_no}_{form_date}"

            record_data = {
                "sample_id": sample_id,
                "source_file": json_path.name,
                "guided_generation": bool(guided),
                "api_base": OPENAI_API_BASE,
                "model": model,
                "metadata": {
                    "cik": rec.get("cik", "N/A"),
                    "ticker": rec.get("ticker", "N/A"),
                    "company": rec.get("name", "N/A"),
                    "form": rec.get("form", "N/A"),
                    "filing_date": rec.get("filling_date", "N/A"),
                    "acc_no": rec.get("acc_no", "N/A"),
                    "sic": rec.get("sic", "N/A"),
                    "text_length": len(rec.get("text", "")),
                },
                "classification": result,
                "error": error_msg,
            }

            await write_jsonl(record_data)
            text_length = record_data["metadata"]["text_length"]
            print(f"  [{idx+1}/{len(records)}] Saved: {sample_id} (text: {text_length } chars)")
            return idx, result, error_msg

    tasks = [asyncio.create_task(run_one(i, r)) for i, r in enumerate(records)]
    await asyncio.gather(*tasks)

# ==========================================
# MAIN
# ==========================================

async def async_main(use_guided: bool):
    ensure_dirs()
    aclient = build_async_client()
    model = await pick_model_async(aclient)

    json_files = sorted(DOCS_DIR.glob("*.json"))
    if not json_files:
        print(f"WARNING: No .json files found in {DOCS_DIR.resolve()}")
        sys.exit(0)

    for json_path in json_files:
        print(f"\n=== Processing: {json_path.name} ===")
        try:
            with json_path.open("r", encoding="utf-8") as f:
                records = json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse {json_path.name}: {e}")
            continue
        if not isinstance(records, list):
            print(f"ERROR: {json_path.name} does not contain a JSON array")
            continue

        print(f"Found {len(records)} record(s) -> processing up to {CONCURRENT_REQUESTS} concurrently")
        await process_records_concurrently(
            client=aclient,
            model=model,
            records=records,
            guided=use_guided,
            concurrency=CONCURRENT_REQUESTS,
            json_path=json_path
        )

    print(f"\n✅ Done. Results saved to: {RESULTS_JSONL.resolve()}")

# ==========================================
# ENTRY POINT
# ==========================================

def main():
    asyncio.run(async_main(USE_GUIDED_GENERATION))

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Execution time: {end - start:.2f} seconds")
