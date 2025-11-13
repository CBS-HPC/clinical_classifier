#!/usr/bin/env python3
"""
Batch classifier for clinical-trial filings using a vLLM OpenAI-compatible server.

- Reads .json files from ./docs/ (expects array of filing records)
- Writes markdown results to ./results/classifier_output.md
- Supports guided (JSON) or plain text output modes

Usage:
    python test_clinical_classifier.py [--no-guided]

Requires:
    pip install openai python-dotenv pydantic
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Literal, Dict

from dotenv import load_dotenv
from openai import OpenAI

from instructions import SYSTEM_PROMPT, GuidedGeneration

# ==========================================
# CONFIGURATION
# ==========================================
load_dotenv(override=False)

OPENAI_API_BASE = "https://app-clinical-classifier.cloud.aau.dk/v1"
OPENAI_API_KEY  = "token-bd2ed34ad7f533587784967938dffb98"


# Toggle guided generation (structured output) via env or default True
USE_GUIDED_GENERATION = True

# Generation parameters
TEMPERATURE = 0.0           # Use 0.0 for deterministic output (recommended for classification)
MAX_TOKENS = 500            # Maximum tokens in response (adjust based on your document length)
TOP_P = 1.0                 # Nucleus sampling (1.0 = disabled, use with temperature > 0)
FREQUENCY_PENALTY = 0.0     # Penalize token repetition (0.0-2.0)
PRESENCE_PENALTY = 0.0      # Penalize new topics (0.0-2.0)
STOP_SEQUENCES = None       # Optional: list of strings to stop generation, e.g., ["\n\n", "END"]

DOCS_DIR    = Path("docs")
RESULTS_DIR = Path("results")
RESULTS_JSONL = RESULTS_DIR / "results.jsonl"
# =========================================
# HELPERS
# ==========================================

def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

def create_user_message(form_date, form_type, ticker, sic, acc_no, filing_text):
    """Create user message in same format as Ollama workflow"""
    user_msg = f"""FORM_DATE: {form_date}
FORM_TYPE: {form_type}
TICKER: {ticker}
SIC: {sic}
ACC_NO: {acc_no}

TEXT:
{filing_text}
"""
    return user_msg

def parse_structured_output(text: str) -> Dict[str, str]:
    """Parse the arrow-based output format into a dictionary."""
    result: Dict[str, str] = {}
    for line in text.strip().splitlines():
        if " -> " in line:
            key, value = line.split(" -> ", 1)
            result[key.strip()] = value.strip()
    return result

def dict_to_arrow_block(d: Dict[str, str]) -> str:
    """Serialize a dict to the exact 8-line arrow block in the required order."""
    fields = [
        "clinical", "summary", "event_type", "phase",
        "outcome", "confidence", "press_release_date", "evidence_spans"
    ]
    lines = [f"{k} -> {d.get(k, 'unknown')}" for k in fields]
    return "\n".join(lines)

# ==========================================
# CLIENT
# ==========================================

def build_client() -> OpenAI:
    print(f"Connecting to: {OPENAI_API_BASE}")
    print(f"Using API key: {OPENAI_API_KEY[:6]}... (hidden)")
    print(f"Guided generation: {'ENABLED' if USE_GUIDED_GENERATION else 'DISABLED'}\n")
    return OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)

def pick_model(client: OpenAI) -> str:
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        if not available:
            raise RuntimeError("No models available on the server.")
        chosen = os.getenv("OPENAI_MODEL", available[0])
        print(f"Available models: {', '.join(available)}")
        print(f"Using model: {chosen}\n")
        return chosen
    except Exception as e:
        print(f"ERROR: Failed to list models: {e}", file=sys.stderr)
        print("   Is the vLLM server running and reachable?", file=sys.stderr)
        sys.exit(2)

# ==========================================
# INFERENCE
# ==========================================

def classify_record(client: OpenAI, model: str, record: dict, guided: bool) -> Dict[str, str]:
    """Classify a single filing record"""
    user_message = create_user_message(
        record.get("filling_date", "unknown"),
        record.get("form", "unknown"),
        record.get("ticker", "unknown"),
        record.get("sic", "unknown"),
        record.get("acc_no", "unknown"),
        record.get("text", "")
    )
    
    try:
        if guided:
            resp = client.chat.completions.create(
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
            return {k: str(v) for k, v in data.items()}
        else:
            resp = client.chat.completions.create(
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
            return parse_structured_output(raw)
    except Exception as e:
        print(f"ERROR: Inference failed: {e}", file=sys.stderr)
        raise

# ==========================================
# MAIN
# ==========================================

def main():
    ensure_dirs()

    client = build_client()
    model = pick_model(client)

    # Gather input files
    json_files = sorted(DOCS_DIR.glob("*.json"))
    if not json_files:
        print(f"WARNING: No .json files found in {DOCS_DIR.resolve()}.")
        print("   Create a file like docs/filings.json and rerun.")
        sys.exit(0)

    for json_path in json_files:
        print(f"\n=== Processing: {json_path.name} ===")

        # Load JSON file
        try:
            with json_path.open("r", encoding="utf-8") as f:
                records = json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse {json_path.name}: {e}")
            continue

        # Ensure records is a list
        if not isinstance(records, list):
            print(f"ERROR: {json_path.name} does not contain a JSON array")
            continue

        print(f"Found {len(records)} record(s)")

        # Process each record
        for idx, record in enumerate(records):

            ticker = record.get("ticker", "UNKNOWN")
            acc_no = record.get("acc_no", "unknown")
            form_date = record.get("filling_date", "unknown")
            sample_id = f"{ticker}_{acc_no}_{form_date}"

            text_len = len(record.get("text", ""))
            print(f"  [{idx+1}/{len(records)}] Classifying: {sample_id} (text: {text_len} chars)")

            result = classify_record(client, model, record, USE_GUIDED_GENERATION)

            classification = {
                "clinical": result.get("clinical", "unknown"),
                "summary": result.get("summary", ""),
                "event_type": result.get("event_type", "unknown"),
                "phase": result.get("phase", "unknown"),
                "outcome": result.get("outcome", "unknown"),
                "confidence": result.get("confidence", "none"),
                "press_release_date": result.get("press_release_date", "unknown"),
                "evidence_spans": result.get("evidence_spans", ""),
            }

            record_data = {
                "sample_id": sample_id,
                "source_file": json_path.name,
                "guided_generation": bool(USE_GUIDED_GENERATION),
                "api_base": OPENAI_API_BASE,
                "model": model,
                "metadata": {
                    "cik": record.get("cik", "N/A"),
                    "ticker": record.get("ticker", "N/A"),
                    "company": record.get("name", "N/A"),
                    "form": record.get("form", "N/A"),
                    "filing_date": record.get("filling_date", "N/A"),
                    "acc_no": record.get("acc_no", "N/A"),
                    "sic": record.get("sic", "N/A"),
                    "text_length": text_len,
                },
                "classification": classification,
            }

            # Append a single JSON line immediately
            with RESULTS_JSONL.open("a", encoding="utf-8") as out_jsonl:
                out_jsonl.write(json.dumps(record_data, ensure_ascii=False) + "\n")

    print(f"\nDone. Results saved to: {RESULTS_JSONL.resolve()}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.2f} seconds")
