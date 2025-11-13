# 🧠 Clinical Trial Classifier — Client & Local Testing

This repository contains **client-side scripts** to classify clinical-trial filings using an **OpenAI-compatible API** served by your vLLM backend (running locally or on UCloud).  
> The server setup lives in **`vllm_server/`** (separate README).  
> This README focuses on **connecting, testing, and running classifications** from your machine.

---

## ⚙️ Step 0 — Set Up the vLLM Server

Before running these client scripts, you must first **deploy the vLLM backend**.  
Follow the setup instructions in:

📄 **[`vllm_server/README.md`](./vllm_server/README.md)**

That guide explains how to:
- Start the vLLM model server (local or on UCloud)
- Configure NGINX for headless/public access
- Choose appropriate GPUs and models (e.g., `qwen25-72b-instruct-awq`)
- Expose the API endpoint (e.g., `https://app-clinical-classifier.cloud.aau.dk/v1`)

Once your backend is running and accessible, continue below.

---

## 🚀 What You Can Do Here
- **Verify** that the vLLM API is reachable and lists available models (`test_connection.py`)
- **Batch-classify** `.txt` filings using a strict **8-field** schema (`test_clinical_classifier.py`)
- Produce a consolidated **Markdown report** under `./results/`

Both scripts work with either:
- a **local** server (e.g., `http://localhost:8000/v1`), or  
- a **remote/headless** server (e.g., `https://app-clinical-classifier.cloud.aau.dk/v1`).

---

## 📦 Requirements
Create a Python environment (using `uv` or `venv`) and install dependencies:

```bash
pip install uv
uv venv
uv add openai python-dotenv pydantic
```

---

## ⚙️ Configure the API connection
Create a `.env` file in this project root with **either** your remote **or** local API:

**Remote / headless (default):**
```env
OPENAI_API_BASE=https://app-clinical-classifier.cloud.aau.dk/v1
OPENAI_API_KEY=your-remote-token
```

**Local (if running vLLM on the same machine):**
```env
OPENAI_API_BASE=http://localhost:8000/v1
OPENAI_API_KEY=your-local-token
```

> Your API key must match the `OPENAI_API_KEY` the server was started with.

---

## 🧪 Quick start

### 1) Connectivity check
Runs a minimal probe: lists models and prints the **first model id**.

```bash
python test_connection.py
```

Expected behavior:
```
<model-id-from-server>
```
(For example: `qwen25-72b-instruct-awq`)

---

### 2) Run the Clinical Trial Classifier
Place one or more `.txt` files in `./docs/`, then run:

```bash
python test_clinical_classifier.py
```

What it does:
- Reads **all** `./docs/*.txt`
- Sends each document with a strict **classification prompt**
- Writes a consolidated report to **`./results/classifier_output.md`**
- Uses **guided (JSON‑structured)** generation by default

Example block from the report:
```text
## Sample: sample_001
clinical -> yes
summary -> Phase III OS benefit for BPI-2000; 450 pts enrolled
event_type -> Results
phase -> 3
outcome -> positive
confidence -> topline
press_release_date -> 2024-03-15
evidence_spans -> "Phase 3", "met its primary endpoint", "topline results"
```

---

## 📂 Inputs & Outputs

**Input folder:** `./docs/` (place `.txt` files here)  
**Output folder:** `./results/`  
**Report file:** `./results/classifier_output.md`

Example layout:
```
project/
├─ vllm_server/ # Backend (vLLM + NGINX)
├─ docs/
│  ├─ sample_001.txt
│  └─ sample_002.txt
├─ results/
│  └─ classifier_output.md  (generated)
├─ test_connection.py
└─ test_clinical_classifier.py
```

---

## 🔧 Script configuration (env vars)

These can go in `.env` or be provided inline in your shell:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_BASE` | `https://app-clinical-classifier.cloud.aau.dk/v1` | Base URL of your vLLM API |
| `OPENAI_API_KEY` | — | API key for the server  |
| `USE_GUIDED_GENERATION` | `1` | `1` = JSON‑guided, `0` = plain‑text parsing |
| `DOCS_DIR` / `RESULTS_DIR` | `docs` / `results` | Input & output directories |
| `TEMPERATURE` | `0.0` | Deterministic classification |
| `MAX_TOKENS` | `500` | Max tokens for completion |
| `TOP_P` | `1.0` | Nucleus sampling (disabled at 1.0) |
| `FREQUENCY_PENALTY` | `0.0` | Penalize repeats |
| `PRESENCE_PENALTY` | `0.0` | Penalize novel topics |

Examples:
```bash
# Force local server
OPENAI_API_BASE=http://localhost:8000/v1 OPENAI_API_KEY=your-local-token python test_connection.py

# Disable guided mode for the classifier
USE_GUIDED_GENERATION=0 python test_clinical_classifier.py
```

---

## 🧰 Notes & Tips
- The **backend** (vLLM + NGINX) lives in `vllm_server/` with its own README.
- If `test_connection.py` shows no models, ensure your server is running and reachable.
- For very short inputs, consider reducing server context to save VRAM (server side).
- All results are Markdown, easy to diff and track in version control.

---

© 2025 — Clinical Trial Classifier (client utilities & local test scripts)
