# 🚀 vLLM Server on SDU Cloud (UCloud)

This folder hosts the **Clinical Trial Classifier API** backend powered by **vLLM** and fronted by **NGINX**. It exposes an **OpenAI‑compatible** `/v1` API you can call from your apps and test scripts.

---

## 🌐 Quick UCloud Deploy (NGINX first)

1. Open the NGINX UCloud application:  
   👉 https://cloud.sdu.dk/app/jobs/create?app=nginx

2. In the job form:
   - **NGINX configuration file:** select or upload the `nginx.conf` from this `vllm_server/` folder.
   - **Configure custom links to your application:** add a public link you created at  
     👉 https://cloud.sdu.dk/app/public-links  
     (e.g. `app-clinical-classifier.cloud.aau.dk`).

3. Choose a **machine** (see GPU guidance below), submit the job, and ensure it’s running.

Your public API will then be reachable at:  
**https://<your-public-link>/v1**

> You’ll run vLLM on the same node and NGINX will reverse-proxy requests to it.

---

## ⚙️ UCloud GPU Machine Options — DeiC Interactive HPC

When launching jobs on [UCloud](https://cloud.sdu.dk/), select a node with enough GPUs for your chosen model.  
Two primary clusters are available: **AAU/K8s (L40)** and **SDU/K8s (H100)**.

---

### 🧩 AAU/K8s: L40 Nodes (`uc1-l40-*`)

| Machine | GPUs (NVIDIA L40) |
|----------|-------------------|
| `uc1-l40-1` | 1 |
| `uc1-l40-2` | 2 |
| `uc1-l40-3` | 3 |
| `uc1-l40-4` | 4 |
| `uc1-l40-5` | 5 |
| `uc1-l40-6` | 6 |
| `uc1-l40-7` | 7 |
| `uc1-l40-8` | 8 |

---

### 🧩 SDU/K8s: H100 Nodes (`u3-gpu-*`)

| Machine | GPUs (NVIDIA H100) |
|----------|-------------------|
| `u3-gpu-1` | 1 |
| `u3-gpu-2` | 2 |
| `u3-gpu-3` | 3 |
| `u3-gpu-4` | 4 |

---

## 🧠 Model-to-GPU Recommendations

Below are all models currently configured in `serve.sh`, including their source repositories, direct Hugging Face links, and recommended GPU configurations for both **AAU/K8s (L40)** and **SDU/K8s (H100)** clusters.

| Model Key | Repository | Hugging Face Link | Recommended GPUs (L40) | Recommended GPUs (H100) | Notes |
|------------|-------------|-------------------|------------------------|--------------------------|-------|
| **qwen25-72b-instruct** | Qwen/Qwen2.5-72B-Instruct | [HF ↗](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) | 8 | 4 | Dense FP16/BF16; excellent accuracy |
| **qwen25-72b-instruct-awq** | Qwen/Qwen2.5-72B-Instruct-AWQ | [HF ↗](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-AWQ) | 4 | 2 | ✅ *Recommended default; AWQ lowers VRAM needs* |
| **llama31-70b-instruct** | meta-llama/Llama-3.1-70B-Instruct | [HF ↗](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) | 8 | 4 | Dense; similar footprint to Qwen 72B |
| **llama33-70b-instruct** | meta-llama/Llama-3.3-70B-Instruct | [HF ↗](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | 8 | 4 | FP8-optimized; ideal for H100 |
| **mixtral-8x22b-instruct** | mistralai/Mixtral-8x22B-Instruct-v0.1 | [HF ↗](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) | 6 | 3 | MoE (8 experts × 22B); only 2 active per token |
| **mixtral-8x22b-instruct-awq** | casperhansen/mixtral-8x22b-instruct-v0.1-awq | [HF ↗](https://huggingface.co/casperhansen/mixtral-8x22b-instruct-v0.1-awq) | 4 | 2 | AWQ MoE; fast and VRAM-efficient |
| **dbrx-instruct** | databricks/dbrx-instruct | [HF ↗](https://huggingface.co/databricks/dbrx-instruct) | 8 | 4 | Large MoE (~132B, 36B active); strong generalization |
| **gpt-oss-120b** | openai/gpt-oss-120b | [HF ↗](https://huggingface.co/openai/gpt-oss-120b) | 8 | 4 | OpenAI OSS 120B; reasoning-focused MoE |
| **qwen25-32b-instruct** | Qwen/Qwen2.5-32B-Instruct | [HF ↗](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | 2 | 1 | Medium dense; good for mid-scale experiments |
| **qwen25-32b-instruct-awq** | Qwen/Qwen2.5-32B-Instruct-AWQ | [HF ↗](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ) | 1 | 1 | Quantized; ideal for testing/dev |
| **llama31-8b-instruct** | meta-llama/Llama-3.1-8B-Instruct | [HF ↗](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 1 | 1 | Lightweight; perfect for local debugging |
| **gpt-oss-20b** | openai/gpt-oss-20b | [HF ↗](https://huggingface.co/openai/gpt-oss-20b) | 1 | 1 | Small dense model; validation and testing |

---

## ⚙️ Install & Run vLLM (on the same node)

Create an isolated Python env **inside this folder** and launch the server:

Only perform the step below the first time:

```bash
cd vllm_server
pip install uv
uv init
uv venv
uv add vllm
chmod +x serve.sh
```

### ▶️ Start with Qwen 2.5‑72B AWQ (recommended example)
# 👉 Example: Qwen 2.5 72B AWQ (great quality, fits on fewer GPUs)

```bash
./serve.sh qwen25-72b-instruct-awq
```

If startup is successful you’ll see logs ending with the served model name and port.  
**Local URL (inside node):** `http://localhost:8000/v1`  
**Public URL (via NGINX):** `https://<your-public-link>/v1`

---

## 🔐 Example `.env`

```env
# tokens & network
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your-strong-random-token

# will be updated by serve.sh when you pick a model
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct-AWQ
```

> Place this file **in `vllm_server/.env`**. Do **not** commit it.

---

## 📦 What’s in this folder?

| File | Description |
|---|---|
| `serve.sh` | Start script for vLLM server (OpenAI‑compatible) |
| `.env` | **Your** runtime secrets and config (not committed) |
| `nginx.conf` | NGINX reverse‑proxy for public access on UCloud |
| `README.md` | This guide |
---
