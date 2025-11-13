#!/usr/bin/env bash
# serve_clinical_classifier.sh — vLLM OpenAI-compatible server for clinical trial classification
# Focus: Large instruction models for multi-GPU setup (L40/L40S, 48GB each; also fine for 4×H100)
# Usage: ./serve_clinical_classifier.sh <model-key>
# Example: ./serve_clinical_classifier.sh qwen25-72b-instruct

set -euo pipefail

# ===== Default runtime settings (single-user friendly) =====


# Fraction of each GPU’s VRAM vLLM pre-allocates for the KV cache (0–1).
# Higher = more memory reserved for concurrent requests (faster, fewer OOMs),
# but leaves less headroom for other processes. Typical: 0.5–0.95.
GPU_UTIL=0.95

# Max tokens the model can see at once ("attention span").
# Larger = longer docs, but more VRAM. Typical: 8192–32768.
DEFAULT_CONTEXT_WINDOW=32768

# Max concurrent requests. 8 is perfect for one user; higher = more VRAM use.
DEFAULT_MAX_SEQS=8

# Auto tensor parallelism: 1 = auto-pick TP up to available GPUs, 0 = manual control.
AUTO_TP=1

# KV-cache precision. fp8 saves VRAM on L40/L40S/H100; use bf16/fp16 if unsupported.
DEFAULT_KV_CACHE_DTYPE="fp8"

# Allow custom model repo code. Keep "false" for safety; set "true" only if required.
DEFAULT_TRUST_REMOTE_CODE="false"

# =========================
# LOAD ENVIRONMENT VARIABLES
# =========================
if [ -f ".env" ]; then
  echo "Loading environment variables from .env..."
  set -o allexport
  # shellcheck source=/dev/null
  source .env
  set +o allexport
else
  echo "ERROR: .env file not found. Please create one with HF_TOKEN (and optionally OPENAI_API_KEY)."
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN is not set in your .env file."
  exit 1
fi

# =========================
# MODEL REGISTRY & SETTINGS
# =========================
declare -A MODEL_REGISTRY=(
  # TOP RECOMMENDED
  ["qwen25-72b-instruct"]="Qwen/Qwen2.5-72B-Instruct"
  ["qwen25-72b-instruct-awq"]="Qwen/Qwen2.5-72B-Instruct-AWQ"
  ["llama31-70b-instruct"]="meta-llama/Llama-3.1-70B-Instruct"
  ["llama33-70b-instruct"]="meta-llama/Llama-3.3-70B-Instruct"

  # MoE for throughput
  ["mixtral-8x22b-instruct"]="mistralai/Mixtral-8x22B-Instruct-v0.1"
  ["mixtral-8x22b-instruct-awq"]="casperhansen/mixtral-8x22b-instruct-v0.1-awq"

  # Bigger additions (8×L40 / 4×H100 friendly)
  ["dbrx-instruct"]="databricks/dbrx-instruct"      # 132B MoE (~36B active)
  ["gpt-oss-120b"]="openai/gpt-oss-120b"            # ~120B, MoE

  # Smaller / testing
  ["qwen25-32b-instruct"]="Qwen/Qwen2.5-32B-Instruct"
  ["qwen25-32b-instruct-awq"]="Qwen/Qwen2.5-32B-Instruct-AWQ"
  ["llama31-8b-instruct"]="meta-llama/Llama-3.1-8B-Instruct"
  ["gpt-oss-20b"]="openai/gpt-oss-20b"              # ~20B
)

# Recommended TP sizes (upper bounds). Can be overridden by env TP_SIZE or AUTO_TP logic.
declare -A TP_SIZES=(
  ["qwen25-72b-instruct"]=8
  ["qwen25-72b-instruct-awq"]=4
  ["llama31-70b-instruct"]=8
  ["llama33-70b-instruct"]=8
  ["mixtral-8x22b-instruct"]=8
  ["mixtral-8x22b-instruct-awq"]=4
  ["dbrx-instruct"]=8
  ["gpt-oss-120b"]=8
  ["gpt-oss-20b"]=1
  ["qwen25-32b-instruct"]=2
  ["qwen25-32b-instruct-awq"]=1
  ["llama31-8b-instruct"]=1
)

# Quantization per model (weight-only)
declare -A QUANTIZATION=(
  ["qwen25-72b-instruct"]=""
  ["qwen25-72b-instruct-awq"]="awq"
  ["llama31-70b-instruct"]=""
  ["llama33-70b-instruct"]=""
  ["mixtral-8x22b-instruct"]=""
  ["mixtral-8x22b-instruct-awq"]="awq"
  ["dbrx-instruct"]=""         # prefer dense bf16 weights + fp8 KV
  ["gpt-oss-120b"]=""          # start dense; add AWQ/FP8 weights if available
  ["gpt-oss-20b"]=""
  ["qwen25-32b-instruct"]=""
  ["qwen25-32b-instruct-awq"]="awq"
  ["llama31-8b-instruct"]=""
)

# DType defaults
declare -A DTYPE=(
  ["qwen25-72b-instruct"]="auto"
  ["qwen25-72b-instruct-awq"]="half"
  ["llama31-70b-instruct"]="auto"
  ["llama33-70b-instruct"]="auto"
  ["mixtral-8x22b-instruct"]="auto"
  ["mixtral-8x22b-instruct-awq"]="half"
  ["dbrx-instruct"]="auto"
  ["gpt-oss-120b"]="auto"
  ["gpt-oss-20b"]="auto"
  ["qwen25-32b-instruct"]="auto"
  ["qwen25-32b-instruct-awq"]="half"
  ["llama31-8b-instruct"]="auto"
)

# Trust remote code (inherits secure default)
declare -A TRUST_REMOTE_CODE=(
  ["qwen25-72b-instruct"]=$DEFAULT_TRUST_REMOTE_CODE
  ["qwen25-72b-instruct-awq"]=$DEFAULT_TRUST_REMOTE_CODE
  ["llama31-70b-instruct"]=$DEFAULT_TRUST_REMOTE_CODE
  ["llama33-70b-instruct"]=$DEFAULT_TRUST_REMOTE_CODE
  ["mixtral-8x22b-instruct"]=$DEFAULT_TRUST_REMOTE_CODE
  ["mixtral-8x22b-instruct-awq"]=$DEFAULT_TRUST_REMOTE_CODE
  ["dbrx-instruct"]=$DEFAULT_TRUST_REMOTE_CODE
  ["gpt-oss-120b"]=$DEFAULT_TRUST_REMOTE_CODE
  ["gpt-oss-20b"]=$DEFAULT_TRUST_REMOTE_CODE
  ["qwen25-32b-instruct"]=$DEFAULT_TRUST_REMOTE_CODE
  ["qwen25-32b-instruct-awq"]=$DEFAULT_TRUST_REMOTE_CODE
  ["llama31-8b-instruct"]=$DEFAULT_TRUST_REMOTE_CODE
)

# Per-model context window defaults (inherit global)
declare -A MAX_MODEL_LEN=(
  ["qwen25-72b-instruct"]=$DEFAULT_CONTEXT_WINDOW
  ["qwen25-72b-instruct-awq"]=$DEFAULT_CONTEXT_WINDOW
  ["llama31-70b-instruct"]=$DEFAULT_CONTEXT_WINDOW
  ["llama33-70b-instruct"]=$DEFAULT_CONTEXT_WINDOW
  ["mixtral-8x22b-instruct"]=$DEFAULT_CONTEXT_WINDOW
  ["mixtral-8x22b-instruct-awq"]=$DEFAULT_CONTEXT_WINDOW
  ["dbrx-instruct"]=$DEFAULT_CONTEXT_WINDOW
  ["gpt-oss-120b"]=$DEFAULT_CONTEXT_WINDOW
  ["gpt-oss-20b"]=$DEFAULT_CONTEXT_WINDOW
  ["qwen25-32b-instruct"]=$DEFAULT_CONTEXT_WINDOW
  ["qwen25-32b-instruct-awq"]=$DEFAULT_CONTEXT_WINDOW
  ["llama31-8b-instruct"]=$DEFAULT_CONTEXT_WINDOW
)

# Per-model MAX_SEQS defaults
declare -A MAX_SEQS_MAP=(
  ["qwen25-72b-instruct"]=$DEFAULT_MAX_SEQS
  ["qwen25-72b-instruct-awq"]=$DEFAULT_MAX_SEQS
  ["llama31-70b-instruct"]=$DEFAULT_MAX_SEQS
  ["llama33-70b-instruct"]=$DEFAULT_MAX_SEQS
  ["mixtral-8x22b-instruct"]=$DEFAULT_MAX_SEQS
  ["mixtral-8x22b-instruct-awq"]=$DEFAULT_MAX_SEQS
  ["dbrx-instruct"]=$DEFAULT_MAX_SEQS
  ["gpt-oss-120b"]=$DEFAULT_MAX_SEQS
  ["gpt-oss-20b"]=$DEFAULT_MAX_SEQS
  ["qwen25-32b-instruct"]=$DEFAULT_MAX_SEQS
  ["qwen25-32b-instruct-awq"]=$DEFAULT_MAX_SEQS
  ["llama31-8b-instruct"]=$DEFAULT_MAX_SEQS
)

# =========================
# INPUTS & DERIVED SETTINGS
# =========================
MODEL_KEY="${1:-}"
if [ -z "$MODEL_KEY" ] || [ -z "${MODEL_REGISTRY[$MODEL_KEY]:-}" ]; then
  echo "ERROR: Invalid or missing model key."
  echo "Usage: $0 <model-key>"
  echo ""
  echo "PRIMARY:"
  echo "  - qwen25-72b-instruct, qwen25-72b-instruct-awq, llama31-70b-instruct, llama33-70b-instruct"
  echo "THROUGHPUT:"
  echo "  - mixtral-8x22b-instruct, mixtral-8x22b-instruct-awq"
  echo "BIGGER (8×L40 / 4×H100):"
  echo "  - dbrx-instruct, gpt-oss-120b, gpt-oss-20b"
  echo "TEST:"
  echo "  - qwen25-32b-instruct, qwen25-32b-instruct-awq, llama31-8b-instruct"
  exit 1
fi

MODEL_NAME="${MODEL_REGISTRY[$MODEL_KEY]}"
TP_SIZE_DEFAULT="${TP_SIZES[$MODEL_KEY]}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-./.cache/models/$MODEL_KEY}"

# Network & server options (override in .env or env)
PORT="${PORT:-8000}"
HOST="${VLLM_HOST:-0.0.0.0}"

# Pull per-model defaults, allow env override
MODEL_LEN="${MODEL_LEN:-${MAX_MODEL_LEN[$MODEL_KEY]}}"
MAX_SEQS="${MAX_SEQS:-${MAX_SEQS_MAP[$MODEL_KEY]}}"
DT="${DTYPE[$MODEL_KEY]}"
QUANT="${QUANTIZATION[$MODEL_KEY]}"
TRC="${TRUST_REMOTE_CODE[$MODEL_KEY]}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-$DEFAULT_KV_CACHE_DTYPE}"

# Preflight behavior
SAFE_VRAM="${SAFE_VRAM:-1}"
VRAM_HEADROOM_SCALE="${VRAM_HEADROOM_SCALE:-1.0}"
SHORTFALL_MARGIN_GB="${SHORTFALL_MARGIN_GB:-8}"

# Auth key for the server
if [ -z "${OPENAI_API_KEY:-}" ]; then
  if command -v openssl >/dev/null 2>&1; then
    OPENAI_API_KEY="token-$(openssl rand -hex 16)"
  else
    OPENAI_API_KEY="token-$(date +%s)"
  fi
  echo "OPENAI_API_KEY=$OPENAI_API_KEY" >> .env
  echo "Generated OPENAI_API_KEY and saved to .env"
fi

# =========================
# UPDATE .env WITH SELECTED MODEL
# =========================
tail -c1 .env | read -r _ || echo >> .env
if grep -q "^MODEL_NAME=" .env; then
  sed -i.bak "s|^MODEL_NAME=.*|MODEL_NAME=${MODEL_NAME}|" .env
else
  echo "MODEL_NAME=${MODEL_NAME}" >> .env
fi

echo "MODEL_NAME=${MODEL_NAME}"


# =========================
# ACTIVATE VENV (IF EXISTS)
# =========================
if [ -d ".venv" ]; then
  echo "Activating virtual environment (.venv)..."
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "WARNING: No .venv directory found. Proceeding without activating a virtual environment."
fi

# =========================
# CHECK GPU AVAILABILITY
# =========================
if command -v nvidia-smi >/dev/null 2>&1; then
  available_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
  echo "Detected $available_gpus GPU(s) available."

  echo ""
  echo "GPU Memory Status:"
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | \
    awk -F', ' '{printf "  GPU %s (%s): %s total, %s free\n", $1, $2, $3, $4}'
  echo ""
else
  echo "WARNING: nvidia-smi not found; skipping GPU count check."
  available_gpus=1
fi

# Determine TP size (env > auto > recommended default)
if [ -n "${TP_SIZE:-}" ]; then
  TP_SIZE="${TP_SIZE}"
else
  if [ "$AUTO_TP" = "1" ]; then
    # We want at least 1, at most recommended, and not more than available GPUs
    if [ "$available_gpus" -lt 1 ]; then available_gpus=1; fi
    rec="${TP_SIZES[$MODEL_KEY]}"
    # min(available_gpus, rec), but at least 1
    if [ "$available_gpus" -le "$rec" ]; then
      TP_SIZE="$available_gpus"
    else
      TP_SIZE="$rec"
    fi
  else
    TP_SIZE="$TP_SIZE_DEFAULT"
  fi
fi

if [ "$available_gpus" -lt "$TP_SIZE" ]; then
  echo "ERROR: Not enough GPUs available. Required TP: $TP_SIZE, Available: $available_gpus"
  echo "   Consider enabling AUTO_TP=1 or lowering TP_SIZE."
  exit 1
fi

# =========================
# CLEAN UP ANY PREVIOUS vLLM INSTANCES
# =========================
existing=$(pgrep -f "vllm\.entrypoints\.openai\.api_server|vllm serve" || true)
if [ -n "$existing" ]; then
  echo "Detected existing vLLM process. Killing: $existing"
  kill -9 $existing || true
  sleep 2
  echo "Running GPU memory cleanup..."
  python3 - <<'PY' || true
import os
try:
    import torch
    torch.cuda.empty_cache(); torch.cuda.ipc_collect()
except Exception:
    pass
PY
else
  echo "No running vLLM server detected."
fi

# =========================
# PREPARE CACHE DIRECTORY
# =========================
mkdir -p "$MODEL_CACHE_DIR"

# =========================
# VRAM PREFLIGHT (dtype-aware, TP-normalized)
# =========================
if [ "$SAFE_VRAM" = "1" ] && command -v nvidia-smi >/dev/null 2>&1; then
  declare -A PARAMS_B=(
    ["qwen25-72b-instruct"]=72
    ["qwen25-72b-instruct-awq"]=72
    ["llama31-70b-instruct"]=70
    ["llama33-70b-instruct"]=70
    ["mixtral-8x22b-instruct"]=141  # 8×22B active params
    ["mixtral-8x22b-instruct-awq"]=141
    ["dbrx-instruct"]=132
    ["gpt-oss-120b"]=120
    ["gpt-oss-20b"]=20
    ["qwen25-32b-instruct"]=32
    ["qwen25-32b-instruct-awq"]=32
    ["llama31-8b-instruct"]=8
  )

  params="${PARAMS_B[$MODEL_KEY]:-0}"
  if [ "$params" = "0" ]; then
    echo "INFO: VRAM preflight: unknown params for $MODEL_KEY — skipping estimate."
  else
    # ---- weights per GPU (GB) ----
    # ~2.0 GB/B for fp16/bf16 weights; ~0.5 GB/B for 4-bit AWQ (+20% overhead).
    if [ -n "${QUANT:-}" ]; then
      weight_per_gpu=$(awk -v p="$params" -v tp="$TP_SIZE" 'BEGIN{printf "%.1f", (p*0.5*1.2)/tp}')
    else
      weight_per_gpu=$(awk -v p="$params" -v tp="$TP_SIZE" 'BEGIN{printf "%.1f", (p*2.0)/tp}')
    fi

    # ---- KV cache per GPU (GB) heuristic at baseline: 8k ctx, 64 seqs, TP=1, fp8 ----
    # ~120–132B MoE: 12.0 | ~70B: 10.0 | ~32B: 6.0 | ~8–13B: 1.2
    if [ "$params" -ge 120 ]; then
      base_kv=12.0
    elif [ "$params" -ge 60 ]; then
      base_kv=10.0
    elif [ "$params" -ge 20 ]; then
      base_kv=6.0
    else
      base_kv=1.2
    fi

    # dtype multiplier: fp8 ~1.0, bf16/half ~2.0
    kv_dtype_lc=$(echo "${KV_CACHE_DTYPE}" | tr '[:upper:]' '[:lower:]')
    if [ "$kv_dtype_lc" = "fp8" ] || [ "$kv_dtype_lc" = "auto" ]; then
      dtype_mult=1.0
    else
      dtype_mult=2.0
    fi

    # context scaling vs 8k
    scale_len=$(awk -v m="$MODEL_LEN" 'BEGIN{printf "%.2f", (m<=8192)?1.0:m/8192}')
    # gentler seq scaling
    if [ "$MAX_SEQS" -le 64 ]; then
      scale_seqs=1.0
    elif [ "$MAX_SEQS" -le 128 ]; then
      scale_seqs=1.3
    else
      scale_seqs=1.6
    fi
    # TP shards KV across GPUs; baseline is TP=1
    tp_scale=$(awk -v tp="$TP_SIZE" 'BEGIN{printf "%.2f", 1.0/tp}')

    reserve_per_gpu=$(awk -v b="$base_kv" -v dl="$dtype_mult" -v sl="$scale_len" -v ss="$scale_seqs" -v ts="$tp_scale" -v k="$VRAM_HEADROOM_SCALE" \
                      'BEGIN{printf "%.1f", b*dl*sl*ss*ts*k}')

    required_per_gpu=$(awk -v w="$weight_per_gpu" -v r="$reserve_per_gpu" 'BEGIN{printf "%.1f", w+r}')

    # Get minimum free VRAM across selected GPUs
    free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -n 1)
    free_gb=$(awk -v m="$free_mb" 'BEGIN{printf "%.1f", m/1024}')

    echo "VRAM preflight (per GPU with TP=$TP_SIZE):"
    echo "    Weights ~ ${weight_per_gpu} GB | KV ~ ${reserve_per_gpu} GB | Required ~ ${required_per_gpu} GB"
    echo "    Min free VRAM on any GPU: ${free_gb} GB"

    shortfall=$(awk -v req="$required_per_gpu" -v free="$free_gb" 'BEGIN{printf "%.1f", req-free}')
    need=$(awk -v sf="$shortfall" -v m="$SHORTFALL_MARGIN_GB" 'BEGIN{print (sf>m)?1:0}')
    if [ "$need" -eq 1 ]; then
      echo "ERROR: VRAM check failed (shortfall ${shortfall} GB > ${SHORTFALL_MARGIN_GB} GB)."
      echo "    Tips:"
      echo "      - KV_CACHE_DTYPE=fp8 (recommended)"
      echo "      - Reduce MAX_SEQS or MODEL_LEN"
      echo "      - Increase TP size (MORE GPUs reduces per-GPU KV)"
      exit 1
    elif awk "BEGIN{exit !(${shortfall} > 0)}"; then
      echo "WARNING: VRAM looks tight (shortfall ~${shortfall} GB <= margin). Proceeding anyway."
    else
      echo "VRAM looks sufficient across $TP_SIZE GPUs."
    fi
  fi
else
  echo "INFO: VRAM preflight skipped (SAFE_VRAM=0 or nvidia-smi not found)."
fi

# =========================
# START SERVER
# =========================
echo ""
echo "Starting vLLM for Clinical Trial Classification"
echo "=================================================="
echo "    Model:           $MODEL_NAME"
echo "    Served name:     $MODEL_KEY"
echo "    Host/Port:       $HOST:$PORT"
echo "    TP Size:         $TP_SIZE GPUs"
echo "    Quantization:    ${QUANT:-none (FP16/BF16)}"
echo "    DType:           $DT"
echo "    KV cache dtype:  $KV_CACHE_DTYPE"
echo "    Max model len:   $MODEL_LEN tokens"
echo "    Max num seqs:    $MAX_SEQS"
echo "    GPU util:        $GPU_UTIL"
echo "=================================================="
echo ""

CMD=(python3 -m vllm.entrypoints.openai.api_server
  --model "$MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --download-dir "$MODEL_CACHE_DIR"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-num-seqs "$MAX_SEQS"
  --max-model-len "$MODEL_LEN"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --dtype "$DT"
  --served-model-name "$MODEL_KEY"
  --api-key "$OPENAI_API_KEY"
  --enable-chunked-prefill
  --disable-log-requests
)

if [ -n "${QUANT:-}" ]; then
  CMD+=(--quantization "$QUANT")
fi
if [ "${TRC,,}" = "true" ] || [ "$TRC" = "1" ]; then
  CMD+=(--trust-remote-code)
fi

# Export tokens for gated models (especially Llama)
export HF_TOKEN
export OPENAI_API_KEY

echo "OPENAI_API_KEY: $OPENAI_API_KEY"
echo ""
echo "Full command:"
printf ' %q' "${CMD[@]}"; echo
echo ""
echo "Usage example:"
echo "   curl http://$HOST:$PORT/v1/models \\"
echo "     -H \"Authorization: Bearer \$OPENAI_API_KEY\""
echo ""

# Exec replaces the shell so we don't leave zombies
exec "${CMD[@]}"