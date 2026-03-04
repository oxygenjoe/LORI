# CUMRAG: Comparative Utilization Metrics for Retrieval-Augmented Generation

## Project Spec v1.0

### What This Is

A benchmark harness that evaluates RAG faithfulness and groundedness across multiple small language models, quantization levels, and hardware tiers — all running on surplus/consumer hardware. The goal is to produce a publishable comparison matrix showing how model size, quantization, and hardware affect RAG output quality, with special focus on faithfulness (does the model trust retrieved context?) vs parametric override (does it hallucinate from training data instead?).

### Thesis

Smaller, "dumber" models may produce more faithful RAG outputs because they have less parametric knowledge to contradict retrieved context. The hardware running the model matters less than the model's architecture for groundedness metrics. We prove this with rigorous benchmarks on eBay-grade hardware.

---

## Hardware Matrix

| Tier ID | Hardware | VRAM/RAM | Software Stack | Status |
|---------|----------|----------|---------------|--------|
| `v100` | NVIDIA V100 SXM2 32GB (in Dell T7820) | 32GB HBM2 | CUDA, llama.cpp | Incoming |
| `mi25` | AMD MI25 16GB | 16GB HBM | ROCm, llama.cpp | Available |
| `1660s` | NVIDIA 1660 Super 6GB | 6GB GDDR6 | CUDA, llama.cpp | Available |
| `fpga` | Inspur FPGA board + 16GB SODIMM | 16GB DDR4 | TBD (BitNet runtime) | Planned |
| `cpu` | Intel E5-2667v4 + 32GB DDR4 | 32GB DDR4 | llama.cpp CPU | Available |
| `optane` | E5-2667v4 + Optane DCPMM 384GB | 384GB Optane | llama.cpp CPU (weights on Optane) | Available |

---

## Model Matrix

Download all models in GGUF format from HuggingFace (bartowski or official quants).

| Model | Parameters | Quantizations to Test | Priority |
|-------|-----------|----------------------|----------|
| Qwen 2.5 14B Instruct | 14B | Q4_K_M, Q8_0, FP16 | HIGH |
| Phi-4 14B | 14B | Q4_K_M, Q8_0, FP16 | HIGH |
| Mistral NeMo 12B Instruct | 12B | Q4_K_M, Q8_0 | HIGH |
| Llama 3.1 8B Instruct | 8B | Q4_K_M, Q8_0, FP16 | HIGH (baseline) |
| Qwen 2.5 7B Instruct | 7B | Q4_K_M, Q8_0 | MEDIUM |
| Gemma 2 9B Instruct | 9B | Q4_K_M, Q8_0 | MEDIUM |
| GLM-4 9B Chat | 9B | Q4_K_M | MEDIUM |
| SmolLM2 1.7B Instruct | 1.7B | Q8_0, FP16 | LOW (floor test) |
| BitNet 1.58b (TBD model) | 3B | 1.58-bit native | LOW (FPGA only) |

**Model-Hardware feasibility:**
- V100 32GB: All models, all quantizations including 14B FP16
- MI25 16GB: 7-8B Q4/Q8, 14B Q4 (tight)
- 1660 Super 6GB: 3B and under, 7B Q4 (very tight, may OOM)
- FPGA: BitNet only
- CPU/Optane: 7B Q4 primary, 14B Q4 if patient

---

## Evaluation Datasets

Download and place in `datasets/` directory.

| Dataset | Purpose | Source |
|---------|---------|--------|
| RGB (Retrieval-Augmented Generation Benchmark) | Noise robustness, negative rejection, information integration, counterfactual robustness | https://github.com/chen700564/RGB |
| Natural Questions (NQ) | General RAG accuracy with human-labeled answers | Via HuggingFace: `nq_open` |
| HaluEval | Hallucination detection with human labels | https://github.com/RUCAIBox/HaluEval |

**Phase 2 (future):**
- MS MARCO for retrieval quality isolation
- RAGAS synthetic generation for domain-specific eval sets
- Custom clinical lab informatics corpus (CUMRAG-Clinical)

---

## Evaluation Frameworks

Install both. RAGAS for quick baseline scores, RAGCHECKER for the deep diagnostics.

### RAGAS (baseline)
- pip install ragas
- Metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- Judge model: Use Claude API or GPT-4 API as the LLM judge
- Docs: https://docs.ragas.io

### RAGCHECKER (primary)
- GitHub: https://github.com/amazon-science/RAGChecker
- Metrics: Context Utilization (CU), Self-Knowledge (SK), Noise Sensitivity (NS)
- These are the core metrics that test our thesis
- Docs: Read the repo README before implementing

---

## Retrieval Pipeline

Keep retrieval constant across all tests. The retrieval component is NOT the variable — the generator model is.

### Embedding Model
- Use `all-MiniLM-L6-v2` from sentence-transformers (small, fast, runs on CPU)
- Same embedding model for all tests, no exceptions

### Vector Store
- ChromaDB (local, file-based, zero config)
- Pre-compute and persist the vector index once
- All models query the same pre-built index

### Chunking
- Fixed chunk size: 512 tokens
- Overlap: 64 tokens
- Document this in results — chunking strategy affects everything

### Retrieval Parameters
- Top-k: 5 chunks retrieved per query
- No reranking in phase 1 (add as variable in phase 2)

---

## Prompt Template

**CRITICAL: This template must be identical across ALL models and ALL hardware tiers. Do not adjust per-model.**

```
<|system|>
You are a helpful assistant. Answer the user's question using ONLY the provided context. If the context does not contain enough information to answer the question, say "I cannot answer this question based on the provided context." Do not use any knowledge from your training data.

<|context|>
{retrieved_chunks}

<|user|>
{query}

<|assistant|>
```

**Generation parameters (locked across all runs):**
- Temperature: 0.0
- Top-p: 1.0
- Max tokens: 512
- Seed: 42 (where supported)
- Stop sequences: model-appropriate

**Note:** Some models use different chat templates (ChatML, Llama, etc.). The llama.cpp server handles template conversion. The CONTENT of the system/context/user message must remain identical. Only the template wrapper changes per model.

---

## Metrics to Record

For every (model × quantization × hardware) combination, record:

### Quality Metrics (from RAGCHECKER/RAGAS)
- `faithfulness`: Ratio of claims supported by retrieved context
- `context_utilization`: % of retrieved context actually used in answer
- `self_knowledge`: Degree model used parametric memory instead of context
- `noise_sensitivity`: Performance degradation with irrelevant chunks
- `answer_relevancy`: Did the answer address the actual question
- `negative_rejection_rate`: % of unanswerable questions correctly refused

### Performance Metrics
- `tokens_per_second`: Generation throughput
- `time_to_first_token`: Latency in ms
- `total_inference_time`: Wall clock per query in ms
- `vram_usage_mb`: Peak VRAM consumption
- `ram_usage_mb`: Peak system RAM consumption

### Hardware Metadata
- `hardware_tier`: Tier ID from hardware matrix
- `gpu_model`: Exact GPU string
- `driver_version`: CUDA/ROCm version
- `inference_framework`: llama.cpp version/commit hash
- `quantization`: Exact quant format string
- `model_file`: Exact GGUF filename with hash

---

## Output Format

Results stored as JSONL (one JSON object per eval run):

```json
{
  "timestamp": "2026-03-01T14:30:00Z",
  "model": "Qwen2.5-14B-Instruct",
  "quantization": "Q4_K_M",
  "hardware_tier": "v100",
  "dataset": "RGB",
  "eval_subset": "noise_robustness",
  "noise_ratio": 0.4,
  "metrics": {
    "faithfulness": 0.82,
    "context_utilization": 0.71,
    "self_knowledge": 0.15,
    "answer_relevancy": 0.88,
    "tokens_per_second": 45.2,
    "time_to_first_token_ms": 120,
    "vram_usage_mb": 9200
  },
  "hardware_meta": {
    "gpu": "NVIDIA Tesla V100-SXM2-32GB",
    "driver": "CUDA 12.4",
    "framework": "llama.cpp b4567",
    "os": "Ubuntu 24.04"
  }
}
```

---

## Statistical Requirements

- Run each (model × quant × hardware × dataset_subset) combination **minimum 3 times** with different random seeds
- Compute **mean and 95% confidence interval** via bootstrap resampling (1000 resamples)
- Report confidence intervals alongside all point estimates
- Any result with CI width > 15% of the mean gets flagged for additional runs
- Use `scipy.stats` or `numpy` for bootstrap — no custom implementations

---

## What Claude Code Should Build

### Phase 1: Scaffold (now, on x99 box)
1. Create the directory structure (see filesystem layout below)
2. Write dataset download scripts (RGB, NQ, HaluEval)
3. Write model download script (HuggingFace GGUF downloads)
4. Set up ChromaDB vector store and indexing pipeline
5. Build the eval harness runner script that:
   - Loads a specified model via llama.cpp server
   - Runs queries from a specified dataset
   - Records all metrics to JSONL
   - Handles the prompt template injection
6. Build the scoring pipeline that feeds outputs to RAGAS/RAGCHECKER
7. Build the results aggregation script (reads JSONL, computes bootstrap CIs, outputs summary tables)
8. Write the standardized hardware profiling logger

### Phase 2: Run (when hardware arrives)
1. Run eval matrix on each hardware tier
2. Generate comparison tables and charts
3. Build README with results

---

## Filesystem Layout

```
cumrag/
├── README.md
├── requirements.txt
├── setup.sh                          # One-shot environment setup
│
├── config/
│   ├── models.yaml                   # Model registry (names, HF paths, quant formats)
│   ├── hardware.yaml                 # Hardware tier definitions
│   ├── eval_config.yaml              # Dataset paths, retrieval params, generation params
│   └── prompt_template.txt           # The sacred prompt template
│
├── scripts/
│   ├── download_datasets.py          # Fetch RGB, NQ, HaluEval
│   ├── download_models.py            # Fetch GGUF files from HuggingFace
│   ├── build_index.py                # Chunk documents, embed, persist to ChromaDB
│   └── profile_hardware.py           # Log GPU/CPU/RAM specs to hardware.yaml
│
├── cumrag/
│   ├── __init__.py
│   ├── retriever.py                  # ChromaDB query interface
│   ├── generator.py                  # llama.cpp server interface (start/stop/query)
│   ├── evaluator.py                  # RAGAS + RAGCHECKER scoring wrapper
│   ├── runner.py                     # Main eval loop: retrieve → generate → score → log
│   ├── aggregator.py                 # Read JSONL results, bootstrap CIs, output tables
│   └── utils.py                      # Hardware logger, timing utils, JSONL I/O
│
├── datasets/
│   ├── rgb/                          # RGB benchmark data
│   ├── nq/                           # Natural Questions
│   └── halueval/                     # HaluEval
│
├── models/                           # GGUF model files (gitignored, large)
│   ├── qwen2.5-14b-instruct-q4_k_m.gguf
│   ├── qwen2.5-14b-instruct-q8_0.gguf
│   └── ...
│
├── index/                            # Persisted ChromaDB vector store
│
├── results/
│   ├── raw/                          # JSONL files per run
│   ├── aggregated/                   # Summary CSVs with CIs
│   └── figures/                      # Generated charts
│
└── docs/
    ├── METHODOLOGY.md                # Statistical methodology writeup
    ├── HARDWARE.md                   # Hardware descriptions, photos, costs, sourcing
    └── RESULTS.md                    # Auto-generated results summary
```

---

## Setup Instructions for x99 Box

### Step 1: System dependencies
```bash
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip git cmake build-essential
```

### Step 2: Create project
```bash
mkdir -p ~/cumrag && cd ~/cumrag
python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 3: Python dependencies
```bash
pip install --upgrade pip
pip install numpy scipy pandas matplotlib seaborn
pip install chromadb sentence-transformers
pip install ragas langchain langchain-community
pip install huggingface_hub requests pyyaml tqdm
```

### Step 4: Build llama.cpp
```bash
cd ~
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=OFF  # CPU-only on x99, rebuild per-hardware later
cmake --build build --config Release -j$(nproc)
```

### Step 5: Download datasets
```bash
cd ~/cumrag
python scripts/download_datasets.py  # Claude Code builds this
```

### Step 6: Download models (start with smallest for testing)
```bash
python scripts/download_models.py --model llama-3.1-8b-instruct --quant Q4_K_M
```

### Step 7: Build vector index
```bash
python scripts/build_index.py --dataset rgb --chunk-size 512 --overlap 64
```

### Step 8: Smoke test
```bash
python -m cumrag.runner \
  --model models/llama-3.1-8b-instruct-q4_k_m.gguf \
  --dataset rgb \
  --subset noise_robustness \
  --hardware-tier cpu \
  --num-queries 10 \
  --output results/raw/smoke_test.jsonl
```

---

## Notes for Claude Code

- Do NOT use LangChain for the core pipeline. Use it only as a thin wrapper for RAGAS integration if needed. The retrieval and generation pipeline should be direct API calls to ChromaDB and llama.cpp server.
- The llama.cpp server exposes an OpenAI-compatible API at localhost:8080. Use that.
- All file I/O uses JSONL (one JSON per line), not monolithic JSON files.
- The harness must be hardware-agnostic. Hardware-specific config (CUDA vs ROCm vs CPU) is handled at llama.cpp build time and in hardware.yaml, not in the Python code.
- Fail loudly. If a model OOMs, log the failure with the exact error and move to the next combination. Don't crash the entire run.
- Every script should be runnable standalone AND importable as a module.
