# CLAUDE.md — LORI (Local Retrieval Intelligence)

## What This Is

Local-first RAG system proving small models (8-30B) + retrieval beat large models without retrieval. Air-gapped operation, no data exfiltration. Based on NeurIPS 2021 Billion-Scale ANN Search Challenge winning architecture (Xeon + Optane + DDR4). A reliable librarian, not a mystical oracle.

## Hardware

**Primary (Dell Precision T7820)** — arriving Monday:
- 1× Xeon Gold 6230 (20C/40T, AVX-512, Cascade Lake)
- NVIDIA V100 32GB HBM2 (SXM2→PCIe adapter, push-pull P8 Max cooling)
- 3× 128GB Intel Optane DCPMM App Direct (384GB persistent)
- 3× 16GB DDR4 RDIMM (48GB)
- 2× 500GB SSD (OS), 1× 2TB HDD (raw corpus)
- 950W PSU

**Staging (Chinese X99 box "compute")** — current preprocessing node:
- E5-2667 v4 (8C/16T)
- 1660 Super (opportunistic compute)
- Linux Mint 22.3
- Tailscale IP: 100.85.113.86
- Cockpit on port 9090, SSH on 22

**Gateway (EliteBook 845 G7)** — future gateway node

**Edge** — old Android phones (future)

## Current State

Wikipedia corpus preprocessed on x99:
- Extracted: ~22GB compressed dump → ~/wiki-extracted (19,357 JSONL files)
- Chunked: ~/wiki-chunked (19,307,247 chunks, section-aware, ~400 words/~520 tokens each)
- Estimated IVF_PQ index: 0.86 GB
- Estimated raw FP32 index: 55.2 GB

Custom extraction and chunking scripts (no WikiExtractor dependency):
- ~/wiki_extract.py — parallel bz2→JSONL extraction, 16 workers
- ~/wiki_chunk.py — section-aware chunking (heading → paragraph → sentence boundaries)

## What's Next (Blocked on Dell + V100)

1. Embed 19.3M chunks using nomic-embed-text-v1.5 (768-dim, GGUF) on V100
2. Build FAISS IVF_PQ index on Optane DAX filesystem
3. Wire RAG pipeline: query → embed → FAISS search → context → llama-server → response
4. Index remaining corpus: Stack Overflow, arXiv, hardware datasheets, Anna's Archive

## Architecture

```
Query → V100 embeds query (<1ms)
      → FAISS searches Optane index (coarse quantizer on GPU, inverted lists on Optane)
      → Top-K chunks returned
      → V100 generates response grounded in retrieved context (25-40 tok/s)
```

Memory hierarchy:
- V100 VRAM (32GB): model weights (~17GB Q4_K_M) + KV cache (~14GB) + FAISS coarse quantizer
- DDR4 (48GB): OS, FAISS query buffers, working memory
- Optane (384GB): persistent FAISS index + chunk text, mmap'd via DAX
- SSD: OS, model GGUFs, staging
- HDD: raw corpus (PDFs, EPUBs, repos, dumps)

## Software Stack

- Ubuntu 24.04 (on Dell), Linux Mint 22.3 (on x99)
- CUDA Toolkit 12.x
- llama.cpp with CUDA backend (LLAMA_CUDA=1)
- FAISS (CPU AVX-512 + faiss-gpu CUDA)
- Qwen3-Coder 30B-A3B (Q4_K_M) — primary model
- nomic-embed-text-v1.5 — embedding model
- FastAPI — query server
- OpenClaw — agentic orchestration (Phase 3)
- Tailscale — mesh networking
- Cockpit — remote admin

## Optane Configuration

App Direct mode (NOT Memory Mode). Explicit control over what lives where.

```bash
ipmctl create -goal PersistentMemoryType=AppDirect
# After reboot:
ndctl create-namespace --mode=fsdax --region=region{0,1,2}
mkfs.ext4 /dev/pmem{0,1,2}
mount -o dax /dev/pmem0 /mnt/optane0  # repeat for 1,2
```

## FAISS Index Design

NeurIPS-validated dual-tier:
- HNSW graph topology → Optane (byte-addressable random access for pointer-chasing)
- Compressed feature vectors → DDR4 (bandwidth for distance computation)
- IVF_PQ mandatory at this corpus scale (~64× compression)
- Optane latency: ~300ns/random access vs NVMe ~100µs (333× advantage for HNSW traversal)

## Chunking Format

JSONL, one object per line:
```json
{"chunk_id": "France::Geography::0", "title": "France", "section": "Geography", "text": "..."}
```

At embedding time, prepend "{title} - {section}: " to text for richer vectors.

## Key Principles

- Retrieval optimization > incremental GPU improvements for this use case
- Beyond a certain corpus size, more storage can degrade performance (search latency, reduced precision)
- KV cache pressure is the major operational risk on V100's 32GB
- KL divergence over logits is optimal drift detection
- ~246 prompts needed for robust golden sets
- HNSW maps well to Optane's byte-addressable random access patterns

## Project Phases

1. **Get It Running** — V100 + CUDA + llama.cpp + Optane DAX ← HERE (waiting on hardware)
2. **Build Knowledge Index** — Wikipedia (done preprocessing), SO, arXiv, datasheets, Anna's Archive
3. **Agentic + Self-Improvement** — OpenClaw, conversational memory (Mem0/Zep), SELF-RAG, ILWS
4. **Quality of Life** — Web UI, CLIP search, voice (Whisper+TTS), Home Assistant, StumbleUpon mode

## Code Style Preferences

- Production-ready, no stubs or placeholders
- Low-level and performant when it matters (C, bare metal), Python for glue/pipelines
- No framework abstractions unless they earn their weight
- Scripts should be parallel by default (this hardware has threads, use them)
- Always include error handling for long-running batch jobs

## Communication Style

- Direct, technical, no preamble
- Assume expert knowledge, skip fundamentals
- Compressed responses, match length to information density
- Flag non-obvious failure modes briefly
- Take your best shot, I'll correct if needed

## Known Gotchas

- Dell T7820 side cover interlock — won't POST with cover removed
- V100 SXM2 adapter cooling: push-pull P8 Max, sealed with HVAC tape, NO fan on top
- PCIe LinkSpeed in BIOS only shows Auto/Gen1/Gen2 — Auto should negotiate Gen3
- "Memory Map IO above 4GB" MUST be enabled for V100's 32GB BAR
- Wiki extraction leaves some ]] artifacts and \n→nn in text (cosmetic, doesn't affect embedding quality much)
- Python 3.12 breaks WikiExtractor regex — we wrote our own scripts instead
