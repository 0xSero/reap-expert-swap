# REAP-swap: Router-Weighted Expert Activation Offloading

**Goal:** Run Kimi-K2.5 on hardware that costs ~$10K USD.
**Gear:** 8x RTX 3090 + 512GB DDR4 + EPYC 7443P + ROMED8-2T

## The idea

Can we use REAP observations to reduce VRAM requirements without pruning and with minimal added latency?

REAP (arXiv:2510.13999) captures per-expert activation mass from calibration prompts, then uses that to prune. We don't prune. Instead we use the same activation mass data to decide which MoE experts to keep GPU-resident versus which to leave in CPU memory, accessed through vLLM's UVA (Unified Virtual Addressing) backend.

Given a personalized calibration dataset, we determine which experts a user is most likely to need, then selectively load those experts up-front into VRAM -- distributing them to best utilize available GPU memory.

## Hypothesis

We can reduce VRAM requirements by 75% by selectively deciding which experts to load into VRAM and which to offload to CPU RAM, with:
- No loss in intelligence (no pruning, no quantization)
- Only a ~2x slowdown for prefill and generation vs full-VRAM serving

This rests on four observations:

1. Not all experts are needed for every forward pass
2. Certain workflows reliably trigger certain experts
3. We can determine which experts matter for which workflows using REAP's observation phase
4. vLLM's UVA backend can offload weights onto CPU memory transparently

By pre-loading the experts a user is most likely to need, we minimize costly CPU-to-GPU transfers. The router can still send tokens to any expert -- it just costs more when that expert lives in CPU memory.

## Experiment design

### Independent variables
- Which experts we preload into VRAM
- If and how often we unload and reload experts
- Which experts we swap in and out
- How we distribute experts across GPUs (relevant under tensor parallelism)

### Controlled variables
- Model: Qwen3.5-35B-A3B (primary), Kimi-K2.5-PRISM-REAP-530B-A32B (secondary)
- Calibration dataset: 2 years of personal AI coding sessions via [ai-data-extraction](https://github.com/0xSero/ai-data-extraction) (supports Claude Code, Codex, Cursor, Opencode, Windsurf, Trae, Continue, Gemini). 2,048 samples, ~187K tokens, 16K-token multiturn format.
- Hardware: 8x RTX 3090 + EPYC 7443P + 512GB DDR4 + ROMED8-2T
- Custom patched vLLM + UVA
- 16K token context

### Metrics

Primary:
- Accuracy on 200 samples across 4 benchmarks
- Similarity to BF16 response at temp=0 (statistical matching of tokens and sequences)
- Prefill tokens/s
- Decode tokens/s
- Resident VRAM

Secondary:
- Swap instances
- Errors
- Experts swapped during multi-turn test

### Success criteria (vs BF16 full-VRAM in stock vLLM)

- 95% accuracy
- 35% of BF16 prefill speed
- 50% of BF16 decode speed
- 100% coherence
- 80% similarity

## Paths explored

### Path 1: Multiple model variants -- FAILED

Created a core resident model (20% of experts) plus specialist variants (coding/agentic, world knowledge, roleplay, etc. at ~10% each). Used vLLM's [sleep mode](https://blog.vllm.ai/2025/10/26/sleep-mode.html) to swap between variants, with a secondary router to pick the right variant per request.

Result: <40% accuracy. The static expert sets couldn't handle nuance. For prompts similar to the calibration data, accuracy hit 85%+, but it wouldn't generalize to benchmarks or novel inputs.

### Path 2: Static floor below 50% -- FAILED

REAP proves you can prune up to 50% of experts with minimal loss on standard benchmarks. Can we go lower with a personalized calibration set?

Using the personal coding session corpus, 37% of experts were responsible for 95% of activation mass for Qwen3.5-35B-A3B.

![Expert activation mass distribution](https://hackmd.io/_uploads/Hy7QgC85We.png)

This works in theory, but it's a one-way trip. You lose 63% of the model's experts and fall into repetition loops on novel scenarios -- which for agentic workflows happen constantly.

### Path 3: vLLM + UVA + REAP-driven resident selection -- CURRENT

Instead of pruning experts or creating model variants, use CPU offloading more intelligently via vLLM's UVA backend.

Stock UVA loads whatever fits in VRAM (typically the tail layers of the model) and spills the rest to CPU memory. This is linear and uninformed -- active experts that happen to be in CPU memory introduce massive latency.

REAP-swap replaces that with activation-mass-informed placement: the experts most likely to be needed (based on REAP observations) start GPU-resident; everything else stays accessible via UVA.

This introduces latency in three places:
1. Secondary router overhead (predicting which experts will be needed)
2. Construction + loading of the resident expert set
3. Router misses that fall back to CPU-resident experts

#### What had to be patched in vLLM/UVA

Stock vLLM doesn't support any of this, so the server monkey-patches vLLM's `build_app` and adds:
1. Selecting which weights are GPU-resident at runtime
2. Loading, unloading, and binding expert tensors during runtime
3. Fallback handling for fused tensors

### What destroyed quality (and how we fixed it)

Two techniques that seemed obvious both wrecked output quality:

- **Expert zeroing**: Zeroing the weights of non-resident experts in VRAM. Caused garbled output.
- **Router masking**: Masking router logits with `-inf` for non-active experts. Also destroyed quality.

The working configuration disables both:
```
REAP_SWAP_MASKS_ONLY=1
REAP_ENABLE_ROUTER_MASKS=0
```

With both disabled, all experts remain in the computation graph. The router can still route tokens anywhere. Experts not in VRAM get fetched from CPU via UVA -- slower, but correct.

## Results

### Speed (Qwen3.5-35B-A3B, static preload)

Comparison: stock UVA (vLLM default offloading, 16 layers resident) vs REAP-swap static preload (16 layers, REAP-selected).

| Metric | Stock UVA | REAP-swap | Change |
|--------|-----------|-----------|--------|
| TTFT | 2.85s | 1.59s | **-44%** |
| Prefill | 116.29 tok/s | 133.18 tok/s | **+14.5%** |
| Generation | 14.13 tok/s | 13.94 tok/s | -1.3% (flat) |

Stock UVA resident layers: 24-39 (tail of the model).
REAP-swap resident layers: 1, 2, 3, 4, 5, 6, 24, 25, 26, 27, 28, 29, 30, 31, 38, 39.

The gain comes from putting early layers (which handle most activation mass for coding workloads) on GPU instead of letting vLLM's default linear allocation waste VRAM on low-traffic tail layers.

Speed is still slower than full-VRAM serving. The claim is not "free speed" -- it's "better speed than naive offloading when the model doesn't fit in VRAM at all."

A hotset controller approach was also tested but performed worse: 1.78s TTFT, 120.04 tok/s prefill.

### Quality (dynamic active-set swaps, strict30-v2 plan)

Tested with per-request dynamic active-set selection (30% resident budget):

| Benchmark | Accuracy | Coherence |
|-----------|----------|-----------|
| ARC Challenge | 100% | 100% |
| GSM8K | 100% | 100% |
| HellaSwag | 100% | 100% |
| MMLU | 80% | 100% |
| WinoGrande | 60% | 100% |
| **Overall** | **88%** | **100%** |

- Router miss ratio: 0.0%
- 25/25 test samples produced unique dynamic signatures
- Zero bytes copied at swap time (masks-only mode -- all experts remain accessible via UVA)

### Models tested

- **Qwen3.5-35B-A3B** -- primary benchmarking target for this research
- **Kimi-K2.5-PRISM-REAP-530B-A32B** ([HuggingFace](https://huggingface.co/Ex0bit/Kimi-K2.5-PRISM-REAP-530B-A32B)) -- 50% REAP-pruned Kimi-K2.5 (289GB INT4, ~530B params, 192 routed experts per MoE layer down from 384). Created by Ex0bit using REAP + PRISM pipeline.

## Key insight

The winning configuration lets vLLM's native UVA handle all expert access while REAP-swap controls only which experts start GPU-resident. No experts are removed from the model's computation graph. The router sends tokens wherever it wants -- REAP-swap just makes sure the most likely destinations are already in VRAM.
