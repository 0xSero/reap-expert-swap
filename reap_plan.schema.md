# REAP plan file format

This documents the structure of the JSON plan file consumed by the REAP-swap multiplex server (`vllm_multiplex_server.py`). The plan tells the server which experts to keep GPU-resident for each model layer, and which specialist slices are available for dynamic per-request selection.

The included example is `example/strict30-v2-plan.json` -- a 30% resident budget plan for Qwen3.5-35B-A3B, built from 2,048 personal coding session samples (187K tokens).

## How the plan gets built

1. Run REAP's observation phase over a calibration corpus. This captures per-expert activation mass across all MoE layers -- how much routing weight each expert receives across your workload.
2. A planner script reads the observation outputs and computes:
   - Which experts are "core" (always resident) based on cumulative activation mass
   - Which groups of co-activated experts form specialist slices
   - Budget constraints derived from available VRAM
3. The output is a single JSON file that the server loads at startup.

The observation data for this plan came from [ai-data-extraction](https://github.com/0xSero/ai-data-extraction), which extracts conversation history from AI coding tools (Claude Code, Cursor, Codex, Opencode, Windsurf, Trae, etc.).

## Top-level structure

```
{
  "mode": "dynamic_core_specialist",
  "model": "<model path or identifier>",
  "signalKey": "reap",
  "selectionStrategy": "support_v1",
  "rotationPolicy": "late_prompt_hash",
  "sourceSummaries": [...],
  "budget": {...},
  "perLayer": {...},
  "scorerArtifacts": {...},
  "summary": {...}
}
```

### `mode`

Must be `"dynamic_core_specialist"`. The server rejects any other mode at startup.

### `sourceSummaries`

Array of objects describing the calibration data sources. Each entry has:

| Field | Type | Description |
|-------|------|-------------|
| `label` | string | Short name for this data source (e.g. `"personal"`) |
| `workflow` | string | Identifier for the specific observation run |
| `processedSamples` | int | Number of calibration samples processed |
| `totalTokens` | int | Total tokens across all samples |

### `budget`

Controls how much of the model stays GPU-resident.

| Field | Type | Description |
|-------|------|-------------|
| `max_resident_ratio` | float | Fraction of full model to keep resident (e.g. `0.3` = 30%) |
| `full_bf16_gib` | float | Full model size in GiB at BF16 precision |
| `always_resident_bytes` | int | Non-expert parameters that are always on GPU |
| `swappable_expert_budget_bytes` | int | Total VRAM budget for expert parameters |
| `core_budget_fraction` | float | How much of the expert budget goes to core experts (e.g. `0.75`) |
| `specialist_budget_fraction` | float | Remainder for specialist slices (e.g. `0.25`) |
| `per_expert_bytes` | int | Size of one expert's parameters |
| `total_active_expert_capacity` | int | Max experts that fit in the budget |
| `core_experts_per_layer_target` | int | Target core experts per layer |
| `specialist_experts_per_layer_target` | int | Target specialist experts per layer |
| `max_refreshes_per_request` | int | How many mid-request active-set swaps are allowed |

### `perLayer`

Object keyed by `"layer_0"` through `"layer_N"`. Each layer contains:

| Field | Type | Description |
|-------|------|-------------|
| `rawLayerKey` | string | Original layer index as string |
| `numExperts` | int | Total experts in this layer (e.g. 256 for Qwen3.5-35B-A3B) |
| `coreExperts` | int[] | Expert indices that are always resident |
| `coreActivationMass` | float | Cumulative activation mass covered by core experts |
| `sliceCatalog` | object[] | Available specialist slices (see below) |
| `coreByteCost` | int | VRAM cost of core experts for this layer |
| `specialistBudgetBytes` | int | VRAM available for specialist slices in this layer |

#### Slice catalog entries

Each slice in `sliceCatalog` represents a group of co-activated experts that can be swapped in as a unit:

| Field | Type | Description |
|-------|------|-------------|
| `sliceId` | string | Unique identifier (e.g. `"layer_0_slice_00"`) |
| `experts` | int[] | Expert indices in this slice (typically ~8 experts) |
| `byteCost` | int | VRAM cost to load this slice |
| `activationMass` | float | Weighted activation mass (task-prior adjusted) |
| `rawActivationMass` | float | Raw activation mass from observations |
| `coactivationScore` | float | How often these experts fire together |
| `signalsBySummary` | object | Per-source activation breakdown |
| `taskPriors` | object | Per-task-family activation weights |

### `summary`

Aggregate stats across the full plan:

| Field | Type | Description |
|-------|------|-------------|
| `layerCount` | int | Number of MoE layers (40 for Qwen3.5-35B-A3B) |
| `totalCoreExperts` | int | Sum of core experts across all layers |
| `totalSlices` | int | Sum of specialist slices across all layers |
| `residentFractionPctMin` | float | Minimum resident fraction across layers |
| `residentFractionPctMax` | float | Maximum resident fraction across layers |

### `scorerArtifacts`

Metadata about how slices were scored and selected. Includes the selection strategy configuration, layer importance weights, task family priors, and normalization parameters. This is preserved for reproducibility -- you can re-derive the plan from the same observation data using these settings.
