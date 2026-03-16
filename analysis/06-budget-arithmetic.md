# Budget Arithmetic: 3 Core/Specialist Split Scenarios

**System:** strict30 (max_resident_ratio = 0.30)
**Model:** Qwen3.5-35B-A3B (bf16)

---

## Shared Constants

| Parameter | Value |
|---|---|
| full_bf16_gib | 63.418583 |
| always_resident_bytes | 3,670,675,456 (3.418583 GiB) |
| per_expert_bytes | 6,291,456 (6 MiB) |
| num_layers | 40 |
| num_experts_per_layer | 256 |
| candidate_pool_multiplier | 1.25 |
| max_refreshes_per_request | 1 |
| hard_cap_bytes | 20,428,555,468 (19.025575 GiB) |
| **swappable_expert_budget_bytes** | **16,757,880,012 (15.607 GiB)** |
| total_active_expert_capacity | 2,663 experts |

> The swappable budget is identical across all three scenarios. The split only controls how that fixed pie is divided between core (always-loaded) and specialist (hot-swapped) experts.

---

## Side-by-Side Comparison

### Table 1: Budget Bytes

| Metric | A: 75/25 | B: 35/65 | C: 50/50 |
|---|---:|---:|---:|
| core_budget_fraction | 0.75 | 0.35 | 0.50 |
| specialist_budget_fraction | 0.25 | 0.65 | 0.50 |
| core_budget_bytes | 12,568,410,009 | 5,865,258,004 | 8,378,940,006 |
| specialist_budget_bytes | 4,189,470,003 | 10,892,622,008 | 8,378,940,006 |

### Table 2: Per-Layer Expert Targets

| Metric | A: 75/25 | B: 35/65 | C: 50/50 |
|---|---:|---:|---:|
| core_experts_per_layer_target | 49 | 23 | 33 |
| specialist_experts_per_layer_target | 16 | 43 | 33 |
| candidate_experts_per_layer_target | 20 | 54 | 42 |
| **Total active per layer** | **65** | **66** | **66** |
| Active as % of 256 | 25.39% | 25.78% | 25.78% |

### Table 3: Compression & Slice Geometry

| Metric | A: 75/25 | B: 35/65 | C: 50/50 |
|---|---:|---:|---:|
| Effective compression ratio | 74.61% | 74.22% | 74.22% |
| Closest REAP paper tier | ~75% | ~75% | ~75% |
| Per-layer specialist budget (bytes) | 104,736,750 | 272,315,550 | 209,473,500 |
| 8-expert slices fitting per layer | 2.08 | 5.41 | 4.16 |
| Rotation window¹ | 4 | 4 | 4 |
| Max unique signatures (20 late layers)² | 4²⁰ ≈ 1.1T | 4²⁰ ≈ 1.1T | 4²⁰ ≈ 1.1T |

¹ `rotation_window = min(max(2, candidate_target), 4)` — all three scenarios exceed 4, so it's clamped to 4.
² Theoretical max with `rotation_window^num_late_layers`. Assumes 20 late layers (last 50%).

### Table 4: All-Layer Totals

| Metric | A: 75/25 | B: 35/65 | C: 50/50 |
|---|---:|---:|---:|
| Total core experts (40 layers) | 1,960 | 920 | 1,320 |
| Total specialist experts (40 layers) | 640 | 1,720 | 1,320 |
| Total experts (all layers) | 2,600 | 2,640 | 2,640 |
| Total expert bytes | 16,357,785,600 | 16,609,443,840 | 16,609,443,840 |
| Fits in swappable budget? | ✅ | ✅ | ✅ |

---

## Key Insight: Same Total, Different Distribution

The total active experts per layer is essentially the same across all three scenarios (~65-66 out of 256). This is because the total swappable budget is fixed by `max_resident_ratio = 0.30`.

**What changes is the distribution:**

| Scenario | Fixed core | Swappable specialist | Meaning |
|---|---:|---:|---|
| A: 75/25 | 49 | 16 | Heavy core, narrow swap window — stable but rigid |
| B: 35/65 | 23 | 43 | Lean core, wide swap window — adaptive but volatile |
| C: 50/50 | 33 | 33 | Balanced — moderate stability with moderate adaptivity |

- **Core experts** are always resident in GPU memory. They never change between requests. Higher core = more predictable baseline quality, but less ability to specialize per-prompt.
- **Specialist experts** are hot-swapped from the candidate pool based on prompt routing. Higher specialist = more prompt-specific adaptation, but the "baseline" set is thinner.
- **Candidate experts** are the superset from which specialists are selected. With `candidate_pool_multiplier = 1.25`, the candidate pool is 25% larger than the specialist target.

### Why A has 65 instead of 66

Scenario A gets 65 total vs 66 for B and C because of integer floor division:
- A: `core = 12,568,410,009 // 251,658,240 = 49`, `spec = 4,189,470,003 // 251,658,240 = 16` → 65
- B: `core = 5,865,258,004 // 251,658,240 = 23`, `spec = 10,892,622,008 // 251,658,240 = 43` → 66
- C: `core = 8,378,940,006 // 251,658,240 = 33`, `spec = 8,378,940,006 // 251,658,240 = 33` → 66

The denominator is `per_expert_bytes × num_layers = 6,291,456 × 40 = 251,658,240`. The 75/25 split loses 1 expert to floor-division waste.

---

## Scenario Trade-off Analysis

### Scenario A: 75/25 (Current Plan)
- **Pros:** 49 core experts provide a very stable baseline. Minimal variance between requests. Good for consistent quality on common tasks.
- **Cons:** Only 16 specialist slots and 20 candidates per layer. Only ~2 eight-expert slices fit, severely limiting task-specific adaptation. The routing system has very little room to specialize.

### Scenario B: 35/65 (Code Default)
- **Pros:** 43 specialist slots and 54 candidates per layer. ~5.4 eight-expert slices fit, giving the router rich material to work with. Best prompt-specific adaptation.
- **Cons:** Only 23 core experts — the always-on baseline is thin. If routing makes a bad choice, there's less safety net. More sensitive to routing quality.

### Scenario C: 50/50 (Proposed Balance)
- **Pros:** 33/33 split — the "sweet spot." 42 candidates provide good routing diversity (~4.2 slices). 33 core experts still provide a solid baseline.
- **Cons:** Neither the best at stability (A wins) nor at adaptation (B wins). A compromise.

---

## Recommendation

**Scenario C (50/50) is the recommended starting point** for the following reasons:

1. **Slice coverage:** 4.16 eight-expert slices per layer is 2× better than A's 2.08. This gives the router meaningful choice without requiring perfect routing decisions.

2. **Baseline stability:** 33 core experts per layer is 43% more than B's 23. This provides a solid safety net for prompts where routing is uncertain.

3. **Diminishing returns on core:** Going from 33→49 core experts (A) adds only 16 more always-on experts but costs 17 specialist slots — a poor trade when the router is working correctly.

4. **Diminishing returns on specialist:** Going from 33→43 specialists (B) adds 10 more swap slots but removes 10 core experts. The marginal value of the 34th-43rd specialist is lower than the marginal value of the 24th-33rd core expert.

5. **Same compression tier:** All three scenarios land in the ~75% compression tier. There is no accuracy cliff to worry about regardless of split choice.

**If routing quality is proven high** → move toward B (35/65) for maximum adaptation.
**If routing quality is uncertain** → stay at C (50/50) or even A (75/25) for safety.

---

*Generated from `compute_dynamic_budget()` in `reap_swap/dynamic_reap.py` with constants from `assets/strict30-best-plan.json`.*
