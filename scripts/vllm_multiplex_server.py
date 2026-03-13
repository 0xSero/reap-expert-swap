import asyncio
import gc
import importlib
import inspect
import json
import os
import time
from typing import Any

import torch
from compact_offload import (
    build_compact_expert_map,
    normalize_dense_local_to_global,
    resolve_compact_indices,
    summarize_compact_delta,
)
from dynamic_swap_delta import build_dense_keep_sets, compute_keep_set_delta
from multiplex_cache import update_loaded_cartridge_order
from router_activity import (
    aggregate_router_activity_results,
    collapse_topk_router_activity,
    empty_layer_activity,
    merge_layer_activity,
)
from vllm.entrypoints.openai import api_server

from dynamic_reap import summarize_router_misses, validate_active_set_payload

original_build_app = api_server.build_app

loaded_cartridges: list[str] = []
MAX_LOADED_CARTRIDGES = int(os.environ.get("REAP_MAX_LOADED_CARTRIDGES", "4"))
ENABLE_ROUTER_MASKS = os.environ.get("REAP_ENABLE_ROUTER_MASKS", "1") != "0"
ENABLE_COMPACT_ACTIVE_EXPERTS = (
    os.environ.get("REAP_COMPACT_ACTIVE_EXPERTS", "1") != "0"
)
DYNAMIC_CONCURRENCY_MODE = "serialized_single_flight"


def get_layer_plan(plan, layer_key):
    per_layer = plan.get("perLayer", {})
    legacy_layers = plan.get("layers", {})
    candidates = [layer_key]
    if layer_key.startswith("layer_"):
        candidates.append(layer_key.split("layer_", 1)[1])
    else:
        candidates.append(f"layer_{layer_key}")

    for candidate in candidates:
        if candidate in per_layer:
            return per_layer[candidate]
        if candidate in legacy_layers:
            return legacy_layers[candidate]
    return None


def get_keep_experts(layer_plan, cartridge_id):
    if not layer_plan:
        return None
    generic = layer_plan.get("cartridges", {}).get(cartridge_id)
    if generic:
        return generic.get("keep") or generic.get("experts")
    if cartridge_id == "cartridge_a":
        return layer_plan.get("codingKeep") or layer_plan.get("coding_half")
    if cartridge_id == "cartridge_b":
        return layer_plan.get("communicationKeep") or layer_plan.get(
            "communication_half"
        )
    return None


def get_cartridge_ids(plan):
    cartridge_ids = plan.get("summary", {}).get("cartridgeIds")
    if cartridge_ids:
        return cartridge_ids
    per_layer = plan.get("perLayer", {})
    for layer_plan in per_layer.values():
        cartridges = layer_plan.get("cartridges", {})
        if cartridges:
            return sorted(cartridges.keys())
    return ["cartridge_a", "cartridge_b"]


class MultiplexWorkerExtension:
    def _get_base_expert_snapshot(self):
        base_snapshot = getattr(self, "_multiplex_base_experts", None)
        if base_snapshot is not None:
            return base_snapshot

        base_snapshot = {}
        for name, param in self.model_runner.model.named_parameters():
            if "experts" not in name:
                continue
            base_snapshot[name] = param.cpu().clone()
        self._multiplex_base_experts = base_snapshot
        return base_snapshot

    _MAX_ROUTER_STAT_ENTRIES = 16

    def _ensure_router_tracking(self):
        if not hasattr(self, "_reap_router_miss_stats"):
            self._reap_router_miss_stats = {}
            self._reap_router_miss_order: list[str] = []

    def _get_base_layer_expert_snapshot(self):
        snapshot = getattr(self, "_reap_base_layer_expert_snapshot", None)
        if snapshot is not None:
            return snapshot
        snapshot = {}
        layers = self._resolve_model_layers()
        if layers is None:
            self._reap_base_layer_expert_snapshot = snapshot
            return snapshot
        local_to_global_by_layer = self._get_layer_local_global_maps()
        for layer_idx, layer in enumerate(layers):
            moe = getattr(getattr(layer, "mlp", None), "experts", None)
            if moe is None or not hasattr(moe, "w13_weight") or not hasattr(moe, "w2_weight"):
                continue
            dense_local_count = int(moe.w13_weight.shape[0])
            dense_local_to_global = normalize_dense_local_to_global(
                local_to_global_by_layer.get(int(layer_idx), {}),
                dense_local_count,
            )
            snapshot[int(layer_idx)] = {
                "w13_weight": moe.w13_weight.detach().cpu().clone().pin_memory(),
                "w2_weight": moe.w2_weight.detach().cpu().clone().pin_memory(),
                "w13_bias": moe.w13_bias.detach().cpu().clone().pin_memory()
                if hasattr(moe, "w13_bias")
                else None,
                "w2_bias": moe.w2_bias.detach().cpu().clone().pin_memory()
                if hasattr(moe, "w2_bias")
                else None,
                "dense_local_to_global": dense_local_to_global,
                "global_num_experts": int(
                    getattr(moe, "global_num_experts", max(dense_local_to_global.values(), default=-1) + 1)
                ),
                "logical_num_experts": int(
                    getattr(moe, "logical_num_experts", max(dense_local_to_global.values(), default=-1) + 1)
                ),
            }
        self._reap_base_layer_expert_snapshot = snapshot
        return snapshot

    def _evict_old_router_stats(self, current_request_id: str | None = None):
        order = getattr(self, "_reap_router_miss_order", [])
        stats = self._reap_router_miss_stats
        max_entries = int(getattr(self, "_MAX_ROUTER_STAT_ENTRIES", 16))
        while len(order) > max_entries:
            oldest = order.pop(0)
            if oldest != current_request_id:
                stats.pop(oldest, None)
        self._reap_router_miss_order = order

    def _resolve_model_layers(self):
        model = self.model_runner.model
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers
        return None

    def _apply_router_masks_and_hooks(
        self,
        layer_keep_sets: dict[int, set[int]],
        *,
        request_id: str | None,
    ) -> int:
        masks_applied = 0
        if not ENABLE_ROUTER_MASKS:
            return 0
        layers = self._resolve_model_layers()
        if layers is None:
            return 0
        self._ensure_router_tracking()

        for layer_idx, keep_set in layer_keep_sets.items():
            if layer_idx >= len(layers):
                continue
            layer = layers[layer_idx]
            if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "gate"):
                continue
            gate = layer.mlp.gate
            local_to_global = self._get_layer_local_global_maps().get(int(layer_idx), {})
            gate_weight = None
            for pname, p in gate.named_parameters():
                if "weight" in pname:
                    gate_weight = p
                    break
            if gate_weight is None:
                continue
            num_experts = gate_weight.shape[0]
            mask = torch.zeros(
                num_experts, dtype=torch.float32, device=gate_weight.device
            )
            for expert_idx in range(num_experts):
                global_expert_idx = int(local_to_global.get(int(expert_idx), int(expert_idx)))
                if global_expert_idx not in keep_set:
                    mask[expert_idx] = float("-inf")

            gate._reap_expert_mask = mask.to(gate_weight.dtype)
            gate._reap_active_keep_set = set(int(idx) for idx in keep_set)
            gate._reap_active_request_id = request_id
            gate._reap_layer_idx = int(layer_idx)
            gate._reap_local_to_global = {
                int(local_idx): int(global_idx)
                for local_idx, global_idx in local_to_global.items()
            }
            gate._reap_router_miss_stats = self._reap_router_miss_stats
            if not getattr(gate, "_reap_hook_installed", False):

                def make_hook():
                    def hook(module, _input, output):
                        mask = getattr(module, "_reap_expert_mask", None)
                        keep_set = getattr(module, "_reap_active_keep_set", set())
                        request_id_local = getattr(module, "_reap_active_request_id", None)
                        layer_idx_local = getattr(module, "_reap_layer_idx", None)
                        local_to_global_local = getattr(module, "_reap_local_to_global", {})
                        router_stats = getattr(module, "_reap_router_miss_stats", None)
                        logits = output[0] if isinstance(output, tuple) else output
                        if (
                            request_id_local
                            and router_stats is not None
                            and layer_idx_local is not None
                            and logits is not None
                            and logits.ndim >= 2
                        ):
                            top_k = int(
                                getattr(module, "top_k", getattr(module, "num_experts_per_tok", 2))
                            )
                            top_k = max(1, min(top_k, logits.shape[-1]))
                            top_values, top_indices = torch.topk(
                                logits.detach().float(), k=top_k, dim=-1
                            )
                            top_probs = torch.softmax(top_values, dim=-1)
                            top_indices_cpu = top_indices.detach().cpu().tolist()
                            top_probs_cpu = top_probs.detach().cpu().tolist()
                            layer_payload = collapse_topk_router_activity(
                                top_indices_cpu,
                                top_probs_cpu,
                                keep_set=set(int(idx) for idx in keep_set),
                                local_to_global=local_to_global_local,
                            )
                            request_stats = router_stats.setdefault(
                                request_id_local, {"by_layer": {}}
                            )
                            layer_key = f"layer_{layer_idx_local}"
                            layer_stats = request_stats["by_layer"].setdefault(
                                layer_key,
                                empty_layer_activity(),
                            )
                            merge_layer_activity(layer_stats, layer_payload)
                        if mask is None:
                            return output
                        if isinstance(output, tuple):
                            masked_logits = output[0] + mask.unsqueeze(0)
                            return (masked_logits,) + output[1:]
                        return output + mask.unsqueeze(0)

                    return hook

                gate.register_forward_hook(make_hook())
                gate._reap_hook_installed = True

            masks_applied += 1
        return masks_applied

    def _layer_keep_sets_from_cartridge(self, cartridge_id, plan):
        layer_keep_sets: dict[int, set[int]] = {}
        if not plan:
            return layer_keep_sets
        per_layer = plan.get("perLayer", {})
        for layer_key in per_layer:
            try:
                layer_idx = int(layer_key.replace("layer_", ""))
                layer_plan = per_layer[layer_key]
                keep_experts = get_keep_experts(layer_plan, cartridge_id)
                if keep_experts is not None:
                    layer_keep_sets[layer_idx] = set(int(idx) for idx in keep_experts)
            except (ValueError, KeyError):
                pass
        return layer_keep_sets

    def _layer_keep_sets_from_active_set(self, payload: dict[str, Any]):
        out: dict[int, set[int]] = {}
        for layer_key, experts in (payload.get("active_set") or {}).items():
            try:
                layer_idx = int(str(layer_key).replace("layer_", ""))
            except ValueError:
                continue
            out[layer_idx] = set(int(expert) for expert in experts)
        return out

    def _get_dense_keep_sets(self, plan: dict[str, Any]) -> dict[int, set[int]]:
        cached = getattr(self, "_reap_dense_keep_sets", None)
        if cached is None:
            cached = build_dense_keep_sets(plan)
            self._reap_dense_keep_sets = cached
        return {layer_idx: set(experts) for layer_idx, experts in cached.items()}

    def _get_current_keep_sets(self, plan: dict[str, Any]) -> dict[int, set[int]]:
        cached = getattr(self, "_reap_current_keep_sets", None)
        if cached is None:
            cached = self._get_dense_keep_sets(plan)
            self._reap_current_keep_sets = cached
        return {layer_idx: set(experts) for layer_idx, experts in cached.items()}

    def _set_current_keep_sets(self, keep_sets: dict[int, set[int]]) -> None:
        self._reap_current_keep_sets = {
            int(layer_idx): set(int(expert) for expert in experts)
            for layer_idx, experts in keep_sets.items()
        }

    def _get_layer_local_global_maps(self) -> dict[int, dict[int, int]]:
        cached = getattr(self, "_reap_local_to_global_by_layer", None)
        if cached is not None:
            return cached
        cached = {}
        layers = self._resolve_model_layers()
        if layers is None:
            self._reap_local_to_global_by_layer = cached
            return cached
        for layer_idx, layer in enumerate(layers):
            mapping: dict[int, int] = {}
            try:
                expert_layer = layer.mlp.experts
                expert_map = getattr(expert_layer, "_expert_map", None)
                if expert_map is not None:
                    for global_idx, mapped_local_idx in enumerate(expert_map.tolist()):
                        if mapped_local_idx >= 0:
                            mapping[int(mapped_local_idx)] = int(global_idx)
            except Exception:
                mapping = {}
            cached[layer_idx] = mapping
        self._reap_local_to_global_by_layer = cached
        return cached

    def _compact_active_set_on_gpu(self, layer_keep_sets: dict[int, set[int]]):
        layers = self._resolve_model_layers()
        if layers is None:
            return {
                "bytes_copied": 0,
                "bytes_zeroed": 0,
                "bytes_touched": 0,
                "zeroed_tensors": 0,
                "masks_applied": 0,
                "active_layer_count": 0,
                "delta": {
                    "added_expert_total": 0,
                    "removed_expert_total": 0,
                    "reused_expert_total": 0,
                },
            }
        base_snapshot = self._get_base_layer_expert_snapshot()
        if not base_snapshot:
            return {
                "bytes_copied": 0,
                "bytes_zeroed": 0,
                "bytes_touched": 0,
                "zeroed_tensors": 0,
                "masks_applied": 0,
                "active_layer_count": 0,
                "delta": {
                    "added_expert_total": 0,
                    "removed_expert_total": 0,
                    "reused_expert_total": 0,
                },
            }

        local_to_global_cache = self._get_layer_local_global_maps()
        bytes_copied = 0
        bytes_zeroed = 0
        zero_count = 0
        delta_totals = {
            "added_expert_total": 0,
            "removed_expert_total": 0,
            "reused_expert_total": 0,
        }
        with torch.no_grad():
            for layer_idx, keep_set in layer_keep_sets.items():
                if layer_idx >= len(layers):
                    continue
                layer = layers[layer_idx]
                moe = getattr(getattr(layer, "mlp", None), "experts", None)
                layer_snapshot = base_snapshot.get(int(layer_idx))
                if moe is None or layer_snapshot is None:
                    continue

                dense_local_to_global = layer_snapshot["dense_local_to_global"]
                dense_local_count = len(dense_local_to_global)
                selected_globals, dense_local_indices = resolve_compact_indices(
                    dense_local_to_global,
                    dense_local_count,
                    keep_set,
                )
                if not selected_globals:
                    raise RuntimeError(f"compact offload produced empty expert set for layer {layer_idx}")

                previous_globals = getattr(moe, "_reap_compact_selected_globals", None)
                delta = summarize_compact_delta(previous_globals, selected_globals)
                for key, value in delta.items():
                    delta_totals[key] += int(value)

                if tuple(selected_globals) == tuple(previous_globals or ()):
                    local_to_global_cache[int(layer_idx)] = {
                        int(local_idx): int(global_idx)
                        for local_idx, global_idx in enumerate(selected_globals)
                    }
                    continue

                index_tensor = torch.tensor(dense_local_indices, dtype=torch.long)

                def _slice_to_device(source_cpu: torch.Tensor | None, current_param: torch.nn.Parameter | None):
                    nonlocal bytes_copied, bytes_zeroed, zero_count
                    if source_cpu is None or current_param is None:
                        return None
                    old_bytes = current_param.numel() * current_param.element_size()
                    compact_cpu = source_cpu.index_select(0, index_tensor).contiguous()
                    compact_gpu = compact_cpu.to(device=current_param.device, non_blocking=False)
                    bytes_copied += compact_gpu.numel() * compact_gpu.element_size()
                    if old_bytes > compact_gpu.numel() * compact_gpu.element_size():
                        bytes_zeroed += old_bytes - (compact_gpu.numel() * compact_gpu.element_size())
                    zero_count += max(0, int(source_cpu.shape[0]) - len(dense_local_indices))
                    return compact_gpu

                new_w13 = _slice_to_device(layer_snapshot["w13_weight"], getattr(moe, "w13_weight", None))
                new_w2 = _slice_to_device(layer_snapshot["w2_weight"], getattr(moe, "w2_weight", None))
                if new_w13 is None or new_w2 is None:
                    continue
                moe.w13_weight.data = new_w13
                moe.w2_weight.data = new_w2
                if hasattr(moe, "w13_bias"):
                    new_w13_bias = _slice_to_device(layer_snapshot["w13_bias"], getattr(moe, "w13_bias", None))
                    if new_w13_bias is not None:
                        moe.w13_bias.data = new_w13_bias
                if hasattr(moe, "w2_bias"):
                    new_w2_bias = _slice_to_device(layer_snapshot["w2_bias"], getattr(moe, "w2_bias", None))
                    if new_w2_bias is not None:
                        moe.w2_bias.data = new_w2_bias

                expert_map_values = build_compact_expert_map(
                    int(layer_snapshot["global_num_experts"]),
                    selected_globals,
                )
                expert_map_tensor = torch.tensor(
                    expert_map_values,
                    dtype=torch.int32,
                    device=new_w13.device,
                )
                if "expert_map" in getattr(moe, "_buffers", {}):
                    moe._buffers["expert_map"] = expert_map_tensor
                else:
                    moe.__dict__["expert_map"] = expert_map_tensor

                if "expert_mask" in getattr(moe, "_buffers", {}) or getattr(moe, "expert_mask", None) is not None:
                    expert_mask = torch.zeros(
                        (int(layer_snapshot["global_num_experts"]) + getattr(moe, "num_fused_shared_experts", 0) + 1,),
                        dtype=torch.int32,
                        device=new_w13.device,
                    )
                    expert_mask[-1] = 0
                    expert_mask[: int(layer_snapshot["global_num_experts"])] = (
                        expert_map_tensor > -1
                    ).to(torch.int32)
                    if getattr(moe, "num_fused_shared_experts", 0):
                        start = int(layer_snapshot["global_num_experts"])
                        end = start + int(getattr(moe, "num_fused_shared_experts", 0))
                        expert_mask[start:end] = 1
                    if "expert_mask" in getattr(moe, "_buffers", {}):
                        moe._buffers["expert_mask"] = expert_mask
                    else:
                        moe.__dict__["expert_mask"] = expert_mask

                moe.local_num_experts = len(selected_globals)
                if hasattr(moe, "moe_config"):
                    moe.moe_config.num_local_experts = len(selected_globals)
                moe._reap_compact_selected_globals = tuple(selected_globals)
                local_to_global_cache[int(layer_idx)] = {
                    int(local_idx): int(global_idx)
                    for local_idx, global_idx in enumerate(selected_globals)
                }

        self._reap_local_to_global_by_layer = local_to_global_cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        masks_applied = self._apply_router_masks_and_hooks(layer_keep_sets, request_id=None)
        return {
            "bytes_copied": bytes_copied,
            "bytes_zeroed": bytes_zeroed,
            "bytes_touched": bytes_copied + bytes_zeroed,
            "zeroed_tensors": zero_count,
            "masks_applied": masks_applied,
            "active_layer_count": len(layer_keep_sets),
            "delta": delta_totals,
            "compact_offload": True,
        }

    def multiplex_load_cartridge(self, cartridge_id, plan):
        local_cartridges = getattr(self, "_multiplex_cartridges", {})
        cartridge_masks = getattr(self, "_multiplex_cartridge_masks", {})
        base_snapshot = self._get_base_expert_snapshot()
        cartridge = {}
        zero_count = 0
        layer_to_local_global = {}

        def global_expert_index(layer_idx: int, local_expert_idx: int) -> int:
            if layer_idx not in layer_to_local_global:
                try:
                    expert_layer = self.model_runner.model.model.layers[layer_idx].mlp.experts
                    expert_map = getattr(expert_layer, "_expert_map", None)
                    if expert_map is not None:
                        mapping = {}
                        for global_idx, mapped_local_idx in enumerate(expert_map.tolist()):
                            if mapped_local_idx >= 0:
                                mapping[int(mapped_local_idx)] = int(global_idx)
                        layer_to_local_global[layer_idx] = mapping
                    else:
                        layer_to_local_global[layer_idx] = {}
                except Exception:
                    layer_to_local_global[layer_idx] = {}
            return layer_to_local_global[layer_idx].get(local_expert_idx, local_expert_idx)

        layer_keep_sets = self._layer_keep_sets_from_cartridge(cartridge_id, plan)

        for name, base_param in base_snapshot.items():
            cloned_param = base_param.clone()
            if plan:
                parts = name.split(".")
                try:
                    layer_idx = int(parts[parts.index("layers") + 1])
                    keep_experts = layer_keep_sets.get(layer_idx)
                    if keep_experts is not None and cloned_param.ndim > 0:
                        local_expert_dim = cloned_param.shape[0]
                        for local_expert_idx in range(local_expert_dim):
                            global_expert_idx = global_expert_index(layer_idx, local_expert_idx)
                            if global_expert_idx not in keep_experts:
                                cloned_param[local_expert_idx].zero_()
                                zero_count += 1
                except (ValueError, IndexError, RuntimeError):
                    pass

            cartridge[name] = cloned_param.pin_memory()

        local_cartridges[cartridge_id] = cartridge
        cartridge_masks[cartridge_id] = layer_keep_sets
        self._multiplex_cartridges = local_cartridges
        self._multiplex_cartridge_masks = cartridge_masks
        return {
            "rank": getattr(self, "rank", 0),
            "tensor_count": len(cartridge),
            "zeroed_tensors": zero_count,
            "masked_layers": len(layer_keep_sets),
        }

    def multiplex_swap_cartridge(self, cartridge_id):
        local_cartridges = getattr(self, "_multiplex_cartridges", {})
        if cartridge_id not in local_cartridges:
            raise RuntimeError(
                f"Cartridge {cartridge_id} not loaded on rank {getattr(self, 'rank', 0)}"
            )

        cartridge = local_cartridges[cartridge_id]
        cartridge_masks = getattr(self, "_multiplex_cartridge_masks", {})
        layer_keep_sets = cartridge_masks.get(cartridge_id, {})

        bytes_copied = 0
        with torch.no_grad():
            for name, param in self.model_runner.model.named_parameters():
                if name in cartridge:
                    source = cartridge[name]
                    if param.shape != source.shape:
                        raise RuntimeError(
                            f"Shape mismatch during swap for {name}: param={tuple(param.shape)} source={tuple(source.shape)}"
                        )
                    if param.dtype != source.dtype:
                        raise RuntimeError(
                            f"Dtype mismatch during swap for {name}: param={param.dtype} source={source.dtype}"
                        )
                    try:
                        param.data.copy_(source, non_blocking=False)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Swap copy failed for {name} on rank {getattr(self, 'rank', 0)} device={param.device} shape={tuple(param.shape)} dtype={param.dtype}"
                        ) from exc
                    bytes_copied += param.numel() * param.element_size()

        masks_applied = self._apply_router_masks_and_hooks(
            layer_keep_sets,
            request_id=None,
        )
        torch.cuda.synchronize()
        return {
            "rank": getattr(self, "rank", 0),
            "bytes_copied": bytes_copied,
            "masks_applied": masks_applied,
        }

    def multiplex_swap_active_set(self, payload: dict[str, Any], plan: dict[str, Any]):
        validated = validate_active_set_payload(payload, plan)
        if ENABLE_COMPACT_ACTIVE_EXPERTS:
            layer_keep_sets = self._layer_keep_sets_from_active_set(validated)
            request_id = str(validated["request_id"])
            self._ensure_router_tracking()
            self._evict_old_router_stats(request_id)
            if request_id not in self._reap_router_miss_stats:
                self._reap_router_miss_stats[request_id] = {"by_layer": {}}
                self._reap_router_miss_order.append(request_id)
            result = self._compact_active_set_on_gpu(layer_keep_sets)
            self._set_current_keep_sets(layer_keep_sets)
            return {
                "rank": getattr(self, "rank", 0),
                "request_id": request_id,
                **result,
            }
        base_snapshot = self._get_base_expert_snapshot()
        layer_keep_sets = self._layer_keep_sets_from_active_set(validated)
        request_id = str(validated["request_id"])
        dense_keep_sets = self._get_dense_keep_sets(plan)
        current_keep_sets = self._get_current_keep_sets(plan)
        keep_set_delta = compute_keep_set_delta(
            current_keep_sets=current_keep_sets,
            desired_keep_sets=layer_keep_sets,
            dense_keep_sets=dense_keep_sets,
        )
        layer_delta_by_idx = {
            int(str(layer_key).replace("layer_", "")): delta
            for layer_key, delta in keep_set_delta["by_layer"].items()
        }
        layer_to_local_global = self._get_layer_local_global_maps()

        zero_count = 0
        bytes_copied = 0
        bytes_zeroed = 0
        with torch.no_grad():
            for name, param in self.model_runner.model.named_parameters():
                if "experts" not in name:
                    continue
                parts = name.split(".")
                try:
                    layer_idx = int(parts[parts.index("layers") + 1])
                except (ValueError, IndexError):
                    layer_idx = None
                if layer_idx is None:
                    continue
                source = base_snapshot[name]
                if param.shape != source.shape:
                    raise RuntimeError(
                        f"Shape mismatch during active set swap for {name}: param={tuple(param.shape)} source={tuple(source.shape)}"
                    )
                if param.dtype != source.dtype:
                    raise RuntimeError(
                        f"Dtype mismatch during active set swap for {name}: param={param.dtype} source={source.dtype}"
                    )
                if source.ndim <= 0:
                    continue
                layer_delta = layer_delta_by_idx.get(layer_idx, {})
                added = set(int(expert) for expert in layer_delta.get("added", []))
                removed = set(int(expert) for expert in layer_delta.get("removed", []))
                if not added and not removed:
                    continue
                local_to_global = layer_to_local_global.get(layer_idx, {})
                local_expert_dim = source.shape[0]
                for local_expert_idx in range(local_expert_dim):
                    global_expert_idx = local_to_global.get(local_expert_idx, local_expert_idx)
                    slice_bytes = source[local_expert_idx].numel() * source[local_expert_idx].element_size()
                    if global_expert_idx in removed:
                        param.data[local_expert_idx].zero_()
                        zero_count += 1
                        bytes_zeroed += slice_bytes
                    elif global_expert_idx in added:
                        param.data[local_expert_idx].copy_(
                            source[local_expert_idx].to(param.device),
                            non_blocking=False,
                        )
                        bytes_copied += slice_bytes

        self._ensure_router_tracking()
        self._evict_old_router_stats(request_id)
        if request_id not in self._reap_router_miss_stats:
            self._reap_router_miss_stats[request_id] = {"by_layer": {}}
            self._reap_router_miss_order.append(request_id)
        masks_applied = self._apply_router_masks_and_hooks(
            layer_keep_sets,
            request_id=request_id,
        )
        self._set_current_keep_sets(layer_keep_sets)
        torch.cuda.synchronize()
        return {
            "rank": getattr(self, "rank", 0),
            "request_id": request_id,
            "bytes_copied": bytes_copied,
            "bytes_zeroed": bytes_zeroed,
            "bytes_touched": bytes_copied + bytes_zeroed,
            "zeroed_tensors": zero_count,
            "masks_applied": masks_applied,
            "active_layer_count": len(layer_keep_sets),
            "delta": keep_set_delta["summary"],
        }

    def multiplex_get_router_misses(self, request_id: str, reset: bool = False):
        self._ensure_router_tracking()
        payload = self._reap_router_miss_stats.get(request_id, {"by_layer": {}})
        result = {
            "rank": getattr(self, "rank", 0),
            "request_id": request_id,
            "payload": payload,
            "summary": summarize_router_misses(payload),
        }
        if reset:
            self._reap_router_miss_stats.pop(request_id, None)
            order = getattr(self, "_reap_router_miss_order", [])
            if request_id in order:
                order.remove(request_id)
        return result

    def multiplex_reset_router_misses(self, request_id: str):
        self._ensure_router_tracking()
        self._reap_router_miss_stats[request_id] = {"by_layer": {}}
        return {"rank": getattr(self, "rank", 0), "request_id": request_id, "reset": True}

    def multiplex_unload_cartridge(self, cartridge_id):
        local_cartridges = getattr(self, "_multiplex_cartridges", {})
        cartridge_masks = getattr(self, "_multiplex_cartridge_masks", {})
        removed = cartridge_id in local_cartridges
        if removed:
            del local_cartridges[cartridge_id]
        if cartridge_id in cartridge_masks:
            del cartridge_masks[cartridge_id]
        self._multiplex_cartridges = local_cartridges
        self._multiplex_cartridge_masks = cartridge_masks
        gc.collect()
        return {
            "rank": getattr(self, "rank", 0),
            "removed": removed,
            "remaining_cartridges": sorted(local_cartridges.keys()),
        }


def _install_worker_extensions() -> None:
    candidate_specs = [
        ("vllm.v1.worker.gpu_worker", "Worker"),
        ("vllm.v1.worker.cpu_worker", "CPUWorker"),
        ("vllm.v1.worker.xpu_worker", "XPUWorker"),
    ]
    extension_methods = {
        name: member
        for name, member in inspect.getmembers(MultiplexWorkerExtension, predicate=inspect.isfunction)
        if not name.startswith("__")
    }
    installed = []
    for module_name, class_name in candidate_specs:
        try:
            module = importlib.import_module(module_name)
            worker_cls = getattr(module, class_name, None)
            if worker_cls is None:
                continue
            for method_name, method in extension_methods.items():
                setattr(worker_cls, method_name, method)
            installed.append(f"{module_name}.{class_name}")
        except Exception as exc:
            print(f"WARNING: failed to install REAP worker extensions on {module_name}.{class_name}: {exc}")
    if installed:
        print(f"Installed REAP worker extensions on: {', '.join(installed)}")
    else:
        print("WARNING: no worker classes were patched with REAP worker extensions")


def aggregate_router_miss_results(worker_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer = aggregate_router_activity_results(worker_results)
    payload = {
        "by_layer": {
            layer_key: {
                "inactive_mass": layer_row.get("inactive_mass", 0.0),
                "observed_mass": layer_row.get("observed_mass", 0.0),
                "inactive_experts": layer_row.get("inactive_experts", []),
            }
            for layer_key, layer_row in by_layer.items()
        }
    }
    return {
        "by_layer": by_layer,
        "summary": summarize_router_misses(payload),
    }


MAX_DYNAMIC_REQUEST_STATE_ENTRIES = 64


def _evict_dynamic_request_state(state_dict: dict[str, Any]) -> None:
    if len(state_dict) <= MAX_DYNAMIC_REQUEST_STATE_ENTRIES:
        return
    excess = len(state_dict) - MAX_DYNAMIC_REQUEST_STATE_ENTRIES
    for key in list(state_dict.keys())[:excess]:
        del state_dict[key]


def build_app_with_swap(args, supported_tasks=None):
    app = original_build_app(args, supported_tasks)
    app.state.event_loop = asyncio.get_running_loop()
    app.state.dynamic_request_state = {}
    app.state.dynamic_request_signatures = {}
    app.state.dynamic_active_signature = None
    app.state.dynamic_swap_lock = asyncio.Lock()
    app.state.dynamic_concurrency_mode = DYNAMIC_CONCURRENCY_MODE
    plan_file = os.environ.get("REAP_PLAN_FILE")
    if not plan_file:
        raise RuntimeError(
            "REAP_PLAN_FILE must be set for dynamic runtime; startup aborted before readiness"
        )
    if not os.path.exists(plan_file):
        raise RuntimeError(
            f"REAP_PLAN_FILE path does not exist: {plan_file}; startup aborted before readiness"
        )
    print(f"Loading REAP plan from {plan_file}...")
    try:
        with open(plan_file) as f:
            app.state.reap_plan = json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"REAP_PLAN_FILE is not valid JSON: {plan_file}; startup aborted before readiness"
        ) from exc
    plan = app.state.reap_plan
    if plan.get("mode") != "dynamic_core_specialist":
        raise RuntimeError(
            f"REAP_PLAN_FILE mode must be dynamic_core_specialist, got: {plan.get('mode')!r}; startup aborted before readiness"
        )
    if not isinstance(plan.get("budget"), dict) or not plan["budget"]:
        raise RuntimeError(
            "REAP_PLAN_FILE missing non-empty budget object; startup aborted before readiness"
        )
    if not isinstance(plan.get("perLayer"), dict) or not plan["perLayer"]:
        raise RuntimeError(
            "REAP_PLAN_FILE missing non-empty perLayer object; startup aborted before readiness"
        )
    for layer_key, layer in plan["perLayer"].items():
        if not isinstance(layer, dict):
            raise RuntimeError(
                f"REAP_PLAN_FILE layer {layer_key} is not an object; startup aborted before readiness"
            )
        if not isinstance(layer.get("coreExperts"), list):
            raise RuntimeError(
                f"REAP_PLAN_FILE layer {layer_key} missing coreExperts list; startup aborted before readiness"
            )
        if not isinstance(layer.get("sliceCatalog"), list):
            raise RuntimeError(
                f"REAP_PLAN_FILE layer {layer_key} missing sliceCatalog list; startup aborted before readiness"
            )

    def build_plan_identity() -> dict[str, Any]:
        budget = plan.get("budget") if isinstance(plan, dict) else {}
        return {
            "plan_mode": plan.get("mode") if isinstance(plan, dict) else None,
            "plan_path": plan_file,
            "swappable_expert_budget_bytes": int(
                (budget or {}).get("swappable_expert_budget_bytes", 0)
            ),
            "max_refreshes_per_request": int(
                (budget or {}).get("max_refreshes_per_request", 1)
            ),
        }

    @app.post("/swap_cartridge/{cartridge_id}")
    async def swap_cartridge(cartridge_id: str):
        engine = app.state.engine_client
        plan = getattr(app.state, "reap_plan", None)
        cache_update = update_loaded_cartridge_order(
            loaded_cartridges,
            cartridge_id,
            max_loaded=MAX_LOADED_CARTRIDGES,
        )
        for evicted_cartridge in cache_update["evicted"]:
            print(f"Evicting {evicted_cartridge} from pinned CPU cache...")
            unload_results = await engine.collective_rpc(
                "multiplex_unload_cartridge",
                args=(evicted_cartridge,),
                timeout=600,
            )
            removed_workers = sum(1 for result in unload_results if result.get("removed"))
            print(
                f"Evicted {evicted_cartridge} across {removed_workers} workers (cache limit {MAX_LOADED_CARTRIDGES})."
            )
        loaded_cartridges[:] = cache_update["order"]
        if not cache_update["already_loaded"]:
            print(f"Lazy-loading {cartridge_id} into pinned CPU memory...")
            load_results = await engine.collective_rpc(
                "multiplex_load_cartridge",
                args=(cartridge_id, plan),
                timeout=1800,
            )
            tensor_count = sum(result["tensor_count"] for result in load_results)
            zero_count = sum(result["zeroed_tensors"] for result in load_results)
            masks_count = sum(result.get("masked_layers", 0) for result in load_results)
            print(
                f"Loaded {cartridge_id} across {len(load_results)} workers ({tensor_count} pinned tensors, {zero_count} zeroed expert tensors, {masks_count} layers with router masks prepared)."
            )

        start_time = time.time()
        print(
            f"Swapping to {cartridge_id} (live copy, no sleep, router_masks={ENABLE_ROUTER_MASKS})..."
        )

        swap_start = time.time()
        swap_results = await engine.collective_rpc(
            "multiplex_swap_cartridge",
            args=(cartridge_id,),
            timeout=600,
        )
        swap_duration = time.time() - swap_start

        total_time = time.time() - start_time
        bytes_copied = sum(result["bytes_copied"] for result in swap_results)
        masks_applied = sum(result.get("masks_applied", 0) for result in swap_results)
        gb_copied = bytes_copied / (1024**3)
        print(
            f"Swap to {cartridge_id} complete: {swap_duration:.2f}s, {gb_copied:.2f} GiB, {gb_copied / swap_duration if swap_duration > 0 else 0:.1f} GB/s, {masks_applied} router masks applied"
        )
        return {
            "status": "success",
            "cartridge_id": cartridge_id,
            "swap_time_s": swap_duration,
            "total_time_s": total_time,
            "bandwidth_gb_s": gb_copied / swap_duration if swap_duration > 0 else 0,
            "masks_applied": masks_applied,
            "worker_results": swap_results,
        }

    @app.post("/swap_active_set")
    async def swap_active_set(payload: dict[str, Any]):
        async with app.state.dynamic_swap_lock:
            engine = app.state.engine_client
            plan = getattr(app.state, "reap_plan", None)
            if not plan or plan.get("mode") != "dynamic_core_specialist":
                return {
                    "status": "error",
                    "error": "dynamic core/specialist plan is required",
                    "concurrency_mode": DYNAMIC_CONCURRENCY_MODE,
                    "plan_identity": build_plan_identity(),
                }
            try:
                validated = validate_active_set_payload(payload, plan)
            except Exception as exc:
                return {
                    "status": "error",
                    "error": str(exc),
                    "concurrency_mode": DYNAMIC_CONCURRENCY_MODE,
                    "plan_identity": build_plan_identity(),
                }
            request_id = str(validated["request_id"])
            phase = validated["phase"]
            active_set_signature = validated.get("active_set_signature")
            max_refreshes = int(plan.get("budget", {}).get("max_refreshes_per_request", 1))
            state = app.state.dynamic_request_state.get(request_id, {"refreshes_used": 0})
            last_signature = app.state.dynamic_request_signatures.get(request_id)
            if phase == "prefill":
                await engine.collective_rpc(
                    "multiplex_reset_router_misses",
                    args=(request_id,),
                    timeout=60,
                )
                state = {"refreshes_used": 0}
            elif state.get("refreshes_used", 0) >= max_refreshes:
                return {
                    "status": "error",
                    "error": f"refresh budget exhausted for request {request_id}",
                    "request_id": request_id,
                    "phase": phase,
                    "refreshes_used": state.get("refreshes_used", 0),
                    "active_set_signature": active_set_signature,
                    "concurrency_mode": DYNAMIC_CONCURRENCY_MODE,
                    "plan_identity": build_plan_identity(),
                }
            if phase == "decode_refresh":
                state["refreshes_used"] = state.get("refreshes_used", 0) + 1
            if active_set_signature and active_set_signature == last_signature:
                state.update(
                    {
                        "request_id": request_id,
                        "last_phase": phase,
                        "budget_bytes": validated["budget_bytes"],
                        "selected_slice_ids": validated.get("selected_slice_ids", {}),
                        "active_set_signature": active_set_signature,
                    }
                )
                app.state.dynamic_request_state[request_id] = state
                app.state.dynamic_request_signatures[request_id] = active_set_signature
                _evict_dynamic_request_state(app.state.dynamic_request_state)
                return {
                    "status": "success",
                    "request_id": request_id,
                    "phase": phase,
                    "swap_time_s": 0.0,
                    "budget_bytes": validated["budget_bytes"],
                    "bytes_copied": 0,
                    "bytes_zeroed": 0,
                    "bytes_touched": 0,
                    "zeroed_tensors": 0,
                    "masks_applied": 0,
                    "delta_added_experts": 0,
                    "delta_removed_experts": 0,
                    "delta_reused_experts": 0,
                    "refreshes_used": state.get("refreshes_used", 0),
                    "no_op_reuse": True,
                    "active_set_signature": active_set_signature,
                    "concurrency_mode": DYNAMIC_CONCURRENCY_MODE,
                    "plan_identity": build_plan_identity(),
                    "worker_results": [],
                }
            swap_start = time.time()
            swap_results = await engine.collective_rpc(
                "multiplex_swap_active_set",
                args=(validated, plan),
                timeout=600,
            )
            swap_duration = time.time() - swap_start
            state.update(
                {
                    "request_id": request_id,
                    "last_phase": phase,
                    "budget_bytes": validated["budget_bytes"],
                    "selected_slice_ids": validated.get("selected_slice_ids", {}),
                    "active_set_signature": active_set_signature,
                }
            )
            app.state.dynamic_request_state[request_id] = state
            app.state.dynamic_request_signatures[request_id] = active_set_signature
            _evict_dynamic_request_state(app.state.dynamic_request_state)
            app.state.dynamic_active_signature = active_set_signature
            bytes_copied = sum(result.get("bytes_copied", 0) for result in swap_results)
            bytes_zeroed = sum(result.get("bytes_zeroed", 0) for result in swap_results)
            bytes_touched = sum(result.get("bytes_touched", 0) for result in swap_results)
            zeroed_tensors = sum(result.get("zeroed_tensors", 0) for result in swap_results)
            masks_applied = sum(result.get("masks_applied", 0) for result in swap_results)
            delta_added_experts = sum(
                int(result.get("delta", {}).get("added_expert_total", 0)) for result in swap_results
            )
            delta_removed_experts = sum(
                int(result.get("delta", {}).get("removed_expert_total", 0)) for result in swap_results
            )
            delta_reused_experts = sum(
                int(result.get("delta", {}).get("reused_expert_total", 0)) for result in swap_results
            )
            return {
                "status": "success",
                "request_id": request_id,
                "phase": phase,
                "swap_time_s": swap_duration,
                "budget_bytes": validated["budget_bytes"],
                "bytes_copied": bytes_copied,
                "bytes_zeroed": bytes_zeroed,
                "bytes_touched": bytes_touched,
                "zeroed_tensors": zeroed_tensors,
                "masks_applied": masks_applied,
                "delta_added_experts": delta_added_experts,
                "delta_removed_experts": delta_removed_experts,
                "delta_reused_experts": delta_reused_experts,
                "refreshes_used": state.get("refreshes_used", 0),
                "concurrency_mode": DYNAMIC_CONCURRENCY_MODE,
                "active_set_signature": active_set_signature,
                "plan_identity": build_plan_identity(),
                "worker_results": swap_results,
            }

    @app.get("/router_misses/{request_id}")
    async def router_misses(request_id: str, reset: bool = False):
        engine = app.state.engine_client
        worker_results = await engine.collective_rpc(
            "multiplex_get_router_misses",
            args=(request_id, reset),
            timeout=120,
        )
        aggregate = aggregate_router_miss_results(worker_results)
        refresh_state = app.state.dynamic_request_state.get(request_id, {"refreshes_used": 0})
        return {
            "status": "success",
            "request_id": request_id,
            "refreshes_used": refresh_state.get("refreshes_used", 0),
            "concurrency_mode": DYNAMIC_CONCURRENCY_MODE,
            **aggregate,
            "worker_results": worker_results,
        }

    print("Lazy cartridge loading enabled; cartridges will be pinned on first swap.")
    return app


_install_worker_extensions()
api_server.build_app = build_app_with_swap

if __name__ == "__main__":
    import uvloop
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    api_server.cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = api_server.make_arg_parser(parser)
    args = parser.parse_args()
    api_server.validate_parsed_serve_args(args)
    uvloop.run(api_server.run_server(args))
