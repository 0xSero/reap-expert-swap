#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


REAP_SRC = "/home/ser/glm-47/reap/src"
if REAP_SRC not in sys.path:
    sys.path.insert(0, REAP_SRC)

from reap.model_util import MODEL_ATTRS  # type: ignore
from reap.observer import MoETransformerObserver, MoETransformerObserverConfig  # type: ignore


@dataclass
class DeepseekV3ObserverHookConfig(MoETransformerObserverConfig):
    module_class_name_to_hook_regex: str | None = "DeepseekV2MoE|DeepseekV3MoE"
    num_experts_attr_name: str = "experts_per_rank"
    top_k_attr_name: str = "num_experts_per_tok"
    fused_experts: bool = False


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def tensor_to_list(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def build_expert_table(observer_data: dict[int, dict[str, Any]]):
    experts: list[dict[str, Any]] = []
    for layer, layer_state in observer_data.items():
        num_experts = len(layer_state["expert_frequency"])
        total_tokens = int(layer_state["total_tokens"])
        for expert in range(num_experts):
            frequency = int(layer_state["expert_frequency"][expert])
            reap_score = float(layer_state.get("reap", [0.0] * num_experts)[expert])
            experts.append(
                {
                    "key": f"{layer}:{expert}",
                    "layer": int(layer),
                    "expert": int(expert),
                    "totalTokens": total_tokens,
                    "expertFrequency": frequency,
                    "expertProbability": 0.0 if total_tokens == 0 else round(frequency / total_tokens, 8),
                    "reap": round(reap_score, 8),
                }
            )
    return experts


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Kimi/DeepseekV3 REAP observation summaries from packed JSONL prompts.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workflow-name", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    MODEL_ATTRS["DeepseekV3ForCausalLM"] = {
        "moe_block": "mlp",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "n_routed_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    observer_state_path = output_dir / f"{args.workflow_name}-observer-state.pt"
    observer_summary_path = output_dir / f"{args.workflow_name}-observer-summary.json"
    expert_table_path = output_dir / f"{args.workflow_name}-expert-table.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    model.eval()

    observer = MoETransformerObserver(
        model=model,
        hook_config=DeepseekV3ObserverHookConfig(
            distance_measure="cosine",
            record_pruning_metrics_only=True,
        ),
    )

    rows = list(iter_jsonl(Path(args.dataset_jsonl)))[: args.max_samples]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    processed_samples = 0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(rows), args.batch_size):
            batch_rows = rows[i : i + args.batch_size]
            texts = [str(row.get(args.text_field) or "").strip() for row in batch_rows]
            texts = [text for text in texts if text]
            if not texts:
                continue
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_tokens,
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            total_tokens += int(encoded["attention_mask"].sum().item())
            processed_samples += len(texts)
            model(**encoded)
            print(f"processed {processed_samples}/{len(rows)} samples", flush=True)

    observer_data = observer.report_state()
    torch.save(observer_data, observer_state_path)

    layers = {
        str(layer): {
            key: tensor_to_list(value)
            for key, value in layer_state.items()
            if key in {"total_tokens", "expert_frequency", "ean_sum", "ean_mean", "weighted_ean_sum", "reap", "reap_l2", "weighted_ean_sum_l2"}
        }
        for layer, layer_state in observer_data.items()
    }
    observer_summary = {
        "workflow": args.workflow_name,
        "model": args.model,
        "datasetJsonl": str(Path(args.dataset_jsonl).resolve()),
        "processedSamples": processed_samples,
        "totalTokens": total_tokens,
        "layers": layers,
    }
    observer_summary_path.write_text(json.dumps(observer_summary, indent=2) + "\n", encoding="utf-8")

    expert_table = build_expert_table(observer_data)
    expert_table_path.write_text("".join(json.dumps(row) + "\n" for row in expert_table), encoding="utf-8")
    print(json.dumps({
        "workflow": args.workflow_name,
        "processedSamples": processed_samples,
        "totalTokens": total_tokens,
        "observerSummaryPath": str(observer_summary_path),
        "expertTablePath": str(expert_table_path),
    }, indent=2))


if __name__ == "__main__":
    main()
