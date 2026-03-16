#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def render_turn(row: dict[str, Any]) -> str:
    text = str(row.get("prompt_text") or "").strip()
    if not text:
        return ""
    return f"user: {text}"


def truncate_text_to_token_budget(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    max_chars = max_tokens * 4
    return text[:max_chars].rstrip() + "\n[truncated]"


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def build_conversations(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        conv_id = str(row.get("conversation_id") or "unknown")
        grouped[conv_id].append(row)

    conversations: list[list[dict[str, Any]]] = []
    for _, convo_rows in grouped.items():
        convo_rows.sort(key=lambda r: str(r.get("timestamp") or ""))
        filtered = [r for r in convo_rows if render_turn(r)]
        if filtered:
            conversations.append(filtered)
    conversations.sort(key=lambda convo: len(convo), reverse=True)
    return conversations


def pack_conversations(
    conversations: list[list[dict[str, Any]]],
    *,
    target_tokens: int,
    max_turns_per_sample: int,
    max_conversations_per_sample: int,
    max_tokens_per_conversation: int,
    max_tokens_per_turn: int,
) -> list[dict[str, Any]]:
    packed: list[dict[str, Any]] = []
    current_blocks: list[str] = []
    current_turns = 0
    current_conversations = 0
    current_tokens = 0
    sources: list[str] = []
    conversation_ids: list[str] = []

    def flush() -> None:
        nonlocal current_blocks, current_turns, current_conversations, current_tokens, sources, conversation_ids
        if not current_blocks:
            return
        text = "\n\n".join(current_blocks).strip()
        packed.append(
            {
                "id": f"packed-{len(packed):06d}",
                "text": text,
                "estimated_tokens": current_tokens,
                "turn_count": current_turns,
                "conversation_count": current_conversations,
                "sources": sorted(set(sources)),
                "conversation_ids": conversation_ids,
            }
        )
        current_blocks = []
        current_turns = 0
        current_conversations = 0
        current_tokens = 0
        sources = []
        conversation_ids = []

    for convo in conversations:
        turn_buffer: list[str] = []
        turn_count = 0
        for row in convo[:max_turns_per_sample]:
            rendered = render_turn(row)
            if not rendered:
                continue
            rendered = truncate_text_to_token_budget(rendered, max_tokens_per_turn)
            projected = estimate_tokens("\n".join(turn_buffer + [rendered]))
            if turn_buffer and projected > max_tokens_per_conversation:
                block = "\n".join(turn_buffer)
                block_tokens = estimate_tokens(block)
                if current_blocks and (
                    current_tokens + block_tokens > target_tokens
                    or current_conversations >= max_conversations_per_sample
                    or current_turns + turn_count > max_turns_per_sample
                ):
                    flush()
                current_blocks.append(block)
                current_tokens += block_tokens
                current_turns += turn_count
                current_conversations += 1
                sources.extend(str(item.get("source") or "unknown") for item in convo)
                conversation_ids.append(str(convo[0].get("conversation_id") or "unknown"))
                turn_buffer = [rendered]
                turn_count = 1
                if current_tokens >= target_tokens:
                    flush()
                continue
            turn_buffer.append(rendered)
            turn_count += 1

        if not turn_buffer:
            continue

        block = "\n".join(turn_buffer)
        block_tokens = estimate_tokens(block)
        if current_blocks and (
            current_tokens + block_tokens > target_tokens
            or current_conversations >= max_conversations_per_sample
            or current_turns + turn_count > max_turns_per_sample
        ):
            flush()

        current_blocks.append(block)
        current_tokens += block_tokens
        current_turns += turn_count
        current_conversations += 1
        sources.extend(str(row.get("source") or "unknown") for row in convo)
        conversation_ids.append(str(convo[0].get("conversation_id") or "unknown"))

        if current_tokens >= target_tokens:
            flush()

    flush()
    return packed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build packed 16k-style personal multi-turn corpus for Kimi REAP calibration.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--target-tokens", type=int, default=16000)
    parser.add_argument("--max-turns-per-sample", type=int, default=256)
    parser.add_argument("--max-conversations-per-sample", type=int, default=32)
    parser.add_argument("--max-tokens-per-conversation", type=int, default=4096)
    parser.add_argument("--max-tokens-per-turn", type=int, default=1024)
    parser.add_argument("--limit-rows", type=int)
    args = parser.parse_args()

    rows = list(iter_jsonl(Path(args.input_jsonl)))
    if args.limit_rows:
        rows = rows[: args.limit_rows]

    conversations = build_conversations(rows)
    packed = pack_conversations(
        conversations,
        target_tokens=args.target_tokens,
        max_turns_per_sample=args.max_turns_per_sample,
        max_conversations_per_sample=args.max_conversations_per_sample,
        max_tokens_per_conversation=args.max_tokens_per_conversation,
        max_tokens_per_turn=args.max_tokens_per_turn,
    )

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in packed), encoding="utf-8")

    summary = {
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        "rows_read": len(rows),
        "conversation_count": len(conversations),
        "packed_sample_count": len(packed),
        "target_tokens": args.target_tokens,
        "max_turns_per_sample": args.max_turns_per_sample,
        "max_conversations_per_sample": args.max_conversations_per_sample,
        "max_tokens_per_conversation": args.max_tokens_per_conversation,
        "max_tokens_per_turn": args.max_tokens_per_turn,
        "avg_estimated_tokens": round(sum(item["estimated_tokens"] for item in packed) / max(len(packed), 1), 2),
        "max_estimated_tokens": max((item["estimated_tokens"] for item in packed), default=0),
    }
    Path(args.output_summary_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
