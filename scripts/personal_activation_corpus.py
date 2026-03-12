#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from dynamic_reap import infer_domain_tags, normalize_prompt_text, stable_prompt_id


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield payload


def iter_json(path: Path) -> Iterable[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        yield payload
        return
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            if isinstance(item, dict):
                yield item
            else:
                raise ValueError(f"{path}:{index} is not a JSON object")
        return
    raise ValueError(f"{path} is not a JSON object or array")


def _extract_timestamp(record: dict[str, Any]) -> str | None:
    for key in ("timestamp", "created_at", "extracted_at", "unixMs", "ts", "time_created"):
        value = record.get(key)
        if value is not None:
            return str(value)
    conversation = record.get("conversation")
    if isinstance(conversation, dict):
        for key in ("timestamp", "created_at"):
            value = conversation.get(key)
            if value is not None:
                return str(value)
    data = record.get("data")
    if isinstance(data, dict):
        for key in ("timestamp", "unixMs"):
            value = data.get(key)
            if value is not None:
                return str(value)
    message = record.get("message")
    if isinstance(message, dict):
        for key in ("timestamp",):
            value = message.get(key)
            if value is not None:
                return str(value)
    payload = record.get("payload")
    if isinstance(payload, dict):
        for key in ("timestamp",):
            value = payload.get(key)
            if value is not None:
                return str(value)
    return None


def _extract_conversation_id(record: dict[str, Any]) -> str:
    for key in ("conversation_id", "workspace_id", "generationUUID", "id", "session_id", "sessionId"):
        value = record.get(key)
        if value:
            return str(value)
    conversation = record.get("conversation")
    if isinstance(conversation, dict):
        for key in ("id", "chat_title", "chatTitle"):
            value = conversation.get(key)
            if value:
                return str(value)
    data = record.get("data")
    if isinstance(data, dict):
        for key in ("generationUUID", "conversation_id", "id"):
            value = data.get(key)
            if value:
                return str(value)
    payload = record.get("payload")
    if isinstance(payload, dict):
        payload_id = payload.get("id")
        if payload_id:
            return str(payload_id)
    return "unknown"


def _looks_like_user_role(value: Any) -> bool:
    if value is None:
        return False
    lowered = str(value).strip().lower()
    return lowered in {"user", "human", "prompt", "assistant_user", "input"}


def _candidate_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_candidate_strings(item))
        return out
    if isinstance(value, dict):
        out: list[str] = []
        for key in (
            "text",
            "content",
            "prompt",
            "value",
            "markdown",
            "display",
            "textDescription",
            "command",
            "message",
        ):
            if key in value:
                out.extend(_candidate_strings(value[key]))
        return out
    return []


def _clean_prompts(prompts: list[str]) -> list[str]:
    cleaned: list[str] = []
    for prompt in prompts:
        normalized = normalize_prompt_text(prompt)
        if len(normalized) < 8:
            continue
        if normalized.lower() in {"user", "assistant", "system"}:
            continue
        if re.fullmatch(r"[\W_]+", normalized):
            continue
        cleaned.append(normalized)
    seen = set()
    out: list[str] = []
    for prompt in cleaned:
        if prompt not in seen:
            seen.add(prompt)
            out.append(prompt)
    return out


def extract_user_prompts(record: dict[str, Any]) -> list[str]:
    prompts: list[str] = []
    source = str(record.get("source") or record.get("source_type") or "").lower()

    conversation = record.get("conversation")
    if isinstance(conversation, dict):
        if isinstance(conversation.get("display"), str):
            prompts.append(conversation["display"])
        for key in ("chat_title", "chatTitle", "prompt", "text", "content"):
            if isinstance(conversation.get(key), str):
                prompts.append(conversation[key])
        tabs = conversation.get("tabs")
        if isinstance(tabs, list):
            for tab in tabs:
                if not isinstance(tab, dict):
                    continue
                for bubble in tab.get("bubbles", []) if isinstance(tab.get("bubbles"), list) else []:
                    if not isinstance(bubble, dict):
                        continue
                    if any(_looks_like_user_role(bubble.get(key)) for key in ("role", "type", "sender", "author")):
                        prompts.extend(_candidate_strings(bubble))
        messages = conversation.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and _looks_like_user_role(message.get("role")):
                    prompts.extend(_candidate_strings(message.get("content")))

    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and _looks_like_user_role(message.get("role")):
                prompts.extend(_candidate_strings(message.get("content")))

    prompt_obj = record.get("prompt")
    if isinstance(prompt_obj, dict):
        prompts.extend(_candidate_strings(prompt_obj.get("text")))
    elif isinstance(prompt_obj, str):
        prompts.append(prompt_obj)

    data = record.get("data")
    if isinstance(data, dict):
        if str(record.get("type") or data.get("type") or "").lower().startswith("composer"):
            prompts.extend(_candidate_strings(data.get("textDescription")))
        for key in ("prompt", "text", "display", "textDescription"):
            prompts.extend(_candidate_strings(data.get(key)))

    nested_message = record.get("message")
    if isinstance(nested_message, dict) and _looks_like_user_role(nested_message.get("role")):
        prompts.extend(_candidate_strings(nested_message.get("content")))
    elif isinstance(nested_message, dict):
        role = nested_message.get("role")
        if _looks_like_user_role(role):
            prompts.extend(_candidate_strings(nested_message))

    payload = record.get("payload")
    if isinstance(payload, dict):
        if record.get("type") == "response_item" and payload.get("type") == "message" and _looks_like_user_role(payload.get("role")):
            prompts.extend(_candidate_strings(payload.get("content")))
        if record.get("type") == "event_msg" and payload.get("type") == "user_message":
            prompts.extend(_candidate_strings(payload.get("message")))

    if record.get("type") == "message" and isinstance(record.get("command"), str):
        prompts.append(str(record["command"]))

    for key in ("display", "text", "chat_title", "chatTitle", "command"):
        if isinstance(record.get(key), str):
            prompts.append(str(record[key]))

    if not prompts and source.startswith("codex") and isinstance(conversation, dict):
        prompts.extend(_candidate_strings(conversation))

    return _clean_prompts(prompts)


def iter_source_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    path_str = str(path)
    if suffix == ".json":
        yield from iter_json(path)
        return
    if suffix == ".jsonl":
        yield from iter_jsonl(path)
        return
    raise ValueError(f"Unsupported input format for {path_str}")


def iter_opencode_sqlite_prompts(path: Path) -> Iterable[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cursor = connection.cursor()
        query = """
        SELECT
          message.session_id,
          message.time_created,
          session.directory,
          session.title,
          part.data
        FROM message
        JOIN session ON session.id = message.session_id
        JOIN part ON part.message_id = message.id
        WHERE json_extract(message.data, '$.role') = 'user'
        ORDER BY message.time_created ASC, part.time_created ASC
        """
        for session_id, time_created, directory, title, part_data_text in cursor.execute(query):
            try:
                part_data = json.loads(part_data_text)
            except Exception:
                continue
            if part_data.get("type") != "text":
                continue
            if part_data.get("synthetic"):
                continue
            prompt_strings = _clean_prompts(_candidate_strings(part_data.get("text")))
            for prompt_text in prompt_strings:
                yield {
                    "source": "opencode",
                    "timestamp": str(time_created),
                    "conversation_id": str(session_id),
                    "session_id": str(session_id),
                    "workspace_id": str(directory or title or "unknown"),
                    "prompt_text": prompt_text,
                }
    finally:
        connection.close()


def infer_source_label(path: Path, raw_record: dict[str, Any]) -> str:
    path_str = str(path)
    if path.name == "opencode.db":
        return "opencode"
    if "/.factory/" in path_str:
        return "factory"
    if "/.claude/" in path_str:
        return "claude-live"
    if "/.pi/agent/sessions/" in path_str:
        return "pi"
    if "/.codex/sessions/" in path_str or "/.codex/archived_sessions/" in path_str:
        return "codex-live"
    return str(raw_record.get("source") or raw_record.get("source_type") or path.stem)

def build_activation_corpus(
    input_paths: list[Path],
    *,
    max_prompts: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_prompt_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    domain_tag_counts: Counter[str] = Counter()
    duplicate_count = 0

    for path in input_paths:
        if not path.exists():
            continue
        if path.suffix.lower() == ".db" and path.name == "opencode.db":
            iterator = iter_opencode_sqlite_prompts(path)
        else:
            iterator = iter_source_records(path)
        for raw_record in iterator:
            if raw_record.get("source") == "opencode" and raw_record.get("prompt_text"):
                prompt_texts = [str(raw_record["prompt_text"])]
                source = str(raw_record.get("source") or path.stem)
                timestamp = raw_record.get("timestamp")
                conversation_id = str(raw_record.get("conversation_id") or raw_record.get("session_id") or "unknown")
            else:
                source = infer_source_label(path, raw_record)
                timestamp = _extract_timestamp(raw_record)
                conversation_id = _extract_conversation_id(raw_record)
                prompt_texts = extract_user_prompts(raw_record)
            for prompt_text in prompt_texts:
                prompt_id = stable_prompt_id(prompt_text)
                if prompt_id in seen_prompt_ids:
                    duplicate_count += 1
                    continue
                domain_tags = infer_domain_tags(prompt_text)
                row = {
                    "prompt_id": prompt_id,
                    "source": source,
                    "timestamp": timestamp,
                    "prompt_text": prompt_text,
                    "conversation_id": conversation_id,
                    "domain_tags": domain_tags,
                }
                records.append(row)
                seen_prompt_ids.add(prompt_id)
                source_counts[source] += 1
                for tag in domain_tags:
                    domain_tag_counts[tag] += 1
                if max_prompts is not None and len(records) >= max_prompts:
                    summary = {
                        "record_count": len(records),
                        "duplicate_prompts_removed": duplicate_count,
                        "source_counts": dict(sorted(source_counts.items())),
                        "domain_tag_counts": dict(sorted(domain_tag_counts.items())),
                        "input_paths": [str(path) for path in input_paths],
                    }
                    return records, summary

    summary = {
        "record_count": len(records),
        "duplicate_prompts_removed": duplicate_count,
        "source_counts": dict(sorted(source_counts.items())),
        "domain_tag_counts": dict(sorted(domain_tag_counts.items())),
        "input_paths": [str(path) for path in input_paths],
    }
    return records, summary
