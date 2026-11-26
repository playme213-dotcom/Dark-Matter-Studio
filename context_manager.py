#!/usr/bin/env python3
"""
Colibri Studio - conversation context manager.

Measures how much of the model's context window a conversation is using, and
when it crosses a configured percentage, folds the oldest exchanges into a
running digest so the agent can keep working on long jobs.

Two rules are absolute:

  1. Nothing is ever deleted. Folding only changes what is *sent* to the model.
     Every original message stays in chat_history.db and every stored
     conversation is still replayed in full when it is opened.
  2. The work in progress is never folded away. The active exchange, its tool
     calls and its tool results are always preserved verbatim.

The module has no Flask, network or backend dependencies so it can be tested
directly.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time

# ---------------------------------------------------------------------------
# Defaults and limits
# ---------------------------------------------------------------------------

DEFAULT_WINDOW = 32768
DEFAULT_THRESHOLD_PERCENT = 78.0
MIN_THRESHOLD_PERCENT = 10.0
MAX_THRESHOLD_PERCENT = 95.0

# OpenAI-style chat framing overhead that a token counter on the raw text alone
# would miss. Deliberately a little generous: over-counting compacts slightly
# early, under-counting overflows the window.
PER_MESSAGE_OVERHEAD = 8
PER_IMAGE_TOKENS = 85

MIN_RECENT_MESSAGES = 6
MAX_PROTECTED_MESSAGES = 48
MIN_OUTPUT_RESERVE = 1024
DIGEST_BUDGET_RATIO = 0.25
MIN_DIGEST_BUDGET_TOKENS = 512

DIGEST_HEADER = "[Colibri conversation digest]"
DIGEST_ROLE_DEFAULT = os.environ.get("LITELLM_DIGEST_ROLE", "system").strip().lower()

_TOKENIZER_NAME = None
_tiktoken_encoding = None
_tiktoken_checked = False

_WORD_RUN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_TOOL_RESPONSE_RE = re.compile(r"^\s*<tool_response\b", re.IGNORECASE)
_ERROR_LINE_RE = re.compile(r"\b(error|failed|failure|traceback|denied|blocked|timeout|not found|refused)\b", re.IGNORECASE)

# Sentences the agent loop injects between steps. They belong to the exchange
# that produced them and are never treated as a new user turn.
_SYNTHETIC_USER_MARKERS = (
    "Continue the same job.",
    "You are the selected provider collaborating with the Colibri agent",
    "Your last response did not include the required task status marker",
    "The previous response attempted an action using incorrect",
    "The previous response produced no visible reply",
    "The previous response did not request any tools",
    "Review the complete transcript and prior results",
)

_USER_SNIPPET_CHARS = 320
_ASSISTANT_SNIPPET_CHARS = 320
_TOOL_SNIPPET_CHARS = 180


# ---------------------------------------------------------------------------
# Token measurement
# ---------------------------------------------------------------------------

def tokenizer_name() -> str:
    estimate_tokens("")  # ensure the lookup has happened
    return _TOKENIZER_NAME or "colibri-heuristic-v1"


def _encoding():
    global _tiktoken_encoding, _tiktoken_checked, _TOKENIZER_NAME
    if _tiktoken_checked:
        return _tiktoken_encoding
    _tiktoken_checked = True
    try:
        import tiktoken

        _tiktoken_encoding = tiktoken.get_encoding("cl100k_base")
        _TOKENIZER_NAME = "tiktoken:cl100k_base"
    except Exception:
        _tiktoken_encoding = None
        _TOKENIZER_NAME = "colibri-heuristic-v1"
    return _tiktoken_encoding


def _is_cjk_run(word: str) -> bool:
    sample = word[:8]
    return any("\u3040" <= ch <= "\ud7af" or "\uf900" <= ch <= "\ufaff" for ch in sample)


def heuristic_token_count(text: str) -> int:
    """Dependency-free estimate used when no real tokenizer is installed."""
    if not text:
        return 0
    total = 0
    for run in _WORD_RUN_RE.findall(text):
        if len(run) == 1:
            total += 1
        elif _is_cjk_run(run):
            total += max(1, math.ceil(len(run) * 1.3))
        else:
            total += max(1, math.ceil(len(run) / 4))
    # Structural characters and run-together paths tokenize worse than prose.
    total += text.count("/") + text.count("\n")
    return total


def estimate_tokens(text) -> int:
    if text is None:
        return 0
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return 0
    enc = _encoding()
    if enc is not None:
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    return heuristic_token_count(text)


def message_text(message) -> str:
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def message_tokens(message) -> int:
    if not isinstance(message, dict):
        return estimate_tokens(message)
    tokens = PER_MESSAGE_OVERHEAD
    tokens += estimate_tokens(message.get("role") or "user")
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image_url", "image"}:
                tokens += PER_IMAGE_TOKENS
            elif isinstance(part, dict) and part.get("type") == "file":
                tokens += PER_IMAGE_TOKENS
        tokens += estimate_tokens(message_text(message))
    else:
        tokens += estimate_tokens(content)
    if message.get("tool_calls"):
        tokens += estimate_tokens(message["tool_calls"])
    return tokens


def messages_tokens(messages) -> int:
    return sum(message_tokens(message) for message in messages or [])


def system_tokens(system_prompt: str, tools_spec=None) -> int:
    total = estimate_tokens(system_prompt or "")
    if tools_spec:
        try:
            import json

            total += estimate_tokens(json.dumps(tools_spec, sort_keys=True))
        except Exception:
            total += estimate_tokens(str(tools_spec))
    return total


# ---------------------------------------------------------------------------
# Window resolution
# ---------------------------------------------------------------------------

def clamp_percent(value, default: float = DEFAULT_THRESHOLD_PERCENT) -> float:
    try:
        percent = float(value)
    except (TypeError, ValueError):
        return default
    if percent <= 0:
        return default
    return max(MIN_THRESHOLD_PERCENT, min(MAX_THRESHOLD_PERCENT, percent))


def clamp_window(value, default: int = DEFAULT_WINDOW) -> int:
    try:
        window = int(value)
    except (TypeError, ValueError):
        return default
    if window <= 0:
        return default
    return max(2048, min(4_000_000, window))


def window_from_model_info(info) -> int:
    """Pull a context window out of whatever shape the provider reported."""
    if not isinstance(info, dict):
        return 0
    candidates = []
    for key in ("max_input_tokens", "context_window", "context_length", "max_context_length",
                "n_ctx", "num_ctx", "max_tokens", "window"):
        value = info.get(key)
        if isinstance(value, (int, float)) and int(value) > 0:
            candidates.append(int(value))
    details = info.get("model_info")
    if isinstance(details, dict):
        for key, value in details.items():
            if str(key).endswith(".context_length") and isinstance(value, (int, float)) and int(value) > 0:
                candidates.append(int(value))
    return max(candidates) if candidates else 0


class ContextBudget:
    """How much room the model actually has for conversation."""

    def __init__(self, window: int, threshold_percent: float, output_reserve: int,
                 fixed_input_tokens: int = 0, recent_floor_tokens: int = 0):
        self.window = clamp_window(window)
        self.threshold_percent = clamp_percent(threshold_percent)
        self.output_reserve = max(MIN_OUTPUT_RESERVE, int(output_reserve))
        self.fixed_input_tokens = max(0, int(fixed_input_tokens))
        self.recent_floor_tokens = max(0, int(recent_floor_tokens))

    @property
    def input_ceiling(self) -> int:
        return max(1024, self.window - self.output_reserve)

    @property
    def threshold_tokens(self) -> int:
        return int(self.input_ceiling * self.threshold_percent / 100.0)

    @property
    def headroom_tokens(self) -> int:
        return max(0, self.input_ceiling - self.threshold_tokens)

    def snapshot(self) -> dict:
        return {
            "context_window": self.window,
            "threshold_percent": round(self.threshold_percent, 2),
            "threshold_tokens": self.threshold_tokens,
            "input_ceiling_tokens": self.input_ceiling,
            "output_reserve_tokens": self.output_reserve,
            "fixed_input_tokens": self.fixed_input_tokens,
            "recent_floor_tokens": self.recent_floor_tokens,
        }


def build_budget(window: int, threshold_percent: float, max_tokens: int,
                 fixed_input_tokens: int = 0, window_source: str = "fallback") -> ContextBudget:
    reserve = max(MIN_OUTPUT_RESERVE, int(max_tokens or 0))
    # Reserve enough room for the reply the model is about to write, but never
    # so much that the conversation gets squeezed out of the window.
    reserve = min(reserve, max(MIN_OUTPUT_RESERVE, self_fraction(window)))
    return ContextBudget(window, threshold_percent, reserve, fixed_input_tokens)


def self_fraction(window: int) -> int:
    return int(clamp_window(window) * 0.25)


# ---------------------------------------------------------------------------
# Conversation structure
# ---------------------------------------------------------------------------

def is_synthetic_user(message) -> bool:
    """True for agent-loop messages that are not a real user turn."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    text = message_text(message)
    if _TOOL_RESPONSE_RE.match(text):
        return True
    head = text[:400]
    return any(marker in head for marker in _SYNTHETIC_USER_MARKERS)


def is_turn_start(message) -> bool:
    return isinstance(message, dict) and message.get("role") == "user" and not is_synthetic_user(message)


def head_end_index(messages) -> int:
    index = 0
    while index < len(messages) and isinstance(messages[index], dict) and messages[index].get("role") == "system":
        index += 1
    return index


def group_turns(messages, start: int, end: int):
    """Split messages[start:end] into user-initiated exchanges."""
    turns = []
    index = max(0, start)
    end = min(end, len(messages))
    while index < end:
        cursor = index + 1
        while cursor < end and not is_turn_start(messages[cursor]):
            cursor += 1
        turns.append((index, cursor))
        index = cursor
    return turns


def protected_boundary(messages, min_recent: int = MIN_RECENT_MESSAGES) -> int:
    """First message index that must be preserved verbatim.

    Walks back from the end until it has at least ``min_recent`` messages and
    has landed on a real exchange start, so tool calls and their results are
    never separated and the active turn is never cut in half.
    """
    total = len(messages)
    head = head_end_index(messages)
    if total <= head:
        return total
    min_recent = max(2, int(min_recent or MIN_RECENT_MESSAGES))
    candidate = max(head, total - min_recent)
    hard_floor = max(head, total - MAX_PROTECTED_MESSAGES)

    steps = 0
    while candidate > hard_floor and not is_turn_start(messages[candidate]):
        candidate -= 1
        steps += 1
        if candidate <= head:
            candidate = head
            break
    return candidate


def turn_key(turn_messages) -> str:
    digest = hashlib.sha1()
    for message in turn_messages:
        digest.update(str(message.get("role", "")).encode("utf-8", "replace"))
        digest.update(b"\0")
        digest.update(message_text(message).encode("utf-8", "replace"))
        digest.update(b"\0")
    return digest.hexdigest()[:20]


# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------

def _snippet(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _tool_names(text: str):
    names = re.findall(r"<tool_response\s+name=\"([^\"]+)\"", text)
    if names:
        return names
    names = re.findall(r"<tool_call>\s*([\w.\-]+)", text)
    seen = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


def digest_entry_for_turn(turn_messages, turn_index: int) -> dict:
    """A lossy but faithful record of one exchange."""
    lines = []
    user_lines = []
    assistant_lines = []
    tool_lines = []
    error_lines = []

    for message in turn_messages:
        role = str(message.get("role") or "user")
        text = message_text(message)
        if not text:
            continue
        if role == "user" and not is_synthetic_user(message):
            user_lines.append(_snippet(text, _USER_SNIPPET_CHARS))
        elif role == "assistant":
            cleaned = re.sub(r"<task_status>\s*\w+\s*</task_status>", "", text, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if cleaned:
                assistant_lines.append(_snippet(cleaned, _ASSISTANT_SNIPPET_CHARS))
            for name in _tool_names(text):
                target = ""
                match = re.search(r'"(?:path|command|pattern|query|url|relative_path)"\s*:\s*"([^"]{0,80})"', text)
                if match:
                    target = f" -> {match.group(1)}"
                tool_lines.append(f"called {name}{target}")
        elif role == "user":
            for name in _tool_names(text):
                status_match = re.search(r'name="%s">\s*Status:\s*(\w+)' % re.escape(name), text)
                status = status_match.group(1) if status_match else "reported"
                body = re.sub(r"<tool_response[^>]*>", "", text)
                tool_lines.append(f"{name} -> {status}: {_snippet(body, _TOOL_SNIPPET_CHARS)}")
            if not tool_lines:
                assistant_lines.append(_snippet(text, _TOOL_SNIPPET_CHARS))
        for line in text.splitlines():
            if len(error_lines) < 4 and _ERROR_LINE_RE.search(line) and len(line) < 400:
                error_lines.append(_snippet(line, 160))

    if user_lines:
        lines.append("user: " + " | ".join(user_lines[:3]))
    if assistant_lines:
        lines.append("assistant: " + " | ".join(assistant_lines[:3]))
    if tool_lines:
        lines.append("tools: " + "; ".join(tool_lines[:12]))
    if error_lines:
        lines.append("issues: " + " | ".join(error_lines))
    if not lines:
        lines.append("no substantive content")

    summary = "\n".join(lines)
    return {
        "index": int(turn_index),
        "key": turn_key(turn_messages),
        "summary": summary,
        "tokens": estimate_tokens(summary),
        "messages": len(turn_messages),
    }


def render_digest(entries, budget_tokens: int, role: str = DIGEST_ROLE_DEFAULT):
    """Render folded exchanges as one digest message, newest first on pressure."""
    budget_tokens = max(MIN_DIGEST_BUDGET_TOKENS, int(budget_tokens))
    entries = sorted(entries or [], key=lambda entry: entry.get("index", 0))
    if not entries:
        return None, 0

    header_lines = [
        DIGEST_HEADER,
        f"{len(entries)} earlier exchange(s) were compacted so the agent could keep working.",
        "The full originals remain in the session database; this digest is what the model sees.",
    ]
    header_text = "\n".join(header_lines)

    kept = []
    used = estimate_tokens(header_text)
    dropped = 0
    for entry in reversed(entries):
        line = f"#{entry.get('index', 0)} (turn {entry.get('index', 0)}): {entry.get('summary', '')}"
        cost = estimate_tokens(line)
        if kept and used + cost > budget_tokens:
            dropped += 1
            continue
        kept.append(line)
        used += cost

    kept.reverse()
    if dropped:
        kept.insert(0, f"[{dropped} older compacted exchange(s) dropped from the digest to stay inside the window]")

    body = "\n".join([header_text] + kept)
    if role == "user":
        body = body + "\n\nUse the digest above as established context. Do not repeat work already recorded there."
    return {"role": role, "content": body}, estimate_tokens(body)


def load_stored_entries(conn, session_id: str):
    """Previously persisted digests, keyed by exchange so re-runs reuse them."""
    if conn is None or not session_id:
        return {}
    try:
        rows = conn.execute(
            "SELECT turn_key, turn_index, summary, tokens FROM context_summaries WHERE session_id = ? ORDER BY turn_index ASC",
            (session_id,),
        ).fetchall()
    except Exception:
        return {}
    stored = {}
    for row in rows:
        stored[row["turn_key"]] = {
            "index": row["turn_index"],
            "key": row["turn_key"],
            "summary": row["summary"],
            "tokens": row["tokens"],
            "stored": True,
        }
    return stored


def persist_entries(conn, session_id: str, entries) -> int:
    """Store digests without touching the original messages. Returns rows written."""
    if conn is None or not session_id or not entries:
        return 0
    written = 0
    now = time.time()
    for entry in entries:
        if not entry.get("key"):
            continue
        try:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO context_summaries (session_id, turn_key, turn_index, summary, tokens, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, entry["key"], int(entry.get("index", 0)), entry.get("summary", ""),
                 int(entry.get("tokens", 0)), now),
            )
            written += cursor.rowcount or 0
        except Exception:
            continue
    try:
        conn.commit()
    except Exception:
        pass
    return written


def clear_stored_entries(conn, session_id: str) -> int:
    if conn is None or not session_id:
        return 0
    try:
        cursor = conn.execute("DELETE FROM context_summaries WHERE session_id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount or 0
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure(system_prompt: str, tools_spec, messages, window: int,
            threshold_percent: float, max_tokens: int, window_source: str = "fallback") -> dict:
    budget = build_budget(window, threshold_percent, max_tokens,
                          system_tokens(system_prompt, tools_spec), window_source)
    history_tokens = messages_tokens((messages or [])[head_end_index(messages):])
    prompt_tokens = budget.fixed_input_tokens + history_tokens
    percent = (prompt_tokens / budget.input_ceiling * 100.0) if budget.input_ceiling else 0.0
    return {
        "measured_at": time.time(),
        "tokenizer": tokenizer_name(),
        "estimated_only": tokenizer_name().startswith("colibri"),
        "window_source": window_source,
        "messages": len(messages or []),
        "history_tokens": history_tokens,
        "prompt_tokens": prompt_tokens,
        "percent_of_window": round(percent, 2),
        "percent_of_input_ceiling": round(percent, 2),
        "over_threshold": prompt_tokens > budget.threshold_tokens,
        "exceeds_ceiling": prompt_tokens > budget.input_ceiling,
        "budget": budget.snapshot(),
    }


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

def compact(messages, system_prompt: str, tools_spec, window: int,
            threshold_percent: float, max_tokens: int, *,
            window_source: str = "fallback",
            min_recent_messages: int = MIN_RECENT_MESSAGES,
            digest_role: str = DIGEST_ROLE_DEFAULT,
            stored_entries=None) -> dict:
    """Compact old complete exchanges while retaining all unsummarized turns.

    The original transcript is never mutated. Previously persisted digests are
    reused by turn hash; only turns represented by a digest are omitted from
    the model-facing copy.
    """
    started = time.time()
    original = list(messages or [])
    budget = build_budget(window, threshold_percent, max_tokens,
                          system_tokens(system_prompt, tools_spec), window_source)
    head_end = head_end_index(original)
    head = original[:head_end]
    boundary = protected_boundary(original, min_recent_messages)
    turns = group_turns(original, head_end, boundary)

    stored = dict(stored_entries or {})
    entries = sorted((entry for entry in stored.values() if entry.get("stored")),
                     key=lambda entry: entry.get("index", 0))
    folded_keys = {entry.get("key") for entry in entries if entry.get("key")}

    def compose(entry_list):
        represented = {entry.get("key") for entry in entry_list if entry.get("key")}
        digest_message, digest_cost = render_digest(
            entry_list, int(budget.input_ceiling * DIGEST_BUDGET_RATIO), digest_role
        )
        remaining = []
        for start, end in turns:
            turn_messages = original[start:end]
            if turn_key(turn_messages) not in represented:
                remaining.extend(turn_messages)
        remaining.extend(original[boundary:])
        # The system prompt is already counted in fixed_input_tokens; do not
        # count the preserved leading system message a second time.
        total = (budget.fixed_input_tokens
                 + (digest_cost if digest_message else 0) + messages_tokens(remaining))
        rebuilt = head + ([digest_message] if digest_message else []) + remaining
        return digest_message, digest_cost, total, remaining, rebuilt

    digest_message, digest_cost, total, remaining, rebuilt = compose(entries)
    before_tokens = budget.fixed_input_tokens + messages_tokens(original[head_end:])
    before_percent = (before_tokens / budget.input_ceiling * 100.0) if budget.input_ceiling else 0.0
    report = {
        "ok": True, "compacted": False,
        "model_window_source": window_source,
        "tokenizer": tokenizer_name(),
        "estimated_only": tokenizer_name().startswith("colibri"),
        "messages_before": len(original),
        "prompt_tokens_before": before_tokens,
        "percent_before": round(before_percent, 2),
        "threshold_percent": round(budget.threshold_percent, 2),
        "threshold_tokens": budget.threshold_tokens,
        "input_ceiling_tokens": budget.input_ceiling,
        "context_window": budget.window,
        "turns_folded": 0,
        "turns_available": len(turns),
        "digest_tokens": digest_cost,
        "digest_entries": len(entries),
        "digest_role": digest_role,
        "preserved_messages": len(original) - boundary,
        "preserved_tokens": messages_tokens(original[boundary:]),
        "preserved_turns": len(group_turns(original, boundary, len(original))),
        "stored_reused": len(entries), "stored_written": 0,
        "digest_dropped_entries": 0,
        "information_loss": False, "mid_turn_split": False,
        "fits": total <= budget.input_ceiling,
        "overflow": total > budget.input_ceiling,
        "reason": "", "elapsed_ms": 0,
    }

    newly_folded = []
    if total > budget.threshold_tokens:
        for start, end in turns:
            turn_messages = original[start:end]
            key = turn_key(turn_messages)
            if key in folded_keys:
                continue
            entry = digest_entry_for_turn(turn_messages, len(entries) + len(newly_folded))
            newly_folded.append(entry)
            folded_keys.add(key)
            candidate_entries = entries + newly_folded
            digest_message, digest_cost, total, remaining, rebuilt = compose(candidate_entries)
            report["turns_folded"] += 1
            if total <= budget.threshold_tokens:
                break

    entries = entries + newly_folded
    digest_message, digest_cost, total, remaining, rebuilt = compose(entries)
    if digest_message:
        report["digest_dropped_entries"] = digest_message["content"].count("dropped from the digest")
    report.update({
        "compacted": bool(newly_folded),
        "messages_after": len(rebuilt),
        "prompt_tokens_after": total,
        "percent_after": round((total / budget.input_ceiling * 100.0) if budget.input_ceiling else 0.0, 2),
        "digest_tokens": digest_cost,
        "digest_entries": len(entries),
        "stored_reused": len([entry for entry in entries if entry.get("stored")]),
        "stored_written": len(newly_folded),
        "entries": entries,
        "digest_budget_tokens": int(budget.input_ceiling * DIGEST_BUDGET_RATIO),
        "fits": total <= budget.input_ceiling,
        "overflow": total > budget.input_ceiling,
        "reason": ("under threshold" if total <= budget.threshold_tokens else
                   "no older exchanges remain to fold"),
        "elapsed_ms": int((time.time() - started) * 1000),
        "messages": rebuilt,
    })
    report["information_loss"] = bool(report["digest_dropped_entries"])
    return report


# ---------------------------------------------------------------------------
# Cache diagnostics
# ---------------------------------------------------------------------------

def prompt_signature(messages) -> str:
    digest = hashlib.sha256()
    for message in messages or []:
        digest.update(str(message.get("role", "")).encode("utf-8", "replace"))
        digest.update(b"\0")
        digest.update(message_text(message).encode("utf-8", "replace"))
        digest.update(b"\0")
    return digest.hexdigest()[:32]


def stable_prefix_tokens(previous_messages, current_messages) -> int:
    """Tokens identical between two consecutive prompts.

    Providers charge far less (and answer far faster) for the part of a prompt
    that is byte-identical to the previous one, so this is the local,
    network-independent half of prefix-cache diagnostics.
    """
    previous = list(previous_messages or [])
    current = list(current_messages or [])
    shared = 0
    tokens = 0
    for previous_message, current_message in zip(previous, current):
        if not isinstance(previous_message, dict) or not isinstance(current_message, dict):
            break
        if previous_message.get("role") != current_message.get("role"):
            break
        if message_text(previous_message) != message_text(current_message):
            break
        tokens += message_tokens(current_message)
        shared += 1
    return tokens


def prefix_stability(previous_messages, current_messages) -> dict:
    shared = stable_prefix_tokens(previous_messages, current_messages)
    total = messages_tokens(current_messages)
    return {
        "shared_prefix_tokens": shared,
        "prompt_tokens": total,
        "shared_prefix_percent": round((shared / total * 100.0) if total else 0.0, 2),
        "messages_shared": min(len(previous_messages or []), len(current_messages or [])),
        "prompt_signature": prompt_signature(current_messages),
        "previous_signature": prompt_signature(previous_messages),
        "note": "Identical leading messages are what a prefix cache can reuse.",
    }


def provider_cache_report(usage, estimated_prompt_tokens: int = 0) -> dict:
    """Interpret whatever usage counters the provider actually returned."""
    report = {
        "reported": False,
        "cached_tokens": 0,
        "cache_hit_percent": 0.0,
        "new_prompt_tokens": 0,
        "reason": "The provider returned no usage counters for this request.",
    }
    if not isinstance(usage, dict) or not usage:
        return report

    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached = 0
    if isinstance(details, dict):
        for key in ("cached_tokens", "cache_read_input_tokens", "cache_read_tokens"):
            value = details.get(key)
            if isinstance(value, (int, float)) and int(value) > 0:
                cached = int(value)
                break
    cache_write = 0
    if isinstance(details, dict):
        for key in ("cache_creation_input_tokens", "cache_write_tokens"):
            value = details.get(key)
            if isinstance(value, (int, float)) and int(value) > 0:
                cache_write = int(value)
                break

    if not prompt_tokens:
        return report

    report.update({
        "reported": True,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "cached_tokens": cached,
        "cache_write_tokens": cache_write,
        "cache_hit_percent": round((cached / int(prompt_tokens) * 100.0) if prompt_tokens else 0.0, 2),
        "new_prompt_tokens": max(0, int(prompt_tokens) - cached),
        "reason": (
            "Provider reported cached prompt tokens." if cached else
            "Provider reports usage but no cache counters; prefix caching may be disabled upstream."
        ),
    })
    if estimated_prompt_tokens:
        report["estimated_prompt_tokens"] = int(estimated_prompt_tokens)
        report["estimate_accuracy_percent"] = round(
            (int(prompt_tokens) / max(1, int(estimated_prompt_tokens))) * 100.0, 2)
    return report


def probe_prefix_cache(send_request, system_prompt: str, tools_spec, messages, max_tokens: int = 16) -> dict:
    """Optional live probe: send the same prefix twice and compare usage.

    ``send_request(messages)`` must return the provider ``usage`` dict. Only runs
    when the user asks for it, because it costs real requests.
    """
    if not callable(send_request):
        return {"tested": False, "reason": "No request hook available."}
    probe_messages = list(messages or [])
    if not probe_messages:
        return {"tested": False, "reason": "No conversation to probe."}
    cold_head = probe_messages[:1]
    warm = probe_messages[:1] + probe_messages[1:2]
    results = []
    for label, subset in (("prefix", cold_head), ("extended", warm)):
        try:
            started = time.time()
            usage = send_request(subset, max_tokens)
            results.append({
                "stage": label,
                "messages": len(subset),
                "latency_ms": int((time.time() - started) * 1000),
                "usage": usage if isinstance(usage, dict) else {},
            })
        except Exception as error:
            return {"tested": False, "reason": f"Probe request failed: {error}", "stages": results}
    second = provider_cache_report(results[-1].get("usage"))
    return {
        "tested": True,
        "stages": results,
        "cached_tokens": second.get("cached_tokens", 0),
        "cache_hit_percent": second.get("cache_hit_percent", 0.0),
        "verdict": (
            "The provider reused the shared prefix." if second.get("cached_tokens") else
            "No cache reuse was reported for the shared prefix."
        ),
    }


def recommendations(snapshot: dict, cache_report: dict, stability: dict) -> list:
    advice = []
    if snapshot.get("percent_of_input_ceiling", 0) >= snapshot.get("budget", {}).get("threshold_percent", 78):
        advice.append("Compaction is active: the oldest exchanges are folded into a digest each request.")
    if cache_report and not cache_report.get("reported"):
        advice.append("The backend exposes no usage counters, so KV/prefix-cache hits can only be inferred from prompt stability.")
    if stability and stability.get("shared_prefix_percent", 0) < 60:
        advice.append("The prompt changes a lot between turns; keeping the system prompt and digest byte-stable raises the cache hit rate.")
    if snapshot.get("estimated_only"):
        advice.append("Install tiktoken for exact token counts; the heuristic estimate is currently in use.")
    if not advice:
        advice.append("Prompt structure is stable and inside budget; no action needed.")
    return advice
