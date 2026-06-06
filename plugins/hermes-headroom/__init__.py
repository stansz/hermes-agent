"""
Headroom context compression plugin for Hermes Agent.

Hooks into transform_tool_result to compress large tool outputs before they
enter the LLM context window.  Uses the Headroom SDK's multi-strategy
compression pipeline (SmartCrusher for JSON, AST-aware for code, specialised
compressors for logs/diffs/search results).  Originals are stored in the SDK's
CCR store and retrievable via the headroom_retrieve tool.

Install (into Hermes venv):
    ./venv/bin/python3 -m pip install "headroom-ai[proxy]"
    # The [proxy] extra is REQUIRED — pulls in transformers + onnxruntime
    # needed by the Kompress text compression strategy. Without it, only
    # JSON/code compression works; text content passes through uncompressed.

Enable in ~/.hermes/config.yaml:
    plugins:
      enabled:
        - hermes-headroom
"""

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from headroom.compress import compress
from headroom.cache.compression_store import get_compression_store

logger = logging.getLogger("hermes-headroom")

# ── Stats ──────────────────────────────────────────────────────────────
_stats = {
    "compressions": 0,
    "original_tokens": 0,
    "compressed_tokens": 0,
    "tokens_saved": 0,
    "full_retrievals": 0,
    "search_retrievals": 0,
    "tokens_added_by_full": 0,
    "tokens_added_by_search": 0,
}
_stats_lock = threading.RLock()
STATS_FILE = os.path.expanduser("~/.hermes/headroom_stats.json")


def _save_stats(increments=None):
    with _stats_lock:
        import tempfile
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(STATS_FILE))
        with os.fdopen(fd, "w") as f:
            json.dump(_stats, f)
        os.replace(tmp, STATS_FILE)
        if increments:
            for k, v in increments.items():
                if k in _stats:
                    _stats[k] += v


def _load_config():
    """Load plugin config from ~/.hermes/config.yaml (hermes-headroom key)."""
    import yaml
    defaults = {
        "min_tokens": 500,       # skip outputs smaller than this
        "target_ratio": None,    # None = let SDK decide; e.g. 0.3 = keep 30%
        "protect_recent": 4,     # don't compress last N tool results (SDK default)
        "timeout": 15,           # seconds per compression attempt
        "model": "deepseek-v4-pro",  # used for token counting / context limits
    }
    config_path = os.path.expanduser("~/.hermes/config.yaml")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            user = cfg.get("hermes-headroom") or {}
            if isinstance(user, dict):
                defaults.update({k: v for k, v in user.items() if v is not None})
        except Exception:
            pass
    return defaults


_config = _load_config()

# ── Counter for protect_recent ──────────────────────────────────────────
_result_count = 0
_result_lock = threading.Lock()

# ── Excluded tools (never compress these) ───────────────────────────────
_EXCLUDE = {
    "write_file", "patch", "read_file",
    "memory", "memory_save", "memory_search",
    "delegate_task", "clarify", "todo", "cronjob",
    "headroom_retrieve",
    "vision_analyze",
}


def _quick_token_estimate(text: str) -> int:
    """Rough token count — good enough for min_tokens gating."""
    return len(text) // 4


# ── Hook: transform_tool_result ─────────────────────────────────────────

def _on_transform_tool_result(
    tool_name: str = "",
    tool_call_id: str = "",
    result: str = "",
    args: dict | None = None,
    **kwargs,
) -> str | None:
    """Compress large tool outputs before they enter context.

    Returns the compressed result (or None to leave unchanged on failure).
    """
    global _result_count

    if tool_name in _EXCLUDE:
        return None

    if not result or len(result) < 200:
        return None

    with _result_lock:
        _result_count += 1
        count = _result_count

    # Honour protect_recent
    if count <= _config["protect_recent"]:
        return None

    rough_tokens = _quick_token_estimate(result)
    if rough_tokens < _config["min_tokens"]:
        return None

    try:
        # The SDK expects messages in Anthropic format
        messages = [{"role": "tool", "content": result, "name": tool_name}]

        compress_opts = {}
        if _config.get("target_ratio") is not None:
            compress_opts["target_ratio"] = _config["target_ratio"]

        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(
                compress,
                messages,
                model=_config["model"],
                **compress_opts,
            )
            try:
                cr = fut.result(timeout=_config["timeout"])
            except FutureTimeout:
                logger.warning("headroom: compression timed out for %s", tool_name)
                return None

        if cr.tokens_saved <= 0:
            return None

        # Store original in CCR so it can be retrieved
        store = get_compression_store()
        hash_key = store.store(
            original=result,
            compressed=cr.messages[0]["content"],
            original_tokens=cr.tokens_before,
            compressed_tokens=cr.tokens_after,
            tool_name=tool_name,
        )

        with _stats_lock:
            _stats["compressions"] += 1
            _stats["original_tokens"] += cr.tokens_before
            _stats["compressed_tokens"] += cr.tokens_after
            _stats["tokens_saved"] += cr.tokens_saved
        _save_stats(increments={
            "compressions": 1,
            "original_tokens": cr.tokens_before,
            "compressed_tokens": cr.tokens_after,
            "tokens_saved": cr.tokens_saved,
        })

        compressed_content = cr.messages[0]["content"]
        return (
            f"{compressed_content}\n\n"
            f"[Content compressed. Use headroom_retrieve(hash=\"{hash_key}\") "
            f"to see full original ({cr.tokens_saved:,} tokens saved)]"
        )

    except Exception as exc:
        logger.error("headroom: compression error for %s: %s", tool_name, exc)
        return None


# ── Command: /headroom ──────────────────────────────────────────────────

def _cmd_headroom(args: str) -> str:
    """Show session compression stats."""
    with _stats_lock:
        s = dict(_stats)
    c = s["compressions"]
    orig = s["original_tokens"]
    comp = s["compressed_tokens"]
    saved = s["tokens_saved"]
    added = s["tokens_added_by_full"] + s["tokens_added_by_search"]
    net = saved - added
    pct = (net / orig * 100) if orig > 0 else 0

    return (
        f"🗜️ **Headroom Session Stats**\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Compressions: {c}\n"
        f"Tokens seen:  {orig:,}\n"
        f"Tokens sent:  {comp:,}\n"
        f"Gross saved:  {saved:,}\n"
        f"Retrieved:    {added:,}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**Net savings: {net:,} tokens ({pct:.1f}%)**"
    )


# ── Tool: headroom_retrieve ─────────────────────────────────────────────

def _headroom_retrieve_handler(*args, **kwargs) -> str:
    """Retrieve original content by hash, optionally with keyword/line-range."""
    if args and isinstance(args[0], dict):
        kwargs.update(args[0])

    hash_key = kwargs.get("hash", "").strip()
    if not hash_key:
        return "Error: 'hash' parameter is required."

    store = get_compression_store()

    start_line = kwargs.get("start_line")
    end_line = kwargs.get("end_line")
    query = kwargs.get("query", "").strip() or None
    keyword_query = kwargs.get("keyword_query", "").strip() or None

    # Line-range extraction
    if start_line is not None and end_line is not None:
        entry = store.retrieve(hash_key)
        if not entry:
            return f"Error: No content for hash={hash_key[:12]}..."
        lines = entry.original_content.split("\n")
        snippet = "\n".join(lines[max(0, start_line - 1):min(len(lines), end_line)])
        return f"# Lines {start_line}-{end_line} of hash={hash_key[:12]}...\n\n{snippet}"

    # Full retrieval (no query)
    if not query and not keyword_query:
        entry = store.retrieve(hash_key)
        if not entry:
            return f"Error: No content for hash={hash_key[:12]}..."
        toks = len(entry.original_content) // 4
        with _stats_lock:
            _stats["full_retrievals"] += 1
            _stats["tokens_added_by_full"] += toks
        _save_stats({"full_retrievals": 1, "tokens_added_by_full": toks})
        return entry.original_content

    # Keyword / semantic search
    search_query: str = query or keyword_query  # type: ignore[assignment]
    assert search_query is not None  # guarded by check above
    results = store.search(hash_key, search_query)
    if not results:
        return f"No matches for '{search_query}' in hash={hash_key[:12]}..."

    formatted = []
    total_tokens = 0
    for r in (results or [])[:10]:
        text = r.get("text", r.get("content", str(r))) if isinstance(r, dict) else str(r)
        formatted.append(f"[Match]\n{text}")
        total_tokens += len(text) // 4

    with _stats_lock:
        _stats["search_retrievals"] += 1
        _stats["tokens_added_by_search"] += total_tokens
    _save_stats({"search_retrievals": 1, "tokens_added_by_search": total_tokens})

    return f"# Results for '{search_query}' in hash={hash_key[:12]}...\n\n" + "\n\n".join(formatted)


# ── Plugin entry point ──────────────────────────────────────────────────

def register(ctx):
    """Register hooks, tool, and command with Hermes."""
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)

    ctx.register_tool(
        name="headroom_retrieve",
        toolset="headroom",
        schema={
            "parameters": {
                "type": "object",
                "properties": {
                    "hash": {
                        "type": "string",
                        "description": "Hash key from compressed content marker."
                    },
                    "query": {
                        "type": "string",
                        "description": "Optional keyword/semantic search within original."
                    },
                    "keyword_query": {
                        "type": "string",
                        "description": "Optional exact keyword search (BM25)."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional. Extract from this line (1-indexed)."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional. Extract to this line (1-indexed)."
                    },
                },
                "required": ["hash"],
            }
        },
        handler=_headroom_retrieve_handler,
        description="Retrieve original content from compressed tool outputs.",
        emoji="🗜️",
    )

    ctx.register_command(
        name="headroom",
        handler=_cmd_headroom,
        description="View Headroom compression stats",
        args_hint="",
    )
