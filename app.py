import os
if __name__ == "__main__" and hasattr(os, "geteuid") and os.geteuid() == 0:
    raise SystemExit("Dark Matter Studio must run as the normal desktop user, never with sudo/root.")
import re
import json
import time
import subprocess
import uuid
import sqlite3
import requests
from flask import Flask, request, Response, jsonify, send_from_directory, render_template_string
from context_manager import compact as compact_context, load_stored_entries, persist_entries, measure as measure_context
from agent_tools import (
    SCREENSHOT_DIR,
    get_workspace_dir,
    set_workspace_dir,
    list_available_projects,
    get_coding_skills_for_task,
    dispatch_tool_call,
    extract_all_tool_calls
)

# =====================================================================
# CONFIGURATION
# =====================================================================
COLIBRI_BASE_URL   = os.environ.get("COLIBRI_BASE_URL", "http://127.0.0.1:8000")
LITELLM_BASE_URL   = os.environ.get("LITELLM_BASE_URL", "http://127.0.0.1:4000")
COLIBRI_MODEL      = os.environ.get("COLIBRI_MODEL", "qwen3-coder-30b-a3b-colibri")
COLIBRI_MODELS_DIR = os.environ.get("COLIBRI_MODELS_DIR", "/media/local-ai/Model/models")
COLIBRI_MODEL_DIR  = os.environ.get("COLIBRI_MODEL_DIR", "/media/local-ai/Model/models/qwen3-coder-30b")
COLIBRI_MAX_TOKENS = 2048
MAX_HISTORY_MESSAGES = 14

THINKING_BUDGET_MAP = {
    "off": 0,
    "low": 1024,
    "medium": 4096,
    "high": 16384
}

# Single source of truth for inference authentication. One API key is used
# for both the Colibri and LiteLLM backends; it is sent as a Bearer header on
# every request. The value is configured via DARK_MATTER_API_KEY (falling back
# to COLIBRI_API_KEY / LITELLM_API_KEY for compatibility) and can also be set
# through the Studio UI.
_API_KEY_ENV_VARS = ("DARK_MATTER_API_KEY", "COLIBRI_API_KEY", "LITELLM_API_KEY", "LITELLM_MASTER_KEY")

def _resolve_api_key():
    for var in _API_KEY_ENV_VARS:
        val = os.environ.get(var)
        if val:
            return val
    return ""

API_KEY = _resolve_api_key()

def _auth_headers():
    return {"Authorization": "Bearer " + API_KEY} if API_KEY else {}

INITIAL_LOADER = os.environ.get("DARK_MATTER_DEFAULT_LOADER", os.environ.get("COLIBRI_DEFAULT_LOADER", "Colibri"))
if INITIAL_LOADER not in ("Colibri", "LiteLLM"):
    INITIAL_LOADER = "Colibri"
INITIAL_MODEL = COLIBRI_MODEL
LOADER_STATE = {
    "loader": INITIAL_LOADER,
    "model": INITIAL_MODEL,
    "loaded": False,
    "api_key": API_KEY,
    "settings": {
        "temperature": 0.2,
        "top_p": 0.9,
        "num_ctx": 8192,
        "num_predict": COLIBRI_MAX_TOKENS,
        "question_mode": "manual"
    }
}
LOADER_URLS = {"Colibri": COLIBRI_BASE_URL, "LiteLLM": LITELLM_BASE_URL}

def _loader_base_url(loader=None):
    loader = loader or LOADER_STATE.get("loader", "Colibri")
    return LOADER_URLS.get(loader, COLIBRI_BASE_URL).rstrip("/")

def _chat_completions_url(loader=None):
    return _loader_base_url(loader) + "/v1/chat/completions"

def _models_url(loader=None):
    return _loader_base_url(loader) + "/v1/models"

CONTEXT_WINDOW_DEFAULT = int(os.environ.get("COLIBRI_CONTEXT_WINDOW", "65536"))
CONTEXT_COMPACT_PERCENT = float(os.environ.get("COLIBRI_COMPACT_PERCENT", "75"))
PYTHON_ENV_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".python_environment.json")
CONTEXT_LAST_REPORT = {}
CONTEXT_SUMMARY_EXTRA_PROMPT = os.environ.get("COLIBRI_CONTEXT_SUMMARY_PROMPT", "").strip()

def _python_environment_candidates():
    root = os.path.realpath(get_workspace_dir())
    studio_root = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(root, ".venv", "bin", "python"),
        os.path.join(root, "venv", "bin", "python"),
        os.path.join(root, "env", "bin", "python"),
        os.path.join(studio_root, ".venv", "bin", "python"),
        os.path.join(studio_root, "venv", "bin", "python"),
        os.path.join(studio_root, "env", "bin", "python"),
        os.sys.executable,
        "/home/local-ai/venv/bin/python"
    ]
    found = []
    for path in candidates:
        path = os.path.abspath(os.path.expanduser(path))
        if os.path.isfile(path) and os.access(path, os.X_OK) and path not in found:
            found.append(path)
    return found


def _get_selected_python():
    candidates = _python_environment_candidates()
    try:
        saved = json.load(open(PYTHON_ENV_CONFIG, encoding="utf-8"))
        path = os.path.abspath(os.path.expanduser(saved.get("python", "")))
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    except Exception:
        pass
    return candidates[0] if candidates else os.path.realpath(os.sys.executable)


def _tool_catalogue_summary():
    paths = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools_skills.md"),
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills", "tools_skills.md")]
    chunks = []
    for path in paths:
        try:
            content = open(path, encoding="utf-8").read().strip()
            if content:
                chunks.append(content)
                break
        except OSError:
            continue
    if not chunks:
        return "Tool catalogue: use list_coding_skills to discover available tools; never invent tool names."
    # Keep a compact, complete reference without duplicating the much larger initialization guide.
    return "\n".join(chunks)[:14000]


def _compact_messages_for_request(session_id, messages, model, settings):
    """Compact model-facing history; original chat rows remain untouched."""
    global CONTEXT_LAST_REPORT
    active_loader = LOADER_STATE.get("loader", "Colibri")
    window = int(settings.get("num_ctx", CONTEXT_WINDOW_DEFAULT) or CONTEXT_WINDOW_DEFAULT)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        stored = load_stored_entries(conn, session_id)
        report = compact_context(messages, messages[0].get("content", "") if messages else "", None,
                                 window, CONTEXT_COMPACT_PERCENT, int(settings.get("num_predict", COLIBRI_MAX_TOKENS) or COLIBRI_MAX_TOKENS),
                                 window_source="loader_setting", digest_role="system", min_recent_messages=2, stored_entries=stored)
        if report.get("entries"):
            persist_entries(conn, session_id, report["entries"])
        conn.commit()
    rebuilt = report.get("messages", list(messages))
    system_parts = [m.get("content", "") for m in rebuilt if m.get("role") == "system"]
    non_system = [m for m in rebuilt if m.get("role") != "system"]
    if system_parts:
        rebuilt = [{"role": "system", "content": "\n\n".join(system_parts)}] + non_system

    CONTEXT_LAST_REPORT = {k: v for k, v in report.items() if k not in ("messages", "entries")}
    return rebuilt


COLIBRI_PROCESS = None
COLIBRI_START_SCRIPT = os.environ.get("COLIBRI_START_SCRIPT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "start_colibri.sh"))

APP_PORT = 5000
DB_PATH = os.path.join(get_workspace_dir(), "chat_history.db")

DARK_MATTER_SYSTEM_PROMPT = """You are an autonomous Senior Systems, Coding, & Automation Engineer (Qwen3-Coder 30B on Dark Matter).
Active Workspace Directory: {workspace}

# Tool Skill Calling Format
The exact tool-call format is loader-specific. Follow the active loader's instructions in the SHARED AGENT↔MODEL JOB LOOP section below. Do not mix tool-call formats. Wait for the matching `<tool_response>` before assuming an operation succeeded.

## Available Tool Skills
PERMISSION AND OWNERSHIP SAFETY (MANDATORY): All file/folder reads, writes, creates, backups, diffs, renames, moves and deletes run as the ordinary Studio user. Never use sudo/root for app.py, shell commands, file contents, or file/folder operations. Never retry a failed operation as root. Diagnose access with inspect_workspace_permissions first. Do not automatically chmod/chown, do not use 777, and never recurse. Only repair_workspace_metadata may request an explicitly approved, single-target chmod/chown through PolicyKit; after repair, retry the original operation as the ordinary user.
1. `read_workspace_file`: Read UTF-8 text with line numbers and return `sha256`, byte size, and line count.
   - Parameters: `path` (str), `line_start` (int, default 1), `line_count` (int, default 400; maximum 2000)
2. `write_workspace_file`: Create or atomically replace a UTF-8 text file in workspace.
   - Parameters: `path`, `content`, optional `expected_sha256`, `create_only` (default false), `backup` (default true). Refuses stale versions; existing files get timestamped backups and a unified diff is returned.
3. `patch_workspace_file`: Exact surgical replacement.
   - Parameters: `path`, `search_pattern`, `replacement`, optional `expected_sha256`, `replace_all` (default false), `backup` (default true). Ambiguous multi-matches are rejected by default; returns diff/hash/backup.
4. `edit_workspace_lines`: Atomically replace validated 1-based line range `[start_line, end_line]`.
   - Parameters: `path`, `start_line`, `end_line`, `new_content`, optional `expected_sha256`, `backup` (default true). Returns diff/hash/backup.
5. `list_workspace_files`: Directory tree with file sizes.
   - Parameters: `relative_path` (str, default "."), `max_depth` (int, default 2)
6. `grep_workspace`: Search code across files using regex.
   - Parameters: `pattern` (str), `path` (str, default "."), `file_pattern` (str, default "*")
7. `find_workspace_files`: Find files by glob filename pattern.
   - Parameters: `pattern` (str), `path` (str, default ".")
8. `check_python_syntax`: AST syntax verifier for Python files.
   - Parameters: `path` (str)
9. `git_workspace_status`: Run git operations (`status`, `diff`, `log`, `branch`).
   - Parameters: `action` (str), `args` (str)
10. `execute_terminal`: Execute bash/shell commands inside the active workspace.
    - Parameters: `command` (str), `timeout` (int, default 120), `cwd` (str, optional workspace-relative directory), `confirm` (bool, default false), `max_output_chars` (int, default 24000).
    - Destructive/system-impacting commands return `confirmation_required` without execution. Ask the user for explicit approval, then repeat the exact command with `confirm=true`. Never set confirm=true based on model judgment alone.
    - Commands are constrained to an existing working directory inside the active workspace. Timed-out process groups are terminated and output is bounded.
11. `web_search`: Live DuckDuckGo web search.
    - Parameters: `query` (str), `max_results` (int, default 5)
12. `fetch_web_page`: Scrape readable text from a URL.
    - Parameters: `url` (str)
13. `test_chrome_extension`: Launch Chromium with unpacked extension & capture console logs.
    - Parameters: `extension_dir` (str), `target_url` (str)
14. `capture_desktop`: Capture X11 primary monitor screenshot.
    - Parameters: `filename` (str, optional)
15. `execute_pc_action`: Control X11 mouse/keyboard.
    - Parameters: `action_type` (`click`, `double_click`, `right_click`, `drag`, `scroll_up`, `scroll_down`, `type`, `key`), `x` (int), `y` (int), `text` (str)
16. `check_python_dependencies`: Inspect `requirements.txt` and `requirements-dev.txt` for missing installed packages without changing the environment.
   - Parameters: `path` (str, default ".")
17. `install_python_package`: Install a named PyPI package with pip after dependency checking.
   - Parameters: `package` (str, e.g. `requests>=2.31`), `upgrade` (bool, default false), `timeout` (int, default 300)
18. `read_studio_skill`: Read a bundled Studio text guide or language skill on demand.
   - Parameters: `name` (str: `initialization`, `python`, `html`, `javascript`, or `tools_skills.md`)
19. `install_missing_python_module`: Install a missing Python import into the selected interpreter and verify it imports.
    - Parameters: `module` (str, required import name such as `yaml` or `PIL.Image`), `package` (str, optional PyPI distribution name when it differs), `timeout` (int, default 300).
20. `inspect_workspace_permissions`: Read ownership and effective access for a workspace path; makes no changes.
    - Parameters: `path` (str, default `.`)
21. `repair_workspace_metadata`: Request a single-path chmod/chown through PolicyKit. Requires explicit user approval; never set `approved=true` unless the user approved that specific change.
    - Parameters: `path`, `action` (`chmod` or `chown`), `approved`, and the relevant `mode` or `owner`/`group`.
22. `run_project_checks`: Check Python syntax and JSON validity in the workspace.
    - Parameters: `path` (str, default `.`)
23. `inspect_project`: Identify project root, languages, and important files.
    - Parameters: `path` (str, default `.`)
24. `run_project_tests`: Run project tests using the selected Python environment.
    - Parameters: `path`, optional `command`, `timeout`.
25. `lint_project`: Run lint checks using the selected Python environment.
    - Parameters: `path`, optional `command`, `timeout`.
26. `search_workspace_code`: Search source code for a text query.
    - Parameters: `query`, `path`, optional `max_results`.
27. `git_status`, `git_diff`, `git_workspace_status`: Read Git status/diff or run a supported Git inspection action.
28. `compare_python_api`: Compare public Python signatures between two workspace files.
    - Parameters: `reference_path`, `candidate_path`.
29. `list_coding_skills`, `load_coding_skill`, `read_studio_skill`: Discover/load task skills or read a Studio guide.
30. `read_project_memory`, `write_project_memory`: Read/write non-secret project notes; write only useful durable facts.
31. `diagnose_tool_error`: Analyze a prior tool error.
    - Parameters: `tool_name`, `target`, and the prior `result` object.

## Dependency Workflow
When Python execution reports `ModuleNotFoundError` or `ImportError`, inspect the project's requirements and the missing import. Then call `install_missing_python_module` with the import name and its PyPI distribution (omit `package` only when the names are the same). This installs into the selected Python environment and verifies the import. Rerun the failed command or test after a successful install; do not stop at installation. For other command failures, use the returned exit code, command, working directory, interpreter, and output to diagnose and continue the same task. A timeout is a failed execution and must be reported and recovered from, not treated as success.

Always read or inspect existing code before patching. After all tool steps succeed, provide a clear final answer with no `<tool_call>` blocks."""

app = Flask(__name__)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                updated_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS context_summaries (
                session_id TEXT NOT NULL,
                turn_key TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                summary TEXT NOT NULL,
                tokens INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                PRIMARY KEY (session_id, turn_key)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS context_model_summaries (
                session_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                source_hash TEXT NOT NULL DEFAULT '',
                updated_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                files_json TEXT,
                tools_json TEXT,
                metrics_json TEXT,
                created_at REAL,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)
        conn.commit()

init_db()
try:
    with sqlite3.connect(DB_PATH) as _settings_conn:
        _saved_prompt = _settings_conn.execute("SELECT value FROM app_settings WHERE key='context_summary_extra_prompt'").fetchone()
        if _saved_prompt:
            CONTEXT_SUMMARY_EXTRA_PROMPT = _saved_prompt[0]
except Exception:
    pass

# =====================================================================
# COLIBRI PAYLOAD VERIFICATION & FORMATTER
# =====================================================================

def detect_colibri_model():
    global DETECTED_COLIBRI_MODEL
    try:
        r = requests.get(f"{COLIBRI_BASE_URL}/v1/models", timeout=3.0)
        if r.status_code != 200:
            return
        raw = r.json().get("data", [])
        ids = [item.get("id") for item in raw if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("id").strip()]
        if DEFAULT_MODEL in ids:
            DETECTED_COLIBRI_MODEL = DEFAULT_MODEL
        elif len(ids) == 1:
            DETECTED_COLIBRI_MODEL = ids[0]
        elif len(ids) > 1:
            DETECTED_COLIBRI_MODEL = DEFAULT_MODEL
    except Exception:
        pass


def verify_colibri_payload(payload: dict) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "Payload must be a dictionary."
    if "model" not in payload or not isinstance(payload["model"], str):
        return False, "Missing or invalid 'model' string in payload."
    if "messages" not in payload or not isinstance(payload["messages"], list):
        return False, "Missing or invalid 'messages' list in payload."
    messages = payload["messages"]
    if len(messages) == 0:
        return False, "Messages array cannot be empty."
    if not isinstance(messages[0], dict) or messages[0].get("role") != "system":
        return False, "First message must be a system message for the configured Colibri chat template."

    last_role = None
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return False, f"Message at index {idx} is not a dictionary."
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("system", "user", "assistant"):
            return False, f"Invalid message role '{role}' at index {idx}."
        if idx > 0 and role == "system":
            return False, f"System message is only supported at index 0 (found at index {idx})."
        if not isinstance(content, str):
            return False, f"Message content at index {idx} must be a string."
        if role != "system" and not content.strip():
            return False, f"Empty content string for role '{role}' at index {idx}."
        if role in ("user", "assistant"):
            if last_role == role:
                return False, f"Consecutive duplicate roles ('{role}') detected at indices {idx-1} and {idx}. Strict alternation required."
            last_role = role
    return True, "Payload format verified successfully."


SKILL_FILES = {
    "python_skills.md": "Python",
    "HTML_skills.md": "HTML",
    "JS_skills.md": "JavaScript",
    "tools_skills.md": "Agent Tools",
}
SKILLS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skills")
MAX_SKILL_CHARS = 18000


def load_selected_skill_context(selected_skill_files):
    if not isinstance(selected_skill_files, list):
        selected_skill_files = list(SKILL_FILES.keys())
    blocks = []
    total_chars = 0
    for filename in selected_skill_files:
        if filename not in SKILL_FILES:
            continue
        path = os.path.join(SKILLS_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read().strip()
        except OSError:
            continue
        if not content:
            continue
        remaining = MAX_SKILL_CHARS - total_chars
        if remaining <= 0:
            break
        content = content[:remaining]
        blocks.append(f"## {SKILL_FILES[filename]} skill ({filename})\n{content}")
        total_chars += len(content)
    return "\n\n".join(blocks)


def _workspace_inventory(max_entries: int = 100) -> str:
    root = os.path.realpath(get_workspace_dir())
    ignored = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build"}
    lines = [f"Workspace root: {root}"]
    if not os.path.isdir(root):
        return "\n".join(lines + ["Workspace status: directory is missing or not accessible."])
    count = 0
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in ignored and not d.startswith("."))
        rel_dir = os.path.relpath(current, root)
        depth = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1
        if depth > 3:
            dirs[:] = []
            continue
        for name in sorted(dirs):
            rel = os.path.relpath(os.path.join(current, name), root)
            lines.append(f"DIR  {rel}/")
            count += 1
            if count >= max_entries:
                lines.append("... inventory truncated; use list_workspace_files for more.")
                return "\n".join(lines)
        for name in sorted(files):
            if name.startswith(".") and name not in {".env.example", ".gitignore"}:
                continue
            rel = os.path.relpath(os.path.join(current, name), root)
            lines.append(f"FILE {rel}")
            count += 1
            if count >= max_entries:
                lines.append("... inventory truncated; use list_workspace_files for more.")
                return "\n".join(lines)
    if count == 0:
        lines.append("No visible files found at the workspace root yet.")
    return "\n".join(lines)


def is_exact_duplicate_completed_response(previous: str | None, current: str | None) -> bool:
    """True only for consecutive, fully completed, byte-for-byte identical responses."""
    return bool(current) and previous is not None and current == previous


_QUESTION_PATTERNS = [
    re.compile(r"\b(should\s+i|do\s+you\s+want\s+me\s+to|would\s+you\s+like\s+me\s+to|shall\s+i|may\s+i)\b", re.IGNORECASE),
    re.compile(r"\b(please\s+(confirm|approve|say\s+yes|reply|let\s+me\s+know))\b", re.IGNORECASE),
    re.compile(r"\b(need\s+your\s+(approval|confirmation|input|answer|decision))\b", re.IGNORECASE),
    re.compile(r"\b(waiting\s+for\s+your\s+(reply|answer|response|input|confirmation))\b", re.IGNORECASE),
    re.compile(r"\b(ask\s+the\s+user|question\s+for\s+the\s+user)\b", re.IGNORECASE),
    re.compile(r"\b(need\s+to\s+ask\s+the\s+user|must\s+ask\s+the\s+user)\b", re.IGNORECASE),
    re.compile(r"\b(please\s+answer|please\s+choose|please\s+select|please\s+decide)\b", re.IGNORECASE),
    re.compile(r"\b(are\s+you\s+sure|do\s+you\s+agree|is\s+that\s+correct|is\s+this\s+ok)\b", re.IGNORECASE),
    re.compile(r"\b(yes\s*/\s*no|yes\s+or\s+no|yes/no)\b", re.IGNORECASE),
    re.compile(r"\b(should\s+we|can\s+we|do\s+you\s+prefer|which\s+option)\b", re.IGNORECASE),
]


def detect_user_question(text: str) -> str | None:
    """Return the matched question phrase if the model output asks the user a question, else None."""
    if not text:
        return None
    for pattern in _QUESTION_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def detect_message_done(text: str) -> bool:
    """Detect explicit 'message done' / 'task done' / 'job done' completion signals."""
    if not text:
        return False
    return bool(re.search(r"\b(message\s+done|task\s+done|job\s+done|all\s+done|finished\s+the\s+(task|job|work))\b", text, re.IGNORECASE))


# --- Loop-control helpers (question pause / auto-yes / plan checklist) ---

_QUESTION_MARKER_RE = re.compile(
    r"(should i|do you want|can i|may i|shall i|would you like|want me to|"
    r"please confirm|yes or no|confirm: yes|shall we|do you need)",
    re.IGNORECASE,
)
_TOOL_CALL_BLOCK_RE = re.compile("<" + "tool_call>.*?<" + "/tool_call>", re.DOTALL | re.IGNORECASE)
_TASK_STATUS_BLOCK_RE = re.compile("<" + "task_status>.*?<" + "/task_status>", re.DOTALL | re.IGNORECASE)


def looks_like_user_question(text: str) -> bool:
    """Return True when model output is a genuine yes/no question to the user.

    Strips tool-call and task_status blocks first, rejects long outputs, and
    requires both a question mark and an explicit asking phrase so that plain
    statements never pause the loop.
    """
    if not text:
        return False
    cleaned = _TOOL_CALL_BLOCK_RE.sub(" ", text)
    cleaned = _TASK_STATUS_BLOCK_RE.sub(" ", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return False
    if len(cleaned.split()) > 150:
        return False
    if "?" not in cleaned:
        return False
    return bool(_QUESTION_MARKER_RE.search(cleaned))


def looks_like_message_done(text: str) -> bool:
    """Return True when output signals completion via tag or explicit phrase."""
    if not text:
        return False
    if re.search("<" + r"task_status>\s*done\s*</" + "task_status>", text, re.IGNORECASE):
        return True
    return detect_message_done(text)
def build_plan_request_instruction() -> str:
    """Instruction appended when a prompt warrants a plan: ask for a checklist."""
    return (
        "This task warrants a plan. Before taking action, output a brief plan as a "
        "markdown checklist using lines of the form '- [ ] step description'. Keep it "
        "to 3-8 concrete steps. Then begin executing the first step."
    )


def build_plan_generation_instruction() -> str:
    """Instruction asking the model to emit a plan checklist from its reply."""
    return (
        "Convert your approach into a short markdown checklist now. Output only lines "
        "of the form '- [ ] step description' (3-8 steps). Do not execute them yet."
    )


def build_auto_yes_instruction() -> str:
    """Instruction telling the model to proceed autonomously without pausing."""
    return (
        "Auto-continue is enabled. Do not ask yes/no confirmation questions; make the "
        "most reasonable decision, proceed, and only stop when the task is complete or "
        "you are genuinely blocked. Signal completion with the completion tag."
    )


_OPEN_QUESTION_RE = re.compile(
    r"^\s*(who|what|when|where|why|how|which|whom|whose)\b", re.IGNORECASE
)


def auto_yes_reply(text: str) -> str:
    """Return the synthetic user reply Auto-Yes injects for a paused question.

    Yes/no confirmation questions get "yes"; open-ended questions (who/what/...)
    and empty text get "continue" so the model keeps working autonomously.
    """
    if not text:
        return "continue"
    cleaned = _TOOL_CALL_BLOCK_RE.sub(" ", text)
    cleaned = _TASK_STATUS_BLOCK_RE.sub(" ", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return "continue"
    if _OPEN_QUESTION_RE.search(cleaned):
        return "continue"
    if _QUESTION_MARKER_RE.search(cleaned):
        return "yes"
    return "continue"


def normalise_question_mode(value) -> str:
    """Normalise a question-mode setting to 'auto' or 'manual' (default manual)."""
    if not isinstance(value, str):
        return "manual"
    token = value.strip().lower().replace("-", "_")
    if token in ("auto", "auto_yes", "autoyes"):
        return "auto"
    return "manual"


def build_continue_instruction() -> str:
    """Instruction injected on each controller-driven continuation turn."""
    return (
        "Continue the same task from the verified tool results. Do not repeat completed "
        "work. Update your plan checklist, mark finished steps as done, and keep going "
        "until the task is complete or you are blocked."
    )


def build_plan_checklist(total: int, current_step: int, status: str) -> list[dict]:
    """Derive a plan checklist from loop progress.

    ``current_step`` is the number of steps already completed. Steps at or
    below ``current_step`` are ``done``; the next step (``current_step + 1``)
    is ``active`` unless the overall ``status`` is ``complete`` (then it is
    ``done``); later steps are ``pending``. The returned ``status`` mirrors the
    caller-supplied ``status`` for the active step so callers can pass through
    values such as ``running``.
    """
    total = max(0, int(total))
    completed = max(0, int(current_step))
    status = str(status or "").strip().lower()
    items: list[dict] = []
    for index in range(1, total + 1):
        if index <= completed:
            step_status = "done"
        elif index == completed + 1:
            step_status = "done" if status == "complete" else (status or "active")
        else:
            step_status = "pending"
        items.append({"step": index, "status": step_status})
    return items


_PLAN_STATUS_RE = re.compile(r"^\s*[-*]\s*\[([ xX✓])\]\s*(.+?)\s*$")
_PLAN_PLAIN_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.+?)\s*$")


def extract_plan_items(text: str) -> list[dict]:
    """Extract a plan/progress checklist from model output.

    Recognizes markdown checklist lines like ``- [x] step`` / ``- [ ] step``
    and plain numbered/bulleted plan lines. Returns a list of
    ``{"text": str, "status": "done"|"active"|"pending"}`` dicts. The first
    pending item is marked ``active`` so the UI can highlight current progress.
    """
    if not text:
        return []
    items: list[dict] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        status_match = _PLAN_STATUS_RE.match(line)
        if status_match:
            mark, label = status_match.group(1), status_match.group(2).strip()
            status = "done" if mark.lower() in ("x", "✓") else "pending"
            if label:
                items.append({"text": label, "status": status})
            continue
        plain_match = _PLAN_PLAIN_RE.match(line)
        if plain_match:
            label = plain_match.group(1).strip()
            # Ignore lines that are clearly not plan steps (tool calls, code fences).
            if label and not label.startswith("<") and len(label) <= 160:
                items.append({"text": label, "status": "pending"})
    # Mark the first pending item as active to show current progress.
    for item in items:
        if item["status"] == "pending":
            item["status"] = "active"
            break
    return items


def build_shared_agent_loop_instructions(loader: str = "", model: str = "") -> str:
    """Small bootstrap pointer; detailed operating rules live in a local text skill file."""
    if str(loader).lower() == "colibri":
        read_call = "<tool_call><function=read_studio_skill><parameter=name>initialization</parameter></function></tool_call>"
    else:
        read_call = '<tool_call>{"name":"read_studio_skill","arguments":{"name":"initialization"}}</tool_call>'
    return (
        f"\n\n# Colibri Agent Loop ({loader} / {model})\n"
        "For project/code/tool tasks, first load the compact shared guide with " + read_call + ". "
        "Follow its tool-call, recovery, safety, and verification rules. The guide is a local "
        "plain-text file, not a repeated long system prompt. Continue the same job after tool "
        "results; verify acceptance criteria before declaring completion. The orchestrator "
        f"continues until verified completion, cancellation, a blocker, or exact repeated-response detection. "
        "Signal completion with <task_status>done</task_status> (or <task_status>complete</task_status>). "
        "When you need a yes/no decision from the user, ask a single clear yes/no question and pause; "
        "in auto mode the loop answers yes/no automatically and continues.\n"
    )


def build_colibri_messages(session_id: str, current_user_msg: str, selected_skill_files=None, thinking_level="off") -> list[dict]:
    workspace_root = get_workspace_dir()
    sys_prompt = DARK_MATTER_SYSTEM_PROMPT.format(workspace=workspace_root)
    active_loader = LOADER_STATE.get("loader", "Colibri")
    active_model = LOADER_STATE.get("model", DEFAULT_MODEL)
    ensure_initialization_text_file(active_loader)
    sys_prompt += build_shared_agent_loop_instructions(active_loader, active_model)
    sys_prompt += "\n\n# Live Project Directory Connection\nThe active project directory is connected to your local file tools. This is the actual directory you can inspect and edit; do not claim you cannot access it without trying a tool. A live inventory follows. For questions about the project, use this inventory and call list_workspace_files/read_workspace_file/grep_workspace as needed. For edits, inspect the existing file first, then use the relevant write/patch/edit tool. Always use paths relative to the active workspace unless a tool explicitly permits otherwise.\n\n" + _workspace_inventory()
    selected_python = _get_selected_python()
    sys_prompt += f"\n\n# Active Project and Python Runtime (MANDATORY)\nSelected project root: {os.path.realpath(workspace_root)}\nSelected Python interpreter: {selected_python}\nAlways treat this project path as the working directory and this interpreter as the default runtime for Python commands, tests, imports, and package checks. Use the selected interpreter explicitly (for example, '{selected_python}' -m pytest). Do not silently switch to system Python or another virtual environment. If it is unavailable, report that and ask before changing the selection.\n"
    sys_prompt += "\n\n# Complete Tool Catalogue Summary\nUse only the tools listed in the initialized Studio guide. Select the narrowest tool for each action, wait for its actual result, and verify changes.\n" + _tool_catalogue_summary()
    
    if thinking_level != "off":
            depth_guidance = {
                "low": "\n[Reasoning Guideline: Keep chain-of-thought concise, focused, and under 150 words inside <think> tags before acting.]",
                "medium": "\n[Reasoning Guideline: Provide standard structured analytical chain-of-thought inside <think> tags before acting.]",
                "high": "\n[Reasoning Guideline: Provide deep, thorough step-by-step reasoning inside <think> tags, analyzing edge cases and validation plans before executing actions.]"
            }
            sys_prompt += depth_guidance.get(thinking_level, "")

    skill_context = load_selected_skill_context(selected_skill_files)
    if skill_context:
        sys_prompt += "\n\n# User-Enabled Skill Instructions\nUse these instructions when relevant to the current task. They guide your behavior; use the available agent tools when an action is needed.\n\n" + skill_context
    active_skills = get_coding_skills_for_task(current_user_msg)
    if active_skills:
        sys_prompt += "\n\n# Task-Specific Built-in Guidance\n" + active_skills

    formatted = [{"role": "system", "content": sys_prompt}]
    raw_history = []

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT role, content, tools_json FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        )
        rows = cur.fetchall()

    if rows and rows[-1]["role"] == "user":
        rows = rows[:-1]

    rows = rows[-MAX_HISTORY_MESSAGES:]

    for r in rows:
        role = r["role"] if r["role"] in ("user", "assistant") else "user"
        content = (r["content"] or "").strip()
        if role == "assistant":
            content = re.sub(r"<think\b[^>]*>[\s\S]*?</think>", "", content).strip()

        if not content:
            continue

        if raw_history and raw_history[-1]["role"] == role:
            raw_history[-1]["content"] += "\n\n" + content
        else:
            raw_history.append({"role": role, "content": content})

    while raw_history and raw_history[0]["role"] != "user":
        raw_history.pop(0)

    clean_current = (current_user_msg or "Continue.").strip()
    if raw_history and raw_history[-1]["role"] == "user":
        raw_history[-1]["content"] += "\n\n" + clean_current
    else:
        raw_history.append({"role": "user", "content": clean_current})

    formatted.extend(raw_history)
    return formatted

# =====================================================================
# API ROUTES
# =====================================================================

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


def _human_size(num_bytes):
    value = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024


def _stop_colibri_managed(timeout=12):
    global COLIBRI_PROCESS
    proc = COLIBRI_PROCESS
    if proc is None:
        return False, "No Studio-managed Colibri process is recorded."
    if proc.poll() is not None:
        COLIBRI_PROCESS = None
        return True, "Studio-managed Colibri process was already stopped."
    try:
        try:
            os.killpg(proc.pid, __import__('signal').SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, __import__('signal').SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait(timeout=5)
        COLIBRI_PROCESS = None
        return True, "Stopped the Studio-managed Colibri process group; process memory is released by the OS."
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"Could not fully stop Colibri: {exc}"


def _release_linux_page_cache():
    notes = []
    try:
        subprocess.run(["sync"], check=True, timeout=30)
        notes.append("Disk writes synced.")
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"Could not sync disk writes: {exc}"
    if os.name != "posix" or not os.path.exists("/proc/sys/vm/drop_caches"):
        notes.append("Linux page-cache release is unavailable on this system.")
        return True, " ".join(notes)
    if os.geteuid() != 0:
        notes.append("Page-cache drop skipped (Studio is not root). Model RAM is still released by unloading/stopping the backend.")
        return True, " ".join(notes)
    try:
        with open("/proc/sys/vm/drop_caches", "w", encoding="ascii") as cache_file:
            cache_file.write("3\n")
        notes.append("Linux page cache, dentries, and inodes were requested to be dropped after sync.")
        return True, " ".join(notes)
    except (OSError, PermissionError) as exc:
        notes.append(f"Page-cache drop failed: {exc}")
        return False, " ".join(notes)


def _folder_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            try:
                total += os.path.getsize(os.path.join(root, filename))
            except OSError:
                pass
    return total


def _colibri_folder_models():
    items = []
    try:
        entries = sorted(os.scandir(COLIBRI_MODELS_DIR), key=lambda e: e.name.lower())
    except OSError:
        entries = []
    for entry in entries:
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
            size = _folder_size(entry.path)
            if size <= 0:
                continue
            path = os.path.realpath(entry.path)
            items.append({"id": path, "name": entry.name, "size_bytes": size,
                          "size": _human_size(size), "label": f"{_human_size(size)} — {entry.name}",
                          "path": path, "source": "folder"})
        except OSError:
            continue
    return items


@app.route("/api/loader", methods=["GET"])
def loader_info():
    loader = LOADER_STATE["loader"]
    base = _loader_base_url(loader).rstrip("/")
    try:
        if loader == "Colibri":
            models = _colibri_folder_models()
        else:
            headers = _auth_headers()
            models_url = base + "/models"
            r = requests.get(models_url, headers=headers, timeout=8)
            r.raise_for_status()
            api_models = [m.get("id") for m in r.json().get("data", [])
                          if isinstance(m, dict) and m.get("id")]
            models = [{"id": model_id, "name": model_id, "size_bytes": None,
                       "size": "size unknown", "label": f"size unknown — {model_id}",
                       "source": "api"} for model_id in api_models]
        status = "online"
    except (requests.RequestException, ValueError):
        models, status = [], "offline"
    return jsonify({**LOADER_STATE, "models": models, "connection": status})


@app.route("/api/loader/thinking", methods=["GET"])
def loader_thinking_capability():
    loader = LOADER_STATE.get("loader", "Colibri")
    model = (request.args.get("model") or LOADER_STATE.get("model", "")).strip()
    if not model:
        return jsonify({"available": False, "levels": ["off"], "reason": "No model selected."})
    name = model.lower()
    known = any(token in name for token in ("qwen3", "qwq", "deepseek-r1", "deepseek-v3.1", "gpt-oss"))
    return jsonify({
        "available": known,
        "levels": ["off", "low", "medium", "high"] if known else ["off"],
        "reason": "Known thinking model family." if known else "Backend does not report native thinking capability."
    })


@app.route("/api/loader/hardware", methods=["GET"])
def loader_hardware_capabilities():
    try:
        affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0
    except OSError:
        affinity = 0
    cpu_count = os.cpu_count() or 1
    cpu_threads_max = max(1, min(256, affinity or cpu_count))

    gpu = {"available": False, "name": None, "memory_total_mb": None, "memory_free_mb": None, "kind": "none"}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True)
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if rows:
            fields = [v.strip() for v in rows[0].split(",", 2)]
            gpu.update({"available": True, "kind": "NVIDIA", "name": fields[0]})
            if len(fields) > 1: gpu["memory_total_mb"] = int(fields[1])
            if len(fields) > 2: gpu["memory_free_mb"] = int(fields[2])
    except Exception:
        pass

    model = (request.args.get("model") or LOADER_STATE.get("model") or "").strip()
    gpu_layers_max = 99 if gpu["available"] else 0
    return jsonify({"cpu_threads_max": cpu_threads_max, "cpu_threads_detected": cpu_count,
                    "cpu_affinity_threads": affinity or None, "gpu": gpu,
                    "gpu_layers_max": gpu_layers_max, "model_layers": model_layers,
                    "model_layers_source": model_source,
                    "note": "GPU layer maximum is the model layer count, not a VRAM-fit guarantee."})


@app.route("/api/loader/limits", methods=["GET"])
def loader_model_limits():
    loader = request.args.get("loader", LOADER_STATE["loader"])
    model = (request.args.get("model") or LOADER_STATE.get("model") or "").strip()
    limits = {"num_ctx_min": 512, "num_ctx_max": 131072, "num_predict_min": 1, "num_predict_max": 131072, "source": "fallback"}
    if loader in ("Colibri", "LiteLLM"):
        limits.update({"num_ctx_max": 262144, "num_predict_max": 262144, "source": "colibri-default" if loader == "Colibri" else "litellm-config"})
    return jsonify(limits)


@app.route("/api/loader", methods=["POST"])
def configure_loader():
    data = request.json or {}
    action = data.get("action", "configure")
    global COLIBRI_PROCESS
    if action == "start_colibri":
        if LOADER_STATE["loader"] != "Colibri":
            return jsonify({"status":"error", "output":"Switch Loader to Colibri before starting its server."}), 400
        base = LOADER_URLS["Colibri"].rstrip("/")
        try:
            check = requests.get(base + "/v1/models", timeout=2)
            if check.ok:
                return jsonify({"status":"success", "output":"Colibri is already running.", **LOADER_STATE})
        except requests.RequestException:
            pass
        model = data.get("model") or COLIBRI_MODEL_DIR
        if not os.path.isabs(model) or not os.path.isdir(model):
            return jsonify({"status":"error", "output":f"Choose a valid Colibri model folder. Not found: {model}"}), 400
        if not os.path.isfile(COLIBRI_START_SCRIPT):
            return jsonify({"status":"error", "output":f"Colibri start script not found: {COLIBRI_START_SCRIPT}."}), 500
        env = os.environ.copy()
        env.update({"COLIBRI_MODEL_DIR":model, "COLIBRI_CTX":str(int(data.get("num_ctx",65536))), "COLIBRI_CAP":str(int(data.get("colibri_cap",48))), "COLIBRI_PORT":str(int(data.get("colibri_port",8000)))})
        try:
            COLIBRI_PROCESS = subprocess.Popen([COLIBRI_START_SCRIPT, model], cwd="/media/local-ai/Model/colibri", env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            deadline = time.time() + 90
            while time.time() < deadline:
                if COLIBRI_PROCESS.poll() is not None:
                    return jsonify({"status":"error", "output":f"Colibri exited during startup (code {COLIBRI_PROCESS.returncode})."}), 502
                try:
                    ready = requests.get(base + "/v1/models", timeout=2)
                    if ready.ok:
                        LOADER_STATE["model"] = model
                        LOADER_STATE["settings"].update({"num_ctx":int(data.get("num_ctx",65536)), "colibri_cap":int(data.get("colibri_cap",48)), "colibri_port":int(data.get("colibri_port",8000))})
                        return jsonify({"status":"success", "output":"Colibri server is responding. Model folder: " + model, **LOADER_STATE})
                except requests.RequestException:
                    pass
                time.sleep(1)
            return jsonify({"status":"error", "output":"Colibri start timed out after 90 seconds."}), 504
        except Exception as exc:
            return jsonify({"status":"error", "output":f"Could not start Colibri: {exc}"}), 500
    if action == "stop_colibri":
        ok, message = _stop_colibri_managed()
        if ok:
            LOADER_STATE["loaded"] = False
            cache_ok, cache_message = _release_linux_page_cache()
            return jsonify({"status": "success" if cache_ok else "warning", "output": message + " " + cache_message, **LOADER_STATE})
        return jsonify({"status": "error", "output": message, **LOADER_STATE}), 409
    if action == "unload":
        if LOADER_STATE["loader"] == "Colibri":
            ok, message = _stop_colibri_managed()
            if not ok:
                return jsonify({"status": "error", "output": message, **LOADER_STATE}), 409
        LOADER_STATE["loaded"] = False
        cache_ok, cache_message = _release_linux_page_cache()
        return jsonify({"status": "success" if cache_ok else "warning", "output": message + " " + cache_message, **LOADER_STATE})
    if LOADER_STATE["loaded"]:
        return jsonify({"status": "error", "output": "Unload the current model before changing loader or model."}), 409
    loader = data.get("loader", LOADER_STATE["loader"])
    if loader not in ("Colibri", "LiteLLM"):
        return jsonify({"status": "error", "output": "Unknown loader. Use Colibri or LiteLLM."}), 400
    api_key = data.get("api_key") or LOADER_STATE.get("api_key", "")
    if api_key:
        LOADER_STATE["api_key"] = api_key
    model = data.get("model", LOADER_STATE["model"]).strip()
    if not model:
        return jsonify({"status": "error", "output": "Select a model first."}), 400
    base = _loader_base_url(loader).rstrip("/")
    try:
        headers = _auth_headers()
        models_url = base + "/models"
        models_resp = requests.get(models_url, headers=headers, timeout=10)
        models_resp.raise_for_status()
        available = [m.get("id") for m in models_resp.json().get("data", []) if isinstance(m, dict) and m.get("id")]
        if available and model not in available:
            model = available[0]
    except Exception as exc:
        return jsonify({"status": "error", "output": f"Cannot reach {loader} at {base}: {exc}"}), 502
    LOADER_STATE["loader"], LOADER_STATE["model"], LOADER_STATE["loaded"] = loader, model, True
    LOADER_STATE["settings"].update({k: data[k] for k in ("temperature", "top_p", "num_ctx", "num_predict", "question_mode") if k in data})
    return jsonify({"status": "success", **LOADER_STATE})

@app.route("/api/skills", methods=["GET"])
def skills_info():
    return jsonify({"skills": [{"name": label, "file": filename} for filename, label in SKILL_FILES.items()]})

@app.route("/screenshots/<path:filename>")
def serve_screenshot(filename):
    return send_from_directory(SCREENSHOT_DIR, filename)

@app.route("/api/python-environment", methods=["GET", "POST"])
def python_environment_config():
    if request.method == "GET":
        candidates = _python_environment_candidates()
        selected = _get_selected_python()
        return jsonify({"selected": selected, "candidates": candidates, "project": get_workspace_dir(),
                        "studio_root": os.path.dirname(os.path.abspath(__file__))})
    data = request.json or {}
    requested = os.path.abspath(os.path.expanduser(str(data.get("python", "")).strip()))
    if os.path.isdir(requested):
        search_dirs = [requested] if os.path.basename(requested) == "bin" else [os.path.join(requested, "bin")]
        requested = next((os.path.join(directory, name)
                          for directory in search_dirs
                          for name in ("python", "python3", "python3.12")
                          if os.path.isfile(os.path.join(directory, name))
                          and os.access(os.path.join(directory, name), os.X_OK)), "")
    if not requested or not os.path.isfile(requested) or not os.access(requested, os.X_OK):
        return jsonify({"status": "error", "output": "Choose an existing executable Python interpreter."}), 400
    # Explicit paths are allowed so users can point to a custom environment; verify it is Python.
    try:
        probe = subprocess.run([requested, "-c", "import sys; print(sys.executable)"], capture_output=True, text=True, timeout=5)
        if probe.returncode != 0:
            raise RuntimeError(probe.stderr.strip() or "Python interpreter probe failed")
    except Exception as exc:
        return jsonify({"status": "error", "output": f"Interpreter validation failed: {exc}"}), 400
    with open(PYTHON_ENV_CONFIG, "w", encoding="utf-8") as handle:
        json.dump({"python": os.path.abspath(probe.stdout.strip() or requested)}, handle, indent=2)
    return jsonify({"status": "success", "selected": _get_selected_python(), "project": get_workspace_dir()})


@app.route("/api/context/status", methods=["GET"])
def context_status():
    return jsonify({"threshold_percent": CONTEXT_COMPACT_PERCENT, "window_default": CONTEXT_WINDOW_DEFAULT,
                    "summary_extra_prompt": CONTEXT_SUMMARY_EXTRA_PROMPT,
                    "last_report": CONTEXT_LAST_REPORT})


@app.route("/api/context/summary-prompt", methods=["POST"])
def set_context_summary_prompt():
    global CONTEXT_SUMMARY_EXTRA_PROMPT
    data = request.get_json(silent=True) or {}
    value = str(data.get("prompt", ""))[:12000]
    changed = value.strip() != CONTEXT_SUMMARY_EXTRA_PROMPT
    CONTEXT_SUMMARY_EXTRA_PROMPT = value.strip()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO app_settings(key,value) VALUES('context_summary_extra_prompt',?) "
                     "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (CONTEXT_SUMMARY_EXTRA_PROMPT,))
        if changed:
            conn.execute("DELETE FROM context_model_summaries")
        conn.commit()
    return jsonify({"status":"saved", "characters":len(CONTEXT_SUMMARY_EXTRA_PROMPT), "existing_summaries_invalidated":changed})


@app.route("/api/context/reset-summary/<session_id>", methods=["POST"])
def reset_context_summary(session_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM context_model_summaries WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM context_summaries WHERE session_id=?", (session_id,))
        conn.commit()
    return jsonify({"status":"reset", "session_id":session_id,
                    "note":"Saved summaries cleared. Original chat messages were not deleted."})


@app.route("/api/workspace", methods=["GET"])
def get_workspace():
    current = get_workspace_dir()
    try:
        projects = list_available_projects()
    except Exception:
        projects = [current] if os.path.isdir(current) else []
    if current and current not in projects:
        projects.insert(0, current)
    return jsonify({"current": current, "projects": projects})

@app.route("/api/workspace", methods=["POST"])
def switch_workspace():
    data = request.json or {}
    new_path = data.get("path", "").strip()
    if not new_path:
        return jsonify({"status": "error", "output": "Directory path cannot be empty"}), 400
    result = set_workspace_dir(new_path)
    status_code = 200 if result.get("status") == "success" else 400
    return jsonify({**result, "current": get_workspace_dir(), "projects": list_available_projects()}), status_code

@app.route("/api/sessions", methods=["GET"])
def list_sessions():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT id, title, updated_at FROM sessions ORDER BY updated_at DESC")
        rows = [dict(r) for r in cur.fetchall()]
    return jsonify(rows)

@app.route("/api/sessions/new", methods=["POST"])
def create_session():
    sid = uuid.uuid4().hex[:10]
    now = time.time()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO sessions (id, title, updated_at) VALUES (?, ?, ?)", (sid, "New Chat", now))
        conn.commit()
    return jsonify({"id": sid, "title": "New Chat", "updated_at": now})

@app.route("/api/sessions/<session_id>", methods=["GET"])
def get_session(session_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT role, content, files_json, tools_json, metrics_json, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        )
        messages = [{
            "role": r["role"],
            "content": r["content"],
            "files": json.loads(r["files_json"]) if r["files_json"] else [],
            "tools": json.loads(r["tools_json"]) if r["tools_json"] else [],
            "metrics": json.loads(r["metrics_json"]) if r["metrics_json"] else None
        } for r in cur.fetchall()]
    return jsonify({"session_id": session_id, "messages": messages})

@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM context_summaries WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
    return jsonify({"status": "deleted", "id": session_id})

@app.route("/api/stop/<session_id>", methods=["POST"])
def stop_session(session_id):
    ACTIVE_ABORTS.add(session_id)
    return jsonify({"status": "abort_signaled", "session_id": session_id})


def should_request_plan(prompt: str) -> bool:
    """Detect whether a user prompt warrants generating an execution plan."""
    text = str(prompt or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if re.search(r"\b(plan|steps|checklist|roadmap|upgrade|refactor|implement|build|create|migrate|fix\s+all|multi-step)\b", lowered):
        return True
    if len(text.split()) >= 14 or "\n" in text or text.count(",") >= 2 or " and " in lowered:
        return True
    return False


def stream_colibri_completion(messages: list[dict], model: str, options: dict | None = None):
    """Stream text deltas from the active inference provider (Colibri or LiteLLM)."""
    opts = dict(options or {})
    active_loader = str(opts.get("loader") or LOADER_STATE.get("loader", "Colibri"))
    active_model = str(model or LOADER_STATE.get("model", DETECTED_COLIBRI_MODEL))
    settings = dict(LOADER_STATE.get("settings") or {})
    settings.update(opts)
    session_id = str(opts.get("session_id") or "")
    thinking_level = str(opts.get("thinking_level") or "off")
    thinking_budget = THINKING_BUDGET_MAP.get(thinking_level, 0)
    streamed_tool_calls_out = opts.get("streamed_tool_calls_out")
    metrics_out = opts.get("metrics_out")

    temperature = float(settings.get("temperature", 0.2))
    top_p = float(settings.get("top_p", 0.9))

    payload = {
        "model": active_model,
        "messages": messages,
        "max_tokens": int(settings.get("num_predict", COLIBRI_MAX_TOKENS)),
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    import hashlib
    prefix = str((messages[0] if messages else {}).get("content", ""))
    payload["prompt_cache_key"] = hashlib.sha256((session_id + prefix).encode("utf-8")).hexdigest()[:48]
    payload["cache_prompt"] = True
    if thinking_level != "off":
        payload["reasoning_effort"] = thinking_level
        payload["chat_template_kwargs"] = {
            "enable_thinking": True,
            "thinking_budget": thinking_budget
        }
    is_valid, err_msg = verify_colibri_payload(payload)
    if not is_valid:
        raise ValueError(f"Payload verification error: {err_msg}")

    base = _loader_base_url(active_loader).rstrip("/")
    resp = requests.post(
        base + "/chat/completions",
        json=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **_auth_headers(),
        },
        stream=True,
        timeout=None
    )
    if resp.status_code != 200:
        err_body = getattr(resp, "text", "")[:1500]
        raise RuntimeError(f"{active_loader} HTTP {resp.status_code} (model: {active_model}): {err_body}")

    try:
        for line in resp.iter_lines():
            if session_id and session_id in ACTIVE_ABORTS:
                break
            if not line:
                continue
            line_str = line.decode("utf-8", errors="ignore").strip() if isinstance(line, bytes) else str(line).strip()
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                if isinstance(chunk.get("usage"), dict) and isinstance(metrics_out, dict):
                    metrics_out["provider_usage"] = chunk["usage"]
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or choice.get("message") or {}
                token = delta.get("content")
                if token:
                    yield token
                tc_deltas = delta.get("tool_calls")
                if tc_deltas and isinstance(tc_deltas, list) and isinstance(streamed_tool_calls_out, dict):
                    for tcd in tc_deltas:
                        idx = tcd.get("index", 0)
                        if idx not in streamed_tool_calls_out:
                            streamed_tool_calls_out[idx] = {
                                "id": tcd.get("id", f"call_{idx}"),
                                "function": {"name": "", "arguments": ""}
                            }
                        fn_delta = tcd.get("function") or {}
                        if fn_delta.get("name"):
                            streamed_tool_calls_out[idx]["function"]["name"] += fn_delta["name"]
                        if fn_delta.get("arguments"):
                            streamed_tool_calls_out[idx]["function"]["arguments"] += fn_delta["arguments"]
            except Exception:
                pass
    finally:
        try:
            resp.close()
        except Exception:
            pass


class _DualStream:
    """Supports both iterating SSE chunks (`for raw in event_stream(...)`) and `.split(...)` string access."""
    def __init__(self, gen):
        self._gen = gen
        self._chunks = None

    def __iter__(self):
        if self._chunks is not None:
            return iter(self._chunks)
        return self._gen

    def _materialize(self):
        if self._chunks is None:
            self._chunks = list(self._gen)
        return "".join(self._chunks)

    def split(self, *args, **kwargs):
        return self._materialize().split(*args, **kwargs)

    def __str__(self):
        return self._materialize()


def event_stream(prompt_or_kind: str, user_prompt_or_session: str, session_id_or_thinking: str = "off",
                 resume_answer: str | None = None, *, attached_files=None, selected_skill_files=None,
                 thinking_level: str | None = None):
    """Module-level agent loop SSE stream supporting both direct route calls and test harnesses."""
    if session_id_or_thinking in THINKING_BUDGET_MAP and thinking_level is None:
        user_prompt = str(prompt_or_kind or "").strip()
        session_id = str(user_prompt_or_session or uuid.uuid4().hex[:10])
        resolved_thinking = session_id_or_thinking
    else:
        user_prompt = str(user_prompt_or_session or "").strip()
        session_id = str(session_id_or_thinking or uuid.uuid4().hex[:10])
        resolved_thinking = thinking_level or "off"

    attached_files = attached_files or []
    if selected_skill_files is None:
        selected_skill_files = list(SKILL_FILES.keys())

    def _gen():
        executed_tools = []
        all_step_replies = []
        last_completed_response = None
        plan_items_state: list[dict] = []
        step = 0

        try:
            file_context_blocks = [
                f'<file path="{f.get("path","file")}">\n```\n{f.get("content","").rstrip()}\n```\n</file>'
                for f in attached_files
            ]
            effective_prompt = user_prompt
            if resume_answer is not None and str(resume_answer).strip():
                ans = str(resume_answer).strip()
                effective_prompt = (
                    (effective_prompt + "\n\n" if effective_prompt else "")
                    + f"USER ANSWER TO PAUSED QUESTION: {ans}\nContinue the task based on this answer."
                )
            if should_request_plan(effective_prompt) and "markdown checklist" not in effective_prompt.lower():
                effective_prompt = (
                    effective_prompt + "\n\n" + build_plan_request_instruction()
                    if effective_prompt else build_plan_request_instruction()
                )

            full_user_msg = (
                "\n\n".join(file_context_blocks) + ("\n\n" + effective_prompt if effective_prompt else "")
                if file_context_blocks else effective_prompt
            )

            messages = build_colibri_messages(session_id, full_user_msg, selected_skill_files, thinking_level=resolved_thinking)
            base_system_prompt = messages[0].get("content", "") if messages else ""

            while True:
                step += 1
                if session_id in ACTIVE_ABORTS:
                    yield f"data: {json.dumps({'type': 'execution_state', 'state': 'CANCELLED', 'iteration': step})}\n\n"
                    yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                    break

                active_loader = LOADER_STATE.get("loader", "Colibri")
                active_provider_label = "Dark Matter Agent"
                brain_label = f"{active_provider_label} ↔ Dark Matter Tools • Iteration {step}"
                yield f"data: {json.dumps({'type': 'execution_state', 'state': 'RUNNING', 'iteration': step, 'doing': f'Thinking & planning (step {step})...'})}\n\n"
                yield f"data: {json.dumps({'type': 'iteration_start', 'iteration': step, 'brain': brain_label})}\n\n"
                yield f"data: {json.dumps({'type': 'brain_switch', 'brain': brain_label})}\n\n"

                try:
                    if messages and messages[0].get("role") == "system":
                        messages[0]["content"] = base_system_prompt
                    messages = _compact_messages_for_request(
                        session_id, messages,
                        LOADER_STATE.get("model", DETECTED_COLIBRI_MODEL),
                        LOADER_STATE.get("settings", {})
                    )
                except Exception as compact_error:
                    CONTEXT_LAST_REPORT.update({"error": str(compact_error), "compacted": False})

                active_loader = LOADER_STATE.get("loader", "Colibri")
                active_model = LOADER_STATE.get("model", DETECTED_COLIBRI_MODEL)
                settings = dict(LOADER_STATE.get("settings") or {})

                streamed_tool_calls_map: dict = {}
                metrics_holder: dict = {}
                step_tokens: list[str] = []
                display_buffer = ""

                try:
                    stream_opts = {
                        **settings,
                        "loader": active_loader,
                        "session_id": session_id,
                        "thinking_level": resolved_thinking,
                        "streamed_tool_calls_out": streamed_tool_calls_map,
                        "metrics_out": metrics_holder,
                    }
                    for token in stream_colibri_completion(messages, active_model, stream_opts):
                        if session_id in ACTIVE_ABORTS:
                            yield f"data: {json.dumps({'type': 'execution_state', 'state': 'CANCELLED', 'iteration': step})}\n\n"
                            yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                            break
                        if not token:
                            continue
                        step_tokens.append(token)
                        display_buffer += token
                        if len(display_buffer) > 96:
                            safe_piece = display_buffer[:-96]
                            display_buffer = display_buffer[-96:]
                            yield f"data: {json.dumps({'type': 'token', 'content': safe_piece})}\n\n"
                except ValueError as ve:
                    yield f"data: {json.dumps({'type': 'execution_state', 'state': 'ERROR', 'iteration': step})}\n\n"
                    yield f"data: {json.dumps({'type': 'error', 'content': str(ve)})}\n\n"
                    break
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'execution_state', 'state': 'ERROR', 'iteration': step})}\n\n"
                    yield f"data: {json.dumps({'type': 'error', 'content': f'Inference connection failed: {e}'})}\n\n"
                    break

                if display_buffer:
                    visible_tail = re.sub(
                        r"<task_status>\s*(?:complete|continue|blocked)\s*</task_status>",
                        "", display_buffer, flags=re.I
                    )
                    if visible_tail:
                        yield f"data: {json.dumps({'type': 'token', 'content': visible_tail})}\n\n"

                if session_id in ACTIVE_ABORTS:
                    break

                ollama_metrics = metrics_holder.get("ollama_metrics")
                provider_usage = metrics_holder.get("provider_usage")
                if ollama_metrics:
                    yield f"data: {json.dumps({'type': 'model_metrics', **ollama_metrics})}\n\n"
                elif provider_usage or active_loader == "LiteLLM":
                    usage = provider_usage or {}
                    context_limit = int(settings.get("num_ctx") or settings.get("context_window") or CONTEXT_WINDOW_DEFAULT)
                    yield f"data: {json.dumps({'type': 'model_metrics', 'prompt_tokens': usage.get('prompt_tokens'), 'output_tokens': usage.get('completion_tokens'), 'total_tokens': usage.get('total_tokens'), 'context_limit': context_limit, 'provider': 'LiteLLM'})}\n\n"

                step_reply = "".join(step_tokens)
                streamed_tcs = list(streamed_tool_calls_map.values())
                tool_calls = extract_all_tool_calls(step_reply, streamed_tcs)

                # Extract or advance plan checklist on every turn
                extracted_plan = extract_plan_items(step_reply)
                if extracted_plan:
                    plan_items_state = extracted_plan
                    yield f"data: {json.dumps({'type': 'plan_update', 'items': plan_items_state, 'turn': step})}\n\n"
                elif plan_items_state:
                    advanced = False
                    for item in plan_items_state:
                        if item.get("status") == "active":
                            item["status"] = "done"
                            advanced = True
                            break
                    if advanced:
                        for item in plan_items_state:
                            if item.get("status") == "pending":
                                item["status"] = "active"
                                break
                        yield f"data: {json.dumps({'type': 'plan_update', 'items': plan_items_state, 'turn': step})}\n\n"

                if not tool_calls:
                    status_match = re.search(r"<task_status>\s*(complete|continue|blocked)\s*</task_status>", step_reply, re.I)
                    task_status = status_match.group(1).lower() if status_match else ""
                    clean_reply = re.sub(r"<task_status>\s*(?:complete|continue|blocked)\s*</task_status>", "", step_reply, flags=re.I).strip()
                    clean_reply = re.sub(r"<tool_call>[\s\S]*?(?:<\/tool_call>|$)", "", clean_reply).strip()

                    if is_exact_duplicate_completed_response(last_completed_response, clean_reply):
                        all_step_replies.append(clean_reply)
                        yield f"data: {json.dumps({'type': 'task_status', 'status': 'duplicate_stop', 'turn': step, 'reason': 'Loop stopped: identical response detected.'})}\n\n"
                        break
                    if clean_reply:
                        last_completed_response = clean_reply

                    if task_status == "complete":
                        if clean_reply:
                            all_step_replies.append(clean_reply)
                        if plan_items_state:
                            for item in plan_items_state:
                                item["status"] = "done"
                            yield f"data: {json.dumps({'type': 'plan_update', 'items': plan_items_state, 'turn': step})}\n\n"
                        yield f"data: {json.dumps({'type': 'execution_state', 'state': 'COMPLETED', 'iteration': step, 'doing': 'Completed'})}\n\n"
                        yield f"data: {json.dumps({'type': 'task_status', 'status': 'complete', 'turn': step})}\n\n"
                        yield f"data: {json.dumps({'type': 'complete', 'turn': step})}\n\n"
                        break

                    if task_status == "blocked":
                        if clean_reply:
                            all_step_replies.append(clean_reply)
                        yield f"data: {json.dumps({'type': 'execution_state', 'state': 'STOPPED', 'iteration': step, 'doing': 'Blocked'})}\n\n"
                        yield f"data: {json.dumps({'type': 'task_status', 'status': 'blocked', 'turn': step})}\n\n"
                        break

                    # --- "message done" / explicit completion signal ---
                    if detect_message_done(clean_reply) or looks_like_message_done(step_reply):
                        if clean_reply:
                            all_step_replies.append(clean_reply)
                        if plan_items_state:
                            for item in plan_items_state:
                                item["status"] = "done"
                            yield f"data: {json.dumps({'type': 'plan_update', 'items': plan_items_state, 'turn': step})}\n\n"
                        yield f"data: {json.dumps({'type': 'execution_state', 'state': 'COMPLETED', 'iteration': step, 'doing': 'Completed'})}\n\n"
                        yield f"data: {json.dumps({'type': 'task_status', 'status': 'complete', 'turn': step, 'reason': 'message_done_signal'})}\n\n"
                        yield f"data: {json.dumps({'type': 'complete', 'turn': step})}\n\n"
                        break

                    # --- User question detection ---
                    question_phrase = detect_user_question(clean_reply)
                    if question_phrase and not (step == 1 and resume_answer is not None and str(resume_answer).strip()):
                        question_mode = normalise_question_mode(LOADER_STATE.get("settings", {}).get("question_mode", "manual"))
                        if question_mode == "auto":
                            if clean_reply:
                                all_step_replies.append(clean_reply)
                            assistant_text = step_reply.strip() or clean_reply or "No response text was returned."
                            messages.append({"role": "assistant", "content": assistant_text})
                            synthetic = auto_yes_reply(clean_reply)
                            synthetic_text = "Yes. Proceed with the suggested action." if synthetic == "yes" else "Continue the task autonomously; make the most reasonable decision and keep going."
                            messages.append({"role": "user", "content": synthetic_text})
                            yield f"data: {json.dumps({'type': 'task_status', 'status': 'auto_yes', 'turn': step, 'question': question_phrase, 'auto_reply': synthetic})}\n\n"
                            continue
                        else:
                            if clean_reply:
                                all_step_replies.append(clean_reply)
                            assistant_text = step_reply.strip() or clean_reply or "No response text was returned."
                            messages.append({"role": "assistant", "content": assistant_text})
                            yield f"data: {json.dumps({'type': 'user_question', 'question': clean_reply or question_phrase, 'matched_phrase': question_phrase, 'full_text': clean_reply, 'turn': step})}\n\n"
                            yield f"data: {json.dumps({'type': 'task_status', 'status': 'awaiting_user', 'turn': step})}\n\n"
                            break

                    if clean_reply:
                        all_step_replies.append(clean_reply)
                    assistant_text = step_reply.strip() or clean_reply or "No response text was returned."
                    messages.append({"role": "assistant", "content": assistant_text})

                    if task_status == "continue":
                        messages.append({"role": "user", "content": (
                            "Continue the same job. Review the complete transcript and prior results, "
                            "then perform the next useful action. Do not restart completed work. "
                            "Use tools when needed. Continue until all user requirements are verified; "
                            "then emit <task_status>complete</task_status>. If more work remains without "
                            "a tool call, emit <task_status>continue</task_status>."
                        )})
                        yield f"data: {json.dumps({'type': 'task_status', 'status': 'continuing', 'turn': step})}\n\n"
                        continue

                    messages.append({"role": "user", "content": (
                        "Assess the original acceptance criteria against the work and tool results. "
                        "If any requirement remains, continue with the next concrete action. "
                        "Only when all requirements are verified, emit <task_status>complete</task_status>. "
                        "If genuinely blocked, explain why and emit <task_status>blocked</task_status>. "
                        "Do not repeat your previous response; make progress."
                    )})
                    yield f"data: {json.dumps({'type': 'task_status', 'status': 'continuing', 'turn': step})}\n\n"
                    continue

                tool_cycle_text = re.sub(r"<tool_call>[\s\S]*?(?:</tool_call>|$)", "", step_reply).strip()
                if tool_cycle_text:
                    all_step_replies.append(tool_cycle_text)

                think_blocks = re.findall(r"<think\b[^>]*>[\s\S]*?(?:<\/think>|$)", step_reply, flags=re.IGNORECASE)
                for tb in think_blocks:
                    tb_clean = tb.strip()
                    if not tb_clean:
                        continue
                    if not tb_clean.lower().endswith("</think>"):
                        tb_clean += "</think>"
                    if tb_clean not in all_step_replies:
                        all_step_replies.append(tb_clean)

                yield f"data: {json.dumps({'type': 'step_reset'})}\n\n"
                tool_response_blocks = []
                for raw_name, tool_args in tool_calls:
                    if session_id in ACTIVE_ABORTS:
                        break
                    canonical_preview = raw_name.lower()
                    target_preview = str(
                        tool_args.get("path")
                        or tool_args.get("command")
                        or tool_args.get("pattern")
                        or tool_args.get("query")
                        or tool_args.get("url")
                        or tool_args.get("relative_path")
                        or tool_args
                    )[:80]

                    yield f"data: {json.dumps({'type': 'execution_state', 'state': 'WAITING_FOR_TOOL', 'iteration': step, 'doing': f'Running {canonical_preview}: {target_preview}'})}\n\n"
                    yield f"data: {json.dumps({'type': 'tool_start', 'name': canonical_preview, 'target': target_preview})}\n\n"
                    ui_name, ui_target, res = dispatch_tool_call(raw_name, tool_args)
                    executed_tools.append({"name": canonical_preview, "target": ui_target, "result": res})
                    yield f"data: {json.dumps({'type': 'execution_state', 'state': 'PROCESSING_TOOL_RESULT', 'iteration': step, 'doing': f'Processing result from {canonical_preview}'})}\n\n"
                    yield f"data: {json.dumps({'type': 'tool_result', 'name': canonical_preview, 'result': res})}\n\n"
                    if session_id in ACTIVE_ABORTS:
                        yield f"data: {json.dumps({'type': 'execution_state', 'state': 'CANCELLED', 'iteration': step})}\n\n"
                        yield f"data: {json.dumps({'type': 'stopped'})}\n\n"
                        break

                    out_text = str(res.get("output", ""))
                    metadata_lines = []
                    if res.get("command") and canonical_preview not in {"execute_terminal", "terminal", "bash"}:
                        metadata_lines.append(f"Command: {res['command']}")
                    if res.get("exit_code") is not None and canonical_preview not in {"execute_terminal", "terminal", "bash"}:
                        metadata_lines.append(f"Exit code: {res['exit_code']}")
                    if res.get("cwd"):
                        metadata_lines.append(f"Working directory: {res['cwd']}")
                    if metadata_lines:
                        out_text = "\n".join(metadata_lines) + "\n" + out_text
                    if canonical_preview in {"execute_terminal", "terminal", "bash"}:
                        failure_details = []
                        if tool_args.get("command"):
                            failure_details.append(f"Command: {tool_args['command']}")
                        if res.get("exit_code") is not None:
                            failure_details.append(f"Exit code: {res['exit_code']}")
                        if res.get("cwd"):
                            failure_details.append(f"Working directory: {res['cwd']}")
                        if res.get("timed_out"):
                            failure_details.append(f"Timed out after {res.get('timeout_seconds', 'unknown')} seconds.")
                        if failure_details:
                            out_text = "\n".join(failure_details) + "\n" + out_text
                    if res.get("python"):
                        out_text = f"Python interpreter: {res['python']}\n" + out_text
                    if res.get("missing_python_module"):
                        out_text = f"Missing Python module: {res['missing_python_module']}\nRecovery: {res.get('recovery', '')}\n" + out_text
                    if res.get("error_type"):
                        out_text = f"Error type: {res['error_type']}\n" + out_text
                    if res.get("diff"):
                        out_text += (f"\nLine changes: +{res.get('added_lines', 0)} / -{res.get('removed_lines', 0)}\n"
                                     "Unified diff (+ added, - removed):\n" + str(res["diff"]))
                    if len(out_text) > 12000:
                        out_text = out_text[:12000] + "\n...[Output truncated]"
                    tool_response_blocks.append(
                        f'<tool_response name="{raw_name}">\nStatus: {res.get("status")}\n{out_text}\n</tool_response>'
                    )

                if session_id in ACTIVE_ABORTS or not tool_response_blocks:
                    break

                assistant_history_text = step_reply.strip() or f"<tool_call>{json.dumps(tool_calls)}</tool_call>"
                messages.append({"role": "assistant", "content": assistant_history_text})
                yield f"data: {json.dumps({'type': 'execution_state', 'state': 'CONTINUING', 'iteration': step, 'doing': 'Continuing after tool execution...'})}\n\n"
                messages.append({
                    "role": "user",
                    "content": "\n\n".join(tool_response_blocks) + "\n\nYou are the selected provider collaborating with the Dark Matter agent. Analyze every tool result above and continue the SAME job. A success status alone does not prove the user's overall task is complete. If any requested work remains, make the next tool call or emit <task_status>continue</task_status>. Only after all acceptance criteria and verification checks are satisfied, provide the final summary and end with <task_status>complete</task_status>. For errors, diagnose and recover using corrected arguments or another safe method. Do not stop after the first successful tool."
                })

            final_saved_reply = "\n\n".join(all_step_replies).strip()
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    conn.execute(
                        "INSERT INTO messages (session_id, role, content, tools_json, created_at) VALUES (?, 'assistant', ?, ?, ?)",
                        (session_id, final_saved_reply, json.dumps(executed_tools), time.time())
                    )
                    conn.commit()
            except Exception:
                pass

            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        finally:
            ACTIVE_ABORTS.discard(session_id)
            with ACTIVE_RUNS_LOCK:
                ACTIVE_RUNS.discard(session_id)

    return _DualStream(_gen())


@app.route("/api/connection-test", methods=["GET", "POST"])
def api_connection_test():
    """Verify live reachability of the configured inference backend (Colibri / LiteLLM)."""
    loader = LOADER_STATE.get("loader", "Colibri")
    start = time.time()
    try:
        base = _loader_base_url(loader).rstrip("/")
        target = base + "/models"
        r = requests.get(target, headers=_auth_headers(), timeout=8)
        latency_ms = round((time.time() - start) * 1000, 1)
        return jsonify({"status": "ok" if r.ok else "error", "loader": loader, "url": target, "http_status": r.status_code, "latency_ms": latency_ms}), (200 if r.ok else 502)
    except Exception as exc:
        latency_ms = round((time.time() - start) * 1000, 1)
        return jsonify({"status": "error", "loader": loader, "error": str(exc), "latency_ms": latency_ms}), 502


@app.route("/api/run", methods=["POST"])
def run_turn():
    data = request.json or {}
    session_id = data.get("session_id")
    user_prompt = data.get("prompt", "").strip()
    resume_answer = data.get("resume_answer")
    attached_files = data.get("files", [])
    selected_skill_files = data.get("skills", list(SKILL_FILES.keys()))
    if not isinstance(selected_skill_files, list):
        selected_skill_files = list(SKILL_FILES.keys())
    selected_skill_files = [name for name in selected_skill_files if name in SKILL_FILES]

    gui_workspace = (data.get("workspace") or "").strip()
    if gui_workspace and os.path.isdir(gui_workspace):
        set_workspace_dir(gui_workspace)

    thinking_level = str(data.get("thinking_level", "")).lower().strip()
    if not thinking_level:
        thinking_level = "medium" if data.get("thinking") is True else "off"
    if thinking_level not in THINKING_BUDGET_MAP:
        thinking_level = "off"

    if not session_id:
        session_id = uuid.uuid4().hex[:10]
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, updated_at) VALUES (?, ?, ?)",
                (session_id, user_prompt[:32] or "New Chat", time.time())
            )
            conn.commit()

    with ACTIVE_RUNS_LOCK:
        if session_id in ACTIVE_RUNS:
            return jsonify({"status": "error", "output": "A task is already running for this conversation."}), 409
        ACTIVE_RUNS.add(session_id)
    ACTIVE_ABORTS.discard(session_id)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, files_json, created_at) VALUES (?, 'user', ?, ?, ?)",
            (session_id, user_prompt, json.dumps(attached_files), time.time())
        )
        cur = conn.execute("SELECT title FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        if row and row[0] == "New Chat" and user_prompt:
            new_title = (user_prompt[:30] + "..") if len(user_prompt) > 30 else user_prompt
            conn.execute("UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?", (new_title, time.time(), session_id))
        else:
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (time.time(), session_id))
        conn.commit()

    stream = event_stream(
        user_prompt,
        session_id,
        thinking_level,
        resume_answer=resume_answer,
        attached_files=attached_files,
        selected_skill_files=selected_skill_files,
    )
    return Response(iter(stream), content_type="text/event-stream")



# =====================================================================
# SWARM MULTI-AGENT API (OpenAI Swarm pattern: Agent + handoffs)
# =====================================================================
import os as _os
import json as _json
from pathlib import Path as _Path

SWARM_AGENTS_FILE = _os.path.join(get_workspace_dir(), "swarm_agents.json")
TOOL_TOGGLES_FILE = _os.path.join(get_workspace_dir(), "tool_toggles.json")
CODA_FILE = _os.path.join(get_workspace_dir(), "coda.json")

# Default coda settings from /home/local-ai/.unsloth/coda.json
DEFAULT_CODA = {"loops": 3, "enabled": True}

def _load_json_file(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return default

def _save_json_file(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(data, f, indent=2, ensure_ascii=False)
    _os.replace(tmp, path)

def _get_tool_toggles():
    return _load_json_file(TOOL_TOGGLES_FILE, {})

def _is_tool_enabled(tool_name: str) -> bool:
    toggles = _get_tool_toggles()
    # Default: all tools enabled unless explicitly disabled
    return toggles.get(tool_name, True) is not False

@app.route("/api/swarm/agents", methods=["GET"])
def swarm_list_agents():
    agents = _load_json_file(SWARM_AGENTS_FILE, {})
    return jsonify({"agents": {name: {"name": a.get("name"), "model": a.get("model"), "instructions": a.get("instructions"), "tools": a.get("tools", []), "handoffs": a.get("handoffs", [])} for name, a in agents.items()}})

@app.route("/api/swarm/agents", methods=["POST"])
def swarm_create_agent():
    data = request.json or {}
    name = data.get("name")
    if not name:
        return jsonify({"status": "error", "output": "Agent name required"}), 400
    agents = _load_json_file(SWARM_AGENTS_FILE, {})
    agents[name] = {
        "name": name,
        "model": data.get("model", LOADER_STATE.get("model", COLIBRI_MODEL)),
        "instructions": data.get("instructions", f"You are {name}, a helpful agent."),
        "tools": data.get("tools", []),
        "handoffs": data.get("handoffs", []),
    }
    _save_json_file(SWARM_AGENTS_FILE, agents)
    return jsonify({"status": "success", "agent": agents[name]})

@app.route("/api/swarm/agents/<name>", methods=["DELETE"])
def swarm_delete_agent(name):
    agents = _load_json_file(SWARM_AGENTS_FILE, {})
    if name in agents:
        del agents[name]
        _save_json_file(SWARM_AGENTS_FILE, agents)
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "output": f"Agent '{name}' not found"}), 404

@app.route("/api/swarm/board", methods=["GET"])
def swarm_board():
    agents = _load_json_file(SWARM_AGENTS_FILE, {})
    nodes = []
    edges = []
    for name, a in agents.items():
        nodes.append({"id": name, "label": name, "model": a.get("model"), "tools": len(a.get("tools", []))})
        for h in a.get("handoffs", []):
            edges.append({"from": name, "to": h})
    return jsonify({"nodes": nodes, "edges": edges})

@app.route("/api/swarm/handoffs", methods=["GET"])
def swarm_handoffs():
    agents = _load_json_file(SWARM_AGENTS_FILE, {})
    return jsonify({"agents": list(agents.keys())})

@app.route("/api/swarm/run", methods=["POST"])
def swarm_run():
    data = request.json or {}
    agent_name = data.get("agent_name")
    prompt = data.get("prompt", "").strip()
    max_turns = int(data.get("max_turns", 10))
    context_variables = data.get("context_variables", {})
    if not agent_name:
        return jsonify({"status": "error", "output": "agent_name required"}), 400
    if not prompt:
        return jsonify({"status": "error", "output": "prompt required"}), 400

    agents = _load_json_file(SWARM_AGENTS_FILE, {})
    if agent_name not in agents:
        return jsonify({"status": "error", "output": f"Agent '{agent_name}' not found"}), 404

    agent_def = agents[agent_name]

    # Build Agent objects for all defined agents (for handoff resolution)
    agent_objs = {}
    for name, a in agents.items():
        agent_objs[name] = create_swarm_agent(
            name=a.get("name", name),
            model=a.get("model"),
            instructions=a.get("instructions"),
            tools=a.get("tools", []),
            handoff_agents=[create_swarm_agent(name=h) for h in a.get("handoffs", [])],
        )

    start_agent = agent_objs[agent_name]

    from swarm import Swarm, Agent as SwarmAgent

    # Wrap studio tools as swarm-callable functions
    def _make_tool_fn(tname):
        def _fn(**kwargs):
            return {"tool": tname, "args": kwargs}
        _fn.__name__ = tname
        return _fn

    # Replace tool name strings with callable functions
    for name, a in agent_objs.items():
        wrapped = []
        for t in agents[name].get("tools", []):
            wrapped.append(_make_tool_fn(t))
        a.functions = wrapped

    swarm = Swarm(debug=False)
    response = swarm.run(
        agent=start_agent,
        messages=[{"role": "user", "content": prompt}],
        context_variables=context_variables,
        max_turns=max_turns,
        stream=False,
    )

    return jsonify({
        "status": "success",
        "agent": response.agent.name,
        "context_variables": response.context_variables,
        "messages": response.messages,
    })


@app.route("/api/tool-toggles", methods=["GET"])
def tool_toggles_get():
    return jsonify(_get_tool_toggles())

@app.route("/api/tool-toggles", methods=["POST"])
def tool_toggles_set():
    data = request.json or {}
    toggles = _get_tool_toggles()
    for k, v in data.items():
        toggles[k] = bool(v)
    _save_json_file(TOOL_TOGGLES_FILE, toggles)
    return jsonify({"status": "success", "toggles": toggles})

@app.route("/api/coda", methods=["GET"])
def coda_get():
    return jsonify(_load_json_file(CODA_FILE, DEFAULT_CODA))

@app.route("/api/coda", methods=["POST"])
def coda_set():
    data = request.json or {}
    current = _load_json_file(CODA_FILE, DEFAULT_CODA)
    current.update(data)
    _save_json_file(CODA_FILE, current)
    return jsonify({"status": "success", "coda": current})

@app.route("/api/update/check", methods=["GET"])
def update_check():
    """Check for updates (placeholder - compares local version to manifest)."""
    version_file = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "VERSION")
    local_version = "1.0.0"
    try:
        with open(version_file, "r") as f:
            local_version = f.read().strip()
    except Exception:
        pass
    return jsonify({"status": "ok", "local_version": local_version, "update_available": False})

@app.route("/api/update/apply", methods=["POST"])
def update_apply():
    """Apply updates (placeholder - requires user confirmation, backs up first)."""
    data = request.json or {}
    if not data.get("confirmed"):
        return jsonify({"status": "error", "output": "Update requires confirmed=true"}), 400
    return jsonify({"status": "error", "output": "Auto-update not yet wired to a remote manifest. Use manual update."}), 501


# =====================================================================
# EMBEDDED SINGLE-PAGE UI (RAW STRING PREVENTS JS ESCAPE ERRORS)
# =====================================================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dark Matter Autonomous Studio</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
  <style>
    body { background-color: #080c14; color: #f1f5f9; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    pre { white-space: pre-wrap; word-break: break-word; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #080c14; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #334155; }
    @keyframes pulse-glow {
      0%, 100% { opacity: 1; filter: drop-shadow(0 0 8px rgba(16, 185, 129, 0.6)); }
      50% { opacity: 0.6; filter: drop-shadow(0 0 2px rgba(16, 185, 129, 0.2)); }
    }
    .animate-glow { animation: pulse-glow 2s infinite ease-in-out; }
    .session-item.active { background-color: rgba(16, 185, 129, 0.12); border-color: rgba(16, 185, 129, 0.4); color: #34d399; }

    /* Sidebar collapse transition */
    #sidebar { transition: width 0.25s ease, padding 0.25s ease; overflow-x: hidden; }
    #sidebar.collapsed { width: 68px !important; padding: 0.5rem !important; }
    #sidebar.collapsed .text-label, #sidebar.collapsed span:not(.fa-*) { display: none; }
    #sidebar.collapsed #session-list { display: none; }
    #sidebar.collapsed .p-3 { padding: 0.25rem !important; }
    #sidebar.collapsed button { justify-content: center; padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
    #sidebar.collapsed button span { display: none; }

    #agent-dock { transition: right 0.25s ease; }

    /* THINKING / PLANNING BOX */
    .thinking-box {
      background: rgba(30, 27, 75, 0.35);
      border: 1px solid rgba(99, 102, 241, 0.35);
      border-radius: 0.5rem;
      margin: 0.75rem 0;
      overflow: hidden;
    }
    .thinking-header {
      padding: 0.45rem 0.85rem;
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #a5b4fc;
      background: rgba(49, 46, 129, 0.25);
      border-bottom: 1px solid rgba(99, 102, 241, 0.2);
      display: flex;
      align-items: center;
      justify-content: space-between;
      user-select: none;
      cursor: pointer;
    }
    .thinking-header:hover { background: rgba(49, 46, 129, 0.4); }

    /* REASONING / EVALUATION BOX */
    .reasoning-box {
      background: rgba(59, 7, 100, 0.3);
      border: 1px solid rgba(168, 85, 247, 0.35);
      border-radius: 0.5rem;
      margin: 0.75rem 0;
      overflow: hidden;
    }
    .reasoning-header {
      padding: 0.45rem 0.85rem;
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #d8b4fe;
      background: rgba(88, 28, 135, 0.25);
      border-bottom: 1px solid rgba(168, 85, 247, 0.2);
      display: flex;
      align-items: center;
      justify-content: space-between;
      user-select: none;
      cursor: pointer;
    }
    .reasoning-header:hover { background: rgba(88, 28, 135, 0.4); }

    .thought-body { padding: 0.75rem 1rem; }
    .thought-text {
      font-style: italic !important;
      color: #c7d2fe;
      font-size: 0.8125rem;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .thought-text * { font-style: italic !important; }

    /* WRITING / FINAL RESPONSE BLOCK */
    .writing-card {
      margin-top: 0.75rem;
      padding: 0.85rem 1rem;
      background: rgba(6, 78, 59, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.35);
      border-radius: 0.5rem;
    }
    .writing-badge {
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #34d399;
      display: flex;
      align-items: center;
      gap: 0.375rem;
      margin-bottom: 0.5rem;
      user-select: none;
    }
    .writing-content {
      color: #34d399 !important;
      font-size: 0.875rem;
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    /* TOGGLE SWITCH */
    .toggle-switch {
      position: relative; display: inline-block; width: 34px; height: 20px;
    }
    .toggle-switch input { opacity: 0; width: 0; height: 0; }
    .toggle-slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
      background-color: #334155; transition: .2s; border-radius: 20px; }
    .toggle-slider:before { position: absolute; content: ""; height: 14px; width: 14px; left: 3px; bottom: 3px;
      background-color: white; transition: .2s; border-radius: 50%; }
    input:checked + .toggle-slider { background-color: #10b981; }
    input:checked + .toggle-slider:before { transform: translateX(14px); }
  </style>
</head>
<body class="h-screen flex relative overflow-hidden select-none">

  <div id="drop-overlay" class="absolute inset-0 bg-slate-950/90 backdrop-blur-sm border-2 border-dashed border-emerald-500 z-50 flex flex-col items-center justify-center space-y-3 pointer-events-none hidden transition-all duration-200">
    <div class="w-16 h-16 rounded-full bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20">
      <i class="fa-solid fa-cloud-arrow-up text-3xl text-emerald-400 animate-bounce"></i>
    </div>
    <p class="text-base font-bold text-slate-100">Drop files to stage into workspace context</p>
  </div>

  <aside id="sidebar" class="w-64 bg-slate-950 border-r border-slate-800/80 flex flex-col flex-shrink-0 z-20">
    <div class="p-3 border-b border-slate-800/80 space-y-2.5">
      <div class="flex items-center justify-between px-1">
        <span class="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5 text-label">
          <i class="fa-solid fa-box-archive text-emerald-400"></i> Saved Chats
        </span>
        <span id="session-count" class="text-[10px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded">0</span>
      </div>
      <button id="new-chat-btn" type="button" class="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-2 px-3 rounded-lg flex items-center justify-center gap-2 text-xs transition shadow-sm cursor-pointer">
        <i class="fa-solid fa-plus text-xs"></i><span class="text-label">New Chat</span>
      </button>
      <button id="toggle-sidebar-btn" type="button" class="w-full mt-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs py-1.5 rounded-lg border border-slate-700 transition cursor-pointer">
        <i class="fa-solid fa-chevron-left"></i><span class="text-label">Collapse</span>
      </button>
    </div>
    <div id="session-list" class="flex-1 overflow-y-auto p-2 space-y-1"></div>

    <div class="p-3 border-t border-slate-800/80 text-[11px] text-slate-500 flex items-center justify-between">
      <span class="text-label">15 Agent Skills Ready</span>
      <i class="fa-solid fa-screwdriver-wrench text-emerald-500"></i>
    </div>
  </aside>

  <div class="flex-1 flex flex-col min-w-0 h-screen overflow-hidden">
    <header class="bg-slate-900/90 border-b border-slate-800/80 px-6 py-2.5 flex justify-between items-center backdrop-blur-sm z-30">
      <div class="flex items-center space-x-3">
        <button id="toggle-sidebar" type="button" class="text-slate-400 hover:text-slate-200 transition p-1 text-sm mr-1 cursor-pointer">
          <i class="fa-solid fa-bars"></i>
        </button>
        <span class="text-emerald-400 text-lg font-bold tracking-tight flex items-center gap-2">
          <i class="fa-solid fa-bolt-lightning animate-glow"></i> Dark Matter Studio
        </span>
        <span class="text-xs bg-slate-800 text-slate-300 px-2.5 py-0.5 rounded border border-slate-700 hidden sm:inline-block">Qwen3-Coder 30B • Payload Verified</span>
      </div>

      <div class="flex items-center space-x-3 text-xs">
        <button id="connection-test-btn" type="button" class="px-2.5 py-1 rounded bg-slate-800/80 hover:bg-slate-700 border border-slate-700 text-slate-200 flex items-center gap-1.5 cursor-pointer transition" title="Test live connection to active inference endpoint">
          <span id="conn-test-dot" class="w-2 h-2 rounded-full bg-emerald-400"></span>
          <span id="conn-test-label">Connection Test</span>
        </button>

        <div id="project-selector-wrapper" class="relative flex flex-wrap items-center gap-2 min-w-0">
          <span class="text-xs text-slate-400 whitespace-nowrap"><i class="fa-solid fa-folder-tree text-emerald-400"></i> Project</span>
          <button id="proj-selector-btn" type="button" aria-haspopup="true" aria-expanded="false" class="max-w-[280px] flex items-center gap-2 bg-slate-950 border border-slate-700 hover:border-emerald-500/60 rounded-lg px-2.5 py-1.5 text-xs text-slate-100 cursor-pointer">
            <i class="fa-regular fa-folder text-emerald-400"></i><span id="proj-selector-label" class="max-w-[190px] truncate">Loading projects…</span><i id="proj-caret" class="fa-solid fa-chevron-down text-[10px] text-slate-400 transition-transform"></i>
          </button>
          <button id="refresh-proj-btn" type="button" class="bg-slate-800 hover:bg-slate-700 border border-slate-700 px-2 py-1.5 rounded-lg text-xs text-slate-200 cursor-pointer" title="Rescan folders"><i class="fa-solid fa-rotate-right"></i></button>
          <div id="proj-dropdown" class="hidden absolute right-0 top-full mt-2 w-[min(92vw,520px)] max-h-[70vh] overflow-y-auto bg-slate-950 border border-slate-700 rounded-xl shadow-2xl p-3 z-[100] space-y-2">
            <div class="flex items-center justify-between gap-2"><span class="text-[10px] text-slate-400 uppercase tracking-wider">Available project folders</span><span id="proj-count-badge" class="text-[10px] text-slate-500">Loading…</span></div>
            <div id="proj-status" role="status" aria-live="polite" class="text-[11px] text-slate-300"></div>
            <div id="proj-candidates-list" class="space-y-1"></div>
            <details id="project-path-details" class="border-t border-slate-800 pt-2"><summary class="cursor-pointer text-xs text-emerald-300 hover:text-emerald-200">Enter a custom directory path…</summary>
              <div class="pt-2 space-y-2"><label for="custom-path-input" class="text-[10px] text-slate-400 uppercase tracking-wider block">Absolute directory path</label>
                <div class="flex gap-2"><input type="text" id="custom-path-input" placeholder="/home/username/project" class="min-w-0 flex-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-2 text-slate-100 text-xs font-mono"><button id="switch-path-btn" type="button" class="bg-emerald-600 hover:bg-emerald-500 text-white font-bold px-3 py-2 rounded-lg text-xs cursor-pointer">Switch</button></div>
              </div>
            </details>
          </div>
        </div>
      </div>
    </header>

    <section class="bg-slate-950/95 border-b border-slate-800/80 px-6 py-2 backdrop-blur-sm z-10 text-xs">
      <div class="max-w-5xl mx-auto space-y-2">
        <div class="flex flex-wrap items-center justify-between gap-2 text-[10px] text-slate-400">
          <span class="font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5 font-mono">
            <i class="fa-solid fa-chart-simple text-emerald-400"></i> Execution Profiler
          </span>
          <div class="flex flex-wrap items-center gap-2.5">
            <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-sm bg-violet-600 inline-block"></span> Prefill</span>
            <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-sm bg-indigo-500 inline-block"></span> Thinking</span>
            <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-sm bg-purple-500 inline-block"></span> Reasoning</span>
            <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-sm bg-emerald-500 inline-block"></span> Writing</span>
            <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-sm bg-amber-500 inline-block"></span> Shell</span>
            <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-sm bg-cyan-500 inline-block"></span> Code/Files</span>
            <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-sm bg-blue-500 inline-block"></span> Web</span>
            <span class="flex items-center gap-1"><span class="w-2 h-2 rounded-sm bg-rose-500 inline-block"></span> Desktop</span>
          </div>
        </div>
        <div class="space-y-1.5 font-mono">
          <div class="flex items-center gap-3">
            <div class="w-20 text-[11px] text-slate-400 flex items-center justify-between">
              <span class="text-slate-400">Last</span>
              <span id="last-run-total" class="text-emerald-400 font-bold">0.0s</span>
            </div>
            <div id="last-run-bar" class="flex-1 h-3.5 bg-slate-900 rounded overflow-hidden flex border border-slate-800 shadow-inner">
              <div class="text-[10px] text-slate-600 flex items-center justify-center w-full">Awaiting execution...</div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <div id="live-doing-banner" class="hidden bg-slate-900/95 border-b border-emerald-500/30 px-6 py-2 text-xs font-mono flex items-center justify-between gap-3">
      <div class="flex items-center gap-2 min-w-0">
        <i class="fa-solid fa-circle-notch animate-spin text-emerald-400"></i>
        <span class="text-slate-400 uppercase tracking-wider text-[10px]">Currently Doing:</span>
        <span id="live-doing-text" class="text-emerald-300 font-semibold truncate">Idle</span>
      </div>
      <span id="live-doing-iter" class="text-[10px] text-slate-400 shrink-0"></span>
    </div>

    <main id="chat-box" class="flex-1 overflow-y-auto p-6 space-y-4 max-w-5xl w-full mx-auto select-text">
      <div id="welcome-card" class="p-4 rounded-xl border border-slate-800 bg-slate-900/40 text-xs text-slate-400 leading-relaxed">
        <p class="font-bold text-slate-200 mb-1 flex items-center gap-2">
          <i class="fa-solid fa-circle-check text-emerald-400"></i> Payload-Verified Studio Ready
        </p>
        Thinking, Reasoning, and Writing are segregated. Tool execution logs and live progress appear directly in this main chat panel.
      </div>
      <section id="main-tool-log-panel" class="rounded-xl border border-slate-800/90 bg-slate-950/70 p-3 space-y-2">
        <div class="flex items-center justify-between gap-2">
          <span class="text-[11px] font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
            <i class="fa-solid fa-terminal text-amber-400"></i> Tool Execution Log (Main Panel)
          </span>
          <button id="clear-execution-log" type="button" class="px-2 py-0.5 text-[10px] border border-slate-700 rounded text-slate-400 hover:text-slate-200 cursor-pointer">Clear</button>
        </div>
        <div id="execution-log" aria-live="polite" class="grid gap-2 max-h-72 overflow-y-auto">
          <p class="execution-log-empty text-[11px] text-slate-500">Tool execution output appears here in the main chat panel.</p>
        </div>
      </section>
    </main>

    <footer class="bg-slate-900/90 border-t border-slate-800/80 p-4 backdrop-blur-sm z-10">
      <div class="max-w-5xl mx-auto space-y-2.5">
        <div id="file-tray" class="flex flex-wrap gap-2 text-xs empty:hidden"></div>

        <div class="flex space-x-2.5 items-end">
          <label class="cursor-pointer bg-slate-800 hover:bg-slate-700 text-slate-300 p-3 rounded-lg border border-slate-700 transition flex items-center justify-center h-[46px] w-[46px]" title="Attach Workspace File">
            <i class="fa-solid fa-paperclip text-sm"></i>
            <input type="file" id="file-input" multiple class="hidden">
          </label>

          <textarea id="prompt-input" rows="1" placeholder="Ask Dark Matter to search code, read/patch files, run tests, or automate desktop..."
            class="flex-1 bg-slate-950 border border-slate-800 rounded-lg p-3 text-sm focus:outline-none focus:border-emerald-500 text-slate-200 resize-none min-h-[46px] max-h-36"></textarea>

          <div class="flex items-center gap-1.5 bg-slate-950 border border-slate-800 rounded-lg px-2.5 h-[46px] text-xs">
            <i class="fa-solid fa-brain text-indigo-400 text-sm"></i>
            <select id="footer-thinking-level" title="Adjust Model Reasoning Effort" class="bg-transparent text-slate-200 focus:outline-none cursor-pointer text-xs">
              <option value="off">Think: Off</option>
              <option value="low">Think: Low</option>
              <option value="medium" selected>Think: Med</option>
              <option value="high">Think: High</option>
            </select>
          </div>

          <button id="send-btn" type="button" title="Run now, or queue this prompt while another run is active" class="bg-emerald-600 hover:bg-emerald-500 text-white px-5 rounded-lg font-bold transition flex items-center space-x-2 h-[46px] cursor-pointer">
            <span>Run</span>
            <i class="fa-solid fa-arrow-right text-xs"></i>
          </button>

          <button id="stop-btn" type="button" title="Request a graceful stop after the current non-interruptible tool or command finishes" class="bg-rose-600 hover:bg-rose-500 text-white px-5 rounded-lg font-bold transition items-center space-x-2 h-[46px] hidden cursor-pointer">
            <span>Stop</span>
            <i class="fa-solid fa-stop text-xs"></i>
          </button>
        </div>
      </div>
    </footer>
  </div>

  <aside id="agent-dock" style="position:fixed;right:0;top:0;bottom:0;width:430px;z-index:40;background:#0b1220;border-left:1px solid #243244;padding:16px;overflow:auto;font:12px ui-monospace,monospace">
    <h2 style="font-weight:800;color:#6ee7b7;font-size:14px;margin-bottom:12px">AGENT CONTROL</h2>

    <!-- Loader -->
    <label class="text-xs text-slate-400 uppercase tracking-wider">Loader</label>
    <select id="loader-select" style="width:100%;padding:8px;background:#111827;margin:5px 0 10px;border:1px solid #334155;border-radius:6px">
      <option>Colibri</option>
      <option value="LiteLLM" selected>LiteLLM</option>
    </select>

    <!-- API Key -->
    <label class="text-xs text-slate-400 uppercase tracking-wider">API Key</label>
    <div style="display:flex;gap:4px;margin-bottom:10px">
      <input id="api-key-input" type="password" placeholder="sk-…" style="flex:1;padding:6px;background:#0f172a;border:1px solid #334155;border-radius:6px;color:#e2e8f0;font:10px monospace">
      <button id="api-key-show" style="padding:6px 8px;background:#334155;border:1px solid #334155;border-radius:6px;color:#e2e8f0;cursor-pointer;font-size:10px">👁</button>
    </div>

    <!-- Model -->
    <label class="text-xs text-slate-400 uppercase tracking-wider" id="model-label">Model</label>
    <div id="model-select-wrap" style="display:flex;gap:5px;margin:5px 0 10px;align-items:stretch">
      <select id="model-select" size="5" aria-label="Available models" style="min-width:0;flex:1;height:126px;max-height:126px;overflow-y:auto;padding:6px;background:#111827;border:1px solid #334155;border-radius:6px"></select>
      <button id="refresh-models" title="Refresh models" style="align-self:flex-start">↻</button>
    </div>

    <div id="colibri-server-controls" style="display:flex;gap:6px;margin:0 0 8px">
      <button id="start-colibri" style="flex:1;background:#155e75;padding:7px;border-radius:6px">Start Colibri server</button>
      <button id="stop-colibri" style="flex:1;background:#7f1d1d;padding:7px;border-radius:6px">Stop managed server</button>
    </div>
    <div style="display:flex;gap:6px">
      <button id="load-model" style="flex:1;background:#059669;padding:8px;border-radius:6px;font-weight:bold">Load</button>
      <button id="unload-model" style="flex:1;background:#9f1239;padding:8px;border-radius:6px">Unload</button>
    </div>

    <p id="loader-status" style="color:#94a3b8;margin:8px 0 6px">Checking backend…</p>
    <p id="context-compaction-status" style="color:#94a3b8;font-size:10px;margin:0 0 14px">Context auto-compaction: ON · 75% trigger · persistent summaries</p>

    <!-- Tool Toggles -->
    <details style="margin:0 0 10px"><summary style="cursor:pointer;color:#a7f3d0;font-weight:bold">Tool Toggles</summary>
      <div id="tool-toggles" style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:6px"></div>
      <div style="margin-top:6px">
        <label style="display:flex;align-items:center;gap:8px;font-size:10px">
          <input type="checkbox" id="toggle-web-search" checked style="accent-color:#10b981">
          <span>Web Search (network)</span>
        </label>
      </div>
    </details>

    <!-- Coda Refinement -->
    <details style="margin:0 0 10px"><summary style="cursor:pointer;color:#a7f3d0;font-weight:bold">Coda Refinement</summary>
      <div style="display:flex;align-items:center;gap:8px;margin-top:6px">
        <label style="display:flex;align-items:center;gap:8px;font-size:10px">
          <input type="checkbox" id="toggle-coda" checked style="accent-color:#10b981">
          <span>Enable coda refinement loop</span>
        </label>
      </div>
      <div style="display:flex;gap:6px;margin-top:6px">
        <label style="font-size:10px">Coda loops:</label>
        <input id="coda-loops" type="number" value="3" min="1" max="10" style="width:60px;padding:4px;background:#0f172a;border:1px solid #334155;border-radius:4px;color:#e2e8f0">
      </div>
    </details>

    <!-- Loop Count -->
    <div style="margin:0 0 10px">
      <label class="text-xs text-slate-400 uppercase tracking-wider block">Max Loop Turns</label>
      <input id="max-loop-turns" type="number" value="10" min="1" max="100" style="width:100%;padding:6px;background:#0f172a;border:1px solid #334155;border-radius:6px;color:#e2e8f0">
    </div>

    <!-- Parameters -->
    <details id="loader-parameters-details" open style="border-top:1px solid #263244;padding-top:12px"><summary id="loader-param-heading" style="cursor:pointer;color:#cbd5e1;font-weight:bold">Request Parameters</summary>
      <label>Temperature <span id="temp-val">0.2</span></label><input id="temperature" type="range" min="0" max="2" step="0.1" value="0.2" style="width:100%">
      <label>Top-p <span id="top-p-val">0.9</span></label><input id="top-p" type="range" min="0.1" max="1" step="0.05" value="0.9" style="width:100%">
      <label>Thinking Level (Reasoning Effort)</label>
      <select id="dock-thinking-level" style="width:100%;padding:6px;background:#111827;margin:4px 0 8px;border:1px solid #334155;border-radius:6px">
        <option value="off">Off (0 tokens)</option>
        <option value="low">Low (~1,024 budget)</option>
        <option value="medium" selected>Medium (~4,096 budget)</option>
        <option value="high">High (~16,384 budget)</option>
      </select>
      <label>Question Mode (yes/no prompts)</label>
      <select id="question-mode" style="width:100%;padding:6px;background:#111827;margin:4px 0 8px;border:1px solid #334155;border-radius:6px">
        <option value="manual">Manual — ask before destructive actions</option>
        <option value="auto_yes">Auto-Yes — proceed automatically</option>
      </select>
      <div style="font-size:10px;color:#94a3b8;margin:-4px 0 8px">Manual: loop pauses for your yes/no reply. Auto-Yes: loop proceeds without waiting.</div>
      <label>Max output tokens <span id="output-token-note">(num_predict)</span>: <span id="num-predict-val">2048</span></label>
      <input id="num-predict" type="range" value="2048" min="1" max="131072" step="64" style="width:100%;margin:4px 0 8px">
      <button id="save-loader-settings" style="width:100%;background:#334155;padding:8px;border-radius:6px;margin-top:8px">Apply settings</button>
    </details>

    <section style="border-top:1px solid #263244;margin-top:14px;padding-top:12px">
      <h3 style="font-weight:bold;color:#cbd5e1;margin-bottom:6px">Python Environment</h3>
      <input id="python-env-path" type="text" spellcheck="false" placeholder="/path/to/venv or /path/to/venv/bin/python" style="width:100%;padding:7px;background:#111827;border:1px solid #334155;border-radius:6px;color:#e2e8f0;font:11px monospace">
      <div style="display:flex;gap:6px;margin-top:6px"><select id="python-env-select" style="min-width:0;flex:1;padding:6px;background:#111827;border:1px solid #334155;border-radius:6px;color:#e2e8f0"></select><button id="save-python-env" style="padding:6px 9px;background:#334155;border-radius:6px">Use</button></div>
      <p id="python-env-status" style="font-size:10px;color:#94a3b8;margin-top:5px">Selected interpreter is injected into every agent prompt.</p>
    </section>
    <section style="border-top:1px solid #263244;margin-top:16px;padding-top:12px"><h3 style="font-weight:bold;color:#cbd5e1;margin-bottom:8px">Skills</h3><div id="skills-list" style="display:grid;gap:6px"></div></section>
    <section id="right-plan-panel" style="border-top:1px solid #263244;margin-top:16px;padding-top:12px">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px">
        <h3 style="font-weight:bold;color:#6ee7b7;display:flex;align-items:center;gap:6px"><i class="fa-solid fa-list-check"></i> Plan / Progress Logs</h3>
        <span id="right-plan-counter" style="font-size:10px;color:#94a3b8">0/0 done</span>
      </div>
      <div id="right-plan-checklist" aria-live="polite" style="display:grid;gap:6px;max-height:320px;overflow:auto">
        <p id="right-plan-empty" style="color:#64748b;font-size:11px">Plan steps with checkboxes appear here when a multi-step task runs.</p>
      </div>
    </section>
    <pre id="agent-debug" style="white-space:pre-wrap;color:#94a3b8;margin-top:10px;border-top:1px solid #263244;padding-top:10px">Backend diagnostics appear here.</pre>
  </aside>
  <style>
    body { padding-right:430px; }
    #sidebar { width:190px !important; }
    #chat-box { max-width:none !important; }
    @media(max-width:1100px){ #agent-dock{width:360px!important}body{padding-right:360px}}
    @media(max-width:720px){ #agent-dock{position:relative!important;width:100%!important;max-height:45vh;bottom:auto;right:auto}body{padding-right:0;overflow:auto!important}#sidebar{display:none!important}}
    #avg-profile-bar{height:5px;background:#1e293b;border-radius:9px;overflow:hidden;margin-top:5px}#avg-profile-fill{height:100%;width:0;background:#34d399;transition:width .3s}
  </style>

  <script>
    const $ = id => document.getElementById(id);
    let loaded = false;

    window.clearExecutionLog = function() {
      const log = $('execution-log');
      if (log) {
        log.innerHTML = '<p class="execution-log-empty" style="color:#64748b">Tool execution output appears here. Chat replies stay in the left pane.</p>';
      }
    };

    document.addEventListener('click', e => {
      if (e.target && e.target.id === 'clear-executionLog') {
        window.clearExecutionLog();
      }
    });

    // Sidebar collapse
    const sidebar = $('sidebar');
    const toggleSidebar = $('toggle-sidebar');
    const toggleSidebarBtn = $('toggle-sidebar-btn');
    let sidebarCollapsed = localStorage.getItem('darkMatterSidebarCollapsed') === 'true';
    function applySidebarCollapse() {
      if (sidebarCollapsed) {
        sidebar.classList.add('collapsed');
        toggleSidebar.querySelector('i').className = 'fa-solid fa-bars';
      } else {
        sidebar.classList.remove('collapsed');
        toggleSidebar.querySelector('i').className = 'fa-solid fa-bars';
      }
    }
    toggleSidebar.addEventListener('click', () => {
      sidebarCollapsed = !sidebarCollapsed;
      localStorage.setItem('darkMatterSidebarCollapsed', String(sidebarCollapsed));
      applySidebarCollapse();
    });
    toggleSidebarBtn.addEventListener('click', () => {
      sidebarCollapsed = !sidebarCollapsed;
      localStorage.setItem('darkMatterSidebarCollapsed', String(sidebarCollapsed));
      applySidebarCollapse();
    });
    applySidebarCollapse();

    const footerThinking = $('footer-thinking-level');
    const dockThinking = $('dock-thinking-level');
    if (footerThinking && dockThinking) {
      footerThinking.addEventListener('change', e => { dockThinking.value = e.target.value; });
      dockThinking.addEventListener('change', e => { footerThinking.value = e.target.value; });
    }

    // API key show/hide
    const apiKeyShow = $('api-key-show');
    const apiKeyInput = $('api-key-input');
    if (apiKeyShow && apiKeyInput) {
      apiKeyShow.addEventListener('click', () => {
        if (apiKeyInput.type === 'password') {
          apiKeyInput.type = 'text';
          apiKeyShow.textContent = '🙈';
        } else {
          apiKeyInput.type = 'password';
          apiKeyShow.textContent = '👁';
        }
      });
      // Load saved key
      const savedKey = localStorage.getItem('darkMatterApiKey');
      if (savedKey) apiKeyInput.value = savedKey;
      apiKeyInput.addEventListener('input', () => {
        localStorage.setItem('darkMatterApiKey', apiKeyInput.value);
      });
    }

    async function refreshLoader() {
      const loader = $('loader-select')?.value || 'Colibri';
      try {
        const r = await fetch('/api/loader');
        const d = await r.json();
        const activeLoader = d.loader || loader;
        if ($('loader-select')) $('loader-select').value = activeLoader;
        const models = d.models || [];
        const modelSelect = $('model-select');
        if (modelSelect) {
          modelSelect.innerHTML = models.map(m => `<option value="${String(m.id).replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;')}">${String(m.label || m.name || m.id).replaceAll('&','&amp;').replaceAll('<','&lt;')}</option>`).join('');
          if (d.model && models.some(m => m.id === d.model)) modelSelect.value = d.model;
          else if (models.length) modelSelect.selectedIndex = 0;
        }
        if ($('question-mode')) $('question-mode').value = d.settings?.question_mode || 'manual';
        if ($('loader-status')) $('loader-status').textContent = `${d.connection}: ${models.length} model(s)`;
        if ($('agent-debug')) $('agent-debug').textContent = d.connection === 'offline' ? `Backend not reachable at ${activeLoader === 'LiteLLM' ? 'http://127.0.0.1:4000' : 'http://127.0.0.1:8000'}.` : 'Backend reachable.';
        loaded = !!d.loaded;
        setLock();
      } catch (e) {
        if ($('loader-status')) $('loader-status').textContent = e.message;
      }
    }

    function setLock() {
      if ($('loader-select')) $('loader-select').disabled = loaded;
      if ($('model-select')) $('model-select').disabled = loaded;
      if ($('load-model')) $('load-model').disabled = loaded;
      if ($('unload-model')) $('unload-model').disabled = !loaded;
      if ($('loader-status')) $('loader-status').textContent = loaded ? 'Model loaded · loader/model locked until unload' : 'Model unloaded · settings available';
    }

    async function refreshThinkingCapability() {
      try {
        const model = $('model-select')?.value || '';
        const r = await fetch(`/api/loader/thinking?model=${encodeURIComponent(model)}`);
        const d = await r.json();
        const footerSelect = $('footer-thinking-level');
        const dockSelect = $('dock-thinking-level');
        if (!d.available) {
          if (footerSelect) { footerSelect.value = 'off'; footerSelect.title = d.reason || 'Thinking mode unavailable for this model'; }
          if (dockSelect) dockSelect.value = 'off';
        } else {
          if (footerSelect) footerSelect.title = 'Adjust model thinking effort';
        }
      } catch (e) {}
    }

    async function refreshTokenLimits() {
      const loader = $('loader-select')?.value || 'Colibri';
      const model = $('model-select')?.value || '';
      try {
        const r = await fetch(`/api/loader/limits?loader=${encodeURIComponent(loader)}&model=${encodeURIComponent(model)}`);
        const limits = await r.json();
        for (const [id, minKey, maxKey] of [['num-predict', 'num_predict_min', 'num_predict_max']]) {
          const el = $(id);
          if (!el) continue;
          const old = Number(el.value) || Number(el.min);
          el.min = limits[minKey];
          el.max = limits[maxKey];
          el.value = Math.min(Number(el.max), Math.max(Number(el.min), old));
        }
        if ($('num-predict-val')) $('num-predict-val').textContent = $('num-predict')?.value;
      } catch (e) {}
    }

    function settings() {
      const common = { temperature: +$('temperature').value, top_p: +$('top-p').value, question_mode: $('question-mode')?.value || 'manual' };
      return { ...common, num_ctx: 8192, num_predict: +$('num-predict').value };
    }

    async function sendLoaderAction(action) {
      const payload = { action, loader: $('loader-select').value, model: $('model-select')?.value || '', api_key: $('api-key-input')?.value || '', ...settings() };
      const r = await fetch('/api/loader', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      const d = await r.json();
      if (!r.ok) throw Error(d.output || 'Request failed');
      loaded = !!d.loaded;
      setLock();
      return d;
    }

    $('loader-select')?.addEventListener('change', async () => {
      await refreshLoader();
      await refreshTokenLimits();
      await refreshThinkingCapability();
    });

    $('model-select')?.addEventListener('change', () => {
      refreshTokenLimits();
      refreshThinkingCapability();
    });

    $('num-predict')?.addEventListener('input', () => {
      if ($('num-predict-val')) $('num-predict-val').textContent = $('num-predict')?.value;
    });

    if ($('refresh-models')) $('refresh-models').onclick = refreshLoader;
    if ($('start-colibri')) $('start-colibri').onclick = async () => {
      try {
        const d = await sendLoaderAction('start_colibri');
        if ($('agent-debug')) $('agent-debug').textContent = d.output || 'Colibri server started.';
        await refreshLoader();
      } catch (e) {
        if ($('agent-debug')) $('agent-debug').textContent = e.message;
      }
    };
    if ($('stop-colibri')) $('stop-colibri').onclick = async () => {
      try {
        const d = await sendLoaderAction('stop_colibri');
        if ($('agent-debug')) $('agent-debug').textContent = d.output || 'Managed Colibri server stopped.';
        await refreshLoader();
      } catch (e) {
        if ($('agent-debug')) $('agent-debug').textContent = e.message;
      }
    };
    if ($('load-model')) $('load-model').onclick = async () => {
      try { await sendLoaderAction('load'); refreshLoader(); }
      catch (e) { if ($('agent-debug')) $('agent-debug').textContent = e.message; }
    };
    if ($('unload-model')) $('unload-model').onclick = async () => {
      try {
        const d = await sendLoaderAction('unload');
        if ($('agent-debug')) $('agent-debug').textContent = d.output || 'Model unloaded.';
        await refreshLoader();
      } catch (e) {
        if ($('agent-debug')) $('agent-debug').textContent = e.message;
      }
    };
    if ($('save-loader-settings')) $('save-loader-settings').onclick = async () => {
      try {
        await sendLoaderAction('configure');
        if ($('agent-debug')) $('agent-debug').textContent = 'Loader parameters applied.';
      } catch (e) {
        if ($('agent-debug')) $('agent-debug').textContent = e.message;
      }
    };

    $('temperature')?.addEventListener('input', () => { if ($('temp-val')) $('temp-val').textContent = $('temperature').value; });
    $('top-p')?.addEventListener('input', () => { if ($('top-p-val')) $('top-p-val').textContent = $('top-p').value; });

    applyLoaderProfile = () => {}; // no-op kept for compatibility
    refreshLoader().then(refreshTokenLimits);

    // ===================================================================
    // MAIN INTERACTION & CHAT PIPELINE
    // ===================================================================
    const chatBox           = $('chat-box');
    const promptInput       = $('prompt-input');
    const sendBtn           = $('send-btn');
    const stopBtn           = $('stop-btn');
    const fileInput         = $('file-input');
    const fileTray          = $('file-tray');
    const dropOverlay       = $('drop-overlay');
    const sessionList       = $('session-list');
    const newChatBtn        = $('new-chat-btn');
    const sessionCount      = $('session-count');
    const sidebar           = $('sidebar');
    const toggleSidebar     = $('toggle-sidebar');

    const projBtn           = $('proj-selector-btn');
    const projDropdown      = $('proj-dropdown');
    const projLabel         = $('proj-selector-label');
    const customPathInput   = $('custom-path-input');
    const switchPathBtn     = $('switch-path-btn');
    const projCountBadge    = $('proj-count-badge');
    const refreshProjBtn    = $('refresh-proj-btn');
    const projStatus        = $('proj-status');
    const projCandidates    = $('proj-candidates-list');
    const projCaret         = $('proj-caret');

    let activeSessionId = null;
    let stagedFiles = [];
    let dragCounter = 0;
    let abortController = null;
    let currentWorkspaceDir = '';
    let runInProgress = false;
    let queuedRuns = [];
    let queuedSubmission = null;

    if (toggleSidebar && sidebar) {
      toggleSidebar.addEventListener('click', () => sidebar.classList.toggle('hidden'));
    }

    function setProjectStatus(message, isError = false) {
      if (!projStatus) return;
      projStatus.textContent = message || '';
      projStatus.className = message
        ? `text-[11px] rounded-lg px-2 py-1 ${isError ? 'bg-rose-950/50 text-rose-300 border border-rose-800/60' : 'bg-slate-900 text-slate-300 border border-slate-800'}`
        : 'hidden';
    }

    function toggleProjDropdown(forceOpen) {
      if (!projDropdown || !projBtn) return;
      const shouldOpen = typeof forceOpen === 'boolean' ? forceOpen : projDropdown.classList.contains('hidden');
      projDropdown.classList.toggle('hidden', !shouldOpen);
      projBtn.setAttribute('aria-expanded', String(shouldOpen));
      if (projCaret) projCaret.classList.toggle('rotate-180', shouldOpen);
      if (shouldOpen) fetchWorkspaceInfo();
    }

    if (projBtn) {
      projBtn.addEventListener('click', e => {
        e.stopPropagation();
        toggleProjDropdown();
      });
    }

    if (projDropdown) {
      projDropdown.addEventListener('click', e => e.stopPropagation());
    }

    document.addEventListener('click', () => toggleProjDropdown(false));
    document.addEventListener('keydown', e => { if (e.key === 'Escape') toggleProjDropdown(false); });

    async function fetchWorkspaceInfo() {
      if (!projBtn || !projLabel || !projCandidates) return;

      const savedWorkspace = localStorage.getItem('selected_workspace');
      if (savedWorkspace && savedWorkspace !== currentWorkspaceDir) {
        currentWorkspaceDir = savedWorkspace;
        await fetch('/api/workspace', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: savedWorkspace })
        });
      }

      projBtn.disabled = true;
      projLabel.textContent = 'Loading projects…';
      projCandidates.innerHTML = '<div class="text-[11px] text-slate-500 italic p-2">Reading available directories…</div>';
      setProjectStatus('Loading project folders…');
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 7000);
        let res;
        try {
          res = await fetch('/api/workspace', { cache: 'no-store', signal: controller.signal });
        } finally {
          clearTimeout(timer);
        }
        const data = await res.json();
        if (!res.ok) throw new Error(data.output || `Request failed (${res.status})`);
        currentWorkspaceDir = data.current || '';
        if (customPathInput) customPathInput.value = currentWorkspaceDir;
        projLabel.textContent = currentWorkspaceDir.split('/').filter(Boolean).pop() || currentWorkspaceDir || 'Select project';
        projLabel.title = currentWorkspaceDir;

        const projects = [...new Set((data.projects || []).filter(p => typeof p === 'string' && p.length))].sort((a, b) => a.localeCompare(b));
        if (currentWorkspaceDir && !projects.includes(currentWorkspaceDir)) projects.unshift(currentWorkspaceDir);
        projCandidates.replaceChildren();

        if (!projects.length) {
          projCandidates.innerHTML = '<div class="text-[11px] text-slate-500 italic p-2">No readable folders found.</div>';
        } else {
          projects.forEach(path => {
            const active = path === currentWorkspaceDir;
            const item = document.createElement('button');
            item.type = 'button';
            item.className = `w-full text-left flex items-start gap-2 p-2 rounded-lg border transition cursor-pointer ${active ? 'bg-emerald-950/40 border-emerald-500/50 text-emerald-300' : 'border-slate-800/80 hover:bg-slate-900 hover:border-slate-700 text-slate-300'}`;
            const icon = document.createElement('i');
            icon.className = `fa-solid ${active ? 'fa-folder-open text-emerald-400' : 'fa-folder text-slate-500'} mt-0.5`;
            const content = document.createElement('span');
            content.className = 'min-w-0 flex-1';
            const name = document.createElement('span');
            name.className = 'block text-xs font-semibold';
            name.textContent = path.split('/').filter(Boolean).pop() || path;
            const full = document.createElement('span');
            full.className = 'block text-[10px] text-slate-500 break-all';
            full.textContent = path;
            content.append(name, full);
            item.append(icon, content);
            item.addEventListener('click', () => active ? toggleProjDropdown(false) : selectWorkspace(path));
            projCandidates.appendChild(item);
          });
        }
        if (projCountBadge) projCountBadge.textContent = `${projects.length} found`;
        setProjectStatus(`${projects.length} folder(s) found.`);
      } catch (err) {
        projLabel.textContent = currentWorkspaceDir.split('/').filter(Boolean).pop() || 'Projects';
        setProjectStatus(`Project load failed: ${err.message}`, true);
      } finally {
        projBtn.disabled = false;
      }
    }

    async function selectWorkspace(targetPath) {
      const path = String(targetPath || '').trim();
      if (!path) return;
      if (switchPathBtn) { switchPathBtn.disabled = true; switchPathBtn.textContent = 'Switching…'; }
      if (projBtn) projBtn.disabled = true;
      try {
        const res = await fetch('/api/workspace', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }) });
        const data = await res.json();
        if (!res.ok) throw new Error(data.output || `Request failed (${res.status})`);
        currentWorkspaceDir = data.current || '';
        localStorage.setItem('selected_workspace', currentWorkspaceDir);
        projLabel.textContent = currentWorkspaceDir.split('/').filter(Boolean).pop() || currentWorkspaceDir || 'Select project';
        projLabel.title = currentWorkspaceDir;
        toggleProjDropdown(false);
      } catch (err) {
        setProjectStatus(`Workspace switch failed: ${err.message}`, true);
      } finally {
        if (switchPathBtn) { switchPathBtn.disabled = false; switchPathBtn.textContent = 'Switch'; }
        if (projBtn) projBtn.disabled = false;
      }
    }

    fetch('/api/skills').then(r => r.json()).then(d => {
      if ($('skills-list')) {
        $('skills-list').innerHTML = d.skills.map(s => `<label style="display:flex;gap:7px;align-items:center"><input type="checkbox" class="skill-toggle" data-skill-file="${s.file}" checked> ${s.name} <span style="color:#64748b">(${s.file})</span></label>`).join('');
      }
    }).catch(() => {});

    window.selectedSkillFiles = function() {
      return [...document.querySelectorAll('.skill-toggle:checked')].map(el => el.dataset.skillFile);
    };

    refreshLoader().then(refreshTokenLimits);
  </script>
</body>
</html>"""

if __name__ == "__main__":
    detect_colibri_model()
    print(f"\n=== Dark Matter Autonomous Studio (Payload Verified Edition) ===")
    print(f"  - Web Interface:     http://127.0.0.1:{APP_PORT}")
    print(f"  - Loader:            {LOADER_STATE['loader']} → {LOADER_STATE.get('model','(none)' )}")
    print(f"  - Max Tokens/Turn:   {COLIBRI_MAX_TOKENS}")
    print(f"  - Active Workspace:  {get_workspace_dir()}\n")
    app.run(host="127.0.0.1", port=APP_PORT, debug=False, threaded=True)
