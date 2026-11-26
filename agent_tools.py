import os
import pwd
from pathlib import Path
import re
import ast
import json
import time
import shlex
import shutil
import fnmatch
import difflib
import hashlib
import tempfile
import subprocess
import urllib.parse

# Optional integrations degrade gracefully so core coding tools always load
try:
    import mss
except ImportError:
    mss = None
try:
    from PIL import Image
except ImportError:
    Image = None
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

# =====================================================================
# PERSISTENT WORKSPACE & PERMISSIONS MANAGEMENT
# =====================================================================

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".workspace_config.json")

def _load_initial_workspace() -> str:
    """Load previously selected project directory or default to /home/local-ai."""
    if os.path.isfile(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f).get("active_workspace")
                if saved and os.path.isdir(saved):
                    return os.path.abspath(saved)
        except Exception:
            pass
    return os.path.abspath("/home/local-ai")

WORKSPACE_DIR = _load_initial_workspace()
SCREENSHOT_DIR = os.path.join(WORKSPACE_DIR, "screenshots")


def _normalize_permissions(filepath: str):
    """Compatibility no-op. Studio never silently changes ownership or permissions."""
    return None


def inspect_workspace_permissions(path: str) -> dict:
    """Report ownership, mode bits, and effective access without changing metadata."""
    target, safe = _resolve_safe_path(path)
    if not safe:
        return {"status": "error", "output": "Access denied: path is outside the active workspace."}
    try:
        st = os.lstat(target)
        if os.path.islink(target):
            return {"status": "error", "output": "Refusing to inspect a symbolic-link target."}
        try:
            owner = pwd.getpwuid(st.st_uid).pw_name
        except KeyError:
            owner = str(st.st_uid)
        try:
            import grp
            group = grp.getgrgid(st.st_gid).gr_name
        except (KeyError, ImportError):
            group = str(st.st_gid)
        return {
            "status": "success", "path": os.path.relpath(target, get_workspace_dir()),
            "owner": owner, "uid": st.st_uid, "group": group, "gid": st.st_gid,
            "mode": format(st.st_mode & 0o777, "04o"),
            "is_directory": os.path.isdir(target), "is_symlink": False,
            "effective": {"read": os.access(target, os.R_OK), "write": os.access(target, os.W_OK),
                          "execute": os.access(target, os.X_OK)},
            "running_uid": os.geteuid(),
            "output": f"{target}\nOwner: {owner} ({st.st_uid})\nGroup: {group} ({st.st_gid})\nMode: {format(st.st_mode & 0o777, '04o')}\nEffective access: read={os.access(target, os.R_OK)}, write={os.access(target, os.W_OK)}, execute={os.access(target, os.X_OK)}"
        }
    except OSError as exc:
        return {"status": "error", "output": f"Cannot inspect permissions: {exc}"}


def repair_workspace_metadata(path: str, action: str, approved: bool = False,
                              mode: str = "", owner: str = "", group: str = "") -> dict:
    """Run only a narrowly scoped chmod/chown through pkexec after explicit approval."""
    if approved is not True:
        return {"status": "confirmation_required", "output": "No metadata change was made. Obtain explicit user approval, then retry with approved=true."}
    target, safe = _resolve_safe_path(path)
    if not safe or os.path.islink(target) or not os.path.lexists(target):
        return {"status": "error", "output": "Target must exist, be inside the active workspace, and not be a symbolic link."}
    action = str(action or "").lower().strip()
    if action == "chmod":
        raw = str(mode).strip()
        if not re.fullmatch(r"0?[0-7]{3}", raw):
            return {"status": "error", "output": "Mode must be an explicit octal permission such as 0644 or 0750; special bits are not allowed."}
        numeric_mode = int(raw, 8)
        if numeric_mode & ~0o777:
            return {"status": "error", "output": "Special permission bits are not allowed."}
        argv = ["/usr/bin/chmod", format(numeric_mode, "03o"), "--", target]
        description = f"chmod {format(numeric_mode, '04o')} {target}"
    elif action == "chown":
        if not owner and not group:
            return {"status": "error", "output": "Specify an owner and/or group."}
        import grp
        uid = str(pwd.getpwnam(owner).pw_uid) if owner else ""
        gid = str(grp.getgrnam(group).gr_gid) if group else ""
        spec = (uid if uid else "") + (":" + gid if gid else (":" if uid else ""))
        argv = ["/usr/bin/chown", "--", spec, target]
        description = f"chown {spec} {target}"
    else:
        return {"status": "error", "output": "Only chmod and chown metadata operations are permitted."}
    if not shutil.which("pkexec"):
        return {"status": "error", "output": "pkexec/PolicyKit is unavailable. No change was made; configure a narrowly scoped OS authorization mechanism first."}
    try:
        proc = subprocess.run(["pkexec", *argv], capture_output=True, text=True, timeout=60, check=False, shell=False)
        if proc.returncode != 0:
            return {"status": "error", "exit_code": proc.returncode, "output": (proc.stderr or proc.stdout or "Authorization or metadata operation failed.")[-4000:]}
        result = inspect_workspace_permissions(path)
        result["output"] = f"Applied approved metadata operation: {description}\n" + result.get("output", "")
        return result
    except Exception as exc:
        return {"status": "error", "output": f"Metadata operation failed: {type(exc).__name__}: {exc}"}


os.makedirs(SCREENSHOT_DIR, exist_ok=True)
_normalize_permissions(SCREENSHOT_DIR)


# =====================================================================
# DYNAMIC WORKSPACE DIRECTORY MANAGEMENT
# =====================================================================

def get_workspace_dir() -> str:
    """Returns the current active project root directory."""
    global WORKSPACE_DIR
    return WORKSPACE_DIR


def set_workspace_dir(path: str) -> dict:
    """Hot-swaps the active project workspace directory and persists the selection."""
    global WORKSPACE_DIR, SCREENSHOT_DIR
    try:
        clean = os.path.abspath(os.path.expanduser(os.path.expandvars(path.strip())))
        if not os.path.exists(clean):
            return {"status": "error", "output": f"Project directory does not exist: {clean}."}
        if not os.path.isdir(clean):
            return {"status": "error", "output": f"Project path is not a directory: {clean}"}
        if not os.access(clean, os.R_OK | os.X_OK):
            return {"status": "error", "output": f"Project directory is not accessible by Studio: {clean}"}

        WORKSPACE_DIR = clean
        SCREENSHOT_DIR = os.path.join(WORKSPACE_DIR, "screenshots")
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)

        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump({"active_workspace": WORKSPACE_DIR}, f, indent=2)
        except Exception:
            pass

        return {"status": "success", "workspace": WORKSPACE_DIR}
    except Exception as e:
        return {"status": "error", "output": f"Could not change workspace: {e}"}


def list_available_projects() -> list:
    """Fast, bounded list of selectable project directories for the UI."""
    candidates = set()
    current = os.path.abspath(os.path.expanduser(get_workspace_dir()))
    home = os.path.abspath(os.path.expanduser("~"))

    def add_dir(path):
        try:
            expanded = os.path.abspath(os.path.expandvars(os.path.expanduser(str(path or ""))))
            if expanded and os.path.isdir(expanded) and os.access(expanded, os.R_OK | os.X_OK):
                candidates.add(expanded)
        except (OSError, ValueError, TypeError):
            pass

    add_dir(current)
    add_dir(home)
    user_dirs = {
        "DESKTOP": os.path.join(home, "Desktop"),
        "DOCUMENTS": os.path.join(home, "Documents"),
        "DOWNLOAD": os.path.join(home, "Downloads"),
        "MUSIC": os.path.join(home, "Music"),
        "PICTURES": os.path.join(home, "Pictures"),
        "VIDEOS": os.path.join(home, "Videos"),
        "PUBLICSHARE": os.path.join(home, "Public"),
    }
    try:
        xdg_file = os.path.join(home, ".config", "user-dirs.dirs")
        with open(xdg_file, "r", encoding="utf-8") as fh:
            for line in fh:
                match = re.match(r'\s*XDG_(DESKTOP|DOCUMENTS|DOWNLOAD|MUSIC|PICTURES|VIDEOS|PUBLICSHARE)_DIR="(.+)"\s*$', line)
                if match:
                    key, value = match.groups()
                    user_dirs[key] = value.replace("$HOME", home)
    except (OSError, UnicodeError):
        pass
    for path in user_dirs.values():
        add_dir(path)

    roots = {current, os.path.dirname(current), home}
    for root in roots:
        try:
            with os.scandir(root) as entries:
                for entry in entries:
                    try:
                        if entry.name.startswith(".") or entry.name == "screenshots":
                            continue
                        if entry.is_dir(follow_symlinks=False) and os.access(entry.path, os.R_OK | os.X_OK):
                            candidates.add(os.path.abspath(entry.path))
                    except (OSError, PermissionError):
                        continue
        except (OSError, PermissionError, ValueError):
            continue
    return sorted(candidates, key=lambda path: (path.casefold() != home.casefold(), path.casefold()))


def _resolve_safe_path(rel_or_abs_path: str) -> tuple[str, bool]:
    """Resolves a path safely; absolute paths are accepted only inside active WORKSPACE_DIR."""
    active_root = os.path.realpath(get_workspace_dir())
    raw = os.path.expandvars(os.path.expanduser(str(rel_or_abs_path or "").strip()))
    if not raw:
        return active_root, False
    target = os.path.realpath(raw if os.path.isabs(raw) else os.path.join(active_root, raw))
    try:
        return target, os.path.commonpath([active_root, target]) == active_root
    except ValueError:
        return target, False


# =====================================================================
# 1. WORKSPACE FILE & TERMINAL EXECUTION
# =====================================================================

def _shell_command_risks(command: str) -> list[str]:
    """Return reasons a shell command needs explicit user confirmation."""
    checks = [
        (r"(^|[;&|]\s*)sudo\b", "privilege escalation (sudo)"),
        (r"\brm\s+[^\n]*(-[a-zA-Z]*r[a-zA-Z]*f|-rf|-fr)\b", "recursive/force deletion"),
        (r"\b(rm|rmdir)\s+", "file or directory deletion"),
        (r"\b(mkfs(?:\.[\w]+)?|fdisk|parted|wipefs|dd)\b", "disk or filesystem modification"),
        (r"\b(shutdown|reboot|poweroff|halt)\b", "system power action"),
        (r"\b(chmod|chown)\s+[^\n]*-R\b", "recursive permission/ownership change"),
        (r"\bgit\s+reset\s+--hard\b|\bgit\s+clean\s+-[a-zA-Z]*f", "destructive Git operation"),
        (r"\b(systemctl|service)\s+(stop|disable|restart)\b", "system service change"),
        (r"\b(killall|pkill)\b|\bkill\s+-9\b", "process termination"),
        (r"\b(apt|apt-get|dnf|yum|pacman)\s+(remove|purge|autoremove)\b", "system package removal"),
    ]
    found = []
    for pattern, reason in checks:
        if re.search(pattern, command, flags=re.IGNORECASE):
            found.append(reason)
    return list(dict.fromkeys(found))


def _python_runtime_environment() -> tuple[str, dict]:
    """Return the selected interpreter and a shell environment that keeps its venv first on PATH."""
    python = _project_python()
    if not os.path.isfile(python) or not os.access(python, os.X_OK):
        raise FileNotFoundError(f"Selected Python interpreter is unavailable: {python}")
    env = os.environ.copy()
    bin_dir = os.path.dirname(os.path.abspath(python))
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    env["PYTHON"] = os.path.abspath(python)
    venv_root = os.path.dirname(bin_dir)
    if os.path.isfile(os.path.join(venv_root, "pyvenv.cfg")):
        env["VIRTUAL_ENV"] = venv_root
    else:
        env.pop("VIRTUAL_ENV", None)
    return python, env


def _missing_python_module(output: str) -> str:
    raw = str(output or "")
    match = re.search(r"(?:ModuleNotFoundError|ImportError):\s*No module named\s+['\"]?([A-Za-z_][A-Za-z0-9_.]*)", raw)
    if not match:
        match = re.search(r"\bNo module named\s+['\"]?([A-Za-z_][A-Za-z0-9_.]*)", raw)
    return match.group(1) if match else ""


def _add_python_dependency_recovery(result: dict) -> dict:
    """Attach a concrete dependency recovery action to Python execution failures."""
    module = _missing_python_module(result.get("output", ""))
    if module:
        result["missing_python_module"] = module
        result["recovery"] = (
            f"Check project requirements for the PyPI distribution that provides '{module}'. "
            "Call install_missing_python_module with the missing import and distribution name, "
            "then rerun the failed command using the selected Python environment."
        )
    return result


def execute_terminal(command: str, timeout: int = 120, cwd: str = "", confirm: bool = False,
                     max_output_chars: int = 24000) -> dict:
    """Run a shell command in the workspace with bounded output and process-group cleanup.

    Potentially destructive commands are not executed until the caller repeats the
    command with confirm=True after obtaining the user's approval.
    """
    command = str(command or "").strip()
    if not command:
        return {"status": "error", "exit_code": -1, "output": "Command cannot be empty."}

    risks = _shell_command_risks(command)
    if risks and not confirm:
        return {
            "status": "confirmation_required", "exit_code": None,
            "requires_confirmation": True, "risk_reasons": risks,
            "command": command,
            "output": "This command was NOT executed. Ask the user for explicit approval, then repeat the exact command with confirm=true. Risks: " + ", ".join(risks)
        }

    active_root = os.path.realpath(get_workspace_dir())
    requested_cwd = os.path.expandvars(os.path.expanduser(str(cwd or ".").strip()))
    target_cwd = os.path.realpath(requested_cwd if os.path.isabs(requested_cwd) else os.path.join(active_root, requested_cwd))
    try:
        if os.path.commonpath([active_root, target_cwd]) != active_root or not os.path.isdir(target_cwd):
            return {"status": "error", "exit_code": -1, "output": f"Invalid working directory: {cwd!r}. It must be an existing directory inside {active_root}."}
    except ValueError:
        return {"status": "error", "exit_code": -1, "output": f"Invalid working directory: {cwd!r}."}

    try:
        timeout = max(1, min(int(timeout), 1800))
        max_output_chars = max(1000, min(int(max_output_chars), 100000))
        try:
            selected_python, runtime_env = _python_runtime_environment()
        except Exception as exc:
            return {"status": "error", "exit_code": -1, "command": command, "cwd": target_cwd,
                    "error_type": type(exc).__name__, "output": f"Python environment setup failed: {exc}"}
        runtime_env["COLIBRI_WORKSPACE"] = active_root
        runtime_env["COLIBRI_PROJECT_ROOT"] = active_root

        started_at = time.monotonic()
        with tempfile.TemporaryFile(mode="w+b") as output_file:
            proc = subprocess.Popen(
                "set -o pipefail\n" + command, shell=True, cwd=target_cwd, env=runtime_env,
                stdout=output_file, stderr=subprocess.STDOUT,
                executable="/bin/bash" if os.path.exists("/bin/bash") else None,
                start_new_session=True
            )
            try:
                proc.wait(timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(proc.pid, 15)
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        os.killpg(proc.pid, 9)
                    except Exception:
                        pass
                    proc.wait()
            duration = round(time.monotonic() - started_at, 3)
            output_file.seek(0, os.SEEK_END)
            total_bytes = output_file.tell()
            read_bytes = min(total_bytes, max_output_chars)
            output_file.seek(max(0, total_bytes - read_bytes))
            output = output_file.read(read_bytes).decode("utf-8", errors="replace")

        truncated = total_bytes > max_output_chars
        if truncated:
            output = f"...[showing last {max_output_chars} bytes of {total_bytes} output bytes]\n" + output
        output = output or "(No output)"
        if timed_out:
            return {
                "status": "timeout", "exit_code": -1, "timed_out": True,
                "timeout_seconds": timeout, "cwd": target_cwd, "duration_seconds": duration,
                "command": command, "python": selected_python,
                "output": output + f"\n\n[Colibri terminated the command process group after {timeout}s.]"
            }
        return {
            "status": "success" if proc.returncode == 0 else "error",
            "exit_code": proc.returncode, "cwd": target_cwd,
            "command": command, "python": selected_python,
            "output": output, "output_truncated": truncated,
            "output_bytes": total_bytes, "duration_seconds": duration,
            "running_as_root": os.geteuid() == 0
        }
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return {"status": "error", "exit_code": -1, "command": command, "cwd": target_cwd,
                "error_type": type(exc).__name__, "output": f"Shell execution failed: {exc}"}


def _file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_workspace_write(target: str, content: str, backup: bool = True) -> tuple[str, str]:
    """Durably replace a workspace file and optionally keep a timestamped backup."""
    parent = os.path.dirname(target)
    os.makedirs(parent, exist_ok=True)
    backup_path = ""
    if os.path.isfile(target) and backup:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup_path = f"{target}.agent-backup-{stamp}-{os.getpid()}"
        shutil.copy2(target, backup_path)
    old_mode = None
    if os.path.isfile(target):
        try:
            old_mode = os.stat(target, follow_symlinks=False).st_mode & 0o777
        except OSError:
            pass
    fd, temp_path = tempfile.mkstemp(prefix=".agent-edit-", dir=parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if old_mode is not None:
            os.chmod(temp_path, old_mode)
        os.replace(temp_path, target)
        try:
            dir_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return backup_path, _file_sha256(content.encode("utf-8"))


def _edit_result(path: str, old_content: str, new_content: str, backup_path: str, new_hash: str) -> dict:
    diff_lines = list(difflib.unified_diff(
        old_content.splitlines(), new_content.splitlines(),
        fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""
    ))
    diff = "\n".join(diff_lines)
    added_lines = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    removed_lines = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))
    if len(diff) > 16000:
        diff = diff[:16000] + "\n...[diff truncated]"
    result = {
        "status": "success", "path": path,
        "backup": backup_path or None,
        "sha256": new_hash,
        "changed": old_content != new_content,
        "added_lines": added_lines,
        "removed_lines": removed_lines,
        "diff": diff or "(No content changes)",
        "output": f"Updated {path}; changes: +{added_lines} / -{removed_lines} lines; SHA-256: {new_hash}" + (f"; backup: {backup_path}" if backup_path else "")
    }
    if path.lower().endswith(".py"):
        check = check_python_file(path)
        result["syntax_check"] = check
        result["output"] += f"\n[Python syntax] {check.get('output', check.get('status'))}"
        if check.get("status") != "success":
            result["status"] = "error"
            result["validation_error"] = True
    return result


def write_workspace_file(path: str, content: str, expected_sha256: str = "",
                         create_only: bool = False, backup: bool = True) -> dict:
    """Create or atomically replace a workspace file with optional optimistic concurrency protection."""
    active_root = get_workspace_dir()
    try:
        target, is_safe = _resolve_safe_path(path)
        if not is_safe:
            return {"status": "error", "output": f"Access denied: '{path}' traverses outside workspace."}
        if os.path.islink(target):
            return {"status": "error", "output": "Refusing to write through a symbolic link."}
        exists = os.path.isfile(target)
        if create_only and os.path.exists(target):
            return {"status": "error", "output": f"File already exists: {path}. No changes made."}
        old_bytes = Path(target).read_bytes() if exists else b""
        old_hash = _file_sha256(old_bytes) if exists else ""
        if expected_sha256 and old_hash != str(expected_sha256).strip():
            return {"status": "conflict", "output": "File changed since it was read. Re-read it before applying edits.", "expected_sha256": expected_sha256, "actual_sha256": old_hash}
        old_content = old_bytes.decode("utf-8") if exists else ""
        backup_path, new_hash = _atomic_workspace_write(target, str(content), backup=bool(backup and exists))
        rel = os.path.relpath(target, active_root)
        return _edit_result(rel, old_content, str(content), backup_path, new_hash)
    except UnicodeDecodeError:
        return {"status": "error", "output": f"{path} is not UTF-8 text; refusing text overwrite."}
    except Exception as exc:
        return {"status": "error", "output": f"File write error: {type(exc).__name__}: {exc}"}


def read_workspace_file(path: str, line_start: int = 1, line_count: int = 400) -> dict:
    """Read a bounded line range and return a content hash for safe follow-up edits."""
    try:
        target, is_safe = _resolve_safe_path(path)
        if not is_safe:
            return {"status": "error", "output": f"Access denied: '{path}' traverses outside workspace."}
        if not os.path.isfile(target) or os.path.islink(target):
            return {"status": "error", "output": f"File does not exist or is not a regular file: {path}"}
        raw = Path(target).read_bytes()
        content = raw.decode("utf-8")
        lines = content.splitlines()
        total_lines = len(lines)
        start_idx = max(1, int(line_start)) - 1
        end_idx = min(total_lines, start_idx + max(1, min(int(line_count), 2000)))
        formatted = [f"{start_idx + i + 1:5d} | {line}" for i, line in enumerate(lines[start_idx:end_idx])]
        rel = os.path.relpath(target, get_workspace_dir())
        return {
            "status": "success", "path": rel, "total_lines": total_lines,
            "size_bytes": len(raw), "sha256": _file_sha256(raw),
            "output": f"=== File: {rel} (lines {start_idx+1}-{end_idx} of {total_lines}; {len(raw)} bytes) ===\n" + "\n".join(formatted)
        }
    except UnicodeDecodeError:
        return {"status": "error", "output": f"File is not UTF-8 text: {path}"}
    except Exception as exc:
        return {"status": "error", "output": f"File read error: {type(exc).__name__}: {exc}"}


def patch_workspace_file(path: str, search_pattern: str, replacement: str,
                         expected_sha256: str = "", replace_all: bool = False,
                         backup: bool = True) -> dict:
    """Apply an exact text patch; reject ambiguous matches and stale file versions."""
    try:
        target, safe = _resolve_safe_path(path)
        if not safe:
            return {"status": "error", "output": f"Access denied: '{path}' traverses outside workspace."}
        if not os.path.isfile(target) or os.path.islink(target):
            return {"status": "error", "output": f"File not found or unsafe: {path}"}
        if not search_pattern:
            return {"status": "error", "output": "Patch rejected: search_pattern cannot be empty."}
        raw = Path(target).read_bytes()
        content = raw.decode("utf-8")
        current_hash = _file_sha256(raw)
        if expected_sha256 and current_hash != str(expected_sha256).strip():
            return {"status": "conflict", "output": "File changed since it was read. Re-read before patching.", "expected_sha256": expected_sha256, "actual_sha256": current_hash}
        count = content.count(search_pattern)
        if count == 0:
            return {"status": "error", "output": f"Exact pattern not found in {path}. No changes made."}
        if count > 1 and not replace_all:
            return {"status": "ambiguous", "matches": count, "output": f"Pattern occurs {count} times in {path}. Supply a more specific pattern or set replace_all=true."}
        new_content = content.replace(search_pattern, replacement, -1 if replace_all else 1)
        if new_content == content:
            return {"status": "success", "path": path, "changed": False, "sha256": current_hash, "output": "Patch matched but produced no changes."}
        backup_path, new_hash = _atomic_workspace_write(target, new_content, backup=backup)
        return _edit_result(os.path.relpath(target, get_workspace_dir()), content, new_content, backup_path, new_hash)
    except UnicodeDecodeError:
        return {"status": "error", "output": f"File is not UTF-8 text: {path}"}
    except Exception as exc:
        return {"status": "error", "output": f"File patch error: {type(exc).__name__}: {exc}"}


def edit_workspace_lines(path: str, start_line: int, end_line: int, new_content: str,
                         expected_sha256: str = "", backup: bool = True) -> dict:
    """Atomically replace a validated 1-based inclusive line range."""
    try:
        target, safe = _resolve_safe_path(path)
        if not safe:
            return {"status": "error", "output": f"Access denied: '{path}' traverses outside workspace."}
        if not os.path.isfile(target) or os.path.islink(target):
            return {"status": "error", "output": f"File not found or unsafe: {path}"}
        raw = Path(target).read_bytes()
        content = raw.decode("utf-8")
        current_hash = _file_sha256(raw)
        if expected_sha256 and current_hash != str(expected_sha256).strip():
            return {"status": "conflict", "output": "File changed since it was read. Re-read before editing lines.", "expected_sha256": expected_sha256, "actual_sha256": current_hash}
        lines = content.splitlines(keepends=True)
        start_line, end_line = int(start_line), int(end_line)
        if start_line < 1 or end_line < start_line or end_line > len(lines):
            return {"status": "error", "output": f"Invalid line range {start_line}..{end_line} for {len(lines)} lines. No changes made."}
        newline = "\r\n" if b"\r\n" in raw else "\n"
        replacement_text = new_content.replace("\r\n", "\n").replace("\r", "\n").replace("\n", newline)
        original_range_ended_with_newline = lines[end_line-1].endswith(("\n", "\r"))
        if replacement_text and original_range_ended_with_newline and not replacement_text.endswith(newline):
            replacement_text += newline
        updated = "".join(lines[:start_line-1]) + replacement_text + "".join(lines[end_line:])
        if updated == content:
            return {"status": "success", "path": path, "changed": False, "sha256": current_hash, "output": "Line edit produced no changes."}
        backup_path, new_hash = _atomic_workspace_write(target, updated, backup=backup)
        return _edit_result(os.path.relpath(target, get_workspace_dir()), content, updated, backup_path, new_hash)
    except UnicodeDecodeError:
        return {"status": "error", "output": f"File is not UTF-8 text: {path}"}
    except Exception as exc:
        return {"status": "error", "output": f"Line edit error: {type(exc).__name__}: {exc}"}


def list_workspace_files(relative_path: str = ".", max_depth: int = 2) -> dict:
    """Provides a structural directory tree of the workspace."""
    try:
        target, is_safe = _resolve_safe_path(relative_path or ".")
        if not is_safe:
            return {"status": "error", "output": f"Access denied: '{relative_path}' traverses outside workspace."}

        if not os.path.exists(target):
            return {"status": "error", "output": f"Directory not found: {relative_path}"}

        tree_lines = []
        base_depth = target.rstrip(os.path.sep).count(os.path.sep)
        max_d = int(max_depth)

        for root, dirs, files in os.walk(target):
            dirs[:] = [d for d in sorted(dirs) if not d.startswith(".") and d not in _IGNORED_DIRS]
            current_depth = root.count(os.path.sep) - base_depth
            if current_depth > max_d:
                dirs.clear()
                continue

            indent = "  " * current_depth
            folder_name = os.path.basename(root) or "."
            tree_lines.append(f"{indent}📂 {folder_name}/")

            sub_indent = "  " * (current_depth + 1)
            for f in sorted(files):
                if f.startswith("."):
                    continue
                f_path = os.path.join(root, f)
                f_size = os.path.getsize(f_path) if os.path.exists(f_path) else 0
                tree_lines.append(f"{sub_indent}📄 {f} ({f_size:,} bytes)")

        return {
            "status": "success",
            "output": "\n".join(tree_lines) or "(Empty directory)"
        }
    except Exception as e:
        return {"status": "error", "output": f"Directory listing error: {e}"}


# =====================================================================
# 1B. DEBUGGING, CODE CHECKING, PARITY & SELF-REPAIR SUPPORT
# =====================================================================

def check_python_file(path: str) -> dict:
    """Compile/parse a Python file without executing it; report syntax location."""
    target, safe = _resolve_safe_path(path)
    if not safe:
        return {"status": "error", "output": f"Access denied: {path} is outside workspace."}
    if not os.path.isfile(target):
        return {"status": "error", "output": f"Python file not found: {path}"}
    try:
        with open(target, "r", encoding="utf-8") as source_file:
            source = source_file.read()
        ast.parse(source, filename=target)
        return {"status": "success", "output": f"Syntax OK: {path}", "checks": {"ast_parse": "passed"}}
    except SyntaxError as e:
        return {
            "status": "error",
            "output": f"SyntaxError in {path}:{e.lineno}:{e.offset}: {e.msg}",
            "line": e.lineno,
            "column": e.offset
        }
    except Exception as e:
        return {"status": "error", "output": f"Code check failed for {path}: {type(e).__name__}: {e}"}


check_python_syntax = check_python_file


def diagnose_tool_error(tool_name: str, target: str, result: dict) -> dict:
    """Turn failed tool results into actionable diagnostics."""
    if result.get("status") == "success":
        return {"status": "success", "output": "No tool error detected.", "retry_safe": False}
    raw = str(result.get("output", "unknown tool failure"))
    lower = raw.lower()
    suggestions = []
    if "not found" in lower or "no such file" in lower:
        suggestions += [
            "List the active workspace tree and confirm the exact filename.",
            "Use read_file with the path relative to the active workspace."
        ]
    if "permission denied" in lower or "access denied" in lower:
        suggestions.append("Check workspace root and file permissions.")
    if not suggestions:
        suggestions.append("Inspect the full error, validate arguments, and retry once only if safe.")
    return {
        "status": "error",
        "tool": tool_name,
        "target": target,
        "error": raw,
        "diagnosis": suggestions,
        "retry_safe": tool_name in {"read_file", "list_files", "check_python"}
    }


def run_project_checks(path: str = ".") -> dict:
    """Run lightweight, non-executing project checks: Python AST syntax and JSON validity."""
    target, safe = _resolve_safe_path(path or ".")
    if not safe or not os.path.isdir(target):
        return {"status": "error", "output": f"Invalid or inaccessible project directory: {path}"}
    checked, failures = [], []
    for root, dirs, files in os.walk(target):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in _IGNORED_DIRS]
        for name in files:
            if name.startswith("."):
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, target)
            if name.endswith(".py"):
                result = check_python_file(full)
                checked.append({"file": rel, "kind": "python", "status": result["status"], "output": result["output"]})
                if result["status"] != "success":
                    failures.append(rel)
            elif name.endswith(".json") and os.path.getsize(full) <= 2_000_000:
                try:
                    with open(full, encoding="utf-8") as f:
                        json.load(f)
                    checked.append({"file": rel, "kind": "json", "status": "success", "output": "Valid JSON"})
                except Exception as e:
                    checked.append({"file": rel, "kind": "json", "status": "error", "output": str(e)})
                    failures.append(rel)
    return {
        "status": "error" if failures else "success",
        "checked_count": len(checked),
        "failure_count": len(failures),
        "failures": failures,
        "results": checked,
        "output": f"Checked {len(checked)} files; {len(failures)} issue(s)."
    }


# =====================================================================
# 1C. CODING AGENT TOOLKIT: SEARCH, DISCOVERY, TESTS, LINT, GIT, PARITY
# =====================================================================

_IGNORED_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
    "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    "dist", "build", "coverage", ".next", ".nuxt", "vendor", "screenshots",
    ".cache", ".local", ".config", ".npm", ".cargo", ".rustup", "models"
}
_TEXT_SUFFIXES = {
    ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
    ".html", ".css", ".scss", ".json", ".toml", ".yaml", ".yml",
    ".md", ".txt", ".sh", ".bash", ".sql", ".java", ".kt", ".go",
    ".rs", ".c", ".h", ".cpp", ".hpp", ".php", ".rb", ".swift",
    ".vue", ".svelte", ".xml", ".ini", ".cfg", ".conf", ".env.example"
}


def _walk_project_files(root: str, max_files: int = 5000):
    """Walk project files efficiently, pruning hidden/heavy directories and binary artifacts."""
    count = 0
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d not in _IGNORED_DIRS)
        for name in sorted(files):
            if name.startswith("."):
                continue
            full = os.path.join(current, name)
            try:
                if os.path.islink(full) or os.path.getsize(full) > 2_000_000:
                    continue
            except OSError:
                continue
            count += 1
            if count > max_files:
                return
            yield full


def search_workspace_code(query: str, path: str = ".", max_results: int = 30, case_sensitive: bool = False) -> dict:
    target, safe = _resolve_safe_path(path or ".")
    query = str(query or "").strip()
    if not safe or not os.path.isdir(target):
        return {"status": "error", "output": f"Invalid or inaccessible search directory: {path}"}
    if not query:
        return {"status": "error", "output": "Search query cannot be empty."}
    max_results = max(1, min(int(max_results), 200))
    needle = query if case_sensitive else query.casefold()
    hits = []
    scanned = 0
    for full in _walk_project_files(target):
        if os.path.splitext(full)[1].lower() not in _TEXT_SUFFIXES and os.path.basename(full) not in {
            "Dockerfile", "Makefile", "Justfile", "Procfile", "Gemfile", "Brewfile"
        }:
            continue
        scanned += 1
        try:
            with open(full, "r", encoding="utf-8", errors="replace") as stream:
                for line_no, line in enumerate(stream, 1):
                    haystack = line if case_sensitive else line.casefold()
                    if needle in haystack:
                        hits.append({"file": os.path.relpath(full, target), "line": line_no, "text": line.rstrip()[:500]})
                        if len(hits) >= max_results:
                            break
        except (OSError, UnicodeError):
            continue
        if len(hits) >= max_results:
            break
    formatted = [f"{h['file']}:{h['line']}: {h['text']}" for h in hits]
    return {
        "status": "success",
        "query": query,
        "scanned_files": scanned,
        "count": len(hits),
        "truncated": len(hits) >= max_results,
        "results": hits,
        "output": "\n".join(formatted) if formatted else f"No matches for {query!r} in {target}.",
    }


def grep_workspace(pattern: str, path: str = ".", file_pattern: str = "*", case_sensitive: bool = False, max_results: int = 50) -> dict:
    """Streamlined grep that avoids blocking on large non-text directories or binary files."""
    active_root = get_workspace_dir()
    try:
        target, is_safe = _resolve_safe_path(path or ".")
        if not is_safe or not os.path.exists(target):
            return {"status": "error", "output": f"Invalid search path: {path}"}

        flags = 0 if str(case_sensitive).lower() in ("true", "1") else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error:
            regex = re.compile(re.escape(pattern), flags)

        matches = []
        limit = int(max_results)
        file_stream = [target] if os.path.isfile(target) else _walk_project_files(target)

        for fpath in file_stream:
            ext = os.path.splitext(fpath)[1].lower()
            fname = os.path.basename(fpath)

            if not fnmatch.fnmatch(fname, file_pattern or "*"):
                continue
            if ext not in _TEXT_SUFFIXES and fname not in {"Dockerfile", "Makefile", "Justfile", "Procfile"}:
                continue

            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    for idx, line in enumerate(f, 1):
                        if regex.search(line):
                            rel = os.path.relpath(fpath, active_root)
                            matches.append(f"{rel}:{idx}: {line.strip()[:250]}")
                            if len(matches) >= limit:
                                break
            except Exception:
                continue
            if len(matches) >= limit:
                break

        if not matches:
            return {"status": "success", "count": 0, "output": f"No matches found for pattern '{pattern}' in {path}."}

        header = f"Found {len(matches)} match(es) for '{pattern}':\n"
        return {"status": "success", "count": len(matches), "output": header + "\n".join(matches)}
    except Exception as e:
        return {"status": "error", "output": f"Grep search error: {e}"}


def find_workspace_files(pattern: str = "*", path: str = ".") -> dict:
    active_root = get_workspace_dir()
    try:
        target, is_safe = _resolve_safe_path(path or ".")
        if not is_safe or not os.path.exists(target):
            return {"status": "error", "output": f"Invalid path: {path}"}

        found = []
        for full in _walk_project_files(target):
            if fnmatch.fnmatch(os.path.basename(full).lower(), pattern.lower()):
                found.append(os.path.relpath(full, active_root))

        return {
            "status": "success",
            "count": len(found),
            "output": "\n".join(found) if found else f"No files matching '{pattern}'."
        }
    except Exception as e:
        return {"status": "error", "output": f"Find files error: {e}"}


def inspect_project(path: str = ".") -> dict:
    target, safe = _resolve_safe_path(path or ".")
    if not safe or not os.path.isdir(target):
        return {"status": "error", "output": f"Invalid project directory: {path}"}
    top_names = set(os.listdir(target))
    indicators = {
        "Python": ["pyproject.toml", "requirements.txt", "setup.py", "Pipfile", "pytest.ini"],
        "Node.js / JavaScript": ["package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock"],
        "TypeScript": ["tsconfig.json"],
        "Rust": ["Cargo.toml"],
        "Go": ["go.mod"],
    }
    stacks = [label for label, markers in indicators.items() if any(m in top_names for m in markers)]
    important = ["AGENTS.md", "README.md", "pyproject.toml", "pytest.ini", "package.json", "Cargo.toml", "go.mod"]
    files = sorted(name for name in important if name in top_names)
    return {"status": "success", "root": target, "stacks": stacks, "important_files": files, "output": f"Project root: {target}\nStacks: {stacks}"}


def run_project_tests(path: str = ".", command: str = "", timeout: int = 180) -> dict:
    target, safe = _resolve_safe_path(path or ".")
    if not safe or not os.path.isdir(target):
        return {"status": "error", "output": f"Invalid test project directory: {path}"}
    try:
        python, env = _python_runtime_environment()
        if command.strip():
            argv = shlex.split(command)
            if not argv:
                return {"status": "error", "output": "Test command cannot be empty."}
            if argv[0] in {"python", "python3", "python3.12"}:
                argv[0] = python
            elif os.path.basename(argv[0]) in {"pytest", "pytest.exe"}:
                argv = [python, "-m", "pytest", *argv[1:]]
        else:
            probe = subprocess.run([python, "-c", "import importlib.util; print(int(importlib.util.find_spec('pytest') is not None))"],
                                    cwd=target, capture_output=True, text=True, timeout=10, check=False)
            if probe.returncode != 0:
                return _add_python_dependency_recovery({"status": "error", "exit_code": probe.returncode,
                    "command": shlex.join([python, "-c", "import importlib.util; print(int(importlib.util.find_spec('pytest') is not None))"]),
                    "cwd": target, "python": python, "output": probe.stderr or probe.stdout or "Could not check for pytest."})
            has_pytest = probe.stdout.strip() == "1"
            argv = [python, "-m", "pytest", "-q"] if has_pytest else [python, "-m", "unittest", "discover", "-v"]
        completed = subprocess.run(argv, cwd=target, env=env, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=max(1, min(int(timeout), 1800)), check=False)
        result = {"status": "success" if completed.returncode == 0 else "error", "exit_code": completed.returncode,
                  "command": shlex.join(argv), "cwd": target, "python": python,
                  "output": completed.stdout or "(No output)"}
        return _add_python_dependency_recovery(result)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        result = {"status": "timeout", "timed_out": True, "timeout_seconds": timeout,
                  "command": shlex.join(argv) if "argv" in locals() else command,
                  "cwd": target, "python": locals().get("python"), "output": str(output)[-12000:]}
        return _add_python_dependency_recovery(result)
    except Exception as exc:
        result = {"status": "error", "error_type": type(exc).__name__,
                  "command": command, "cwd": target, "output": f"Test runner failed: {type(exc).__name__}: {exc}"}
        return _add_python_dependency_recovery(result)


def lint_project(path: str = ".", command: str = "", timeout: int = 180) -> dict:
    target, safe = _resolve_safe_path(path or ".")
    if not safe or not os.path.isdir(target):
        return {"status": "error", "output": f"Invalid lint directory: {path}"}
    try:
        python, env = _python_runtime_environment()
        argv = shlex.split(command) if command.strip() else [python, "-m", "ruff", "check", "."]
        if argv and os.path.basename(argv[0]) in {"ruff", "ruff.exe"}:
            argv = [python, "-m", "ruff", *argv[1:]]
        completed = subprocess.run(argv, cwd=target, env=env, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, timeout=max(1, min(int(timeout), 1800)), check=False)
        result = {"status": "success" if completed.returncode == 0 else "error", "exit_code": completed.returncode,
                  "command": shlex.join(argv), "cwd": target, "python": python,
                  "output": completed.stdout or "(No output)"}
        return _add_python_dependency_recovery(result)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return {"status": "timeout", "timed_out": True, "timeout_seconds": timeout,
                "command": shlex.join(argv) if "argv" in locals() else command,
                "cwd": target, "python": locals().get("python"), "output": str(output)[-12000:]}
    except Exception as exc:
        return {"status": "error", "error_type": type(exc).__name__, "command": command,
                "cwd": target, "output": f"Lint failed: {type(exc).__name__}: {exc}"}


def git_status(path: str = ".") -> dict:
    target, safe = _resolve_safe_path(path or ".")
    if not safe or not os.path.isdir(target):
        return {"status": "error", "output": f"Invalid directory: {path}"}
    argv = ["git", "-C", target, "status", "--short", "--branch"]
    try:
        proc = subprocess.run(argv, cwd=target, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10, check=False)
        return {"status": "success" if proc.returncode == 0 else "error", "exit_code": proc.returncode,
                "command": shlex.join(argv), "cwd": target, "output": proc.stdout or "(No output)"}
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes): output = output.decode("utf-8", errors="replace")
        return {"status": "timeout", "timed_out": True, "timeout_seconds": 10,
                "command": shlex.join(argv), "cwd": target, "output": str(output)[-8000:]}
    except OSError as exc:
        return {"status": "error", "error_type": type(exc).__name__, "command": shlex.join(argv),
                "cwd": target, "output": f"Could not start git status: {exc}"}


def git_diff(path: str = ".", staged: bool = False, max_chars: int = 16000) -> dict:
    target, safe = _resolve_safe_path(path or ".")
    if not safe or not os.path.isdir(target):
        return {"status": "error", "output": f"Invalid directory: {path}"}
    argv = ["git", "-C", target, "diff", "--no-ext-diff"]
    if staged:
        argv.append("--cached")
    try:
        proc = subprocess.run(argv, cwd=target, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15, check=False)
        return {"status": "success" if proc.returncode == 0 else "error", "exit_code": proc.returncode,
                "command": shlex.join(argv), "cwd": target, "output": (proc.stdout or "(No output)")[:max_chars]}
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes): output = output.decode("utf-8", errors="replace")
        return {"status": "timeout", "timed_out": True, "timeout_seconds": 15,
                "command": shlex.join(argv), "cwd": target, "output": str(output)[-max_chars:]}
    except OSError as exc:
        return {"status": "error", "error_type": type(exc).__name__, "command": shlex.join(argv),
                "cwd": target, "output": f"Could not start git diff: {exc}"}


def git_workspace_status(action: str = "status", args: str = "") -> dict:
    act = (action or "status").strip().lower()
    if act == "status":
        return git_status(args or ".")
    elif act == "diff":
        return git_diff(args or ".")
    return execute_terminal(f"git {act} {args}".strip(), timeout=30)


def compare_python_api(reference_path: str, candidate_path: str) -> dict:
    """Compare public top-level Python function/class signatures between two files."""
    def public_api(path):
        try:
            p = Path(path)
            if not p.is_absolute():
                p = Path(get_workspace_dir()) / p
            source = p.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(p))
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise ValueError(f"Cannot read/parse {path}: {exc}") from exc
        found = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                found[f"function {node.name}"] = ast.dump(node.args, include_attributes=False)
            elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                found[f"class {node.name}"] = "class"
        return found

    try:
        reference = public_api(reference_path)
        candidate = public_api(candidate_path)
    except ValueError as exc:
        return {"status": "error", "output": str(exc)}
    differences = []
    for name in sorted(set(reference) | set(candidate)):
        if name not in reference:
            differences.append(f"Added public API: {name}")
        elif name not in candidate:
            differences.append(f"Removed public API: {name}")
        elif reference[name] != candidate[name]:
            differences.append(f"Changed signature: {name}")
    if differences:
        return {"status": "error", "output": "API parity differences found:\n" + "\n".join(differences)}
    return {"status": "success", "output": "API parity verified: public functions and classes match."}


# =====================================================================
# 1D. CODING SKILLS & DURABLE PROJECT MEMORY (.agent_memory.json)
# =====================================================================

CODING_SKILLS = {
    "debugging": "DEBUGGING: Reproduce failure, identify root cause, make minimal fix, and verify.",
    "testing": "TESTING: Derive tests from requirements and verify via exit code 0.",
    "api_parity": "API PARITY: Compare exports and signatures.",
    "code_review": "CODE REVIEW: Check correctness, security, and side effects."
}


def list_coding_skills() -> dict:
    names = set(CODING_SKILLS)
    skills_dir = Path(get_workspace_dir()) / ".agent_skills"
    if skills_dir.is_dir():
        names.update(p.name for p in skills_dir.iterdir() if p.is_dir() and (p / "SKILL.md").is_file())
    return {"status": "success", "skills": sorted(names), "output": "\n".join(sorted(names))}


def load_coding_skill(name: str) -> dict:
    key = str(name or "").strip().lower()
    if key in CODING_SKILLS:
        return {"status": "success", "output": CODING_SKILLS[key]}
    skill_file = Path(get_workspace_dir()) / ".agent_skills" / key / "SKILL.md"
    try:
        if skill_file.is_file():
            return {"status": "success", "output": skill_file.read_text(encoding="utf-8")[:12000]}
    except OSError as exc:
        return {"status": "error", "output": f"Could not read skill {key!r}: {exc}"}
    return {"status": "error", "output": f"Unknown skill {name!r}"}


def read_studio_skill(name: str) -> dict:
    """Read one allow-listed Studio Markdown skill on demand."""
    skill_map = {
        "python": "python_skills.md", "python_skills.md": "python_skills.md",
        "html": "HTML_skills.md", "html_skills.md": "HTML_skills.md",
        "javascript": "JS_skills.md", "js": "JS_skills.md", "js_skills.md": "JS_skills.md",
        "agent tools": "tools_skills.md", "tools": "tools_skills.md",
        "tools_skills.md": "tools_skills.md",
        "initialization": "colibri_initialization.txt",
        "colibri_initialization.txt": "colibri_initialization.txt",
        "colibri guide": "colibri_initialization.txt",
    }
    key = str(name or "").strip().lower()
    filename = skill_map.get(key)
    if not filename:
        return {"status": "error", "output": "Unknown Studio skill. Allowed skills: initialization, python, html, javascript, tools_skills.md."}
    skill_path = Path(__file__).resolve().parent / "skills" / filename
    try:
        content = skill_path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"status": "error", "output": f"Could not read Studio skill {filename}: {exc}"}
    return {"status": "success", "skill": filename, "output": content[:24000]}


def get_coding_skills_for_task(task: str) -> str:
    lower = str(task or "").lower()
    chosen = []
    if any(w in lower for w in ("bug", "error", "debug", "fix")):
        chosen.append("debugging")
    if any(w in lower for w in ("test", "pytest", "verify")):
        chosen.append("testing")
    return "\n".join(CODING_SKILLS[name] for name in chosen if name in CODING_SKILLS)


def read_project_memory(key: str = "") -> dict:
    root = get_workspace_dir()
    memory_path = os.path.join(root, ".agent_memory.json")
    if not os.path.isfile(memory_path):
        return {"status": "success", "output": "No project memory saved yet."}
    try:
        with open(memory_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if key and key.strip():
            return {"status": "success", "output": json.dumps({key.strip(): data.get(key.strip())}, indent=2)}
        return {"status": "success", "output": json.dumps(data, indent=2)}
    except Exception as e:
        return {"status": "error", "output": f"Read memory failed: {e}"}


def write_project_memory(key: str, value: str) -> dict:
    key, value = str(key or "").strip(), str(value or "").strip()
    if not key or not value:
        return {"status": "error", "output": "Key and value required."}
    root = get_workspace_dir()
    memory_path = os.path.join(root, ".agent_memory.json")
    try:
        data = {}
        if os.path.isfile(memory_path):
            with open(memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        data[key] = {"value": value, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return {"status": "success", "output": f"Saved memory key {key!r}"}
    except Exception as e:
        return {"status": "error", "output": f"Write memory failed: {e}"}


# =====================================================================
# 2. WEB, EXTENSION & DESKTOP TOOLS
# =====================================================================

def web_search(query: str, max_results: int = 5) -> dict:
    if DDGS is None:
        return {"status": "error", "output": "duckduckgo-search dependency missing."}
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query.strip(), max_results=int(max_results)))
        formatted = [f"{i}. [{r.get('title')}]({r.get('href')})\n   {r.get('body')}" for i, r in enumerate(results, 1)]
        return {"status": "success", "output": "\n\n".join(formatted)}
    except Exception as e:
        return {"status": "error", "output": f"Search error: {e}"}


def fetch_web_page(url: str, max_chars: int = 5000) -> dict:
    if sync_playwright is None:
        return {"status": "error", "output": "playwright dependency missing."}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url.strip(), timeout=20000)
            text = page.inner_text("body")
            browser.close()
        return {"status": "success", "output": text[:max_chars]}
    except Exception as e:
        return {"status": "error", "output": f"Fetch error: {e}"}


def test_chrome_extension(extension_dir: str, target_url: str = "https://example.com") -> dict:
    """Report honestly when browser-extension automation is not implemented."""
    target, safe = _resolve_safe_path(extension_dir)
    if not safe:
        return {"status": "error", "output": "Access denied: extension directory is outside the active workspace."}
    if not os.path.isdir(target):
        return {"status": "error", "output": f"Extension directory not found: {extension_dir}"}
    manifest = os.path.join(target, "manifest.json")
    if not os.path.isfile(manifest):
        return {"status": "error", "output": f"No manifest.json found in extension directory: {extension_dir}"}
    try:
        with open(manifest, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "manifest_version" not in data:
            return {"status": "error", "output": "manifest.json is invalid or missing manifest_version."}
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "error", "output": f"Could not validate extension manifest: {exc}"}
    return {
        "status": "success",
        "output": "Extension manifest validated. Browser launch/install and target-page behavior were NOT tested because automated extension loading is not implemented in this tool.",
        "manifest_version": data.get("manifest_version"),
        "tested_browser_behavior": False,
        "target_url": target_url,
    }


def capture_desktop(filename: str = None) -> dict:
    if mss is None or Image is None:
        return {"status": "error", "output": "mss or Pillow missing."}
    if not filename:
        filename = f"desktop_{int(time.time())}.png"
    filename = os.path.basename(str(filename).strip())
    if not filename or filename in {".", ".."}:
        return {"status": "error", "output": "Invalid screenshot filename."}
    if not filename.lower().endswith(".png"):
        filename += ".png"
    target_path = os.path.realpath(os.path.join(SCREENSHOT_DIR, filename))
    if os.path.commonpath([os.path.realpath(SCREENSHOT_DIR), target_path]) != os.path.realpath(SCREENSHOT_DIR):
        return {"status": "error", "output": "Screenshot path must stay inside the screenshots directory."}
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img.save(target_path)
    return {"status": "success", "output": f"Captured desktop -> /screenshots/{filename}", "screenshot_url": f"/screenshots/{filename}"}


def _ensure_x11_env():
    """Ensure DISPLAY and valid XAUTHORITY are exported in os.environ."""
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"
    current_auth = os.environ.get("XAUTHORITY", "")
    if not current_auth or not os.path.isfile(current_auth):
        for candidate in ["/run/user/1000/gdm/Xauthority", "/home/local-ai/.Xauthority"]:
            if os.path.isfile(candidate):
                os.environ["XAUTHORITY"] = candidate
                break

def execute_pc_action(action_type: str = "click", x: int = None, y: int = None, text: str = None, **kwargs) -> dict:
    """Executes GUI actions (move, click, double_click, right_click, type, press, hotkey, scroll, drag) via PyAutoGUI."""
    _ensure_x11_env()
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
    except ImportError:
        return {"status": "error", "output": "pyautogui is not installed. Run: pip install pyautogui python-xlib"}

    act = str(action_type or "click").strip().lower()

    try:
        if act in ("move", "mousemove", "mouse_move"):
            if x is None or y is None:
                return {"status": "error", "output": "Move action requires both x and y coordinates."}
            pyautogui.moveTo(int(x), int(y))
            return {"status": "success", "output": f"Moved mouse cursor to ({int(x)}, {int(y)})."}

        elif act in ("click", "left_click"):
            if x is not None and y is not None:
                pyautogui.click(int(x), int(y))
                return {"status": "success", "output": f"Clicked at ({int(x)}, {int(y)})."}
            pyautogui.click()
            return {"status": "success", "output": "Clicked at current cursor position."}

        elif act == "double_click":
            if x is not None and y is not None:
                pyautogui.doubleClick(int(x), int(y))
                return {"status": "success", "output": f"Double-clicked at ({int(x)}, {int(y)})."}
            pyautogui.doubleClick()
            return {"status": "success", "output": "Double-clicked at current cursor position."}

        elif act in ("right_click", "context_menu"):
            if x is not None and y is not None:
                pyautogui.rightClick(int(x), int(y))
                return {"status": "success", "output": f"Right-clicked at ({int(x)}, {int(y)})."}
            pyautogui.rightClick()
            return {"status": "success", "output": "Right-clicked at current cursor position."}

        elif act in ("type", "write"):
            if not text:
                return {"status": "error", "output": "Type action requires text."}
            if x is not None and y is not None:
                pyautogui.click(int(x), int(y))
            pyautogui.write(str(text), interval=0.02)
            return {"status": "success", "output": f"Typed text: {repr(str(text)[:50])}"}

        elif act in ("press", "key"):
            key = str(text or kwargs.get("key") or "").strip()
            if not key:
                return {"status": "error", "output": "Press action requires a key name (e.g., 'enter', 'esc', 'tab')."}
            pyautogui.press(key)
            return {"status": "success", "output": f"Pressed key: {key}"}

        elif act in ("hotkey", "shortcut"):
            keys = [k.strip() for k in str(text).split("+")] if text else kwargs.get("keys", [])
            if not keys:
                return {"status": "error", "output": "Hotkey action requires keys (e.g., text='ctrl+c')."}
            pyautogui.hotkey(*keys)
            return {"status": "success", "output": f"Triggered shortcut: {'+'.join(keys)}"}

        elif act in ("scroll",):
            clicks = int(kwargs.get("clicks", kwargs.get("amount", 5)))
            pyautogui.scroll(clicks)
            return {"status": "success", "output": f"Scrolled {clicks} units."}

        elif act in ("drag", "drag_to"):
            if x is None or y is None:
                return {"status": "error", "output": "Drag action requires target x and y coordinates."}
            pyautogui.dragTo(int(x), int(y))
            return {"status": "success", "output": f"Dragged cursor to ({int(x)}, {int(y)})."}

        else:
            return {"status": "error", "output": f"Unknown PC action: '{action_type}'. Supported: move, click, double_click, right_click, type, press, hotkey, scroll, drag."}

    except Exception as e:
        return {"status": "error", "output": f"GUI action execution error: {e}"}


def _ensure_x11_env():
    """Ensure DISPLAY and valid XAUTHORITY are exported in os.environ."""
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"
    current_auth = os.environ.get("XAUTHORITY", "")
    if not current_auth or not os.path.isfile(current_auth):
        for candidate in ["/run/user/1000/gdm/Xauthority", "/home/local-ai/.Xauthority"]:
            if os.path.isfile(candidate):
                os.environ["XAUTHORITY"] = candidate
                break

def execute_pc_action_internal(action_type: str, x=None, y=None, text=None, to_x=None, to_y=None) -> dict:
    action = action_type.strip().lower()
    try:
        if action == "click" and x is not None and y is not None:
            subprocess.run(["xdotool", "mousemove", str(int(x)), str(int(y)), "click", "1"], check=True, timeout=10)
            return {"status": "success", "output": f"Left-clicked at ({x}, {y})"}
        elif action == "type" and text:
            subprocess.run(["xdotool", "type", "--delay", "30", str(text)], check=True, timeout=30)
            return {"status": "success", "output": f"Typed: '{text}'"}
        return {"status": "error", "output": f"Unknown PC action: {action}"}
    except Exception as e:
        return {"status": "error", "output": f"xdotool failure: {e}"}


# =====================================================================
# 4B. DEPENDENCY INSPECTION & SAFE PYTHON PACKAGE INSTALLATION
# =====================================================================

def _project_python() -> str:
    """Use the Studio-selected interpreter, then a project venv, then Studio Python."""
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".python_environment.json")
    try:
        with open(config_file, "r", encoding="utf-8") as handle:
            selected = os.path.abspath(os.path.expanduser(json.load(handle).get("python", "")))
        if selected and os.path.isfile(selected) and os.access(selected, os.X_OK):
            return selected
    except Exception:
        pass
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
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return os.path.realpath(os.sys.executable)


def check_python_dependencies(path: str = ".") -> dict:
    """Check package declarations in requirements files without installing anything."""
    target, safe = _resolve_safe_path(path or ".")
    if not safe or not os.path.isdir(target):
        return {"status": "error", "output": f"Invalid or inaccessible project directory: {path}"}
    req_files = [os.path.join(target, "requirements.txt"), os.path.join(target, "requirements-dev.txt")]
    specs = []
    for req in req_files:
        if not os.path.isfile(req):
            continue
        try:
            for line in Path(req).read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith(("#", "-", "http://", "https://")):
                    continue
                match = re.match(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)", line)
                if match:
                    specs.append((match.group(1), line, os.path.basename(req)))
        except OSError:
            continue

    seen, results = set(), []
    for name, spec, source in specs:
        key = name.casefold().replace("_", "-").replace(".", "-")
        if key in seen:
            continue
        seen.add(key)
        results.append({"package": name, "requirement": spec, "source": source})

    python = _project_python()
    probe_code = (
        "import importlib.metadata as metadata, json, sys\n"
        "names = json.loads(sys.stdin.read())\n"
        "versions = {}\n"
        "for name in names:\n"
        "    try:\n"
        "        versions[name] = metadata.version(name)\n"
        "    except metadata.PackageNotFoundError:\n"
        "        versions[name] = None\n"
        "print(json.dumps(versions))\n"
    )
    try:
        probe = subprocess.run([python, "-c", probe_code], input=json.dumps([item["package"] for item in results]),
                               cwd=target, capture_output=True, text=True, timeout=30, check=False)
        if probe.returncode != 0:
            return {"status": "error", "workspace": target, "python": python,
                    "command": shlex.join([python, "-c", probe_code]), "exit_code": probe.returncode,
                    "output": probe.stderr or probe.stdout or "Could not inspect installed package metadata."}
        versions = json.loads(probe.stdout or "{}")
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return {"status": "timeout", "timed_out": True, "timeout_seconds": 30,
                "workspace": target, "python": python, "output": str(output)[-8000:]}
    except Exception as exc:
        return {"status": "error", "workspace": target, "python": python,
                "error_type": type(exc).__name__, "output": f"Dependency inspection failed: {type(exc).__name__}: {exc}"}

    results = [{**item, "installed": versions.get(item["package"]) is not None,
                "installed_version": versions.get(item["package"])} for item in results]
    missing = [x for x in results if not x["installed"]]
    return {
        "status": "error" if missing else "success",
        "workspace": target,
        "python": python,
        "checked_count": len(results),
        "missing_count": len(missing),
        "dependencies": results,
        "output": f"Checked {len(results)} declared Python dependencies; {len(missing)} missing. No packages were installed."
    }


def install_python_package(package: str, upgrade: bool = False, timeout: int = 300) -> dict:
    """Install one named PyPI package using pip; never invoke a shell or accept pip flags."""
    package = str(package or "").strip()
    if not package or package.startswith("-") or package.startswith((".", "/", "\\")) or re.search(r"[;&|`$<>\n\r]", package):
        return {"status": "error", "output": "Rejected package spec. Supply a PyPI name such as 'requests' or 'Pillow>=10'; pip options, URLs and local paths are not accepted."}
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:\[[A-Za-z0-9_,.-]+\])?(?:\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]+)?", package):
        return {"status": "error", "output": "Invalid package spec. Use a package name with an optional version constraint, e.g. 'requests>=2.31'."}
    python = _project_python()
    cmd = [python, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.append(package)
    try:
        proc = subprocess.run(
            cmd,
            cwd=get_workspace_dir(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=max(10, min(int(timeout), 900)),
            check=False,
            shell=False
        )
        return {
            "status": "success" if proc.returncode == 0 else "error",
            "exit_code": proc.returncode,
            "package": package,
            "python": python,
            "command": shlex.join(cmd),
            "output": (proc.stdout or "(pip produced no output)")[-12000:]
        }
    except subprocess.TimeoutExpired as exc:
        tail = exc.stdout or ""
        if isinstance(tail, bytes):
            tail = tail.decode("utf-8", errors="replace")
        return {"status": "timeout", "timed_out": True, "timeout_seconds": timeout,
                "exit_code": -1, "package": package, "python": python,
                "command": shlex.join(cmd), "output": f"pip timed out after {timeout}s. {tail[-4000:]}"}
    except Exception as exc:
        return {"status": "error", "exit_code": -1, "output": f"pip invocation failed: {type(exc).__name__}: {exc}"}


def install_missing_python_module(module: str, package: str = "", timeout: int = 300) -> dict:
    """Install a missing import into the selected interpreter and verify that it imports."""
    module = str(module or "").strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", module):
        return {"status": "error", "output": "Invalid module name. Supply an import name such as 'yaml' or 'PIL.Image'."}
    distributions = {
        "PIL": "Pillow", "cv2": "opencv-python", "yaml": "PyYAML",
        "sklearn": "scikit-learn", "bs4": "beautifulsoup4", "dotenv": "python-dotenv",
        "serial": "pyserial", "Crypto": "pycryptodome", "gi": "PyGObject"
    }
    package = str(package or distributions.get(module.split(".", 1)[0], module.split(".", 1)[0])).strip()
    installed = install_python_package(package, timeout=timeout)
    if installed.get("status") != "success":
        return {**installed, "module": module, "package": package}
    python = installed.get("python") or _project_python()
    try:
        probe = subprocess.run(
            [python, "-c", "import importlib, sys; importlib.import_module(sys.argv[1]); print('import ok')", module],
            cwd=get_workspace_dir(), capture_output=True, text=True,
            timeout=max(5, min(int(timeout), 60)), check=False, shell=False
        )
        return {
            **installed,
            "status": "success" if probe.returncode == 0 else "error",
            "module": module,
            "package": package,
            "verification_exit_code": probe.returncode,
            "verification_output": (probe.stdout or probe.stderr or "").strip(),
            "output": (installed.get("output", "") + "\n" +
                       (f"Verified import '{module}' with {python}." if probe.returncode == 0 else
                        f"Package '{package}' installed, but import '{module}' still fails with {python}: {probe.stderr.strip()}"))
        }
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return {**installed, "status": "timeout", "timed_out": True,
                "timeout_seconds": timeout, "module": module, "package": package,
                "output": installed.get("output", "") + f"\nImport verification timed out: {output[-4000:]}"}
    except Exception as exc:
        return {**installed, "status": "error", "module": module, "package": package,
                "error_type": type(exc).__name__,
                "output": installed.get("output", "") + f"\nImport verification failed: {type(exc).__name__}: {exc}"}


# =====================================================================
# 5. UNIFIED TOOL DISPATCHER & PARSER
# =====================================================================

TOOL_ALIASES = {
    "inspect_workspace_permissions": "inspect_workspace_permissions", "file_permissions": "inspect_workspace_permissions",
    "diagnose_tool_error": "diagnose_tool_error",
    "repair_workspace_metadata": "repair_workspace_metadata", "change_file_permissions": "repair_workspace_metadata",
    "read_file": "read_workspace_file", "read_workspace_file": "read_workspace_file",
    "write_file": "write_workspace_file", "write_workspace_file": "write_workspace_file", "create_file": "write_workspace_file",
    "patch_file": "patch_workspace_file", "patch_workspace_file": "patch_workspace_file",
    "edit_lines": "edit_workspace_lines", "edit_workspace_lines": "edit_workspace_lines",
    "list_files": "list_workspace_files", "list_workspace_files": "list_workspace_files",
    "grep": "grep_workspace", "ggrep": "grep_workspace", "grep_workspace": "grep_workspace",
    "search_code": "search_workspace_code", "search_workspace_code": "search_workspace_code",
    "find_files": "find_workspace_files", "find_workspace_files": "find_workspace_files",
    "check_syntax": "check_python_file", "check_python": "check_python_file", "check_python_file": "check_python_file",
    "check_project": "run_project_checks", "run_project_checks": "run_project_checks",
    "check_dependencies": "check_python_dependencies", "check_python_dependencies": "check_python_dependencies",
    "install_package": "install_python_package", "pip_install": "install_python_package",
    "install_python_package": "install_python_package",
    "install_missing_python_module": "install_missing_python_module",
    "project_info": "inspect_project", "inspect_project": "inspect_project",
    "run_tests": "run_project_tests", "run_project_tests": "run_project_tests",
    "lint_project": "lint_project", "lint": "lint_project",
    "git_status": "git_status", "git_diff": "git_diff", "git": "git_workspace_status", "git_workspace_status": "git_workspace_status",
    "api_parity": "compare_python_api", "compare_python_api": "compare_python_api",
    "skills": "list_coding_skills", "skill": "load_coding_skill",
    "read_skill": "read_studio_skill", "read_studio_skill": "read_studio_skill",
    "memory_read": "read_project_memory", "memory_write": "write_project_memory",
    "terminal": "execute_terminal", "bash": "execute_terminal", "sh": "execute_terminal", "execute_terminal": "execute_terminal",
    "web_search": "web_search", "search": "web_search",
    "fetch_url": "fetch_web_page", "fetch_web_page": "fetch_web_page",
    "test_extension": "test_chrome_extension",
    "screenshot": "capture_desktop", "capture_desktop": "capture_desktop",
    "gui_action": "execute_pc_action", "execute_pc_action": "execute_pc_action",
}


def dispatch_tool_call(raw_name: str, args: dict) -> tuple[str, str, dict]:
    canonical = TOOL_ALIASES.get(raw_name.strip().lower(), raw_name.strip().lower())
    args = args or {}

    # Tool toggle check: disabled tools are rejected before execution
    try:
        import json, os
        toggles_path = os.path.join(get_workspace_dir(), "tool_toggles.json")
        if os.path.isfile(toggles_path):
            with open(toggles_path, "r", encoding="utf-8") as _tf:
                _toggles = json.load(_tf)
            if canonical in _toggles and _toggles[canonical] is False:
                return raw_name, str(args), {"status": "error", "output": f"Tool '{canonical}' is disabled by tool toggle."}
    except Exception:
        pass

    try:
        if canonical == "diagnose_tool_error":
            name = str(args.get("tool_name") or args.get("name") or "unknown")
            target = str(args.get("target") or "")
            result = args.get("result") if isinstance(args.get("result"), dict) else {"status": str(args.get("status") or "error"), "output": str(args.get("error") or args.get("output") or "unknown tool failure")}
            return "diagnose_error", target or name, diagnose_tool_error(name, target, result)
        elif canonical == "inspect_workspace_permissions":
            path = str(args.get("path") or ".")
            return "inspect_permissions", path, inspect_workspace_permissions(path)
        elif canonical == "repair_workspace_metadata":
            path = str(args.get("path") or "")
            return "repair_metadata", path, repair_workspace_metadata(
                path, str(args.get("action") or ""), approved=args.get("approved") is True,
                mode=str(args.get("mode") or ""), owner=str(args.get("owner") or ""),
                group=str(args.get("group") or ""))
        elif canonical == "read_workspace_file":
            path = str(args.get("path") or args.get("file") or "")
            start = int(args.get("line_start", args.get("start_line", 1)))
            count = int(args.get("line_count", args.get("limit", 400)))
            return "read_file", path, read_workspace_file(path, start, count)
        elif canonical == "write_workspace_file":
            path = str(args.get("path") or args.get("file") or "")
            content = str(args.get("content", ""))
            return "write_file", path, write_workspace_file(path, content, expected_sha256=args.get("expected_sha256", ""), create_only=bool(args.get("create_only", False)), backup=bool(args.get("backup", True)))
        elif canonical == "patch_workspace_file":
            path = str(args.get("path") or args.get("file") or "")
            search_pat = str(args.get("search_pattern") or args.get("search") or "")
            replace_str = str(args.get("replacement") or args.get("replace") or "")
            return "patch_file", path, patch_workspace_file(path, search_pat, replace_str, expected_sha256=args.get("expected_sha256", ""), replace_all=bool(args.get("replace_all", False)), backup=bool(args.get("backup", True)))
        elif canonical == "edit_workspace_lines":
            path = str(args.get("path") or "")
            s_line = int(args.get("start_line", 1))
            e_line = int(args.get("end_line", s_line))
            new_c = str(args.get("new_content") or args.get("content") or "")
            return "edit_lines", f"{path}:{s_line}-{e_line}", edit_workspace_lines(path, s_line, e_line, new_c, expected_sha256=args.get("expected_sha256", ""), backup=bool(args.get("backup", True)))
        elif canonical == "list_workspace_files":
            rel = str(args.get("relative_path") or args.get("path") or ".")
            return "list_files", rel, list_workspace_files(rel, int(args.get("max_depth", 2)))
        elif canonical == "grep_workspace":
            pat = str(args.get("pattern") or args.get("query") or "")
            path = str(args.get("path") or ".")
            return "grep_search", f"'{pat}' in {path}", grep_workspace(pat, path)
        elif canonical == "search_workspace_code":
            q = str(args.get("query") or args.get("pattern") or "")
            path = str(args.get("path") or ".")
            return "search_code", f"'{q}' in {path}", search_workspace_code(q, path)
        elif canonical == "find_workspace_files":
            pat = str(args.get("pattern") or "*")
            path = str(args.get("path") or ".")
            return "find_files", f"{pat} in {path}", find_workspace_files(pat, path)
        elif canonical == "check_python_file":
            path = str(args.get("path") or "")
            return "check_syntax", path, check_python_file(path)
        elif canonical == "run_project_checks":
            path = str(args.get("path") or ".")
            return "check_project", path, run_project_checks(path)
        elif canonical == "check_python_dependencies":
            path = str(args.get("path") or ".")
            return "check_dependencies", path, check_python_dependencies(path)
        elif canonical == "install_python_package":
            package = str(args.get("package") or args.get("name") or "")
            return "install_package", package, install_python_package(
                package, bool(args.get("upgrade", False)), int(args.get("timeout", 300)))
        elif canonical == "install_missing_python_module":
            module = str(args.get("module") or "")
            package = str(args.get("package") or "")
            return "install_module", module, install_missing_python_module(
                module, package, int(args.get("timeout", 300)))
        elif canonical == "inspect_project":
            path = str(args.get("path") or ".")
            return "project_info", path, inspect_project(path)
        elif canonical == "run_project_tests":
            path = str(args.get("path") or ".")
            cmd = str(args.get("command") or "")
            return "run_tests", path, run_project_tests(path, command=cmd)
        elif canonical == "lint_project":
            path = str(args.get("path") or ".")
            cmd = str(args.get("command") or "")
            return "lint_project", path, lint_project(path, command=cmd)
        elif canonical == "git_status":
            path = str(args.get("path") or ".")
            return "git_status", path, git_status(path)
        elif canonical == "git_diff":
            path = str(args.get("path") or ".")
            return "git_diff", path, git_diff(path)
        elif canonical == "git_workspace_status":
            act = str(args.get("action") or "status")
            gargs = str(args.get("args") or "")
            return "git", f"git {act}", git_workspace_status(act, gargs)
        elif canonical == "compare_python_api":
            ref = str(args.get("reference_path") or "")
            cand = str(args.get("candidate_path") or "")
            return "api_parity", f"{ref} -> {cand}", compare_python_api(ref, cand)
        elif canonical == "list_coding_skills":
            return "skills", "all", list_coding_skills()
        elif canonical == "load_coding_skill":
            name = str(args.get("name") or "")
            return "skill", name, load_coding_skill(name)
        elif canonical == "read_studio_skill":
            name = str(args.get("name") or args.get("skill") or args.get("filename") or "")
            return "read_skill", name, read_studio_skill(name)
        elif canonical == "read_project_memory":
            key = str(args.get("key") or "")
            return "memory_read", key or "all", read_project_memory(key)
        elif canonical == "write_project_memory":
            key = str(args.get("key") or "")
            val = str(args.get("value") or "")
            return "memory_write", key, write_project_memory(key, val)
        elif canonical == "execute_terminal":
            cmd = str(args.get("command") or args.get("cmd") or "")
            result = execute_terminal(cmd, int(args.get("timeout", 120)), cwd=str(args.get("cwd") or ""), confirm=bool(args.get("confirm", False)), max_output_chars=int(args.get("max_output_chars", 24000)))
            return "terminal", cmd[:80], _add_python_dependency_recovery(result)
        elif canonical == "web_search":
            q = str(args.get("query") or "")
            return "web_search", q, web_search(q, int(args.get("max_results", 5)))
        elif canonical == "fetch_web_page":
            u = str(args.get("url") or "")
            return "fetch_url", u, fetch_web_page(u)
        elif canonical == "test_chrome_extension":
            ext = str(args.get("extension_dir") or "")
            return "test_extension", ext, test_chrome_extension(ext)
        elif canonical == "capture_desktop":
            fn = args.get("filename")
            return "capture_desktop", fn or "Primary Display", capture_desktop(fn)
        elif canonical == "execute_pc_action":
            act = str(args.get("action_type") or args.get("action") or "click")
            return f"gui_{act}", act, execute_pc_action(act, args.get("x"), args.get("y"), args.get("text"))

        return raw_name, str(args), {"status": "error", "output": f"Unknown tool skill: '{raw_name}'"}
    except Exception as e:
        return raw_name, str(args), {"status": "error", "error_type": type(e).__name__,
                                     "output": f"Tool dispatch error ({type(e).__name__}): {e}"}


def _coerce_tool_value(value: str):
    """Coerce scalar XML parameter values without damaging ordinary strings."""
    value = str(value).strip()
    if not value:
        return ""
    if value[0:1] in ('{', '[') or value in ("true", "false", "null") or re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            pass
    return value


def extract_all_tool_calls(reply_text: str, streamed_tool_calls: list = None) -> list[tuple[str, dict]]:
    """Parse tool calls from OpenAI tool_calls, Qwen XML, JSON wrappers, and legacy blocks."""
    calls = []
    seen = set()

    def add_call(name, args):
        name = str(name or "").strip()
        if not name:
            return
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                args = {}
        if not isinstance(args, dict):
            args = {}
        signature = (name, json.dumps(args, sort_keys=True, ensure_ascii=False, default=str))
        if signature not in seen:
            seen.add(signature)
            calls.append((name, args))

    for tc in (streamed_tool_calls or []):
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments", {})
        add_call(fn.get("name"), raw_args)

    clean_text = re.sub(r"<think\b[^>]*>[\s\S]*?</think\s*>", "", reply_text or "", flags=re.I)

    function_pattern = re.compile(
        r"<function\s*=\s*([a-zA-Z0-9_:-]+)\s*>([\s\S]*?)(?:</function\s*>|(?=</tool_call\s*>|$))",
        re.I,
    )
    for fn_match in function_pattern.finditer(clean_text):
        fn_name, fn_body = fn_match.group(1).strip(), fn_match.group(2)
        params = {}
        parameter_pattern = re.compile(
            r"<parameter\s*=\s*([a-zA-Z0-9_:-]+)\s*>([\s\S]*?)(?:</parameter\s*>|(?=<parameter\s*=|</function\s*>|</tool_call\s*>|$))",
            re.I,
        )
        for p_match in parameter_pattern.finditer(fn_body):
            params[p_match.group(1).strip()] = _coerce_tool_value(p_match.group(2))
        add_call(fn_name, params)

    for fn_match in re.finditer(r"<function\s+name=[\"']([a-zA-Z0-9_:-]+)[\"'][^>]*>([\s\S]*?)</function\s*>", clean_text, re.I):
        params = {}
        for p_match in re.finditer(r"<parameter\s+name=[\"']([a-zA-Z0-9_:-]+)[\"'][^>]*>([\s\S]*?)</parameter\s*>", fn_match.group(2), re.I):
            params[p_match.group(1).strip()] = _coerce_tool_value(p_match.group(2))
        add_call(fn_match.group(1), params)

    for match in re.finditer(r"<tool_call\b[^>]*>\s*(\{[\s\S]*?\})\s*</tool_call\s*>", clean_text, re.I):
        raw = match.group(1)
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                add_call(
                    payload.get("name") or payload.get("function") or payload.get("tool"),
                    payload.get("arguments", payload.get("parameters", payload.get("args", {}))),
                )
        except (ValueError, TypeError):
            pass

    if not calls:
        for m in re.finditer(r"```\s*read_file:([^`\n\r]+?)\s*```", clean_text, re.I):
            raw_spec = m.group(1).strip()
            parts = raw_spec.rsplit(":", 2)
            path, line_start, line_count = raw_spec, 1, 300
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                path, line_start, line_count = parts[0].strip(), int(parts[1]), int(parts[2])
            elif len(parts) >= 2 and parts[-1].isdigit():
                path, line_start = ":".join(parts[:-1]).strip(), int(parts[-1])
            add_call("read_workspace_file", {"path": path.strip("'\""), "line_start": line_start, "line_count": line_count})
        for m in re.finditer(r"```patch:([^\n\r]+)[\r\n]<<<<<<< SEARCH[\r\n]([\s\S]*?)=======[\r\n]([\s\S]*?)>>>>>>>\s*```", clean_text):
            add_call("patch_workspace_file", {"path": m.group(1).strip(), "search_pattern": m.group(2), "replacement": m.group(3)})
        for m in re.finditer(r"```file:([^\n\r]+)[\r\n]([\s\S]*?)```", clean_text):
            add_call("write_workspace_file", {"path": m.group(1).strip(), "content": m.group(2)})
        for m in re.finditer(r"```list_files:?([^\n\r`]*)```", clean_text, re.I):
            add_call("list_workspace_files", {"relative_path": m.group(1).strip() or "."})
        for m in re.finditer(r"```(?:bash|sh)\s+([\s\S]*?)\s*```", clean_text, re.I):
            cmd = m.group(1).strip()
            if cmd:
                add_call("execute_terminal", {"command": cmd})

    return calls

# =====================================================================
# DYNAMIC EXTENSION HOOK FOR ADVANCED TOOLS
# =====================================================================
try:
    import advanced_tools as _adv
    _NEW_ALIASES = {
        'find_text_on_screen': 'find_text_on_screen', 'find_text': 'find_text_on_screen', 'locate_text': 'find_text_on_screen',
        'inspect_screen_errors': 'inspect_screen_errors', 'screen_errors': 'inspect_screen_errors', 'debug_screen': 'inspect_screen_errors',
        'ocr_screen_region': 'ocr_screen_region', 'crop_ocr': 'ocr_screen_region',
        "start_service": "start_background_service", "start_background_service": "start_background_service",
        "service_status": "get_service_status", "get_service_status": "get_service_status", "services": "get_service_status",
        "stop_service": "stop_background_service", "stop_background_service": "stop_background_service",
        "telemetry": "get_system_telemetry", "system_telemetry": "get_system_telemetry", "vram": "get_system_telemetry",
        "get_system_telemetry": "get_system_telemetry",
        "probe_http": "probe_http_endpoint", "http_request": "probe_http_endpoint", "probe_http_endpoint": "probe_http_endpoint",
        "check_port": "check_port_open", "check_port_open": "check_port_open",
        "query_sqlite": "query_sqlite", "sqlite": "query_sqlite",
        "inspect_dataset": "inspect_tabular_file", "inspect_tabular_file": "inspect_tabular_file",
        "find_symbol": "find_symbol_definitions", "find_symbol_definitions": "find_symbol_definitions",
        "outline_file": "outline_python_file", "outline_python_file": "outline_python_file",
        "ocr_screen": "read_screen_region_text", "read_screen_region_text": "read_screen_region_text",
    }
    TOOL_ALIASES.update(_NEW_ALIASES)
    _base_dispatcher = dispatch_tool_call

    def dispatch_tool_call(raw_name: str, args: dict) -> tuple:
        canonical = TOOL_ALIASES.get(str(raw_name).strip().lower(), str(raw_name).strip().lower())
        if hasattr(_adv, canonical):
            fn = getattr(_adv, canonical)
            return canonical, str(args), fn(**(args or {}))
        return _base_dispatcher(raw_name, args)
except Exception:
    pass


# =====================================================================
# CODA REFINEMENT HELPER
# =====================================================================
def coda_refine(stream_completion_fn, messages: list, model: str, options: dict, max_loops: int = 3) -> str:
    """Iteratively refine the assistant's output up to max_loops times.

    Each loop feeds the previous response back with a refinement instruction
    and collects a new, improved version. Returns the final refined text.
    """
    refinement_prompt = (
        "\n\n[CODA REFINEMENT PASS] Review the previous response above. "
        "Improve clarity, correctness, and completeness. Fix any errors, "
        "clarify ambiguous statements, and ensure all task requirements are met. "
        "Output only the refined response."
    )

    current_text = ""
    for loop in range(max_loops):
        if loop == 0:
            # First pass: just collect the normal response
            collected = []
            for token in stream_completion_fn(messages, model, options):
                if token:
                    collected.append(token)
            current_text = "".join(collected)
            messages.append({"role": "assistant", "content": current_text})
        else:
            # Refinement pass: append refinement instruction and re-run
            refine_messages = list(messages) + [
                {"role": "user", "content": refinement_prompt}
            ]
            collected = []
            for token in stream_completion_fn(refine_messages, model, options):
                if token:
                    collected.append(token)
            refined = "".join(collected)
            if refined.strip():
                current_text = refined
                messages.append({"role": "assistant", "content": current_text})

    return current_text
