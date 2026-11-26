import os
import sys
import time
import json
import re
import shutil
import subprocess
import ast
import socket
import sqlite3
import csv
import urllib.request
import urllib.error
from pathlib import Path

def _get_services_dir() -> str:
    from agent_tools import get_workspace_dir, _normalize_permissions
    s_dir = os.path.join(get_workspace_dir(), ".agent_services")
    os.makedirs(s_dir, exist_ok=True)
    _normalize_permissions(s_dir)
    return s_dir

def _get_registry() -> dict:
    reg_file = os.path.join(_get_services_dir(), "registry.json")
    if os.path.isfile(reg_file):
        try:
            with open(reg_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_registry(data: dict):
    from agent_tools import _normalize_permissions
    reg_file = os.path.join(_get_services_dir(), "registry.json")
    with open(reg_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    _normalize_permissions(reg_file)

def start_background_service(command: str = "", cmd: str = "", service_id: str = "", id: str = "", log_file: str = "", **kwargs) -> dict:
    from agent_tools import get_workspace_dir, _normalize_permissions
    command = str(command or cmd or "").strip()
    if not command:
        return {"status": "error", "output": "Command cannot be empty."}
    sid = re.sub(r"[^a-zA-Z0-9_-]", "_", str(service_id or id or f"svc_{int(time.time())}"))
    reg = _get_registry()
    if sid in reg and reg[sid].get("pid"):
        try:
            os.kill(reg[sid]["pid"], 0)
            return {"status": "error", "output": f"Service '{sid}' is already running (PID {reg[sid]['pid']})."}
        except (OSError, ProcessLookupError):
            pass
    sdir = _get_services_dir()
    log_p = os.path.abspath(log_file) if log_file else os.path.join(sdir, f"{sid}.log")
    try:
        log_fd = open(log_p, "a", encoding="utf-8")
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=get_workspace_dir(),
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            executable="/bin/bash"
        )
        _normalize_permissions(log_p)
        reg[sid] = {"pid": proc.pid, "command": command, "log_file": log_p, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        _save_registry(reg)
        return {"status": "success", "service_id": sid, "pid": proc.pid, "log_file": log_p, "output": f"Started '{sid}' (PID {proc.pid}). Logging to {os.path.basename(log_p)}."}
    except Exception as e:
        return {"status": "error", "output": f"Failed to start service: {e}"}

def get_service_status(service_id: str = "", id: str = "", tail_lines: int = 30, **kwargs) -> dict:
    reg = _get_registry()
    sid = str(service_id or id or "").strip()
    if sid:
        if sid not in reg:
            return {"status": "error", "output": f"Service '{sid}' not found."}
        info = reg[sid]
        pid = info.get("pid", 0)
        alive = False
        try:
            os.kill(pid, 0)
            alive = True
        except (OSError, ProcessLookupError):
            pass
        log_p = info.get("log_file", "")
        tail = "(No log file)"
        if os.path.isfile(log_p):
            try:
                with open(log_p, "r", encoding="utf-8", errors="replace") as f:
                    tail = "".join(f.readlines()[-int(tail_lines):])
            except Exception as e:
                tail = f"(Could not read log: {e})"
        status_str = "RUNNING" if alive else "STOPPED"
        return {"status": "success", "service_id": sid, "running": alive, "pid": pid, "output": f"Service: {sid} [{status_str}]\nPID: {pid}\nCommand: {info.get('command')}\nLog tail:\n{tail}"}
    
    summary = []
    for s, info in reg.items():
        alive = False
        try:
            os.kill(info.get("pid", 0), 0)
            alive = True
        except (OSError, ProcessLookupError):
            pass
        summary.append(f"- {s} (PID {info.get('pid')}) [{'RUNNING' if alive else 'STOPPED'}]: {info.get('command', '')[:40]}")
    return {"status": "success", "output": "Background Services:\n" + ("\n".join(summary) if summary else "None registered.")}

def stop_background_service(service_id: str = "", id: str = "", **kwargs) -> dict:
    reg = _get_registry()
    sid = str(service_id or id or "").strip()
    if sid not in reg:
        return {"status": "error", "output": f"Service '{sid}' not found."}
    pid = reg[sid].get("pid", 0)
    try:
        os.kill(pid, 15)
        time.sleep(0.3)
        try: os.kill(pid, 9)
        except (OSError, ProcessLookupError): pass
        return {"status": "success", "output": f"Stopped service '{sid}' (PID {pid})."}
    except (OSError, ProcessLookupError):
        return {"status": "success", "output": f"Service '{sid}' was already stopped."}
    except Exception as e:
        return {"status": "error", "output": f"Failed stopping service: {e}"}

def get_system_telemetry(**kwargs) -> dict:
    from agent_tools import get_workspace_dir
    mem_total, mem_avail = 0, 0
    try:
        with open("/proc/meminfo") as f:
            for l in f:
                if l.startswith("MemTotal:"): mem_total = int(l.split()[1]) // 1024
                elif l.startswith("MemAvailable:"): mem_avail = int(l.split()[1]) // 1024
    except Exception: pass
    mem_used = mem_total - mem_avail
    ram_pct = round((mem_used / mem_total * 100), 1) if mem_total else 0
    
    du = shutil.disk_usage(get_workspace_dir())
    disk_free_gb = round(du.free / (1024**3), 1)
    disk_total_gb = round(du.total / (1024**3), 1)
    
    gpu_info = "None"
    if shutil.which("nvidia-smi"):
        try:
            cmd = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"]
            out = subprocess.check_output(cmd, text=True, timeout=5, stderr=subprocess.DEVNULL).strip()
            gpus = []
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    gpus.append(f"{parts[0]}: {parts[2]}MB/{parts[1]}MB VRAM | {parts[4]}°C | Load {parts[3]}%")
            if gpus: gpu_info = "\n  " + "\n  ".join(gpus)
        except Exception: pass
    
    load = [round(x, 2) for x in os.getloadavg()] if hasattr(os, "getloadavg") else []
    report = (
        f"=== SYSTEM TELEMETRY ===\n"
        f"RAM: {mem_used}MB / {mem_total}MB ({ram_pct}%)\n"
        f"Disk: {disk_free_gb}GB free of {disk_total_gb}GB\n"
        f"CPU: {os.cpu_count()} cores | Load: {load}\n"
        f"GPU: {gpu_info}"
    )
    return {"status": "success", "output": report}

def check_port_open(port: int = 80, host: str = "127.0.0.1", timeout: float = 2.0, **kwargs) -> dict:
    try:
        with socket.create_connection((str(host).strip(), int(port)), timeout=float(timeout)):
            return {"status": "success", "open": True, "output": f"Port {port} on {host} is OPEN."}
    except (socket.timeout, ConnectionRefusedError, OSError):
        return {"status": "success", "open": False, "output": f"Port {port} on {host} is CLOSED."}

def probe_http_endpoint(url: str = "", method: str = "GET", payload: dict = None, headers: dict = None, timeout: int = 10, **kwargs) -> dict:
    url = str(url or "").strip()
    if not url.startswith(("http://", "https://")): url = "http://" + url
    req_data = json.dumps(payload).encode("utf-8") if payload else None
    h = dict(headers or {})
    if payload: h.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=req_data, headers=h, method=method.upper())
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=int(timeout)) as resp:
            ms = round((time.time() - t0) * 1000, 1)
            body = resp.read().decode("utf-8", errors="replace")
            return {"status": "success", "status_code": resp.status, "latency_ms": ms, "output": f"HTTP {resp.status} ({ms}ms)\n{body[:500]}"}
    except urllib.error.HTTPError as e:
        ms = round((time.time() - t0) * 1000, 1)
        err = e.read().decode("utf-8", errors="replace")
        return {"status": "error", "status_code": e.code, "latency_ms": ms, "output": f"HTTP {e.code} ({ms}ms): {err[:300]}"}
    except Exception as e:
        return {"status": "error", "output": f"Connection failed to {url}: {e}"}

def query_sqlite(db_path: str = "", path: str = "", sql_query: str = "", query: str = "", max_rows: int = 25, **kwargs) -> dict:
    from agent_tools import _resolve_safe_path
    db = str(db_path or path or "").strip()
    q = str(sql_query or query or "").strip()
    target, safe = _resolve_safe_path(db)
    if not safe or not os.path.isfile(target):
        return {"status": "error", "output": f"SQLite DB not found: {db}"}
    if not q:
        return {"status": "error", "output": "Query required."}
    try:
        conn = sqlite3.connect(target, timeout=5)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(q)
        if cur.description is None:
            conn.commit()
            rows_aff = conn.total_changes
            conn.close()
            return {"status": "success", "output": f"Query OK. Rows affected: {rows_aff}"}
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(int(max_rows))
        conn.close()
        if not rows:
            return {"status": "success", "count": 0, "output": f"0 rows returned. Columns: {cols}"}
        header = "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |"
        body = "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows)
        return {"status": "success", "count": len(rows), "output": f"{len(rows)} row(s):\n\n{header}\n{body}"}
    except Exception as e:
        return {"status": "error", "output": f"SQLite error: {e}"}

def inspect_tabular_file(path: str = "", file: str = "", max_sample_rows: int = 5, **kwargs) -> dict:
    from agent_tools import _resolve_safe_path
    target, safe = _resolve_safe_path(path or file)
    if not safe or not os.path.isfile(target):
        return {"status": "error", "output": f"File not found: {path or file}"}
    ext = os.path.splitext(target)[1].lower()
    size = os.path.getsize(target)
    try:
        samples = []
        cols = []
        if ext in (".csv", ".tsv", ".txt"):
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                cols = next(reader, [])
                for i, r in enumerate(reader):
                    if i >= int(max_sample_rows): break
                    samples.append(r)
        elif ext == ".jsonl":
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                for i, l in enumerate(f):
                    if i >= int(max_sample_rows): break
                    if l.strip(): samples.append(json.loads(l))
            if samples and isinstance(samples[0], dict): cols = list(samples[0].keys())
        return {"status": "success", "size_bytes": size, "columns": cols, "output": f"File: {os.path.basename(target)} ({size:,} bytes)\nColumns ({len(cols)}): {cols}\nSamples:\n{json.dumps(samples, indent=2)}"}
    except Exception as e:
        return {"status": "error", "output": f"Inspect failed: {e}"}

def find_symbol_definitions(symbol_name: str = "", symbol: str = "", path: str = ".", **kwargs) -> dict:
    from agent_tools import get_workspace_dir, _resolve_safe_path, _walk_project_files
    target, safe = _resolve_safe_path(path or ".")
    q = str(symbol_name or symbol or "").strip().lower()
    if not q: return {"status": "error", "output": "Symbol name required."}
    matches = []
    root = get_workspace_dir()
    for full in _walk_project_files(target):
        if not full.endswith(".py"): continue
        try:
            with open(full, "r", encoding="utf-8", errors="ignore") as f: source = f.read()
            tree = ast.parse(source, filename=full)
            rel = os.path.relpath(full, root)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if q in node.name.lower():
                        k = "class" if isinstance(node, ast.ClassDef) else "def"
                        matches.append(f"{rel}:{node.lineno}: {k} {node.name}")
        except Exception: continue
    return {"status": "success", "count": len(matches), "output": f"Found {len(matches)} match(es):\n" + "\n".join(matches) if matches else f"No symbols matching '{q}'."}

def outline_python_file(path: str = "", file: str = "", **kwargs) -> dict:
    from agent_tools import _resolve_safe_path
    target, safe = _resolve_safe_path(path or file)
    if not safe or not os.path.isfile(target): return {"status": "error", "output": f"File not found: {path or file}"}
    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f: source = f.read()
        tree = ast.parse(source, filename=target)
        out = []
        for n in tree.body:
            if isinstance(n, ast.ClassDef):
                out.append(f"Line {n.lineno:4d}: class {n.name}")
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out.append(f"Line {m.lineno:4d}:   def {m.name}")
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(f"Line {n.lineno:4d}: def {n.name}")
        return {"status": "success", "count": len(out), "output": f"=== Outline: {os.path.basename(target)} ===\n" + ("\n".join(out) if out else "(No top-level functions or classes)")}
    except Exception as e:
        return {"status": "error", "output": f"Outline failed: {e}"}

def read_screen_region_text(image_path: str = None, **kwargs) -> dict:
    from agent_tools import SCREENSHOT_DIR, _resolve_safe_path
    if not image_path:
        shots = sorted(Path(SCREENSHOT_DIR).glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not shots: return {"status": "error", "output": "No screenshots found."}
        target = str(shots[0])
    else:
        target, safe = _resolve_safe_path(image_path)
        if not safe or not os.path.isfile(target): target = os.path.join(SCREENSHOT_DIR, os.path.basename(image_path))
    if not os.path.isfile(target): return {"status": "error", "output": f"Image not found: {target}"}
    tess = shutil.which("tesseract")
    if not tess: return {"status": "error", "output": "tesseract binary not installed (apt install tesseract-ocr)."}
    try:
        out = subprocess.check_output([tess, target, "stdout"], text=True, timeout=15, stderr=subprocess.DEVNULL).strip()
        return {"status": "success", "output": out if out else "(No text recognized)"}
    except Exception as e:
        return {"status": "error", "output": f"OCR failed: {e}"}

# =====================================================================
# SCREEN IDENTIFICATION, TARGETED OCR & VISUAL ERROR INSPECTOR
# =====================================================================

def _run_tesseract_tsv(image_path: str) -> list[dict]:
    """Runs tesseract with TSV output to extract word-level coordinates."""
    tess = shutil.which("tesseract")
    if not tess:
        raise RuntimeError("tesseract-ocr binary is missing. Run: apt install tesseract-ocr")
    
    cmd = [tess, image_path, "stdout", "--psm", "6", "tsv"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=15)
    
    records = []
    lines = proc.stdout.splitlines()
    if len(lines) <= 1:
        return records

    header = lines[0].split("\t")
    for row in lines[1:]:
        cols = row.split("\t")
        if len(cols) != len(header):
            continue
        entry = dict(zip(header, cols))
        text = entry.get("text", "").strip()
        conf = entry.get("conf", "-1")
        try:
            conf_val = float(conf)
        except ValueError:
            conf_val = -1.0

        if text and conf_val > 0:
            records.append({
                "text": text,
                "confidence": conf_val,
                "left": int(entry["left"]),
                "top": int(entry["top"]),
                "width": int(entry["width"]),
                "height": int(entry["height"]),
                "line_num": int(entry.get("line_num", 0)),
                "block_num": int(entry.get("block_num", 0))
            })
    return records

def find_text_on_screen(query: str = "", text: str = "", image_path: str = None, **kwargs) -> dict:
    """Finds exact coordinates of a target string or button on the screen for automated clicking."""
    search_term = str(query or text or "").strip().lower()
    if not search_term:
        return {"status": "error", "output": "Search query/text cannot be empty."}

    from agent_tools import capture_desktop, SCREENSHOT_DIR, _resolve_safe_path
    
    # Capture fresh screenshot if not provided
    if not image_path:
        cap_res = capture_desktop("find_target_temp.png")
        if cap_res.get("status") != "success":
            return {"status": "error", "output": f"Screenshot capture failed: {cap_res.get('output')}"}
        target_img = os.path.join(SCREENSHOT_DIR, "find_target_temp.png")
    else:
        target, safe = _resolve_safe_path(image_path)
        target_img = target if safe and os.path.isfile(target) else os.path.join(SCREENSHOT_DIR, os.path.basename(image_path))

    if not os.path.isfile(target_img):
        return {"status": "error", "output": f"Target image not found: {target_img}"}

    try:
        words = _run_tesseract_tsv(target_img)
    except Exception as e:
        return {"status": "error", "output": f"OCR extraction error: {e}"}

    matches = []
    # 1. Single word match
    for w in words:
        if search_term == w["text"].lower() or search_term in w["text"].lower():
            cx = w["left"] + (w["width"] // 2)
            cy = w["top"] + (w["height"] // 2)
            matches.append({
                "matched_text": w["text"],
                "confidence": round(w["confidence"], 1),
                "box": {"x": w["left"], "y": w["top"], "width": w["width"], "height": w["height"]},
                "center": {"x": cx, "y": cy}
            })

    if not matches:
        return {"status": "success", "count": 0, "output": f"Text '{search_term}' not detected on screen."}

    summary = [
        f"Found {len(matches)} match(es) for '{search_term}':",
        *[f"- Text: '{m['matched_text']}' | Center: ({m['center']['x']}, {m['center']['y']}) | Box: {m['box']}" for m in matches[:10]]
    ]
    return {
        "status": "success",
        "count": len(matches),
        "matches": matches,
        "best_match": matches[0],
        "output": "\n".join(summary)
    }

def inspect_screen_errors(image_path: str = None, **kwargs) -> dict:
    """Scans the screen for stack traces, compiler errors, or modal popups and reports coordinates."""
    from agent_tools import capture_desktop, SCREENSHOT_DIR, _resolve_safe_path

    if not image_path:
        cap_res = capture_desktop("screen_inspect_temp.png")
        if cap_res.get("status") != "success":
            return {"status": "error", "output": f"Screenshot failed: {cap_res.get('output')}"}
        target_img = os.path.join(SCREENSHOT_DIR, "screen_inspect_temp.png")
    else:
        target, safe = _resolve_safe_path(image_path)
        target_img = target if safe and os.path.isfile(target) else os.path.join(SCREENSHOT_DIR, os.path.basename(image_path))

    if not os.path.isfile(target_img):
        return {"status": "error", "output": f"Image file not found: {target_img}"}

    try:
        words = _run_tesseract_tsv(target_img)
    except Exception as e:
        return {"status": "error", "output": f"OCR scanning error: {e}"}

    ERROR_MARKERS = {
        "error", "traceback", "exception", "failed", "fatal",
        "syntaxerror", "indentationerror", "nameerror", "typeerror",
        "connectionrefused", "denied", "segfault", "panic"
    }

    detected_issues = []
    for w in words:
        cleaned = re.sub(r"[^a-zA-Z]", "", w["text"]).lower()
        if cleaned in ERROR_MARKERS:
            cx = w["left"] + (w["width"] // 2)
            cy = w["top"] + (w["height"] // 2)
            detected_issues.append({
                "marker": w["text"],
                "confidence": round(w["confidence"], 1),
                "center": {"x": cx, "y": cy},
                "box": {"x": w["left"], "y": w["top"], "width": w["width"], "height": w["height"]}
            })

    if not detected_issues:
        return {"status": "success", "error_count": 0, "output": "No prominent error markers detected on the screen."}

    out_lines = [f"=== DETECTED {len(detected_issues)} VISUAL ERROR MARKER(S) ==="]
    for issue in detected_issues[:15]:
        out_lines.append(f"- Marker: '{issue['marker']}' at center ({issue['center']['x']}, {issue['center']['y']})")

    return {
        "status": "success",
        "error_count": len(detected_issues),
        "issues": detected_issues,
        "output": "\n".join(out_lines)
    }

def ocr_screen_region(x: int = 0, y: int = 0, width: int = 400, height: int = 300, **kwargs) -> dict:
    """Crops a specific rectangular screen coordinate box and extracts the text within it."""
    from agent_tools import capture_desktop, SCREENSHOT_DIR
    from PIL import Image

    cap_res = capture_desktop("region_temp.png")
    if cap_res.get("status") != "success":
        return {"status": "error", "output": f"Capture failed: {cap_res.get('output')}"}

    target_img = os.path.join(SCREENSHOT_DIR, "region_temp.png")
    crop_path = os.path.join(SCREENSHOT_DIR, "region_crop.png")

    try:
        with Image.open(target_img) as img:
            x0 = max(0, int(x))
            y0 = max(0, int(y))
            x1 = min(img.width, x0 + int(width))
            y1 = min(img.height, y0 + int(height))
            cropped = img.crop((x0, y0, x1, y1))
            cropped.save(crop_path)

        tess = shutil.which("tesseract")
        out = subprocess.check_output([tess, crop_path, "stdout"], text=True, timeout=10, stderr=subprocess.DEVNULL).strip()
        return {
            "status": "success",
            "region": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
            "output": out if out else "(No readable text in specified region)"
        }
    except Exception as e:
        return {"status": "error", "output": f"Region OCR error: {e}"}
