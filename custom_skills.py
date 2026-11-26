"""
Custom Skills System for Dark Matter Studio.

Users can drop .py files into skills/custom/ directory. Each file is:
1. Syntax-checked
2. Test-searched (functions starting with 'test_' or 'run_test_')
3. Imported if tests pass
4. Made available to the agent as callable skills

Skills are auto-discovered on startup and can be managed via API.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

CUSTOM_SKILLS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "skills", "custom"
)

# Registry of loaded custom skills: name -> {"module": module, "functions": {...}, "file": path, "error": str}
CUSTOM_SKILLS_REGISTRY: Dict[str, dict] = {}


def _ensure_dir() -> None:
    os.makedirs(CUSTOM_SKILLS_DIR, exist_ok=True)


def list_custom_skills() -> List[dict]:
    """Return list of custom skill definitions."""
    _ensure_dir()
    result = []
    for fname in sorted(os.listdir(CUSTOM_SKILLS_DIR)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        fpath = os.path.join(CUSTOM_SKILLS_DIR, fname)
        name = fname[:-3]  # strip .py
        entry = {
            "name": name,
            "file": fname,
            "path": fpath,
            "loaded": name in CUSTOM_SKILLS_REGISTRY,
            "error": None,
            "functions": [],
        }
        if name in CUSTOM_SKILLS_REGISTRY:
            reg = CUSTOM_SKILLS_REGISTRY[name]
            entry["functions"] = list(reg.get("functions", {}).keys())
            if reg.get("error"):
                entry["error"] = reg["error"]
        result.append(entry)
    return result


def _find_test_functions(module) -> List[str]:
    """Find test functions in a module (names starting with test_ or run_test_)."""
    test_funcs = []
    for name in dir(module):
        if name.startswith("test_") or name.startswith("run_test_"):
            obj = getattr(module, name)
            if callable(obj):
                test_funcs.append(name)
    return test_funcs


def _run_test_functions(module, test_funcs: List[str]) -> tuple[bool, str]:
    """Run test functions and return (passed, error_message)."""
    errors = []
    for tf_name in test_funcs:
        try:
            tf = getattr(module, tf_name)
            result = tf()
            if result is False:
                errors.append(f"{tf_name} returned False")
        except Exception as exc:
            errors.append(f"{tf_name} raised {type(exc).__name__}: {exc}")
    if errors:
        return False, "; ".join(errors)
    return True, ""


def load_custom_skill(name: str) -> dict:
    """Load a single custom skill by name (without .py extension)."""
    _ensure_dir()
    fpath = os.path.join(CUSTOM_SKILLS_DIR, f"{name}.py")
    if not os.path.isfile(fpath):
        return {"status": "error", "output": f"Skill '{name}' not found"}

    if name in CUSTOM_SKILLS_REGISTRY:
        # Already loaded, return info
        reg = CUSTOM_SKILLS_REGISTRY[name]
        return {
            "status": "success",
            "name": name,
            "functions": list(reg.get("functions", {}).keys()),
            "already_loaded": True,
        }

    # Read and syntax-check
    try:
        source = Path(fpath).read_text(encoding="utf-8")
    except OSError as exc:
        return {"status": "error", "output": f"Cannot read {name}: {exc}"}

    try:
        ast.parse(source)
    except SyntaxError as exc:
        return {
            "status": "error",
            "output": f"Syntax error in {name}: {exc.msg} at line {exc.lineno}",
        }

    # Load module
    spec = importlib.util.spec_from_file_location(f"custom_skill_{name}", fpath)
    if spec is None or spec.loader is None:
        return {"status": "error", "output": f"Cannot create spec for {name}"}

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        return {
            "status": "error",
            "output": f"Import error in {name}: {type(exc).__name__}: {exc}",
        }

    # Find and run tests
    test_funcs = _find_test_functions(module)
    if test_funcs:
        passed, error_msg = _run_test_functions(module, test_funcs)
        if not passed:
            return {
                "status": "error",
                "output": f"Tests failed in {name}: {error_msg}",
            }

    # Collect public functions (not starting with _)
    functions = {}
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        obj = getattr(module, attr_name)
        if callable(obj):
            functions[attr_name] = obj

    CUSTOM_SKILLS_REGISTRY[name] = {
        "module": module,
        "functions": functions,
        "file": fpath,
        "error": None,
    }

    return {
        "status": "success",
        "name": name,
        "functions": list(functions.keys()),
    }


def load_all_custom_skills() -> dict:
    """Load all custom skills from the directory."""
    _ensure_dir()
    results = []
    for fname in sorted(os.listdir(CUSTOM_SKILLS_DIR)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        name = fname[:-3]
        result = load_custom_skill(name)
        results.append(result)
    return results


def unload_custom_skill(name: str) -> dict:
    """Unload a custom skill."""
    if name in CUSTOM_SKILLS_REGISTRY:
        del CUSTOM_SKILLS_REGISTRY[name]
        # Also remove from sys.modules if present
        mod_name = f"custom_skill_{name}"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        return {"status": "success", "name": name}
    return {"status": "error", "output": f"Skill '{name}' not loaded"}


def delete_custom_skill(name: str) -> dict:
    """Delete a custom skill file and unload it."""
    _ensure_dir()
    fpath = os.path.join(CUSTOM_SKILLS_DIR, f"{name}.py")
    if not os.path.isfile(fpath):
        return {"status": "error", "output": f"Skill '{name}' not found"}

    # Unload first
    unload_custom_skill(name)

    try:
        os.remove(fpath)
        return {"status": "success", "name": name}
    except OSError as exc:
        return {"status": "error", "output": f"Cannot delete {name}: {exc}"}


def write_custom_skill(name: str, content: str) -> dict:
    """Write or update a custom skill file."""
    _ensure_dir()
    # Validate name
    if not name or not name.replace("_", "").isalnum():
        return {"status": "error", "output": "Invalid skill name"}

    fpath = os.path.join(CUSTOM_SKILLS_DIR, f"{name}.py")

    # Syntax check
    try:
        ast.parse(content)
    except SyntaxError as exc:
        return {
            "status": "error",
            "output": f"Syntax error: {exc.msg} at line {exc.lineno}",
        }

    try:
        Path(fpath).write_text(content, encoding="utf-8")
    except OSError as exc:
        return {"status": "error", "output": f"Cannot write {name}: {exc}"}

    # Try to load it
    return load_custom_skill(name)


def get_skill_function(name: str, func_name: str):
    """Get a specific function from a loaded custom skill."""
    reg = CUSTOM_SKILLS_REGISTRY.get(name)
    if reg is None:
        return None
    return reg.get("functions", {}).get(func_name)


def call_skill_function(name: str, func_name: str, *args, **kwargs):
    """Call a skill function and return result."""
    fn = get_skill_function(name, func_name)
    if fn is None:
        return {"status": "error", "output": f"Function {name}.{func_name} not found"}
    try:
        result = fn(*args, **kwargs)
        return {"status": "success", "result": result}
    except Exception as exc:
        return {
            "status": "error",
            "output": f"Error calling {name}.{func_name}: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def get_all_skills_for_prompt() -> str:
    """Generate a text description of all loaded custom skills for the system prompt."""
    if not CUSTOM_SKILLS_REGISTRY:
        return ""

    lines = ["\n\n# Custom Skills (User-Defined Python Functions)\n"]
    lines.append(
        "The following custom skills are available as callable Python functions.\n"
        "Each skill was loaded from skills/custom/ and passed its test functions.\n"
        "Use them by calling the function name with appropriate arguments.\n"
    )

    for name, reg in sorted(CUSTOM_SKILLS_REGISTRY.items()):
        funcs = reg.get("functions", {})
        lines.append(f"\n## {name} (from skills/custom/{name}.py)")
        for fn_name, fn_obj in funcs.items():
            # Get docstring
            doc = fn_obj.__doc__ or "No description"
            # Get signature info
            lines.append(f"  - {fn_name}: {doc.strip().split(chr(10))[0]}")

    lines.append("\nTo call a custom skill, use the function name directly in your reasoning.\n")
    return "\n".join(lines)


def reload_all() -> dict:
    """Reload all custom skills (clear registry and reload)."""
    global CUSTOM_SKILLS_REGISTRY
    CUSTOM_SKILLS_REGISTRY = {}
    # Clear sys.modules entries
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("custom_skill_"):
            del sys.modules[mod_name]
    return load_all_custom_skills()
