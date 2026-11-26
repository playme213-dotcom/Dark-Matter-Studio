# Python Skills for Small Local AI Models

## Purpose
Use this guide to write, explain, debug, and safely modify Python code. Follow the user's request precisely and keep solutions small, explicit, and testable.

## Workflow
1. Restate the goal internally as a concrete output or behavior.
2. Inspect the existing code and its surrounding call sites before editing.
3. Identify constraints: Python version, OS, dependencies, entry point, input/output formats, and compatibility requirements.
4. Make the smallest change that solves the problem. Preserve existing behavior unless the user requests otherwise.
5. Validate syntax, run focused tests, and report anything that could not be tested.
6. Summarize changed files, behavior, commands run, and remaining risks.

## Python style
- Prefer readable, direct code over clever abstractions.
- Use descriptive `snake_case` names for variables and functions; use `PascalCase` for classes.
- Add type hints to public functions and non-obvious interfaces.
- Use docstrings for public modules, classes, and functions when they clarify purpose or contracts.
- Keep functions focused. Extract a helper when it removes meaningful duplication or complexity.
- Use f-strings for readable interpolation.
- Use `pathlib.Path` for filesystem paths where practical.
- Use context managers (`with`) for files, locks, and resources.
- Avoid mutable default arguments. Use `None` and initialize inside the function.
- Prefer standard-library solutions unless an external dependency offers a clear benefit.

## Correctness and robustness
- Validate external inputs at boundaries; do not repeatedly validate trusted internal values without reason.
- Handle expected errors specifically. Do not use broad `except Exception` to hide bugs.
- If catching an error, preserve useful context and either recover safely or raise a meaningful error.
- Do not silently ignore failures. Log actionable details without exposing secrets.
- Consider empty input, missing files, invalid encodings, permission errors, timeouts, and malformed data.
- Keep side effects visible. Separate pure logic from filesystem, network, and UI operations when practical.
- Use deterministic behavior where possible.
- Do not invent APIs, library methods, configuration keys, or command-line flags. Verify them from installed versions or project documentation when available.

## Editing existing projects
- Read the relevant files before editing; avoid replacing a whole file when a targeted edit is enough.
- Preserve public function names, file formats, settings keys, routes, and established interfaces unless explicitly asked to change them.
- Search for callers and tests before changing a function signature.
- Do not remove working features to simplify an implementation.
- If a project has a locked baseline or explicit compatibility requirements, treat those as hard constraints.
- Keep patches limited to the requested files. Do not package unrelated files.
- Never claim an edit, test, or command succeeded unless it actually ran and succeeded.

## Testing checklist
- Run syntax validation, for example: `python -m py_compile path/to/file.py`.
- Run relevant unit tests, for example: `python -m unittest` or the project's documented test command.
- Test the normal case and at least one meaningful failure or boundary case.
- For bug fixes, reproduce the original failure when practical and verify the fix.
- Do not describe syntax validation as a full integration test.
- If tests cannot run, explain the exact blocker and give the command the user can run.

## Security
- Never hard-code passwords, API keys, tokens, or private keys.
- Treat user input, files, environment variables, and network responses as untrusted.
- Avoid `eval`, `exec`, unsafe deserialization, and shell commands built by concatenating untrusted input.
- Prefer argument arrays with `subprocess.run([...], check=True, timeout=...)` over `shell=True`.
- Apply least privilege to filesystem and process operations.
- Do not delete, overwrite, upload, or transmit user data unless it is clearly requested and the target is understood.
- Avoid logging secrets or full sensitive payloads.
- For web services, use timeouts, validate inputs, and avoid exposing debug tracebacks to end users.

## Performance
- Measure before optimizing.
- Prefer simple data structures and algorithms appropriate to the input size.
- Avoid needless repeated filesystem scans, network calls, or model/API calls.
- Stream large files rather than loading them fully into memory when suitable.
- Close resources promptly.

## Response format for coding tasks
When useful, report:
- **Plan:** brief steps.
- **Files changed:** exact paths and purpose.
- **Validation:** exact tests/commands and outcomes.
- **Limitations:** untested paths or assumptions.
- **How to run:** concise commands.

## Minimal examples

### Safe file reading
```python
from pathlib import Path

def read_text(path: str | Path) -> str:
    file_path = Path(path)
    return file_path.read_text(encoding="utf-8")
```

### Safe subprocess
```python
import subprocess

def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
```

### Specific error handling
```python
from pathlib import Path

def load_config(path: Path) -> dict:
    try:
        import json
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {path}") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid JSON in config file: {path}") from exc
```

## Final rule
Prefer the smallest correct, secure, maintainable solution. Preserve existing functionality, test changes, and clearly distinguish verified facts from assumptions.
