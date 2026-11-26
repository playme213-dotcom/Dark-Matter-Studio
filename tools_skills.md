# Agent Tools & Skills Reference
## Advanced Tool Suites

### 1. Background Service Management
- **`start_background_service(command, service_id, log_file)`**: Spawns an asynchronous background process/daemon. Output streams to `.agent_services/<service_id>.log` and process lifecycle is tracked in `registry.json`.
- **`get_service_status(service_id, tail_lines)`**: Checks if a background service is active (or lists all registered services) and retrieves the last N lines of stdout/stderr logs.
- **`stop_background_service(service_id)`**: Gracefully terminates a background service via SIGTERM, falling back to SIGKILL.

### 2. Telemetry & Hardware Inspection
- **`get_system_telemetry()`**: Returns live CPU load averages, system RAM utilization, disk headroom, and GPU VRAM usage / temperature via `nvidia-smi`.

### 3. Network & Service Probing
- **`check_port_open(port, host, timeout)`**: Verifies whether a local or remote TCP port is currently listening and accepting connections.
- **`probe_http_endpoint(url, method, payload, headers, timeout)`**: Dispatches structured HTTP requests (GET, POST, PUT, DELETE) and measures round-trip latency and response payloads.

### 4. Structured Data & Tabular Inspection
- **`query_sqlite(db_path, sql_query, max_rows)`**: Executes queries against SQLite databases within the workspace and formats rows into Markdown tables.
- **`inspect_tabular_file(path, max_sample_rows)`**: Analyzes CSV, TSV, or JSONL files to extract column schemas, file size, and sample records.

### 5. AST Symbol Navigation
- **`find_symbol_definitions(symbol_name, path)`**: Uses Python AST to locate exact class, function, and async def definitions across the project without false positives from comments or calls.
- **`outline_python_file(path)`**: Generates a complete hierarchical AST outline of classes, methods, and functions with exact line numbers.

### 6. Screen & Vision OCR
- **`read_screen_region_text(image_path)`**: Runs local `tesseract-ocr` against screenshots or images to extract text from UI elements, dialogs, and terminal buffers.

## File ownership and permissions (safe mode)
- Use `inspect_workspace_permissions` to report owner, group, mode, and effective read/write/execute access.
- If a normal-user file operation fails, diagnose first; never auto-repair and never retry as root.
- `repair_workspace_metadata` supports only a single-path `chmod` or `chown` inside the active workspace, and requires explicit user approval (`approved: true`). It uses PolicyKit (`pkexec`) only for that metadata operation. No recursive operations, special mode bits, arbitrary commands, or `777`.
- All file and folder creation, reads, edits, backups, moves, renames and deletes remain normal-user operations. Never launch Studio with sudo.
