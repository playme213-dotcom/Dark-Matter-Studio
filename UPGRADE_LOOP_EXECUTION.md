# Colibri Studio — Loop Execution Upgrade

## Architecture changes

The existing Colibri Studio architecture is retained. The changes are in `app.py` and the existing loop regression tests; there is no database migration.

- Replaced the fixed 30-step loop with a controller-driven `while True` loop. It exits on explicit completion, blocked state, exact consecutive completed-response duplication, cancellation, or execution error.
- Added a per-conversation active-run guard to prevent concurrent loops for the same session.
- Added explicit execution-state events: `RUNNING`, `WAITING_FOR_TOOL`, `PROCESSING_TOOL_RESULT`, `CONTINUING`, `COMPLETED`, `STOPPED`, `CANCELLED`, and `ERROR`.
- Duplicate tracking is task-local and compares complete, tool-free assistant responses using exact equality only. The repeated response is retained and the UI displays `Loop stopped: identical response detected.`
- If a response lacks a completion marker, the controller asks the model to reassess acceptance criteria and continue rather than treating one response as completion.
- Removed the frontend `step_reset` clearing behavior. Streaming chunks append to the current assistant message, and each later iteration is separated by a newline block.
- Tool-cycle assistant text is persisted in the assistant transcript; tool records remain in `tools_json`.
- Cancellation is observed during streaming and immediately after a synchronous tool call returns. A currently running synchronous tool is not forcibly preempted.
- Added a bottom-left LiteLLM Strata usage panel. LiteLLM requests ask for streamed usage metadata and display prefill/prompt tokens, generation tokens, and configured maximum context when supplied. Existing Ollama metrics remain supported.

## Files changed

- `app.py` — loop controller, duplicate guard, active-run guard, execution states, append-only streaming/persistence, Strata usage panel.
- `test_shared_agent_loop.py` — regression assertions for unbounded loop behavior, exact duplicate comparison, append-only rendering, and usage panel.
- `test_provider_agent_loop.py` — updated assertion that the controller has no fixed turn cap.
- `UPGRADE_LOOP_EXECUTION.md` — this report and operating instructions.

## Test report

Executed successfully:

- Python compilation of `app.py`: passed.
- JavaScript syntax checks (`node --check`) for both embedded scripts: passed.
- Full repository test suite: **37 passed, 9 subtests passed**.
- Mocked route-level streaming tests against `run_turn()`:
  - 12 sequential iterations reached explicit completion; all 11 intermediate outputs were present in the saved assistant transcript.
  - Exact duplicate response stopped execution, displayed the stop reason, and preserved both repeated responses.
  - Two sequential tool calls generated two tool-start and two tool-result events.
  - A tool failure remained visible and the model continued to a recovered completion.
  - Cancellation during streaming emitted `CANCELLED`/`stopped` and released the active-run lock.
  - Cancellation while a synchronous tool call was returning emitted `CANCELLED`/`stopped` and released the active-run lock.
  - A new task in the same conversation began with fresh duplicate tracking.

The route-level checks used a mocked LiteLLM-compatible SSE provider and mocked tools; they did not contact a live model endpoint.

Not verified in this environment:

- Live LiteLLM/Strata compatibility and whether the gateway returns usage metadata for every model.
- A real browser session with a live model and real tools, including visual layout at the user's display resolution.
- Restarting the deployed Studio process against the user's production database. SQLite transcript writes were verified, but not a production process restart.
- Forced interruption of a synchronous tool already in progress. Cancellation is observed after that tool returns.

The test runner did not have Flask installed and could not access `/home/local-ai`. The full suite was executed in an isolated temporary workspace using a minimal Flask API shim for test imports. This is not a substitute for testing with the normal Flask dependency and a live provider.

## Installation / update

Run as the normal Ubuntu account that owns the Studio project (normally `local-ai`), never with `sudo` or as root.

1. Stop Colibri Studio using its normal stop method.
2. Back up the current project directory, or at minimum `app.py`, `test_shared_agent_loop.py`, `test_provider_agent_loop.py`, and the Studio database.
3. Extract this ZIP to a temporary directory.
4. Copy the updated `app.py`, both updated test files, and this document into the corresponding existing Studio directory.
5. Keep the existing `.workspace_config.json`, database, model settings, environment files, workspace contents, and integrations unchanged.
6. Restart Studio with its existing virtual environment and normal-user startup command.
7. Run `python -m pytest -q` from the Studio directory.

No `chmod 777`, recursive `chown`, or permission repair is required.

## Rollback

1. Stop Colibri Studio.
2. Restore the backed-up `app.py`, test files, and any other replaced files.
3. Leave the database and workspace configuration unchanged unless they were separately backed up and intentionally need restoring.
4. Restart Studio using the previous normal-user startup command.

There is no schema migration to reverse.
