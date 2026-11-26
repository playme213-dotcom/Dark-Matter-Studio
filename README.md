# Dark Matter Studio

Autonomous local AI agent studio — a Flask-based desktop orchestration environment that connects a local LLM backend to a rich set of agent tools (filesystem, terminal, Git, tests/lint, web search, desktop GUI control, skills, project memory). Renamed from Colibri Studio.

> WARNING: Dark Matter Studio must run as the normal desktop user. Never run app.py or install.sh with sudo or as root. The app exits automatically if launched as root.

## Features

| Feature | Description |
|---------|-------------|
| API + key only | Two loader engines: Colibri (direct API) and LiteLLM (local proxy). Authentication via DARK_MATTER_API_KEY sent as a Bearer header on every request. Ollama, Web Extension, Strata, and ngrok removed. |
| Swarm multi-agent | OpenAI Swarm pattern port: Agent, Swarm, Response, Result with handoffs. API routes: /api/swarm/agents, /api/swarm/run, /api/swarm/board, /api/swarm/handoffs. State stored in swarm_agents.json. |
| Collapse sidebar | True CSS width transition with localStorage persistence; icon-only collapsed mode. |
| Context compaction 75% | Context auto-compaction triggers at 75% of the window (was 78%). Original chat rows are never deleted. |
| Tool toggles | Per-tool enable/disable via tool_toggles.json. Disabled tools are rejected at dispatch. |
| Web search toggle | Prominent on/off toggle for network-facing web_search / fetch_web_page. |
| Coda refinement | Iterative refinement loop (from coda.json, {"loops": 3}) that re-feeds the assistant output to improve clarity/correctness before completion. |
| Loop count control | Configurable max_loop_turns (default 10) for the main agent loop and swarm runs. |
| Auto-update routes | /api/update/check and /api/update/apply (requires confirmed=true). |
| Shell launchers | dms — launch Dark Matter Studio from anywhere, like codex. |

## Quick Start

### One-shot install

```bash
# Clone
git clone https://github.com/playme213-dotcom/Dark-Matter-Studio.git
cd Dark-Matter-Studio

# Install (configures API key, installs dms launcher, launches Studio)
bash install.sh

# Or install only (no launch)
bash install.sh --no-run
```

### Launch from anywhere

After install, ~/.local/bin must be on your PATH:

```bash
export PATH="$HOME/.local/bin:$PATH"

dms        # launch Dark Matter Studio
```

Then open http://127.0.0.1:5000 in your browser.

### Manual launch

```bash
cd Dark-Matter-Studio
source /home/local-ai/.config/dark-matter/studio.env
export DARK_MATTER_API_KEY="sk-..."
/home/local-ai/venv/bin/python app.py
```

## Shell Launchers

| Launcher | Path | Purpose |
|----------|------|---------|
| dms | ~/.local/bin/dms | Dark Matter Studio launcher |

Both source ~/.config/dark-matter/studio.env for the API key, cd to the studio directory, and run app.py.

Install it manually with:
```bash
bash setup_dark_matter.sh --install-launchers
```

## API Key

Configured via DARK_MATTER_API_KEY environment variable. Priority:
1. DARK_MATTER_API_KEY env var
2. ~/.config/dark-matter/studio.env
3. Fallback: COLIBRI_API_KEY / LITELLM_API_KEY / LITELLM_MASTER_KEY

Sent as Authorization: Bearer <key> on every inference request.

## Project Structure

```
Dark-Matter-Studio/
├── app.py                 # Main Flask app (agent loop, routes, frontend)
├── agent_tools.py         # Agent tool implementations (filesystem, terminal, Git, etc.)
├── advanced_tools.py      # Extended tools (telemetry, SQLite, OCR, etc.)
├── context_manager.py     # Token counting, context budgeting, compaction
├── swarm.py               # OpenAI Swarm port (Agent/Swarm/Response/handoffs)
├── setup_dark_matter.sh   # Studio setup & launcher installer
├── install.sh             # One-shot installer (API key + launchers)
├── VERSION                # Version file (1.0.0)
├── PERMISSIONS_SECURITY.md# File ownership & permission policy
├── UPGRADE_LOOP_EXECUTION.md# Loop execution architecture notes
├── tools_skills.md        # Agent tools & skills reference
├── skills/                # Agent-facing skill guides
│   ├── colibri_initialization.txt
│   ├── python_skills.md
│   ├── HTML_skills.md
│   ├── JS_skills.md
│   └── tools_skills.md
├── web_extension/         # Chrome MV3 extension (disconnected from backend)
│   ├── manifest.json
│   ├── background.js
│   └── content.js
├── screenshots/           # Desktop capture output
└── tests/                 # Python test suite
```

## Configuration

Edit LOADER_STATE in app.py or use the UI dock to set:
- Loader: Colibri or LiteLLM
- Model: your Qwen3-Coder model name
- API key: via DARK_MATTER_API_KEY or the UI
- Temperature / Top-p / Max output tokens
- Question mode: manual or auto_yes
- Thinking level: off / low / medium / high
- Max loop turns: max_loop_turns
- Coda refinement: on/off + loop count
- Tool toggles: per-tool enable/disable

## Safety & Permissions

All file/folder operations run as the ordinary desktop user (local-ai). The app never elevates to root and never retries failed operations as root.

- No chmod 777, no recursive chown
- repair_workspace_metadata allows only single-path chmod/chown with explicit approved=true via PolicyKit (pkexec)
- New files inherit the user's normal umask

See PERMISSIONS_SECURITY.md for the full policy.

## Tests

```bash
cd Dark-Matter-Studio
python -m pytest -q
```

Or with unittest:
```bash
python -m unittest discover
```

## License

MIT — see LICENSE if present.

## Repository

https://github.com/playme213-dotcom/Dark-Matter-Studio
