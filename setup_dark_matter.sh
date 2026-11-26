#!/usr/bin/env bash
# Dark Matter Studio setup & launcher.
# Configures the API key and launches Dark Matter Studio (app.py) as the normal desktop user.
# Usage: ./setup_dark_matter.sh [--no-run] [--install-launchers]
set -Eeuo pipefail

STUDIO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDIO_APP="$STUDIO_DIR/app.py"
STUDIO_PY="${COLIBRI_STUDIO_PYTHON:-/home/local-ai/venv/bin/python}"
ENV_DIR="/home/local-ai/.config/dark-matter"
ENV_FILE="$ENV_DIR/studio.env"
BIN_DIR="/home/local-ai/.local/bin"
NO_RUN=0
INSTALL_LAUNCHERS=0

for arg in "$@"; do
  [[ "$arg" == "--no-run" ]] && NO_RUN=1
  [[ "$arg" == "--install-launchers" ]] && INSTALL_LAUNCHERS=1
done

log(){ echo "[DARK-MATTER] $*"; }
die(){ echo "[DARK-MATTER][ERROR] $*" >&2; exit 1; }

if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  die "Do not run this setup as root. Run it from the normal Ubuntu desktop user account."
fi

command -v python3 >/dev/null || die "python3 is required"

mkdir -p "$ENV_DIR" "$BIN_DIR"
chmod 700 "$ENV_DIR"

# Preserve existing API key
if [[ -f "$ENV_FILE" ]]; then
  _SAVED_KEY="$(grep -E '^export DARK_MATTER_API_KEY=' "$ENV_FILE" | cut -d"'" -f2 || true)"
  [[ -n "${DARK_MATTER_API_KEY:-}" ]] || DARK_MATTER_API_KEY="$_SAVED_KEY"
fi

if [[ -z "${DARK_MATTER_API_KEY:-}" ]]; then
  DARK_MATTER_API_KEY="$(python3 -c 'import secrets; print("sk-" + secrets.token_urlsafe(36))')"
fi
export DARK_MATTER_API_KEY

umask 077
cat > "$ENV_FILE" <<EOF
export DARK_MATTER_API_KEY='${DARK_MATTER_API_KEY}'
EOF
chmod 600 "$ENV_FILE"

log "Dark Matter Studio configured. API key saved to $ENV_FILE"

# Install shell-linkable launchers (agy and dms)
if [[ "$INSTALL_LAUNCHERS" == 1 ]]; then
  for name in agy dms; do
    cat > "$BIN_DIR/$name" <<LAUNCHER
#!/usr/bin/env bash
set -Eeuo pipefail
# $name — Dark Matter Studio launcher (shell-linkable, like codex)
ENV_FILE="$ENV_FILE"
STUDIO_DIR="$STUDIO_DIR"
STUDIO_PY="$STUDIO_PY"

[[ -f "\$ENV_FILE" ]] && source "\$ENV_FILE"

export DARK_MATTER_API_KEY="${DARK_MATTER_API_KEY}"

cd "\$STUDIO_DIR"
exec "\$STUDIO_PY" "\$STUDIO_DIR/app.py" "\$@"
LAUNCHER
    chmod +x "$BIN_DIR/$name"
    log "Installed '$name' launcher to $BIN_DIR/$name"
  done
  log "Make sure $BIN_DIR is on your PATH, then run: agy  or  dms"
fi

if [[ "$NO_RUN" == 1 ]]; then
  log "Setup complete (--no-run, not launching Studio)."
  exit 0
fi

log "Launching Dark Matter Studio on http://127.0.0.1:5000 ..."
exec "$STUDIO_PY" "$STUDIO_APP"
