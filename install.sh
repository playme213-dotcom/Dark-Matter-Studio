#!/usr/bin/env bash
# Dark Matter Studio — install.sh
set -Eeuo pipefail
STUDIO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="/home/local-ai/.config/dark-matter"
ENV_FILE="$ENV_DIR/studio.env"
BIN_DIR="/home/local-ai/.local/bin"
NO_RUN=0
for arg in "$@"; do
  [[ "$arg" == "--no-run" ]] && NO_RUN=1
done
if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
  echo "Do not run install.sh as root." >&2
  exit 1
fi
command -v python3 >/dev/null || { echo "python3 required" >&2; exit 1; }
mkdir -p "$ENV_DIR" "$BIN_DIR"
chmod 700 "$ENV_DIR"
API_KEY="sk-$(python3 -c "import secrets; print(secrets.token_urlsafe(36))")"
printf 'export DARK_MATTER_API_KEY="%s"\n' "$API_KEY" > "$ENV_FILE"
chmod 600 "$ENV_FILE"
echo "API key saved to $ENV_FILE"
for name in agy dms; do
  cat > "$BIN_DIR/$name" <<'LAUNCHER'
#!/usr/bin/env bash
set -Eeuo pipefail
ENV_FILE="{{ENV_FILE}}"
STUDIO_DIR="{{STUDIO_DIR}}"
STUDIO_PY="/home/local-ai/venv/bin/python"
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"
export DARK_MATTER_API_KEY="{{API_KEY}}"
cd "$STUDIO_DIR"
exec "$STUDIO_PY" "$STUDIO_DIR/app.py" "$@"
LAUNCHER
  sed -i "s|{{ENV_FILE}}|$ENV_FILE|g" "$BIN_DIR/$name"
  sed -i "s|{{STUDIO_DIR}}|$STUDIO_DIR|g" "$BIN_DIR/$name"
  sed -i "s|{{API_KEY}}|$API_KEY|g" "$BIN_DIR/$name"
  chmod +x "$BIN_DIR/$name"
done
echo "Installation complete. Run: dms (or agy)"
if [[ "$NO_RUN" == 1 ]]; then
  exit 0
fi
exec /home/local-ai/venv/bin/python "$STUDIO_DIR/app.py"
