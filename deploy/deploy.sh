#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEPLOY_DIR="$ROOT_DIR/deploy"
DIST_DIR="${DIST_DIR:-$ROOT_DIR/dist}"
BUILD_DIR="${BUILD_DIR:-$DIST_DIR/build}"
CACHE_DIR="${CACHE_DIR:-$DIST_DIR/.cache}"
GO_BUILD_CACHE="${GO_BUILD_CACHE:-$CACHE_DIR/go-build}"
GO_MOD_CACHE="${GO_MOD_CACHE:-$CACHE_DIR/go-mod}"

if [[ -f "$DEPLOY_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$DEPLOY_DIR/.env"
  set +a
fi

APP_NAME="llmio"
SERVICE_NAME="llmio"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_PORT="${REMOTE_PORT:-22}"

TARGET_OS="${TARGET_OS:-linux}"
TARGET_ARCH="${TARGET_ARCH:-amd64}"
CGO_ENABLED_VALUE="${CGO_ENABLED_VALUE:-0}"
VERSION="${VERSION:-$(git -C "$ROOT_DIR" rev-parse --short HEAD)}"

REMOTE_APP_DIR="/opt/llmio"
REMOTE_CONFIG_DIR="/etc/llmio"
REMOTE_ENV_FILE="$REMOTE_CONFIG_DIR/llmio.env"
REMOTE_SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
REMOTE_RUN_USER="llmio"
REMOTE_RUN_GROUP="llmio"

PKG_NAME="${APP_NAME}_${VERSION}_${TARGET_OS}_${TARGET_ARCH}"
PKG_ROOT="$BUILD_DIR/$PKG_NAME"
ARCHIVE_PATH="$DIST_DIR/${PKG_NAME}.tar.gz"
REMOTE_TMP_ARCHIVE="/tmp/${PKG_NAME}.tar.gz"

usage() {
  cat <<'EOF'
Usage:
  deploy/deploy.sh package
  deploy/deploy.sh deploy

Environment variables:
  VERSION       Release version label. Default: current git short sha.
  TARGET_OS     Build target OS. Default: linux.
  TARGET_ARCH   Build target arch. Default: amd64.
  REMOTE_HOST   Required for deploy.
  REMOTE_USER   SSH user. Default: root.
  REMOTE_PORT   SSH port. Default: 22.

Examples:
  VERSION=v0.1.0 bash deploy/deploy.sh package
  REMOTE_HOST=10.0.0.8 REMOTE_USER=root VERSION=v0.1.0 bash deploy/deploy.sh deploy
EOF
}

log() {
  printf '[deploy] %s\n' "$*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "missing command: $1" >&2
    exit 1
  }
}

ssh_target() {
  printf '%s@%s' "$REMOTE_USER" "$REMOTE_HOST"
}

package_release() {
  require_cmd go
  require_cmd tar

  rm -rf "$PKG_ROOT"
  mkdir -p "$PKG_ROOT" "$GO_BUILD_CACHE" "$GO_MOD_CACHE"

  log "building ${APP_NAME} for ${TARGET_OS}/${TARGET_ARCH}"
  (
    cd "$ROOT_DIR"
    GOCACHE="$GO_BUILD_CACHE" GOMODCACHE="$GO_MOD_CACHE" \
    CGO_ENABLED="$CGO_ENABLED_VALUE" GOOS="$TARGET_OS" GOARCH="$TARGET_ARCH" \
      go build -o "$PKG_ROOT/$APP_NAME" ./cmd/llmio
  )

  cp "$DEPLOY_DIR/llmio.service" "$PKG_ROOT/llmio.service"
  cp "$DEPLOY_DIR/llmio.env.example" "$PKG_ROOT/llmio.env.example"
  cp "$ROOT_DIR/llmio.json.example" "$PKG_ROOT/llmio.json.example"

  mkdir -p "$DIST_DIR"
  tar -C "$BUILD_DIR" -czf "$ARCHIVE_PATH" "$PKG_NAME"
  log "created archive: $ARCHIVE_PATH"
}

deploy_release() {
  require_cmd ssh
  require_cmd scp
  [[ -n "$REMOTE_HOST" ]] || {
    echo "REMOTE_HOST is required for deploy" >&2
    exit 1
  }

  package_release

  local target
  target="$(ssh_target)"

  log "uploading archive to ${target}:${REMOTE_TMP_ARCHIVE}"
  scp -P "$REMOTE_PORT" "$ARCHIVE_PATH" "${target}:${REMOTE_TMP_ARCHIVE}"

  log "installing release on remote host"
  ssh -p "$REMOTE_PORT" "$target" \
    "APP_NAME='$APP_NAME' \
SERVICE_NAME='$SERVICE_NAME' \
VERSION='$VERSION' \
REMOTE_APP_DIR='$REMOTE_APP_DIR' \
REMOTE_CONFIG_DIR='$REMOTE_CONFIG_DIR' \
REMOTE_ENV_FILE='$REMOTE_ENV_FILE' \
REMOTE_SERVICE_PATH='$REMOTE_SERVICE_PATH' \
REMOTE_RUN_USER='$REMOTE_RUN_USER' \
REMOTE_RUN_GROUP='$REMOTE_RUN_GROUP' \
REMOTE_TMP_ARCHIVE='$REMOTE_TMP_ARCHIVE' \
bash -s" <<'EOF'
set -euo pipefail

RELEASES_DIR="$REMOTE_APP_DIR/releases"
RELEASE_DIR="$RELEASES_DIR/$VERSION"
CURRENT_LINK="$REMOTE_APP_DIR/current"

mkdir -p "$RELEASES_DIR" "$REMOTE_CONFIG_DIR"

if ! getent group "$REMOTE_RUN_GROUP" >/dev/null 2>&1; then
  groupadd --system "$REMOTE_RUN_GROUP"
fi
if ! id -u "$REMOTE_RUN_USER" >/dev/null 2>&1; then
  useradd --system --home "$REMOTE_APP_DIR" --gid "$REMOTE_RUN_GROUP" --shell /usr/sbin/nologin "$REMOTE_RUN_USER"
fi

rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"
tar -C "$RELEASE_DIR" -xzf "$REMOTE_TMP_ARCHIVE" --strip-components=1
chmod +x "$RELEASE_DIR/$APP_NAME"

if [[ ! -f "$REMOTE_ENV_FILE" ]]; then
  cp "$RELEASE_DIR/llmio.env.example" "$REMOTE_ENV_FILE"
fi
if [[ ! -f "$REMOTE_CONFIG_DIR/llmio.json" ]]; then
  cp "$RELEASE_DIR/llmio.json.example" "$REMOTE_CONFIG_DIR/llmio.json"
fi

install -m 0644 "$RELEASE_DIR/llmio.service" "$REMOTE_SERVICE_PATH"
ln -sfn "$RELEASE_DIR" "$CURRENT_LINK"
chown -R "$REMOTE_RUN_USER:$REMOTE_RUN_GROUP" "$REMOTE_APP_DIR" "$REMOTE_CONFIG_DIR"

systemctl daemon-reload
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"
systemctl --no-pager --full status "$SERVICE_NAME"

rm -f "$REMOTE_TMP_ARCHIVE"
EOF

  log "deploy finished: $SERVICE_NAME@$REMOTE_HOST"
}

main() {
  local command="${1:-}"
  case "$command" in
    package)
      package_release
      ;;
    deploy)
      deploy_release
      ;;
    ""|-h|--help|help)
      usage
      ;;
    *)
      echo "unknown command: $command" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
