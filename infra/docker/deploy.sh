#!/usr/bin/env bash
# One-command deploy to a single VM (Docker + Caddy).
#
# Usage:
#   ./infra/docker/deploy.sh root@1.2.3.4 chat.example.com you@example.com
#
# Arg 1: SSH target (user@host)
# Arg 2: DOMAIN (A record must already point at the server)
# Arg 3: CADDY_EMAIL (Let's Encrypt notifications)
#
# Family data and .env are read from $MEMORIAL_DATA_HOME (default ~/MemorialData),
# never from the repo. This sends family material to the server — only run it
# against a machine the family trusts.
#
# Bot data is copied with --ignore-existing: the first deploy seeds the server,
# later deploys never overwrite memories the family saved there.
# Set SYNC_PERSONA=1 to push an edited persona file anyway.

set -euo pipefail

SSH_TARGET="${1:?SSH target required}"
DOMAIN="${2:?DOMAIN required}"
CADDY_EMAIL="${3:?CADDY_EMAIL required}"
REMOTE_DIR="${REMOTE_DIR:-/opt/rony}"
DATA_ROOT="${DATA_ROOT:-/opt/rony-data}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MEMORIAL_DATA_HOME="${MEMORIAL_DATA_HOME:-${HOME}/MemorialData}"
LOCAL_DATA="${MEMORIAL_DATA_HOME}/data"
LOCAL_ENV="${MEMORIAL_DATA_HOME}/.env"

if [[ ! -f "${LOCAL_ENV}" ]]; then
    echo "Missing ${LOCAL_ENV} — copy .env.example there and fill it in." >&2
    exit 1
fi
for key in FAMILY_ACCESS_KEY ADMIN_ACCESS_KEY; do
    if ! grep -qE "^${key}=.+" "${LOCAL_ENV}"; then
        echo "${key} must be set in .env before deploying." >&2
        exit 1
    fi
done

echo "==> Ensuring Docker is installed on ${SSH_TARGET}"
ssh "${SSH_TARGET}" 'command -v docker >/dev/null 2>&1 || curl -fsSL https://get.docker.com | sh'

echo "==> Syncing code to ${SSH_TARGET}:${REMOTE_DIR}"
ssh "${SSH_TARGET}" "mkdir -p ${REMOTE_DIR} ${DATA_ROOT}"
rsync -az --delete \
    --exclude ".git" \
    --exclude ".backups" \
    --exclude "node_modules" \
    --exclude ".next" \
    --exclude ".venv" \
    --exclude "apps/api/data" \
    --exclude ".env" \
    --exclude "infra/docker/.env" \
    --exclude "__pycache__" \
    --exclude "*.pyc" \
    --exclude ".DS_Store" \
    "${REPO_ROOT}/" "${SSH_TARGET}:${REMOTE_DIR}/"
rsync -a --chmod=F600 "${LOCAL_ENV}" "${SSH_TARGET}:${REMOTE_DIR}/.env"

echo "==> Seeding bot data to ${DATA_ROOT} (existing server files are kept)"
for item in uploads whatsapp avatars memories personas persons.json; do
    if [[ -e "${LOCAL_DATA}/${item}" ]]; then
        rsync -az --ignore-existing "${LOCAL_DATA}/${item}" "${SSH_TARGET}:${DATA_ROOT}/"
    fi
done
if [[ "${SYNC_PERSONA:-0}" == "1" ]]; then
    rsync -az "${LOCAL_DATA}/personas/" "${SSH_TARGET}:${DATA_ROOT}/personas/"
fi

echo "==> Writing infra/docker/.env on server"
ssh "${SSH_TARGET}" "cat > ${REMOTE_DIR}/infra/docker/.env" <<EOF
DOMAIN=${DOMAIN}
CADDY_EMAIL=${CADDY_EMAIL}
DATA_ROOT=${DATA_ROOT}
EOF

COMPOSE="cd ${REMOTE_DIR}/infra/docker && docker compose --env-file .env -f docker-compose.prod.yml"

echo "==> Building + starting stack"
ssh "${SSH_TARGET}" "${COMPOSE} up -d --build"
ssh "${SSH_TARGET}" "${COMPOSE} ps"

echo
echo "Done. First deploy only — build the vector index on the server:"
echo "  ssh ${SSH_TARGET} '${COMPOSE} exec api python -m scripts.reingest <person_id>'"
echo "Test: curl https://${DOMAIN}/health"
