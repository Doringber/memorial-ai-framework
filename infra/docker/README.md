# Rony — production deploy (single VM + Docker + Caddy)

One server runs three containers:

- **caddy** — HTTPS (Let's Encrypt). `/api/*` and `/health` go to the API,
  everything else to the web app. Same origin, so no CORS setup is needed.
- **api** — FastAPI. Requires `FAMILY_ACCESS_KEY` and `ADMIN_ACCESS_KEY`
  (`APP_ENV=production` refuses to start without them). API docs are disabled.
- **web** — Next.js standalone server.

> The app is local-first: the default `.env` uses Ollama on `127.0.0.1` and no
> voice. The prod compose has no Ollama container, so a server deploy needs
> either Ollama installed there or `ALLOW_EXTERNAL_SERVICES=true` with a hosted
> provider. Deploying copies family material to the server — get the family's
> agreement first.

All bot data (uploads, memories, personas, `persons.json`, Chroma) lives in one
host directory, `DATA_ROOT` (default `/opt/rony-data`), mounted at `/app/data`.
Hugging Face models are cached in the `hfcache` volume.

## One-time setup

1. Provision a server (e.g. Hetzner CX32 — 4 vCPU / 8 GB RAM). Ubuntu 24.04,
   SSH key auth, ports 22/80/443 open.
2. Point an `A` record for your domain at the server and wait until
   `dig <domain>` returns it (otherwise certificate issuance fails).
3. Fill in `~/MemorialData/.env` (from `.env.example`), including both access keys.

## Deploy

From the repo root on your laptop:

```bash
./infra/docker/deploy.sh root@<SERVER_IP> chat.example.com you@example.com
```

This installs Docker if missing, rsyncs the code to `/opt/rony`, copies
`~/MemorialData/.env` to `/opt/rony/.env`, seeds `DATA_ROOT` from
`~/MemorialData/data`, writes `infra/docker/.env` (domain, email, `DATA_ROOT`)
and runs `docker compose up -d --build`.

Data seeding uses `rsync --ignore-existing`: the first deploy copies your local
data, later deploys never overwrite memories the family saved on the server.
To push an edited persona file anyway:

```bash
SYNC_PERSONA=1 ./infra/docker/deploy.sh root@<SERVER_IP> chat.example.com you@example.com
```

Chroma is not copied. After the first deploy, build the index on the server:

```bash
ssh root@<SERVER_IP>
cd /opt/rony/infra/docker
docker compose --env-file .env -f docker-compose.prod.yml exec api \
    python -m scripts.reingest <person_id>
```

The first build pulls ~1.5 GB of Python deps; the e5 and reranker models
(~3 GB) download on first use, so the first chat after a cold start is slow.

## Check

```bash
curl https://chat.example.com/health
```

Then open the site and enter the family password.

## Local smoke test of the production stack

```bash
ENV_FILE=~/MemorialData/.env DATA_ROOT=~/MemorialData/data docker compose \
    -f infra/docker/docker-compose.prod.yml -f infra/docker/docker-compose.local.yml up --build
```

API on http://127.0.0.1:8080, web on http://127.0.0.1:3000 (this machine only),
Caddy skipped. Ollama is reached on the host via `host.docker.internal`.

## Operations

```bash
cd /opt/rony/infra/docker
C="docker compose --env-file .env -f docker-compose.prod.yml"

$C logs -f api          # logs
$C restart api          # restart API only
$C up -d --build        # rebuild after a deploy
```

## Backups

Everything that matters is in `DATA_ROOT`. Nightly cron on the host:

```bash
tar -C /opt -czf /root/rony-$(date +%F).tar.gz rony-data
```

Ship the tarball off the server (e.g. `rclone` to an S3-compatible bucket).
Memories are also replayable into a fresh index with `scripts.reingest`.
