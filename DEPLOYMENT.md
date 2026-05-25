# Deploying PhotoValid on Render

The deployment is defined by [`render.yaml`](render.yaml) (Infrastructure-as-Code).
**That file is the source of truth** — don't hand-copy its settings into docs, or
they drift. This guide explains the surrounding context only.

## One-time setup

1. Push the repo to GitHub.
2. On https://render.com → **New + → Blueprint**, connect the repo. Render reads
   `render.yaml` and creates the service automatically.
3. Click **Apply**. First build takes ~5–10 min (MediaPipe + OpenCV wheels).

## What `render.yaml` sets

- **Runtime**: Python **3.11.9** (`PYTHON_VERSION` env var; also pinned in `runtime.txt`).
  MediaPipe must be on a supported Python — do not let Render fall back to 3.13.
- **Build**: `pip install -r backend/requirements.txt`
- **Start**: `python -m uvicorn backend.main:app --host 0.0.0.0 --port 10000`
- **PYTHONPATH** = repo root, so `backend.main` and the static `frontend/` resolve.

## After deploy

- **App (frontend + API)**: `https://<service>.onrender.com/`
- **Validation endpoint**: `POST https://<service>.onrender.com/api/validate`
- **Liveness probe**: `GET /` returns `200` (it serves the UI). There is no
  separate JSON health endpoint; `GET /api/validate` returns `404` (the catch-all
  static mount at `/` handles it), so use `GET /` for health checks.

```bash
curl -I https://<service>.onrender.com           # GET / -> 200 (UI loads)
curl -s -o /dev/null -w '%{http_code}\n' \
  https://<service>.onrender.com/api/validate     # GET -> 404 (POST-only endpoint)
```

## Locking down CORS (optional)

The API is credential-free and defaults to `allow_origins=*`. To restrict it,
set the `CORS_ORIGINS` env var (comma-separated) on the Render service, e.g.
`CORS_ORIGINS=https://yourdomain.com`.

## Free-tier notes

| Resource | Free tier |
|----------|-----------|
| Uptime   | sleeps after ~15 min idle (cold start on next request) |
| Memory   | 512 MB |
| Build    | 500 min/month |

For always-on / more memory, upgrade the service plan in Render.

## Troubleshooting

- **Build fails on MediaPipe** → confirm Python resolved to 3.11.9 (check the build
  log's `python --version`); MediaPipe has no wheels for 3.13.
- **App won't start** → verify the start command targets `backend.main:app` and that
  `PYTHONPATH` includes the repo root.
- **Frontend 404** → ensure `frontend/index.html` exists; `backend/main.py` mounts it at `/`.
