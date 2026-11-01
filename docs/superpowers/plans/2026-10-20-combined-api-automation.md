# Combined API and Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Uvicorn and the automation daemon together in the API container, with an in-container command to stop or restart automation independently.

**Architecture:** Supervisor runs in the foreground and manages Uvicorn and `python -m app.automation.runner` as separate child processes. Both start with the container; Supervisor sends SIGTERM to automation when stopped, allowing the runner to stop claiming work and finish an active evaluation. Remove the standalone automation daemon service to keep one scheduler owner.

**Tech Stack:** Docker Compose, Debian Bookworm Supervisor, Uvicorn, Python automation runner.

**Spec:** `docs/superpowers/specs/2026-10-20-combined-api-automation-design.md`

## Global Constraints

- Keep the API bind address at `0.0.0.0:8000` and preserve the published API port.
- Keep the existing non-root container user, bind mounts, environment variables, and CRM cache mount.
- Run exactly one automation daemon against the shared ledger.
- Do not attach automation to FastAPI lifespan or add multiple Uvicorn workers.
- Send SIGTERM to automation so the runner stops claiming jobs and lets an active evaluation finish.
- Keep Supervisor’s control socket private to the container and write process logs to container output.

## Review Focus

- Manual stop must not stop Uvicorn; verify `/health` still returns HTTP 200 after `supervisorctl stop automation`.
- Manual start after stop must bring automation back to RUNNING without recreating the container.
- Container startup must start both services, but must not start a second automation daemon.
- Container shutdown must signal both children and let Supervisor reap them.
- Container user 1000 must be able to access the Supervisor socket and write existing app data/ledger paths.

---

### Task 1: Add Supervisor process definitions to the image

**Files:**
- Modify: `Dockerfile`
- Create: `supervisord.conf`

**Interfaces:**
- Produces process names `api` and `automation` for `supervisorctl` commands.
- `api` runs the existing Uvicorn command with reload enabled.
- `automation` runs `/opt/venv/bin/python -m app.automation.runner`.

- [ ] **Step 1: Add Supervisor to the Debian package install**

Add `supervisor` to the existing `apt-get install --no-install-recommends` list. Keep the reduced ODBC runtime package set and both Microsoft ODBC drivers.

- [ ] **Step 2: Create `supervisord.conf`**

Configure foreground mode, PID/log files in `/tmp`, a Unix control socket at `/tmp/supervisor.sock` with mode `0700`, and `supervisorctl` to use that socket. Define `api` and `automation` with autostart enabled, unexpected-exit restart, and stdout/stderr forwarded to container output. Set automation’s stop signal to SIGTERM and `stopwaitsecs=600` so a running evaluation has time to finish after the daemon stops claiming additional work.

- [ ] **Step 3: Copy the Supervisor config into the image and make it the default process manager**

Copy `supervisord.conf` to `/etc/supervisor/supervisord.conf`. Replace the Dockerfile CMD with `/usr/bin/supervisord -n -c /etc/supervisor/supervisord.conf`.

- [ ] **Step 4: Commit the image process-manager change**

Stage only `Dockerfile` and `supervisord.conf`; commit as `feat: supervise API and automation processes`.

### Task 2: Make the API service the single combined Compose mode

**Files:**
- Modify: `compose.yaml`
- Modify: `README.md`

**Interfaces:**
- `docker compose up -d api` starts both `api` and `automation` processes inside one container.
- `docker compose exec api supervisorctl stop automation` stops only automation.
- `docker compose exec api supervisorctl start automation` restarts automation.
- `docker compose exec api supervisorctl status` reports both process states.

- [ ] **Step 1: Remove the separate Compose automation daemon service**

Delete the `automation` service that launches `python -m app.automation.runner` separately. Preserve the API service’s port, user, environment, bind mounts, and CRM cache mount.

- [ ] **Step 2: Document combined startup and runtime controls**

Update the Docker development section to say the API container starts both processes and document the status, stop, and start commands above. Explain that stopping automation lets an active evaluation finish and can wait up to 600 seconds. Keep the existing API-only local `uvicorn` instructions unchanged. Document that one-shot runner commands remain available with `docker compose exec api python -m app.automation.runner <command>`.

- [ ] **Step 3: Commit the Compose and documentation change**

Stage only `compose.yaml` and `README.md`; commit as `docs: document combined API automation operation`.

### Task 3: Verify container startup and independent automation control

**Files:**
- Verify: `Dockerfile`, `supervisord.conf`, `compose.yaml`

- [ ] **Step 1: Validate Compose rendering**

Run: `docker compose config`
Expected: valid configuration with a single `api` service and no standalone automation service.

- [ ] **Step 2: Build and start the combined container**

Run: `docker compose up --build -d api`
Expected: build succeeds and the container starts.

- [ ] **Step 3: Check both managed processes**

Run: `docker compose exec api supervisorctl status`
Expected: `api` and `automation` both report `RUNNING`.

- [ ] **Step 4: Stop automation and confirm API remains available**

Run: `docker compose exec api supervisorctl stop automation`
Then run: `curl --fail http://localhost:${QA_API_PORT:-8001}/health`
Expected: automation reports `STOPPED`; API health returns HTTP 200.

- [ ] **Step 5: Restart automation without recreating the container**

Run: `docker compose exec api supervisorctl start automation`
Expected: `automation` reports `RUNNING` and Uvicorn stays running.

- [ ] **Step 6: Confirm non-root access to the control socket and ledger path**

Run: `docker compose exec api sh -c 'id && test -S /tmp/supervisor.sock && test -w /workspace/data'`
Expected: the process runs as UID 1000; the private control socket exists; the mounted data directory is writable.

- [ ] **Step 7: Stop the container and confirm both programs shut down**

Run: `docker compose stop api`
Then run: `docker compose logs --tail=100 api`
Expected: Supervisor stops both children, the automation runner logs `automation stopped`, and Uvicorn logs its normal shutdown.

- [ ] **Step 8: Commit any verification fixes with the owning task**

If verification requires a code/config correction, add it to the relevant task commit; do not stage unrelated working-tree changes.
