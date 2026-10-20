# Combined API and Automation Container Design

Date: 2026-10-20

## Goal

Run the QA API (`uvicorn`) and the automation daemon together in one Compose container. Allow an operator to stop or restart automation from inside that container while keeping the API available.

## Current behavior

- Compose starts `api` with Uvicorn.
- The optional `automation` service starts `python -m app.automation.runner` in a separate container.
- The runner already handles `SIGINT` and `SIGTERM`. Its worker loop checks a shutdown flag between jobs, allowing an active evaluation to finish before the daemon exits.
- The runner documentation explicitly avoids starting the daemon inside FastAPI lifespan because multiple Uvicorn workers could create duplicate schedulers.

## Design

Use Supervisor as the container process manager. Keep Uvicorn and the automation daemon as separate child processes in the existing API service. Both programs autostart when the container starts. Do not attach the runner to FastAPI lifespan.

Supervisor will run in foreground as the container command. Its program entries will use the existing Uvicorn reload command and `python -m app.automation.runner`. It will send `SIGTERM` to the automation process and allow the runner to shut down gracefully. Supervisor control socket and logs will live in `/tmp` and standard container output so they work with the configured non-root user and `docker compose logs`.

Remove the separate Compose automation daemon service to prevent two schedulers from running accidentally. Keep one-shot runner commands available inside the API container, for example `docker compose exec api python -m app.automation.runner status`.

## Operator commands

Start both processes:

```bash
docker compose up --build -d api
```

Check process state:

```bash
docker compose exec api supervisorctl status
```

Stop or restart only automation while leaving Uvicorn running:

```bash
docker compose exec api supervisorctl stop automation
docker compose exec api supervisorctl start automation
```

Stopping the container continues to stop both processes through normal Compose lifecycle handling.

## Files expected to change during implementation

- `Dockerfile`: install Supervisor.
- `compose.yaml`: run Supervisor as the API container command and remove the separate daemon service.
- `supervisord.conf`: add API and automation process definitions, signal handling, log forwarding, and control socket settings.
- `README.md`: document combined startup and in-container automation controls.

## Risks and constraints

- Only one automation daemon may run against a shared ledger at a time; the combined service is the single owner.
- Stopping automation requests graceful shutdown. It stops claiming new jobs and lets a currently running evaluation finish; the operator command may wait for that job.
- Supervisor must run without root privileges and expose its control socket only inside the container.
- Existing environment, bind mounts, CRM cache mount, and API port mapping should be preserved.

## Out of scope

- Adding an HTTP endpoint to control automation.
- Changing scheduling, planning, dispatching, worker, or ledger semantics.
- Supporting multiple Uvicorn workers in the combined service.
