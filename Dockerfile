FROM python:3.12.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Both ODBC versions are used by the project's existing passcode files and .env.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg libgomp1 libgssapi-krb5-2 supervisor tzdata unixodbc \
    && install -d -m 0755 /etc/apt/keyrings \
    && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /etc/apt/keyrings/microsoft.gpg \
    && . /etc/os-release \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/${VERSION_ID}/prod ${VERSION_CODENAME} main" \
        > /etc/apt/sources.list.d/microsoft-prod.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql17 msodbcsql18

WORKDIR /workspace
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv "$VIRTUAL_ENV" \
    && python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

# A few existing utilities still use this absolute path. Point it at the
# bind-mounted checkout so those calls see the same live files.
RUN mkdir -p /home/ai/Workspace/Rafik \
    && ln -s /workspace /home/ai/Workspace/Rafik/QA_System-main

# Compose overlays this copy with the local checkout for immediate file updates.
# The .dockerignore keeps credentials and local data out of the image.
COPY . .

COPY supervisord.conf /etc/supervisor/supervisord.conf

EXPOSE 8000
CMD ["/usr/bin/supervisord", "-n", "-c", "/etc/supervisor/supervisord.conf"]
