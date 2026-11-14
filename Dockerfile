# Project Tabarru API and adjuster view. FakeProvider is the default.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Locked runtime dependencies only (no dev group), from uv.lock.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY claims ./claims
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN useradd --create-home --uid 10001 tabarru \
    && mkdir -p /data && chown tabarru:tabarru /data \
    && chmod 0755 /usr/local/bin/entrypoint.sh
USER tabarru

ENV CLAIMS_DATA=/data \
    CLAIMS_REGISTER=/data/policies.jsonl \
    CLAIMS_DB=/data/claims.db

EXPOSE 8000
ENTRYPOINT ["entrypoint.sh"]
CMD ["uvicorn", "--factory", "claims.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
