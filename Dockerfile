FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml .
COPY arbiter/ arbiter/
RUN pip install --no-cache-dir .

# Test stage: install dev deps, run tests — must pass before production builds
FROM base AS test
RUN pip install --no-cache-dir -e ".[dev]"
COPY tests/ tests/
RUN pytest tests/ -v

# Production stage: clean image from base (no dev deps, no test code)
FROM base AS production

EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["python", "-m", "arbiter"]
