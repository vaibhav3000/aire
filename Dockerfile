# Runtime image for the aire-gate CLI. Mount evaluated run directories and
# compare them, for example:
#   docker run --rm -v "$PWD/results:/data:ro" aire /data/v1_baseline /data/v2_improved --ignore latency_ms
# The base image is pinned by digest so rebuilds are reproducible.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 10001 aire
WORKDIR /app

# Build dependencies first: this layer survives source edits. AIRE itself has
# no runtime dependencies, and the dev extras (pytest, ruff, pyright) are not
# installed.
RUN pip install "setuptools>=68" wheel

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-build-isolation . && rm -rf /app/src /app/build

USER aire
ENTRYPOINT ["aire-gate"]
CMD ["--help"]
