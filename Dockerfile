# --- Build stage ---
FROM python:3.12-slim AS builder

WORKDIR /build

COPY pyproject.toml uv.lock* ./
COPY src/ src/

RUN pip install --no-cache-dir build && \
    python -m build --wheel --outdir /build/dist

# --- Runtime stage ---
FROM python:3.12-slim

LABEL maintainer="Tremayne Timms <Ttimmsinternational@gmail.com>"
LABEL org.opencontainers.image.source="https://github.com/omnipotence-eth/godspeed-coding-agent"
LABEL org.opencontainers.image.description="Security-first open-source coding agent"
LABEL org.opencontainers.image.license="MIT"

# Non-root user
RUN groupadd --gid 1000 godspeed && \
    useradd --uid 1000 --gid godspeed --create-home godspeed

WORKDIR /workspace

# Install from built wheel
COPY --from=builder /build/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm -f /tmp/*.whl

# Switch to non-root
USER godspeed

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD ["godspeed", "version"]

ENTRYPOINT ["godspeed"]
