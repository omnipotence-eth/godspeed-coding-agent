# syntax=docker/dockerfile:1
FROM python:3.13-slim

LABEL org.opencontainers.image.title="Godspeed Coding Agent"
LABEL org.opencontainers.image.description="Security-first open-source coding agent"
LABEL org.opencontainers.image.source="https://github.com/omnipotence-eth/godspeed-coding-agent"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY settings.yaml.example ./settings.yaml

# Install dependencies and the package
RUN uv sync --all-extras

# Create non-root user for security
RUN useradd -m -u 1000 godspeed && chown -R godspeed:godspeed /app
USER godspeed

# Default working directory for the agent (mount your project here)
WORKDIR /workspace

# Entrypoint
ENTRYPOINT ["uv", "run", "python", "-m", "godspeed"]
CMD ["--help"]
