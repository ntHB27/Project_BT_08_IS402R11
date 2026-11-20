# ==============================================================================
# Fed-LCCF Production Container Image
# Multi-platform support: linux/amd64, linux/arm64
# Base: Debian 12 Bookworm slim with glibc 2.36
# ==============================================================================

FROM python:3.12-slim

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install system compilation packages for C extensions fallback and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    libxml2-dev \
    zlib1g-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pre-install dependencies to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy complete Fed-LCCF prototype codebase
COPY . /app/

# Ensure runtime directories exist
RUN mkdir -p /app/orchestrator/logs /app/data

EXPOSE 8000

# Container liveness health check
HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default execution entrypoint: Cloud Aggregation Server
CMD ["python", "-m", "cloud_server.server", "--host", "0.0.0.0", "--port", "8000"]
