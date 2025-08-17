# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv and upgrade pip/setuptools
RUN pip install --upgrade pip setuptools wheel && pip install uv

# Copy pyproject.toml and uv.lock
COPY pyproject.toml uv.lock ./

# Create src directory and install dependencies using uv
RUN mkdir -p src && uv sync --frozen --no-dev

# Copy source code
COPY src/ ./src/

# Pre-download models/assets at build time
RUN uv run python src/agent.py download-files

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Expose port for health checks (optional)
EXPOSE 8081

# Run the agent
CMD ["uv", "run", "python", "src/agent.py"]
