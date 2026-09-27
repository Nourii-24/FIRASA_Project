# Firasa scoring API -- Phase 6: containerize the model, expose it via FastAPI.
#
# Build (from the Firasa/ project root, next to this file):
#   docker build -t firasa-api .
#
# Run:
#   docker run --rm -p 8000:8000 firasa-api
#
# Then open http://localhost:8000/docs

FROM python:3.11-slim AS base

# Keep Python from writing .pyc files / buffering stdout, standard for containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code and the trained model artifacts
COPY api/ ./api/
COPY models/ ./models/

# Where the drift monitor (api/drift.py) appends its JSONL log. Created
# here so it exists (and is writable by the non-root user below) even if
# no host volume is mounted over it -- see docker-compose.yml for the
# volume that makes drift history survive a container restart.
RUN mkdir -p /app/logs

# Runs as a non-root user inside the container
RUN useradd --create-home --uid 1000 firasa && chown -R firasa:firasa /app
USER firasa

EXPOSE 8000

# Lets Docker/Azure know the container is actually serving, not just running
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
