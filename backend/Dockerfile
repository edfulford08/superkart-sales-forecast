# Base image: Python 3.13 (same minor version as the training session), pinned to a digest when it could be resolved
ARG BASE_IMAGE=python:3.13-slim@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f
FROM ${BASE_IMAGE}

# Do not write .pyc files and show log lines immediately
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Set the working directory inside the container to /app
WORKDIR /app

# Install dependencies first so that Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 appuser

# Copy the application code, the shared feature module, the metadata, and the serialized model
COPY --chown=appuser:appuser . .
USER appuser

# The Flask API listens on port 7860 (mapped to the same port of the Codespace)
EXPOSE 7860

# Docker reports the container as unhealthy if the health route stops answering
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=4)"

# Serve the Flask app with gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
