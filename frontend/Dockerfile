# Base image: same pinned image as the backend
ARG BASE_IMAGE=python:3.13-slim@sha256:70729b46c69b4f1e97c4822c1af3df53a1476cf5ddc6c087c0c10bc3a5678c2f
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Set the working directory inside the container to /app
WORKDIR /app

# Install dependencies first so that Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 appuser

# Copy all files from the current directory on the host to the container's /app directory
COPY --chown=appuser:appuser . .
USER appuser

# On the shared Docker network the backend container is reachable by its name "backend"
ENV BACKEND_URL=http://backend:7860

# The Streamlit app listens on port 8501 (mapped to the same port of the Codespace)
EXPOSE 8501

# Docker reports the container as unhealthy if Streamlit stops answering
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"

# Run the Streamlit app on port 8501 and make it accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--server.enableCORS=false"]

# NOTE: XSRF protection and CORS are disabled so that CSV uploads for batch predictions work
# through the public forwarded port of a GitHub Codespace (this demo app has no login).
