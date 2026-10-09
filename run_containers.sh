#!/usr/bin/env bash
# Build and start the SuperKart backend and frontend containers (run this inside the GitHub Codespace).
# Usage: bash run_containers.sh
set -euo pipefail
cd "$(dirname "$0")"

fail() { echo "ERROR: $1" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || fail "docker is not installed or not on the PATH"
docker info >/dev/null 2>&1 || fail "the Docker daemon is not running"
[ -f backend/Dockerfile ] && [ -f frontend/Dockerfile ] || fail "run this script from the repository root (backend/ and frontend/ not found)"
ls backend/*.joblib >/dev/null 2>&1 || fail "the model file (.joblib) is missing from the backend folder"

echo "Building the backend image..."
docker build -t superkart-backend ./backend || fail "the backend image did not build (see the messages above)"

echo "Building the frontend image..."
docker build -t superkart-frontend ./frontend || fail "the frontend image did not build (see the messages above)"

# Shared network so the frontend can reach the backend by its container name
docker network inspect superkart-app-network >/dev/null 2>&1 || docker network create superkart-app-network

# Remove containers from an earlier run so that the names and ports are free
docker rm -f backend frontend >/dev/null 2>&1 || true

docker run -d --name backend --network superkart-app-network -p 7860:7860 superkart-backend || fail "the backend container did not start"
docker run -d --name frontend --network superkart-app-network -p 8501:8501 superkart-frontend || fail "the frontend container did not start"

wait_for() {
  # wait_for NAME URL: poll a health URL for up to 60 seconds
  for _ in $(seq 1 30); do
    if curl -fsS "$2" >/dev/null 2>&1; then echo "$1 is up"; return 0; fi
    sleep 2
  done
  echo "$1 did not respond within 60 seconds. Inspect it with: docker logs $3" >&2
  return 1
}

wait_for "Backend" "http://localhost:7860/" backend || fail "backend health check failed"
wait_for "Frontend" "http://localhost:8501/_stcore/health" frontend || fail "frontend health check failed"

docker ps
echo "Both containers are running. In the PORTS tab make ports 7860 and 8501 Public, then open the 8501 address."
