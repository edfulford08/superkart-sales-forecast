# Use a minimal base image with Python 3.9 installed
FROM python:3.9-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Install dependencies first so that Docker can cache this layer
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all files from the current directory on the host to the container's /app directory
COPY . .

# On the shared Docker network the backend container is reachable by its name "backend"
ENV BACKEND_URL=http://backend:7860

# The Streamlit app listens on port 8501 (mapped to the same port of the Codespace)
EXPOSE 8501

# Run the Streamlit app on port 8501 and make it accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--server.enableCORS=false"]

# NOTE: XSRF protection and CORS are disabled so that CSV uploads for batch predictions work
# through the public forwarded port of a GitHub Codespace (this demo app has no login).
