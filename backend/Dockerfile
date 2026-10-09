# Use a minimal base image with Python 3.9 installed
FROM python:3.9-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Install dependencies first so that Docker can cache this layer
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code and the serialized model
COPY . .

# The Flask API listens on port 7860 (mapped to the same port of the Codespace)
EXPOSE 7860

# Serve the Flask app with gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
