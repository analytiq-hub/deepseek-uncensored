# Dockerfile
FROM nvidia/cuda:12.0.0-base-ubuntu22.04

# Install basic dependencies
RUN apt-get update && apt-get install -y python3 python3-pip

# Create app directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose port 8080 for Cloud Run
EXPOSE 8080

# Start server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]