# Use an official lightweight Python runtime as a base image
FROM python:3.10-slim

# Set system environment variables to optimize Python within the container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install essential system-level compilation dependencies required for ChromaDB / SQLite build tasks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Establish the working directory inside the container space
WORKDIR /app

# Copy the requirements manifest first to leverage Docker's caching mechanism
COPY requirements.txt .

# Install all listed Python engineering packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all local project source directories into the container filesystem
COPY . .

# Expose network ports used by FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# The execution target entry point is handled dynamically via Docker Compose profiles
