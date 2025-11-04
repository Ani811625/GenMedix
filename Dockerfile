# Use an official Python 3.11 base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system-level libraries
RUN apt-get update && apt-get install -y \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libffi-dev \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project code into the container
COPY . .

# -----------------
# --- CHANGES ---
# -----------------

# 1. DELETE this old command:
# CMD ["gunicorn", "app:app"]

# 2. ADD these new commands instead:
# This makes your new start.sh script executable
RUN chmod +x ./start.sh

# This tells the container to run your new script on startup
CMD ["./start.sh"]