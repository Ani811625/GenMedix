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

# --- NEW LINES ---
# This tells Docker to expect a build-time argument named DATABASE_URL
ARG DATABASE_URL
# This sets the environment variable inside the container
ENV DATABASE_URL=${DATABASE_URL}
# --- END NEW LINES ---

# Copy your requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project code into the container
COPY . .

# Make the new start script executable
RUN chmod +x ./start.sh

# This tells the container to run your new script on startup
CMD ["./start.sh"]