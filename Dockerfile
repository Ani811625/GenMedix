# Use an official Python 3.11 base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install the system-level libraries that WeasyPrint needs
# This is the most important step
RUN apt-get update && apt-get install -y \
    libgobject-2.0-0 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project code into the container
COPY . .

# Tell the container what command to run when it starts
# We use our Procfile command here, but explicitly
CMD ["gunicorn", "app:app"]