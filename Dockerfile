# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Install system dependencies required for WeasyPrint (PDF generation)
# WeasyPrint needs Pango, GDK-Pixbuf, and other GTK libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-cffi \
    python3-brotli \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libharfbuzz-subset0 \
    libjpeg-dev \
    libopenjp2-7-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file first (for caching)
COPY requirements.txt .

# --- FIX FOR BROKEN PIPE / MEMORY ERRORS ---

# 1. Upgrade pip to the latest version
RUN pip install --upgrade pip

# 2. Install heavy Machine Learning libraries first (individually)
# This prevents RAM spikes that cause Render to kill the connection
RUN pip install --no-cache-dir --default-timeout=1000 numpy pandas scikit-learn

# 3. Install the rest of the requirements
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# --- END OF FIX ---

# Copy the rest of your project code into the container
COPY . .

# Expose port 5000
EXPOSE 5000

# Run the command to start Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]