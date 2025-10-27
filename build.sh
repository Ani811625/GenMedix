#!/usr/bin/env bash
# exit on error
set -o errexit

# --- Set Cargo environment variables ---
export CARGO_HOME=/tmp/.cargo
export CARGO_TARGET_DIR=/tmp/.cargo-target
echo ">>> Setting Cargo directories to /tmp"

# --- Install System Dependencies ---
echo ">>> Updating apt and installing system dependencies..."
apt-get update && apt-get install -y \
  libpango-1.0-0 \
  libpangoft2-1.0-0 \
  libpangocairo-1.0-0 \
  libcairo2 \
  libgdk-pixbuf2.0-0 \
  libffi-dev \
  shared-mime-info \
  build-essential \
  python3-dev \
  --no-install-recommends
echo ">>> System dependencies installed."

# --- Upgrade pip ---
echo ">>> Upgrading pip..."
pip install --upgrade pip
echo ">>> pip upgraded."

# --- Install Python dependencies ---
echo ">>> Installing Python requirements..."
pip install -r requirements.txt
echo ">>> Python requirements installed."

echo ">>> Build script completed successfully!"