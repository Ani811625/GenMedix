#!/usr/bin/env bash
# exit on error
set -o errexit

# Install WeasyPrint system dependencies
apt-get update && apt-get install -y \
  libpango-1.0-0 \
  libpangoft2-1.0-0 \
  libpangocairo-1.0-0 \
  libcairo2 \
  libgdk-pixbuf2.0-0 \
  libffi-dev \
  shared-mime-info \
  --no-install-recommends

# Install Python dependencies
pip install -r requirements.txt