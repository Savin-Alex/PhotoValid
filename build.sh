#!/bin/bash
# Build script for Render deployment with Python 3.11

# Install Python 3.11 if not available
if ! command -v python3.11 &> /dev/null; then
    echo "Installing Python 3.11..."
    # This might not work on Render, but worth trying
    curl -fsSL https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz | tar xz
fi

# Install dependencies
pip install -r backend/requirements.txt
