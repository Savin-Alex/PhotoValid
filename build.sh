#!/bin/bash
set -euo pipefail

python --version
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
