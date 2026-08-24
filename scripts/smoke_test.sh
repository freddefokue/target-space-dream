#!/usr/bin/env bash
set -euo pipefail

python -m dream.cli
python -m pytest

