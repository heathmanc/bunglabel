#!/usr/bin/env bash
set -e
APPDIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$APPDIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$APPDIR"
python3 "$APPDIR/main.py"
