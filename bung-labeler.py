#!/usr/bin/env python3
"""Compatibility launcher for older commands/shortcuts.

Prefer: python main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from bung_labeler.ui.main_window import main

if __name__ == "__main__":
    main()
