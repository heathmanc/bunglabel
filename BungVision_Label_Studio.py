#!/usr/bin/env python3
"""BungVision Label Studio launcher.

Use this file directly from the extracted folder:
    python main.py

This launcher forces the extracted application folder onto sys.path so the
bundled bung_labeler package is found even when launched from a shortcut or
with a different working directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    from bung_labeler.ui.main_window import main
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", "")
    if missing == "bung_labeler":
        raise SystemExit(
            "BungVision Label Studio could not find its bundled 'bung_labeler' folder.\n"
            "Make sure the zip is fully extracted before running, and run from the extracted folder.\n"
            "Recommended: double-click run_label_studio.bat on Windows, or run ./run_label_studio.sh on Linux."
        ) from exc
    raise

if __name__ == "__main__":
    main()
