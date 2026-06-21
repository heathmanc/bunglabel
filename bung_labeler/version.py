"""Single source of truth for the application name and version.

Keeping these here avoids the version-string drift that previously required
editing the window title and the review-stamp string separately.
"""
from __future__ import annotations

APP_NAME = "BungVision Label Studio"
APP_VERSION = "0.9.58"
APP_TITLE = f"{APP_NAME} v{APP_VERSION}"
