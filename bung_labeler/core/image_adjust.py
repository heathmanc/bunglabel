from __future__ import annotations

import cv2
import numpy as np


def apply_adjustments(
    frame_bgr: np.ndarray,
    brightness: int = 0,
    contrast: int = 0,
    gamma: float = 1.0,
    clahe_enabled: bool = False,
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
    sharpen: int = 0,
) -> np.ndarray:
    """Apply non-destructive preview adjustments to a BGR frame.

    brightness: -100..100
    contrast: -100..100
    gamma: 0.20..3.00
    sharpen: 0..100
    """
    if frame_bgr is None:
        raise ValueError("frame_bgr cannot be None")

    img = frame_bgr.copy()

    alpha = 1.0 + (contrast / 100.0)
    beta = brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if gamma <= 0:
        gamma = 1.0
    if abs(gamma - 1.0) > 0.01:
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
        img = cv2.LUT(img, table)

    if clahe_enabled:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        grid = max(2, int(clahe_grid))
        clahe = cv2.createCLAHE(clipLimit=max(0.5, float(clahe_clip)), tileGridSize=(grid, grid))
        l2 = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

    if sharpen > 0:
        amount = sharpen / 100.0
        blurred = cv2.GaussianBlur(img, (0, 0), 1.2)
        img = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)

    return img


def bgr_to_qimage_rgb_bytes(frame_bgr: np.ndarray) -> tuple[bytes, int, int, int]:
    """Return RGB bytes and image dimensions for safe QImage creation."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return rgb.tobytes(), w, h, ch * w
