from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CAPTURE_DIR = DATA_DIR / "captures"
LABEL_DIR = DATA_DIR / "labels"
RECIPE_DIR = DATA_DIR / "recipes"
EXPORT_DIR = DATA_DIR / "exports"
CLASS_CONFIG_PATH = DATA_DIR / "class_config.json"
CAMERA_SETTINGS_PATH = DATA_DIR / "camera_settings.json"
TRAINING_SETTINGS_PATH = DATA_DIR / "training_settings.json"

for d in (CAPTURE_DIR, LABEL_DIR, RECIPE_DIR, EXPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Broad equipment category recipes fall under by default. The default category
# is special-cased so legacy recipes keep their original on-disk safe_name.
DEFAULT_CATEGORY = "General"


DEFAULT_CLASSES = [
    {"id": 0, "name": "battery", "default_tool": "OBB", "enabled": True, "role": "battery"},
    {"id": 1, "name": "bung", "default_tool": "OBB", "enabled": True, "role": "bung"},
    {"id": 2, "name": "retainer", "default_tool": "OBB", "enabled": True, "role": "retainer"},
]

DEFAULT_CAMERA_SETTINGS: dict[str, Any] = {
    "camera_source": "0",
    "camera_backend": "V4L2",
    "width": 2592,
    "height": 1944,
    "fps": 0,
    "preview_scale": "1/2",
    "exposure_auto": True,
    "exposure_us": 0,
    "force_v4l2": True,
    "low_latency": True,
    "threaded_camera": True,
    "mjpg": True,
    "skip_heavy_live": True,
}

def infer_role_and_layout(name: str) -> tuple[str, str]:
    """Infer a simple class role from the class name for old configs and exports.

    The second return value is retained as "none" only so older call sites
    and configs remain compatible.
    """
    n = str(name or "").strip().lower()
    if n == "battery" or n.startswith("battery"):
        return "battery", "none"
    if n == "bung" or n.startswith("bung"):
        return "bung", "none"
    if n == "retainer" or n.startswith("retainer"):
        return "retainer", "none"
    return "custom", "none"

def load_class_config() -> list[dict[str, Any]]:
    if not CLASS_CONFIG_PATH.exists():
        save_class_config(DEFAULT_CLASSES)
        return [dict(c) for c in DEFAULT_CLASSES]
    try:
        data = json.loads(CLASS_CONFIG_PATH.read_text(encoding="utf-8"))
        classes = data.get("classes", data if isinstance(data, list) else [])
        out = []
        used = set()
        for c in classes:
            cid = int(c.get("id", len(out)))
            name = str(c.get("name", f"class_{cid}")).strip() or f"class_{cid}"
            # Drop the old default layout helper classes from v0.9.21-v0.9.23.
            # Battery/bung labeling is now plain OBB, not layout-zone driven.
            if name.lower() in {"battery_6row", "battery_2x3"}:
                continue
            if cid in used:
                continue
            used.add(cid)
            role, _layout = infer_role_and_layout(name)
            default_tool = str(c.get("default_tool", "OBB")).upper()
            if default_tool not in {"OBB", "BOX"}:
                default_tool = "OBB"
            out.append({
                "id": cid,
                "name": name,
                "default_tool": default_tool,
                "enabled": bool(c.get("enabled", True)),
                "role": str(c.get("role", role)).lower(),
                # v0.9.38: old custom classes created before this version were
                # hardcoded as BOX.  tool_locked distinguishes a deliberate
                # v0.9.38+ Box fallback choice from that old default.
                "tool_locked": bool(c.get("tool_locked", False)),
            })
        if not out:
            return [dict(c) for c in DEFAULT_CLASSES]

        # OBB is now the normal workflow.  Built-in part classes always use
        # OBB, and old pre-v0.9.38 custom classes that were automatically
        # saved as BOX are migrated to OBB.  Deliberate v0.9.38+ Box fallback
        # choices are preserved with tool_locked=True.
        for c in out:
            role = str(c.get("role", "")).lower()
            name = str(c.get("name", "")).lower()
            if role in {"battery", "bung", "retainer"} or name in {"battery", "bung", "retainer"}:
                c["default_tool"] = "OBB"
            elif str(c.get("default_tool", "OBB")).upper() == "BOX" and not c.get("tool_locked", False):
                c["default_tool"] = "OBB"

        return sorted(out, key=lambda c: int(c["id"]))
    except Exception:
        return [dict(c) for c in DEFAULT_CLASSES]


def save_class_config(classes: list[dict[str, Any]]) -> Path:
    payload = {"classes": sorted(classes, key=lambda c: int(c.get("id", 0)))}
    CLASS_CONFIG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return CLASS_CONFIG_PATH


def class_names_from_config(classes: list[dict[str, Any]]) -> list[str]:
    enabled = [c for c in sorted(classes, key=lambda c: int(c.get("id", 0))) if c.get("enabled", True)]
    max_id = max([int(c.get("id", 0)) for c in enabled], default=-1)
    names = [f"class_{i}" for i in range(max_id + 1)]
    for c in enabled:
        names[int(c["id"])] = str(c["name"])
    return names or ["battery", "bung"]


def load_camera_settings() -> dict[str, Any]:
    settings = dict(DEFAULT_CAMERA_SETTINGS)
    if CAMERA_SETTINGS_PATH.exists():
        try:
            data = json.loads(CAMERA_SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                settings.update({k: data[k] for k in settings if k in data})
        except Exception:
            pass
    return settings


def save_camera_settings(settings: dict[str, Any]) -> Path:
    payload = dict(DEFAULT_CAMERA_SETTINGS)
    payload.update({k: settings[k] for k in payload if k in settings})
    CAMERA_SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return CAMERA_SETTINGS_PATH


def load_training_settings() -> dict[str, Any]:
    """Last-used YOLO training parameters, persisted between sessions."""
    if TRAINING_SETTINGS_PATH.exists():
        try:
            data = json.loads(TRAINING_SETTINGS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_training_settings(settings: dict[str, Any]) -> Path:
    TRAINING_SETTINGS_PATH.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return TRAINING_SETTINGS_PATH


@dataclass
class Recipe:
    group: str
    model: str
    # Broad equipment category above group/model. Lets one install hold recipes
    # for several machines and load/browse them separately. The default category
    # keeps the legacy on-disk safe_name (group__model) so pre-category captures
    # and labels stay attached to their recipes.
    category: str = DEFAULT_CATEGORY
    expected_bungs: int = 6
    # When False, the recipe is unlocked from the battery/bung quantity check
    # so the tool can label arbitrary object classes (free-form labeling).
    constrained: bool = True
    brightness: int = 0
    contrast: int = 0
    gamma: float = 1.0
    clahe_enabled: bool = False
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    sharpen: int = 0
    notes: str = ""

    @property
    def safe_name(self) -> str:
        def clean(s: str) -> str:
            return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s.strip()) or "Unnamed"
        base = f"{clean(self.group)}__{clean(self.model)}"
        if str(self.category).strip() in ("", DEFAULT_CATEGORY):
            # Legacy form: keeps existing capture/label folders working.
            return base
        return f"{clean(self.category)}__{base}"


def recipe_path(group: str, model: str, category: str = DEFAULT_CATEGORY) -> Path:
    r = Recipe(group=group, model=model, category=category)
    return RECIPE_DIR / f"{r.safe_name}.json"


def save_recipe(recipe: Recipe) -> Path:
    path = RECIPE_DIR / f"{recipe.safe_name}.json"
    path.write_text(json.dumps(asdict(recipe), indent=2), encoding="utf-8")
    return path


def load_recipe(path: Path) -> Recipe:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Older builds stored app-wide camera settings in each recipe. Ignore those
    # keys so loading a recipe never changes the current camera setup.
    recipe_fields = set(Recipe.__dataclass_fields__)
    data = {k: v for k, v in data.items() if k in recipe_fields}
    return Recipe(**data)


def list_recipes() -> list[Recipe]:
    recipes: list[Recipe] = []
    for p in sorted(RECIPE_DIR.glob("*.json")):
        try:
            recipes.append(load_recipe(p))
        except Exception:
            continue
    return recipes


def recipe_category(recipe: Recipe) -> str:
    """Category for a recipe, falling back to the default for legacy recipes."""
    cat = str(getattr(recipe, "category", "") or "").strip()
    return cat or DEFAULT_CATEGORY


def list_categories() -> list[str]:
    """Sorted, de-duplicated categories across all saved recipes (default first)."""
    cats = {recipe_category(r) for r in list_recipes()}
    cats.add(DEFAULT_CATEGORY)
    ordered = sorted(c for c in cats if c != DEFAULT_CATEGORY)
    return [DEFAULT_CATEGORY] + ordered


def capture_folder(recipe: Recipe) -> Path:
    p = CAPTURE_DIR / recipe.safe_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def label_folder(recipe: Recipe) -> Path:
    p = LABEL_DIR / recipe.safe_name
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_capture(recipe: Recipe, frame_bgr: np.ndarray, adjusted_bgr: np.ndarray | None = None) -> tuple[Path, Path | None]:
    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)
    base = f"{recipe.safe_name}_{ts}_{ms:03d}"
    folder = capture_folder(recipe)
    raw_path = folder / f"{base}.jpg"
    cv2.imwrite(str(raw_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    adjusted_path = None
    if adjusted_bgr is not None:
        adjusted_path = folder / f"{base}_adjusted.jpg"
        cv2.imwrite(str(adjusted_path), adjusted_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return raw_path, adjusted_path


IMPORT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def find_sidecar_json(image_src: Path, json_dir: Path | None = None) -> Path | None:
    """Locate a BungVision-style sidecar label JSON for a source image.

    If ``json_dir`` is given the JSON is looked up there (parallel-directory
    layout: images and labels live in separate sibling folders).  Otherwise
    the JSON is looked up next to the image file (co-located layout).

    Supports both ``foo.json`` (stem) and ``foo.jpg.json`` (full-name) naming.
    """
    if json_dir is not None:
        candidates = [
            json_dir / f"{image_src.stem}.json",
            json_dir / f"{image_src.name}.json",
        ]
    else:
        candidates = [
            image_src.with_suffix(".json"),
            Path(str(image_src) + ".json"),
        ]
    for c in candidates:
        if c.exists():
            return c
    return None


def import_images(
    recipe: Recipe,
    paths: list[Path | str],
    json_dir: Path | None = None,
) -> tuple[list[Path], list[str], int]:
    """Copy external images (and any sidecar label JSON) into a recipe.

    Each source image is decoded and re-encoded to JPEG under the recipe's
    normal naming convention so it shows up in the captured-image list.

    If ``json_dir`` is supplied, the matching ``.json`` label file is looked up
    there (parallel-directory layout).  Otherwise the JSON is expected to sit
    next to the image (co-located layout).  When a sidecar is found, its boxes
    and review/source metadata are written into the recipe's label folder under
    the new image name so imported labels appear immediately.

    Returns (imported_paths, errors, label_count).
    """
    folder = capture_folder(recipe)
    imported: list[Path] = []
    errors: list[str] = []
    label_count = 0
    ts = time.strftime("%Y%m%d_%H%M%S")
    for i, src in enumerate(paths):
        src = Path(src)
        try:
            img = cv2.imread(str(src))
            if img is None:
                errors.append(f"Could not read image: {src.name}")
                continue
            base = f"{recipe.safe_name}_import_{ts}_{i:04d}"
            dest = folder / f"{base}.jpg"
            cv2.imwrite(str(dest), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            imported.append(dest)

            sidecar = find_sidecar_json(src, json_dir=json_dir)
            if sidecar is not None:
                try:
                    data = json.loads(sidecar.read_text(encoding="utf-8"))
                    _write_imported_label(dest, img, data)
                    label_count += 1
                except Exception as exc:
                    errors.append(f"{src.name} label JSON: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{src.name}: {exc}")
    return imported, errors, label_count


def _write_imported_label(image_path: Path, img_bgr: "np.ndarray", data: dict[str, Any]) -> Path:
    """Write a sidecar label JSON for an imported image, preserving its content.

    The full source payload is kept (boxes plus any review/source metadata) but
    the image path and dimensions are corrected to the newly imported file.
    """
    h, w = img_bgr.shape[:2]
    payload = dict(data) if isinstance(data, dict) else {}
    payload["image"] = str(image_path)
    try:
        payload["width"] = int(payload.get("width") or w)
        payload["height"] = int(payload.get("height") or h)
    except (TypeError, ValueError):
        payload["width"], payload["height"] = w, h
    payload["boxes"] = payload.get("boxes") or []
    # Record provenance so review tooling treats these as imported.
    payload.setdefault("imported_from", "image_import")
    path = image_label_json_path(image_path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def image_label_json_path(image_path: Path) -> Path:
    # data/labels/<recipe>/<image_stem>.json
    recipe_name = image_path.parent.name
    folder = LABEL_DIR / recipe_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{image_path.stem}.json"


def save_annotations(
    image_path: Path,
    image_w: int,
    image_h: int,
    boxes: list[dict[str, Any]],
    class_names: list[str],
    review: dict[str, Any] | None = None,
    clear_review: bool = False,
) -> Path:
    """Save Label Studio annotations.

    v0.9.28 adds an optional review marker so images imported from
    BungVision can remain visibly "needs review" until an operator
    explicitly saves or marks them reviewed.  When review is not supplied,
    existing review metadata is normally preserved.  v0.9.37 adds
    clear_review=True for the safety case where an already-reviewed image
    is edited into a quantity mismatch; normal Save must then remove the
    old review marker so only Force Review can include the mismatch in
    training/export.
    """
    path = image_label_json_path(image_path)
    previous: dict[str, Any] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}

    payload = {
        "image": str(image_path),
        "width": image_w,
        "height": image_h,
        "classes": class_names,
        "boxes": boxes,
    }

    # Preserve useful source metadata from imported BungVision JSON.
    # Review metadata is preserved only for non-review-changing saves.  When
    # clear_review=True, stale Label Studio review markers are intentionally
    # removed so an image edited into a count mismatch cannot accidentally
    # remain eligible for reviewed-only export/training.
    source_keys = (
        "source",
        "origin",
        "imported_from",
        "imported_from_bungvision",
        "bungvision",
        "capture_source",
    )
    review_keys = (
        "review",
        "reviewed",
        "reviewed_at",
        "reviewed_by",
        "review_status",
        "review_source",
        "review_tool",
        "forced_review",
        "force_reviewed",
    )
    for key in source_keys:
        if key in previous:
            payload[key] = previous[key]
    if review is None and not clear_review:
        for key in review_keys:
            if key in previous:
                payload[key] = previous[key]
    elif review is None and clear_review:
        payload["reviewed"] = False
        payload["review_status"] = "needs_review"

    if review is not None:
        payload["review"] = review
        payload["reviewed"] = bool(review.get("reviewed", False))
        payload["review_source"] = review.get("source", "bungvision_label_studio")
        payload["review_tool"] = review.get("tool", "BungVision Label Studio")
        if review.get("reviewed_at"):
            payload["reviewed_at"] = review.get("reviewed_at")
        if review.get("reviewed_by"):
            payload["reviewed_by"] = review.get("reviewed_by")
        payload["review_status"] = review.get("review_status") or ("reviewed" if payload["reviewed"] else "needs_review")
        if review.get("forced_review") or review.get("force_reviewed"):
            payload["forced_review"] = True
            payload["force_reviewed"] = True

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_annotations(image_path: Path) -> dict[str, Any] | None:
    path = image_label_json_path(image_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
