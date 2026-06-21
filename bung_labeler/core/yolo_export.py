from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from .storage import CAPTURE_DIR, EXPORT_DIR, LABEL_DIR, infer_role_and_layout


def _clean_class_token(text: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(text).strip())
    return cleaned or "Unnamed"


def _canonical_kind(box: dict) -> str:
    label = str(box.get("label", "") or "").strip().lower()
    cls = int(box.get("class_id", -1))

    role, _layout = infer_role_and_layout(label)
    if role in ("battery", "bung", "retainer"):
        return role
    if cls == 0:
        return "battery"
    if cls == 1:
        return "bung"
    if cls == 2:
        return "retainer"
    if label:
        return _clean_class_token(label).lower()
    return f"class_{cls}" if cls >= 0 else "unknown"


def _is_label_studio_review_marker(review: dict | None) -> bool:
    if not isinstance(review, dict) or not bool(review.get("reviewed", False)):
        return False
    text = " ".join(
        str(review.get(k, ""))
        for k in ("source", "tool", "review_source", "reviewed_by", "reviewer", "app")
    ).lower()
    return "bungvision_label_studio" in text or "bung label studio" in text or "label studio" in text


def _annotation_reviewed(data: dict | None) -> bool:
    """Return True only for labels explicitly reviewed inside this labeler.

    BungVision runtime/import JSON may already contain generic fields like
    reviewed=true, review_status=ok/pass, approved, or accepted. Those are
    not operator review markers for training. Legacy v0.9.28-v0.9.30 Label
    Studio markers are accepted because reviewed_by contains "BungVision
    Label Studio".
    """
    if not data:
        return False
    review = data.get("review") if isinstance(data, dict) else None
    if _is_label_studio_review_marker(review):
        return True
    if bool(data.get("reviewed", False)):
        top_level_review = {
            "reviewed": True,
            "source": data.get("review_source") or data.get("source") or data.get("origin") or data.get("imported_from"),
            "tool": data.get("review_tool") or data.get("tool") or data.get("app"),
            "reviewed_by": data.get("reviewed_by"),
        }
        return _is_label_studio_review_marker(top_level_review)
    return False


def _class_name_for_box(box: dict, recipe_safe_name: str, class_mode: str) -> str | None:
    kind = _canonical_kind(box)
    recipe_token = _clean_class_token(recipe_safe_name)
    label_name = _clean_class_token(box.get("label", "") or kind)

    # Keep exact annotation class names when requested.
    if class_mode == "label_names":
        return label_name

    if class_mode == "generic":
        return kind

    if class_mode == "battery_model_generic_bung":
        if kind == "battery":
            return f"battery_{recipe_token}"
        if kind == "bung":
            return "bung"
        if kind == "retainer":
            return f"retainer_{recipe_token}"

    if kind == "battery":
        return f"battery_{recipe_token}"
    if kind == "bung":
        return f"bung_{recipe_token}"
    if kind == "retainer":
        return f"retainer_{recipe_token}"
    return kind if class_mode == "generic" else f"{kind}_{recipe_token}"


def _yolo_line(box: dict, image_w: int, image_h: int, class_id: int) -> str:
    x = float(box["x"])
    y = float(box["y"])
    w = float(box["w"])
    h = float(box["h"])
    cx = (x + w / 2.0) / image_w
    cy = (y + h / 2.0) / image_h
    nw = w / image_w
    nh = h / image_h
    return f"{int(class_id)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"




def _obb_line(box: dict, image_w: int, image_h: int, class_id: int) -> str | None:
    points = box.get("points") or box.get("obb") or []
    if len(points) < 4:
        return None
    vals = []
    for x, y in points[:4]:
        vals.append(max(0.0, min(1.0, float(x) / image_w)))
        vals.append(max(0.0, min(1.0, float(y) / image_h)))
    return f"{int(class_id)} " + " ".join(f"{v:.6f}" for v in vals)


def _box_to_obb_points(box: dict) -> list[list[float]] | None:
    """Return four image-space points for either an OBB label or a legacy box label.

    YOLO OBB labels accept four normalized corner points. For older labels that
    were drawn as normal boxes, export the enclosing rectangle as a zero-rotation
    OBB so the dataset remains usable while the user relabels/rotates as needed.
    """
    kind = _annotation_kind(box)
    if kind == "obb":
        pts = box.get("points") or box.get("obb") or []
        if len(pts) >= 4:
            return [[float(x), float(y)] for x, y in pts[:4]]
        return None
    if kind == "box":
        try:
            x = float(box["x"]); y = float(box["y"]); w = float(box["w"]); h = float(box["h"])
        except Exception:
            return None
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    return None


def _obb_line_from_any_box(box: dict, image_w: int, image_h: int, class_id: int) -> str | None:
    pts = _box_to_obb_points(box)
    if not pts:
        return None
    return _obb_line({"points": pts}, image_w, image_h, class_id)


def _annotation_kind(box: dict) -> str:
    return str(box.get("kind") or box.get("type") or "box").lower()


def _box_for_detect_export(box: dict) -> dict | None:
    """Return a normal YOLO box for detect export.

    Legacy OBB labels are converted to their enclosing rectangle so older
    battery annotations still export when the project is using box detection.
    """
    kind = _annotation_kind(box)
    if kind == "box":
        return box
    if kind == "obb":
        pts = box.get("points") or box.get("obb") or []
        if len(pts) < 4:
            return None
        xs = [float(p[0]) for p in pts[:4]]
        ys = [float(p[1]) for p in pts[:4]]
        out = dict(box)
        out["x"] = min(xs)
        out["y"] = min(ys)
        out["w"] = max(xs) - min(xs)
        out["h"] = max(ys) - min(ys)
        out["kind"] = "box"
        return out
    return None

def _collect_labeled_entries(recipe_safe_name: str, reviewed_only: bool = True):
    image_dir = CAPTURE_DIR / recipe_safe_name
    label_dir = LABEL_DIR / recipe_safe_name
    if not image_dir.exists() or not label_dir.exists():
        return []

    entries = []
    for lp in sorted(label_dir.glob("*.json")):
        try:
            data = json.loads(lp.read_text(encoding="utf-8"))
        except Exception:
            continue

        if reviewed_only and not _annotation_reviewed(data):
            continue

        img = Path(data.get("image", ""))
        if not img.exists():
            img = image_dir / img.name
        if not img.exists():
            continue

        if not data.get("boxes", []):
            continue

        entries.append((recipe_safe_name, img, data))

    return entries


def _split_entries(entries, split_train: float):
    random.shuffle(entries)
    if len(entries) == 1:
        return entries, entries

    n_train = max(1, int(round(len(entries) * split_train)))
    n_train = min(n_train, len(entries) - 1)
    return entries[:n_train], entries[n_train:]


def _class_names_for_entries(entries, class_mode: str) -> list[str]:
    names = []
    seen = set()

    for recipe_safe_name, _img, data in entries:
        for box in data.get("boxes", []):
            name = _class_name_for_box(box, recipe_safe_name, class_mode)
            if name and name not in seen:
                names.append(name)
                seen.add(name)

    def sort_key(name: str):
        if name == "battery" or name.startswith("battery_"):
            return (0, name)
        if name == "bung" or name.startswith("bung_"):
            return (1, name)
        if name == "retainer" or name.startswith("retainer_"):
            return (2, name)
        return (9, name)

    names.sort(key=sort_key)
    return names or ["battery", "bung"]


def _write_dataset(out: Path, entries, class_mode: str, split_train: float, combined: bool) -> Path:
    if out.exists():
        shutil.rmtree(out)

    train, val = _split_entries(entries, split_train)
    class_names = _class_names_for_entries(entries, class_mode)
    class_index = {name: i for i, name in enumerate(class_names)}

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = ["split,recipe,image,box_count,battery_count,bung_count,retainer_count,class_mode"]

    for split, items in (("train", train), ("val", val)):
        for recipe_safe_name, img, data in items:
            out_name = f"{recipe_safe_name}__{img.name}" if combined else img.name
            shutil.copy2(img, out / "images" / split / out_name)

            lines = []
            battery_count = 0
            bung_count = 0
            retainer_count = 0

            for raw_box in data.get("boxes", []):
                box = _box_for_detect_export(raw_box)
                if box is None:
                    continue
                kind = _canonical_kind(box)
                if kind == "battery":
                    battery_count += 1
                elif kind == "bung":
                    bung_count += 1
                elif kind == "retainer":
                    retainer_count += 1

                class_name = _class_name_for_box(box, recipe_safe_name, class_mode)
                if not class_name:
                    continue

                lines.append(
                    _yolo_line(box, int(data["width"]), int(data["height"]), class_index[class_name])
                )

            (out / "labels" / split / f"{Path(out_name).stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            manifest_rows.append(
                f"{split},{recipe_safe_name},{out_name},{len(data.get('boxes', []))},"
                f"{battery_count},{bung_count},{retainer_count},{class_mode}"
            )

    yaml = [
        f"path: {out.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for i, name in enumerate(class_names):
        yaml.append(f"  {i}: {name}")

    (out / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    (out / "manifest.csv").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    (out / "class_mode.txt").write_text(class_mode + "\n", encoding="utf-8")
    return out



def _write_obb_dataset(out: Path, entries, class_mode: str, split_train: float, combined: bool) -> Path:
    """Write a YOLO OBB dataset for all labeled classes.

    BungVision's current direction uses OBB for both battery and bung. This exporter
    keeps the same class-mode choices as detect export but writes labels as:
        class x1 y1 x2 y2 x3 y3 x4 y4
    Legacy box annotations are converted to zero-rotation OBB rectangles so older
    captures are still exportable.
    """
    if out.exists():
        shutil.rmtree(out)

    usable_entries = []
    for recipe_safe_name, img, data in entries:
        boxes = []
        for b in data.get("boxes", []):
            if _box_to_obb_points(b):
                boxes.append(b)
        if boxes:
            filtered = dict(data)
            filtered["boxes"] = boxes
            usable_entries.append((recipe_safe_name, img, filtered))

    if not usable_entries:
        raise FileNotFoundError("No labels found for OBB export. Draw battery/bung OBB labels, save labels, then export again.")

    train, val = _split_entries(usable_entries, split_train)
    class_names = _class_names_for_entries(usable_entries, class_mode)
    class_index = {name: i for i, name in enumerate(class_names)}

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = ["split,recipe,image,obb_count,battery_count,bung_count,retainer_count,class_mode"]

    for split, items in (("train", train), ("val", val)):
        for recipe_safe_name, img, data in items:
            out_name = f"{recipe_safe_name}__{img.name}" if combined else img.name
            shutil.copy2(img, out / "images" / split / out_name)

            lines = []
            battery_count = 0
            bung_count = 0
            retainer_count = 0

            for box in data.get("boxes", []):
                class_name = _class_name_for_box(box, recipe_safe_name, class_mode)
                if not class_name:
                    continue
                if class_name not in class_index:
                    # Should not happen, but avoid crashing an export because of a
                    # malformed late-added label.
                    continue
                line = _obb_line_from_any_box(box, int(data["width"]), int(data["height"]), class_index[class_name])
                if not line:
                    continue
                lines.append(line)

                kind = _canonical_kind(box)
                if kind == "battery":
                    battery_count += 1
                elif kind == "bung":
                    bung_count += 1
                elif kind == "retainer":
                    retainer_count += 1

            (out / "labels" / split / f"{Path(out_name).stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            manifest_rows.append(
                f"{split},{recipe_safe_name},{out_name},{len(lines)},"
                f"{battery_count},{bung_count},{retainer_count},{class_mode}"
            )

    yaml = [
        f"path: {out.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for i, name in enumerate(class_names):
        yaml.append(f"  {i}: {name}")

    (out / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    (out / "manifest.csv").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    (out / "class_mode.txt").write_text(class_mode + "\n", encoding="utf-8")
    (out / "task.txt").write_text("obb\n", encoding="utf-8")
    return out


def export_recipe_obb(
    recipe_safe_name: str,
    split_train: float = 0.8,
    class_mode: str = "label_names",
    reviewed_only: bool = True,
) -> Path:
    reviewed_only = True
    image_dir = CAPTURE_DIR / recipe_safe_name
    if not image_dir.exists():
        raise FileNotFoundError(f"No capture folder exists for {recipe_safe_name}")
    entries = _collect_labeled_entries(recipe_safe_name, reviewed_only=reviewed_only)
    if not entries:
        if reviewed_only:
            raise FileNotFoundError(
                f"No reviewed labeled images found for {recipe_safe_name}. Use Save Labels or Mark Current Reviewed on images you trust."
            )
        raise FileNotFoundError(
            f"No labeled images found for {recipe_safe_name}. Capture images, draw OBB labels, click Save Labels, then export again."
        )
    out = EXPORT_DIR / f"{recipe_safe_name}_obb"
    result = _write_obb_dataset(out, entries, class_mode=class_mode, split_train=split_train, combined=False)
    (result / "review_filter.txt").write_text("reviewed_only\n", encoding="utf-8")
    return result


def export_all_recipes_obb(
    export_name: str = "all_recipes_obb",
    split_train: float = 0.8,
    class_mode: str = "label_names",
    reviewed_only: bool = True,
) -> Path:
    reviewed_only = True
    recipe_names = sorted(
        {p.name for p in CAPTURE_DIR.iterdir() if p.is_dir()}
        | {p.name for p in LABEL_DIR.iterdir() if p.is_dir()}
    )

    all_entries = []
    for recipe_safe_name in recipe_names:
        all_entries.extend(_collect_labeled_entries(recipe_safe_name, reviewed_only=reviewed_only))

    if not all_entries:
        if reviewed_only:
            raise FileNotFoundError(
                "No reviewed labeled images found across any recipe. Use Save Labels or Mark Current Reviewed on images you trust."
            )
        raise FileNotFoundError(
            "No labeled images found across any recipe. Capture images, draw OBB labels, save labels, then export again."
        )

    out = EXPORT_DIR / export_name
    result = _write_obb_dataset(out, all_entries, class_mode=class_mode, split_train=split_train, combined=True)
    (result / "review_filter.txt").write_text("reviewed_only\n", encoding="utf-8")
    return result


def _write_battery_obb_dataset(out: Path, entries, split_train: float, combined: bool) -> Path:
    if out.exists():
        shutil.rmtree(out)

    obb_entries = []
    for recipe_safe_name, img, data in entries:
        obbs = [b for b in data.get("boxes", []) if _annotation_kind(b) == "obb" and _canonical_kind(b) == "battery"]
        if obbs:
            filtered = dict(data)
            filtered["boxes"] = obbs
            obb_entries.append((recipe_safe_name, img, filtered))

    if not obb_entries:
        raise FileNotFoundError("No battery OBB labels found. Draw battery annotations with the OBB / 4-corner tool, save labels, then export again.")

    train, val = _split_entries(obb_entries, split_train)
    class_names = ["battery"]

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = ["split,recipe,image,obb_count"]

    for split, items in (("train", train), ("val", val)):
        for recipe_safe_name, img, data in items:
            out_name = f"{recipe_safe_name}__{img.name}" if combined else img.name
            shutil.copy2(img, out / "images" / split / out_name)
            lines = []
            for box in data.get("boxes", []):
                line = _obb_line(box, int(data["width"]), int(data["height"]), 0)
                if line:
                    lines.append(line)
            (out / "labels" / split / f"{Path(out_name).stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            manifest_rows.append(f"{split},{recipe_safe_name},{out_name},{len(lines)}")

    yaml = [
        f"path: {out.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
        "  0: battery",
    ]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    (out / "manifest.csv").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    (out / "task.txt").write_text("obb\n", encoding="utf-8")
    return out


def export_recipe_battery_obb(recipe_safe_name: str, split_train: float = 0.8, reviewed_only: bool = True) -> Path:
    reviewed_only = True
    image_dir = CAPTURE_DIR / recipe_safe_name
    if not image_dir.exists():
        raise FileNotFoundError(f"No capture folder exists for {recipe_safe_name}")
    entries = _collect_labeled_entries(recipe_safe_name, reviewed_only=reviewed_only)
    if not entries:
        raise FileNotFoundError(f"No reviewed labeled images found for {recipe_safe_name}.")
    out = EXPORT_DIR / f"{recipe_safe_name}_battery_obb"
    return _write_battery_obb_dataset(out, entries, split_train=split_train, combined=False)


def export_all_battery_obb(export_name: str = "all_recipes_battery_obb", split_train: float = 0.8, reviewed_only: bool = True) -> Path:
    reviewed_only = True
    recipe_names = sorted(
        {p.name for p in CAPTURE_DIR.iterdir() if p.is_dir()}
        | {p.name for p in LABEL_DIR.iterdir() if p.is_dir()}
    )
    all_entries = []
    for recipe_safe_name in recipe_names:
        all_entries.extend(_collect_labeled_entries(recipe_safe_name, reviewed_only=reviewed_only))
    if not all_entries:
        raise FileNotFoundError("No reviewed labeled images found across any recipe.")
    out = EXPORT_DIR / export_name
    return _write_battery_obb_dataset(out, all_entries, split_train=split_train, combined=True)


def export_recipe_yolo(
    recipe_safe_name: str,
    split_train: float = 0.8,
    split_val: float = 0.2,
    class_mode: str = "model_specific",
    reviewed_only: bool = True,
) -> Path:
    reviewed_only = True
    image_dir = CAPTURE_DIR / recipe_safe_name
    if not image_dir.exists():
        raise FileNotFoundError(f"No capture folder exists for {recipe_safe_name}")

    entries = _collect_labeled_entries(recipe_safe_name, reviewed_only=reviewed_only)
    if not entries:
        if reviewed_only:
            raise FileNotFoundError(
                f"No reviewed labeled images found for {recipe_safe_name}. Use Save Labels or Mark Current Reviewed on images you trust."
            )
        raise FileNotFoundError(
            f"No labeled images found for {recipe_safe_name}. Capture images, draw boxes, click Save Labels, then export again."
        )

    out = EXPORT_DIR / f"{recipe_safe_name}_yolo"
    result = _write_dataset(out, entries, class_mode=class_mode, split_train=split_train, combined=False)
    (result / "review_filter.txt").write_text("reviewed_only\n", encoding="utf-8")
    return result


def export_all_recipes_yolo(
    export_name: str = "all_recipes_yolo",
    split_train: float = 0.8,
    class_mode: str = "model_specific",
    reviewed_only: bool = True,
) -> Path:
    reviewed_only = True
    recipe_names = sorted(
        {p.name for p in CAPTURE_DIR.iterdir() if p.is_dir()}
        | {p.name for p in LABEL_DIR.iterdir() if p.is_dir()}
    )

    all_entries = []
    for recipe_safe_name in recipe_names:
        all_entries.extend(_collect_labeled_entries(recipe_safe_name, reviewed_only=reviewed_only))

    if not all_entries:
        if reviewed_only:
            raise FileNotFoundError(
                "No reviewed labeled images found across any recipe. Use Save Labels or Mark Current Reviewed on images you trust."
            )
        raise FileNotFoundError(
            "No labeled images found across any recipe. Capture images, draw boxes, save labels, then export again."
        )

    out = EXPORT_DIR / export_name
    result = _write_dataset(out, all_entries, class_mode=class_mode, split_train=split_train, combined=True)
    (result / "review_filter.txt").write_text("reviewed_only\n", encoding="utf-8")
    return result
