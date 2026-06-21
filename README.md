# BungVision Label Studio

**Current target version:** v0.9.48  
**Package reviewed:** `bung_labeling_tool_v0_9_48_train_and_eval.zip`  
**Purpose:** custom PySide6 labeling/capture/training utility for BungVision battery bung inspection.

This is **not** the commercial HumanSignal Label Studio project. It is a custom Python/PySide6 desktop tool used to capture, review, label, and export training images for the BungVision machine-vision system.

---

## 1. Project scope

BungVision Label Studio exists to create reliable training datasets for the BungVision runtime inspection HMI.

The current BungVision model direction is:

- one YOLO **OBB** model
- `battery` labeled as an oriented bounding box
- `bung` labeled as an oriented bounding box
- `retainer` also defaults to OBB
- custom classes default to OBB unless explicitly changed to box fallback
- export to YOLO OBB dataset format
- only images reviewed inside this tool may be exported/trained

The tool also provides:

- Basler/Pylon and OpenCV camera capture
- manual/auto exposure control
- live preview and still capture
- recipe organization by group/model
- reviewed/unreviewed image workflow
- force-review workflow for intentional quantity mismatches
- local YOLO model test/count utilities
- optional local YOLO training and TensorRT export helpers

---

## 2. What this tool is not

Do **not** treat this application as the production inspection runtime. It is a labeling and dataset-preparation tool.

It is not responsible for:

- PLC conveyor control
- production PASS/FAIL latching
- reject output logic
- multi-battery production tracking
- runtime watchdogs
- production operator HMI behavior

Those belong to the separate BungVision runtime HMI project.

The following older directions were intentionally removed and should not be reintroduced unless explicitly requested:

- bung-zone UI
- `battery_6row` / `battery_2x3` zone-layout class workflow
- segmentation masks
- exporting unreviewed imported images
- treating generic BungVision runtime fields like `reviewed: true` as Label Studio review approval

---

## 3. Current workflow

Typical operator flow:

1. Start the app.
2. Create or load a recipe.
3. Configure camera settings if capturing live images.
4. Capture or import images into the recipe.
5. Draw OBB labels:
   - `battery` around the battery/lid area
   - `bung` around each bung
   - optional `retainer` or other custom classes
6. Save labels.
7. Mark images reviewed when the counts are correct.
8. Use **Force Review Current** only for intentional mismatch/fail examples.
9. Export dataset.
10. Train YOLO OBB model using the exported `data.yaml`.

Export is intentionally reviewed-only. There is no normal option to export every labeled image.

---

## 4. Python and library requirements

The current `requirements.txt` is:

```text
PySide6>=6.6
opencv-python>=4.8
numpy>=1.24
ultralytics>=8.3.0
PyYAML>=6.0
pypylon>=3.0
```

Recommended Python versions:

- Python 3.10 or 3.11 preferred
- Python 3.12 may work on some platforms, but camera/YOLO/PySide dependency compatibility is less predictable
- On Jetson, match NumPy/Torch/Ultralytics versions carefully to the installed JetPack/CUDA stack

### Required libraries by subsystem

| Subsystem | Main files | Required libraries |
|---|---|---|
| Qt desktop UI | `bung_labeler/ui/main_window.py`, `bung_labeler/ui/canvas.py` | `PySide6` |
| Image processing | `core/image_adjust.py`, camera/canvas code | `opencv-python`, `numpy` |
| OpenCV camera capture | `core/camera.py` | `opencv-python` |
| Basler camera capture | `core/camera.py` | `pypylon`, Basler Pylon runtime/SDK |
| Dataset export | `core/yolo_export.py` | Python stdlib, `PyYAML` indirectly through Ultralytics use |
| Export diagnostics | `core/export_report.py` (pure, unit-tested in `tests/test_export_report.py`) | Python stdlib |
| Review / quantity logic | `core/review.py` (pure, unit-tested in `tests/test_review.py`) | Python stdlib |
| Geometry (point-in-polygon, angles) | `core/geometry.py` (pure, unit-tested in `tests/test_geometry.py`) | Python stdlib |
| Model test / training | `ui/main_window.py` | `ultralytics`, PyTorch stack installed by Ultralytics |
| TensorRT export | `ui/main_window.py` | Ultralytics export dependencies, NVIDIA TensorRT runtime/tooling |

### Optional OS-level requirements

For Linux/OpenCV USB cameras:

```bash
sudo apt install v4l-utils
```

The app can call `v4l2-ctl` to force camera mode before OpenCV opens the device.

For Basler cameras:

- Install the Basler Pylon runtime/SDK outside Python.
- Verify the camera in Pylon Viewer first.
- Then install `pypylon` in the Python environment used by this app.

---

## 5. Launching the app

Recommended from the extracted project folder:

```bash
python main.py
```

Windows launchers:

```text
RUN_BUNGVISION_LABEL_STUDIO.bat
run_label_studio.bat
```

Linux launcher:

```bash
./run_label_studio.sh
```

Python module launch should also work:

```bash
python -m bung_labeler
```

The real Python package name is `bung_labeler` with an underscore. Do not use a hyphenated import name in Python source code.

---

## 6. Repository/package structure

```text
.
├── main.py
├── BungVision_Label_Studio.py
├── RUN_BUNGVISION_LABEL_STUDIO.bat
├── run_label_studio.bat
├── run_label_studio.sh
├── requirements.txt
├── camera_debug.py
├── install_training_deps.bat
├── install_training_deps.sh
├── bung-labeler.py
├── bung_labeler/
│   ├── __init__.py
│   ├── __main__.py
│   ├── version.py
│   ├── core/
│   │   ├── active_learning.py
│   │   ├── camera.py
│   │   ├── export_report.py
│   │   ├── geometry.py
│   │   ├── image_adjust.py
│   │   ├── evaluation.py
│   │   ├── relabel.py
│   │   ├── review.py
│   │   ├── storage.py
│   │   ├── training.py
│   │   └── yolo_export.py
│   ├── eval_runner.py
│   └── ui/
│       ├── assets/
│       │   └── checkbox_check.svg
│       ├── canvas.py
│       └── main_window.py
├── tests/
│   ├── test_active_learning.py
│   ├── test_export_report.py
│   ├── test_geometry.py
│   ├── test_evaluation.py
│   ├── test_relabel.py
│   ├── test_review.py
│   └── test_training.py
└── data/
    ├── camera_settings.json
    ├── class_config.json
    ├── captures/
    ├── labels/
    ├── recipes/
    └── exports/
```

### Important modules

#### `main.py`

Primary launcher. It ensures the extracted app folder is on `sys.path`, then starts the PySide6 app.

#### `bung_labeler/ui/main_window.py`

Main application controller. It owns most UI wiring and workflows:

- recipe tab
- live capture tab
- camera settings dialog
- adjustment tab
- annotation right panel
- custom class handling
- review / force-review workflow
- model test/count tools
- export buttons
- training helpers
- live detection helpers

This file is large. Be careful when patching it. Prefer small localized changes.

#### `bung_labeler/ui/canvas.py`

Interactive image annotation canvas.

Key responsibilities:

- display loaded/captured image
- draw box and OBB annotations
- pan/zoom image
- drag OBB corner handles
- draw temporary model-test overlays
- emit `boxes_changed` only when appropriate

Current performance-sensitive behavior:

- scaled image pixmap is cached
- panning avoids unnecessary full-resolution rescaling
- OBB handle drag emits `boxes_changed` on mouse release, not every mouse-move

Do not reintroduce autosave/status refresh on every mouse movement.

#### `bung_labeler/core/storage.py`

Filesystem and JSON persistence:

- app data directories
- recipe schema
- class config schema
- camera settings schema
- capture save paths
- annotation JSON save/load
- review marker preservation/clearing

#### `bung_labeler/core/camera.py`

Camera abstraction for:

- OpenCV cameras
- Linux V4L2 mode forcing
- Basler/Pylon cameras
- threaded camera reads
- Basler AOI/ROI width-height handling
- exposure control

#### `bung_labeler/core/yolo_export.py`

Dataset export logic.

Important behavior:

- reviewed-only export is hardcoded
- YOLO OBB export writes `class x1 y1 x2 y2 x3 y3 x4 y4`
- legacy box labels can be converted to zero-rotation OBBs
- generic runtime review fields do not count as Label Studio review approval

---

## 7. Data model

All application data is stored under `data/` inside the extracted project folder.

### Recipes

Location:

```text
data/recipes/<Group>__<Model>.json
```

Example:

```json
{
  "group": "Default",
  "model": "Battery_Model",
  "expected_bungs": 6,
  "brightness": 0,
  "contrast": 0,
  "gamma": 1.0,
  "clahe_enabled": false,
  "clahe_clip": 2.0,
  "clahe_grid": 8,
  "sharpen": 0,
  "notes": ""
}
```

Camera settings are no longer recipe-specific. Loading a recipe should not change the camera source/resolution/exposure.

### Camera settings

Location:

```text
data/camera_settings.json
```

Defaults are defined in `storage.DEFAULT_CAMERA_SETTINGS`:

```json
{
  "camera_source": "0",
  "camera_backend": "V4L2",
  "width": 2592,
  "height": 1944,
  "fps": 0,
  "preview_scale": "1/2",
  "exposure_auto": true,
  "exposure_us": 0,
  "force_v4l2": true,
  "low_latency": true,
  "threaded_camera": true,
  "mjpg": true,
  "skip_heavy_live": true
}
```

v0.9.42 behavior:

- applying/saving changed camera settings reopens the live camera stream
- Basler AOI offsets are reset before applying width/height
- blank or zero Basler width/height means full available sensor AOI

### Class configuration

Location:

```text
data/class_config.json
```

Current default classes:

```json
{
  "classes": [
    {"id": 0, "name": "battery", "default_tool": "OBB", "enabled": true, "role": "battery"},
    {"id": 1, "name": "bung", "default_tool": "OBB", "enabled": true, "role": "bung"},
    {"id": 2, "name": "retainer", "default_tool": "OBB", "enabled": true, "role": "retainer"}
  ]
}
```

Custom classes default to OBB. The UI provides an explicit box fallback option for custom classes that truly should be axis-aligned boxes.

### Captures

Location:

```text
data/captures/<RecipeSafeName>/*.jpg
```

`RecipeSafeName` is generated as:

```text
<Group>__<Model>
```

with unsafe filename characters replaced by underscores.

### Labels

Location:

```text
data/labels/<RecipeSafeName>/<ImageStem>.json
```

Typical annotation payload:

```json
{
  "image": "data/captures/Default__Battery_Model/example.jpg",
  "width": 2592,
  "height": 1944,
  "classes": ["battery", "bung", "retainer"],
  "boxes": [
    {
      "x": 122.0,
      "y": 314.0,
      "w": 1580.0,
      "h": 640.0,
      "class_id": 0,
      "label": "battery",
      "kind": "obb",
      "points": [[122.0, 314.0], [1702.0, 330.0], [1685.0, 954.0], [105.0, 936.0]]
    },
    {
      "x": 510.0,
      "y": 530.0,
      "w": 92.0,
      "h": 84.0,
      "class_id": 1,
      "label": "bung",
      "kind": "obb",
      "points": [[510.0, 530.0], [602.0, 532.0], [600.0, 614.0], [508.0, 612.0]]
    }
  ],
  "review": {
    "reviewed": true,
    "source": "bungvision_label_studio",
    "tool": "BungVision Label Studio",
    "review_status": "reviewed"
  },
  "reviewed": true,
  "review_source": "bungvision_label_studio",
  "review_tool": "BungVision Label Studio",
  "review_status": "reviewed"
}
```

OBB points are image-space coordinates in clockwise order:

```text
top-left, top-right, bottom-right, bottom-left
```

Legacy box annotations use:

```json
{
  "x": 100,
  "y": 200,
  "w": 300,
  "h": 150,
  "class_id": 1,
  "label": "bung",
  "kind": "box"
}
```

Legacy boxes are converted to zero-rotation OBB rectangles during OBB export.

---

## 8. Review and force-review rules

Reviewed-only export is central to this tool. The purpose is to keep imported or partially checked BungVision data out of training until an operator explicitly approves it inside this labeler.

### Normal reviewed image

A normal reviewed image should have:

- at least one battery label
- exactly the recipe's expected number of bungs **inside every battery** (a bung is assigned to a battery when its center falls within that battery's polygon)
- no bung labels outside all batteries
- Label Studio review marker

As of v0.9.43 multiple batteries are supported in a single image: if two batteries
are both fully labeled with the expected bungs, the image passes normal review with
no force-review needed. Force review is only required when at least one battery is
missing/over its expected bung count, or a bung sits outside every battery.

Normal review can be set by:

- Save Labels, when counts match
- Save + Next, when counts match
- Mark Current Reviewed, when counts match

### Mismatch image

If counts do not match, normal Save Labels should:

- save the annotation geometry
- clear stale Label Studio review markers
- mark the image as needing review
- prevent reviewed-only export from including it

This protects against this dangerous case:

1. image was reviewed
2. operator edits labels into a mismatch
3. operator saves labels
4. image accidentally remains reviewed

v0.9.37 fixed this by clearing stale review metadata on mismatch save.

### Force-reviewed image

Use **Force Review Current** for intentional fail/missing-bung examples.

Force-reviewed images are included in reviewed-only export/training, but are marked so they can be identified later.

Typical force-review metadata:

```json
{
  "review": {
    "reviewed": true,
    "reviewed_by": "BungVision Label Studio v0.9.43",
    "source": "bungvision_label_studio",
    "tool": "BungVision Label Studio",
    "forced_review": true,
    "review_status": "forced_reviewed",
    "forced_reason": "quantity_mismatch"
  }
}
```

`_annotation_force_reviewed` also accepts the legacy `force_reviewed: true` key for backward compatibility, but current builds write `forced_review` / `review_status: "forced_reviewed"`.

### Important safety rule

Do not make generic imported fields count as reviewed. These must not be enough for export:

```json
{"reviewed": true}
{"review_status": "ok"}
{"review_status": "approved"}
{"review_status": "accepted"}
```

The export code should only accept review markers created by BungVision Label Studio.

---

## 9. Export behavior

All normal export paths force reviewed-only mode.

Current OBB export format:

```text
class x1 y1 x2 y2 x3 y3 x4 y4
```

Coordinates are normalized to image width/height by the exporter.

Typical output folder:

```text
data/exports/all_recipes_obb/
├── data.yaml
├── manifest.csv
├── class_mode.txt
├── review_filter.txt
├── task.txt
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

`review_filter.txt` should contain:

```text
reviewed_only
```

`task.txt` should contain:

```text
obb
```

Recommended training command:

```bash
yolo obb train model=yolo11s-obb.pt data=data/exports/all_recipes_obb/data.yaml imgsz=736 epochs=100
```

The user commonly prefers `imgsz=736` for BungVision work.

### In-app training (Train tab, v0.9.47)

The **Train** tab launches Ultralytics YOLO training from inside the app instead of requiring a separate terminal:

- Pick **Task** (obb/detect/…), **Base model** (e.g. `yolo11s-obb.pt` or a checkpoint), and **Data YAML** (an export's `data.yaml`; **Latest export** fills the most recent one and matches the task from `task.txt`).
- Set **imgsz**, **batch** (`-1` = Ultralytics auto-batch), **epochs**, **patience**, **workers**, **device**, **output folder**, and **run name**.
- **Start Training** runs the `yolo` CLI in a background `QProcess` — the UI stays responsive, output streams into the log, and **Stop** cancels the run. Best weights land in `<output folder>/<run name>/weights/best.pt`.
- Parameters persist between sessions (`data/training_settings.json`). If `yolo` is not on PATH, set a full path in the **yolo executable** field.

The command is built and validated by the pure, unit-tested `core/training.py`; the app layer only runs it.

### Evaluate and promote (Train tab, v0.9.48)

Below training, the **Evaluate and promote** section closes the loop:

- Pick a trained model (**Use trained** fills `<output folder>/<run name>/weights/best.pt`) and a **split** (val/test/train); the **Data YAML** and **Task** are shared with training above.
- **Evaluate** runs `python -m bung_labeler.eval_runner` in the background. That thin Ultralytics wrapper computes metrics and prints a JSON block between sentinels, which the app parses and renders: overall **mAP50 / mAP50-95 / precision / recall** plus a **per-class precision/recall/mAP50** table.
- **Promote model** copies the evaluated weights into `data/models/` and sets them as the active model for Test Models, Auto-label, Count Test, and the active-learning review queue — so you only adopt a model after seeing its numbers.

`core/evaluation.py` (command building, metrics parsing, formatting) is pure and unit-tested; `eval_runner.py` is the only Ultralytics-dependent piece.

---

## 9.5. Labeling assistance (v0.9.44)

Three features speed up labeling and protect dataset quality. They live under the **Tools** and **Edit** menus and reuse existing logic.

### Model-assisted pre-labeling (Auto-label, Ctrl+L)
Runs the trained model selected in the Model Test tab on the current image and turns its battery/bung detections into **editable labels** rather than draw-from-scratch boxes. The operator corrects predictions, then Save Labels as usual. It honors the Model Test **Confidence** setting, asks before replacing existing labels, and is a single Undo step. It never auto-saves — predictions are not training data until reviewed.

### Annotation validation (Validate, Ctrl+Shift+V)
Lints the on-canvas labels for the geometry-level mistakes the count-based review gate cannot see: degenerate/tiny boxes, boxes outside the image, heavily overlapping (duplicate) bungs, per-battery over/under counts, and bungs outside every battery. Advisory only — it reports issues without blocking. Pure logic lives in `core/review.py::validate_boxes` and is unit-tested.

### Undo / redo (Ctrl+Z / Ctrl+Y)
The canvas keeps a bounded per-image history. Box creation, deletion, handle drags, nudges (coalesced per burst), clear, copy-previous, and auto-label are all undoable. History resets when a different image loads.

### Bulk relabel (Tools → Bulk relabel class…, v0.9.45)
Reassigns every box of one class to another across the current recipe's saved label files — e.g. consolidating a stray `rubber_bung` into `bung`. Shows a dry-run preview ("will change N boxes across M images") before applying. Because changing a class can alter the battery/bung counts an image was reviewed against, every changed image is returned to the review queue (review marker cleared). Pure logic lives in `core/relabel.py` and is unit-tested; it edits sidecars directly and is not undoable from the canvas, so the preview + confirmation are the safeguard.

### Active-learning review queue (Tools → Build review queue, v0.9.46)
Runs the Model Test tab's model across every unreviewed image in the recipe and ranks them by a disagreement score: missing batteries dominate, each battery's |detected − expected| bungs accumulates, stray bungs add uncertainty, and low average confidence nudges borderline images up. The operator then steps through the queue highest-disagreement-first via **Next in review queue** (Ctrl+Shift+N), labeling the most informative images first. Pure scoring lives in `core/active_learning.py` and is unit-tested; the model pass is preview-only and never writes labels.

---

## 10. Camera behavior

The camera layer supports OpenCV cameras and Basler/Pylon cameras.

### Basler/Pylon

Backend should be set to Basler/Pylon in the UI.

Source behavior:

- blank source: use first Pylon-detected Basler camera
- serial/model/friendly name: match that detected camera if possible

Resolution behavior:

- width/height are Basler AOI dimensions
- blank or zero means full available sensor AOI
- v0.9.42 resets `OffsetX` and `OffsetY` before applying width/height so the camera does not stay stuck in a smaller AOI

Exposure behavior:

- Auto exposure uses `ExposureAuto`
- Manual exposure writes `ExposureTime` or `ExposureTimeAbs`
- exposure value is in microseconds for Basler

### OpenCV / V4L2

Source examples:

```text
0
/dev/video0
```

Options include:

- force V4L2 mode
- MJPG mode
- low latency
- threaded camera
- preview scaling

For Linux USB cameras, installing `v4l-utils` is recommended so `v4l2-ctl` is available.

---

## 11. UI and layout constraints

The UI target is a practical industrial workstation display, especially 1920x1080.

Important UI preferences already incorporated:

- compact right-side annotation buttons
- compact count input boxes instead of spinbox arrows
- group/model fields kept compact
- visible checkbox outlines on dark theme
- right panel should not overlap controls
- live capture button height is the visual reference for compact button sizing

Avoid large UI rearrangements unless necessary. Many previous issues were caused by oversized buttons, wrapped text, or panels fighting for vertical space.

---

## 12. Performance constraints

Two different performance problems have been addressed:

### Recipe/file-list performance

v0.9.35 added a cached recipe image index.

Do not reparse every JSON sidecar after every edit. Save/review/delete should invalidate only the current image status cache.

Manual refresh exists for external file changes:

```text
View -> Refresh recipe index
Ctrl+F5
```

### Canvas performance

v0.9.36 added canvas performance improvements.

Do not reintroduce behavior where panning or OBB handle dragging triggers expensive recipe-wide operations.

Performance-sensitive rules:

- cache scaled display pixmap
- do not smooth-rescale the full 5 MP source image on every paint
- do not emit `boxes_changed` on every OBB handle mouse move
- do not autosave on every mouse move
- keep panning lightweight

---

## 13. Model test / count-test behavior

The current model-test direction assumes a single YOLO OBB model containing multiple classes.

v0.9.37 added class filtering so one combined model does not count every OBB as both a battery and a bung.

Default filters:

```text
Battery class: battery,0
Count class: bung,1
```

Rules:

- battery overlays/counts must only use the configured battery class
- bung count candidates must only use the configured count class
- retainer and custom classes must not be counted as battery or bung unless explicitly selected in the filter

---

## 14. Custom classes

v0.9.38 changed custom class behavior.

Current rule:

- new custom classes default to OBB
- built-in classes always use OBB
- existing older custom classes that were auto-created as BOX are migrated to OBB unless explicitly locked as box fallback
- user can intentionally set a selected custom class to Box fallback using the class tool selector

This matters because BungVision is now OBB-first for battery and bungs, and custom classes may also need oriented boxes.

---

## 15. Training and TensorRT helpers

The app includes training/export helpers, but they depend on the local Python environment.

Recommended manual train command:

```bash
yolo obb train model=yolo11s-obb.pt data=data/exports/all_recipes_obb/data.yaml imgsz=736 epochs=100
```

TensorRT export depends on platform-specific NVIDIA packages and should not be assumed to work everywhere.

For Jetson:

- TensorRT must match the JetPack/CUDA stack
- PyTorch/Ultralytics versions must be Jetson-compatible
- avoid casually upgrading NumPy/Torch without verifying Jetson compatibility

For Windows:

- normal labeling/capture may work
- TensorRT export requires a proper NVIDIA CUDA/TensorRT install
- Basler requires Pylon runtime and `pypylon`

---

## 16. Development checklist for Claude Code

Before making changes:

1. Confirm the target version and baseline.
2. Read this README.
3. Search for the relevant workflow in `main_window.py` before patching.
4. Keep changes localized.
5. Do not reintroduce old zone-layout code.
6. Do not add an option to export unreviewed images.
7. Preserve compact 1920x1080 UI layout.
8. Preserve Basler camera AOI/exposure behavior.
9. Preserve OBB-first class behavior.

After making changes, run:

```bash
python -m compileall bung_labeler main.py BungVision_Label_Studio.py camera_debug.py
```

Recommended smoke tests:

- launch with `python main.py`
- create/load recipe
- verify expected bungs field accepts 1-99 without spinbox arrows
- add custom class and confirm default tool is OBB
- draw battery OBB and bung OBBs
- save matching image and confirm it becomes reviewed
- edit reviewed image into mismatch and confirm Save clears review
- force-review mismatch and confirm export includes it
- export all OBB and confirm only reviewed/force-reviewed images are copied
- verify `review_filter.txt` says `reviewed_only`
- run model test with class filters and confirm retainer is not counted as battery/bung
- test Basler camera open after changing resolution
- test manual exposure apply

---

## 17. Known pitfalls / do-not-break list

### Do not make export optional-all

The reviewed-only behavior is intentional. Earlier versions exported too many images and caused confusion.

### Do not count generic runtime review fields

Only Label Studio review markers should make an image trainable.

### Do not preserve review on mismatch save

If labels no longer match expected quantity, normal Save must clear stale reviewed status.

### Do not reintroduce bung zones

The runtime direction moved away from Label Studio zone-layout workflow.

### Do not default custom classes to box

Current OBB workflow requires custom classes to support OBB by default.

### Do not let camera resolution changes silently fail

Camera Settings Apply/Save must reopen/re-negotiate the live stream when backend/source/width/height/FPS/options change.

### Do not save camera setup inside recipes

Camera settings are app-wide in `data/camera_settings.json`.

### Do not make UI controls taller again

The right panel must fit at 1920x1080. Live Capture button height is the reference.

---

## 18. Troubleshooting

### `No module named bung_labeler`

Run from the extracted project root:

```bash
python main.py
```

On Windows, use:

```text
RUN_BUNGVISION_LABEL_STUDIO.bat
```

The launcher scripts set `PYTHONPATH` so Python can import the local `bung_labeler` package.

### Basler test works but live preview does not open

Check:

- Pylon Viewer can see the camera
- `pypylon` is installed in this same Python environment
- backend is Basler/Pylon
- source is blank or a valid serial/model/friendly name
- camera is not already open in another program

### Resolution appears stuck

v0.9.42 should reopen the live stream when camera settings change. For Basler, blank/zero width/height should request full AOI. If stuck:

1. close live camera
2. set width/height
3. apply/save camera settings
4. reopen camera
5. check status message for actual AOI size

### Edits become slow with many images

Use v0.9.35+ recipe index behavior. Avoid code changes that refresh the entire recipe image list after each mouse movement or save.

### Panning slows after adding OBBs

Use v0.9.36+ canvas behavior. Avoid code changes that rescale full images or emit `boxes_changed` continuously during panning/handle dragging.

---

## 19. Version history summary

Key recent versions:

- v0.9.33: launcher path fix
- v0.9.34: force-review for intentional quantity mismatches
- v0.9.35: recipe-index performance cleanup
- v0.9.36: canvas performance cleanup
- v0.9.37: single-model OBB class filtering and review safety
- v0.9.38: custom classes default to OBB / class tool selector
- v0.9.40: compact count inputs, no spinbox arrows
- v0.9.41: compact right-panel button height for 1920x1080
- v0.9.42: camera resolution apply / Basler AOI fix
- v0.9.43: performance tuning (frame-seq dedup, vectorized gamma LUT, hoisted filter parsing, single-pass image list), dead training/detect/TensorRT code removed; pure-logic refactor into `core/review.py`, `core/geometry.py`, `core/export_report.py` with headless tests
- v0.9.44: model-assisted pre-labeling (Auto-label, Ctrl+L), annotation validation/linting (Validate, Ctrl+Shift+V), and canvas undo/redo (Ctrl+Z / Ctrl+Y)
- v0.9.45: bulk relabel — reassign one class to another across a recipe's saved labels (Tools > Bulk relabel), with dry-run preview; changed images return to the review queue. Pure logic in `core/relabel.py`, unit-tested
- v0.9.46: active-learning review queue — run the model across unreviewed images and order them by how much detections disagree with the recipe (Tools > Build review queue / Next in review queue, Ctrl+Shift+N). Pure scoring in `core/active_learning.py`, unit-tested
- v0.9.47: in-app training — launch Ultralytics YOLO training from the Train tab (task, base model, data.yaml, imgsz, batch, epochs, device, output folder, etc.) as a cancelable background process with streamed logs. Pure command builder/validator in `core/training.py`, unit-tested
- v0.9.48: in-app evaluation + promote — score a trained model against a labeled split (mAP50, mAP50-95, per-class precision/recall via `eval_runner.py`) and promote it to the active model for Test/Auto-label/Count/review-queue. Pure command/parse/format in `core/evaluation.py`, unit-tested

---

## 20. Recommended next development priorities

Only if requested by the user:

1. Continue performance tuning if canvas editing still slows on large 5 MP images.
2. Add better operator-visible diagnostics for export counts.
3. Add additional camera troubleshooting status, especially actual AOI, FPS, pixel format, and exposure readback.
4. Stabilize the UI and avoid further layout churn.

Do not add new workflow concepts unless the user explicitly asks. The current goal is to keep the labeler stable, compact, OBB-first, and safe for training export.
