from __future__ import annotations

import sys
import traceback
import time
import math
import json
from pathlib import Path

# Allow this file to be launched directly during troubleshooting, e.g.
# python bung_labeler/ui/main_window.py, without losing access to the
# bundled bung_labeler package.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QAction, QKeySequence, QIntValidator
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QScrollArea,
    QSizePolicy,
)

from bung_labeler.core.camera import CameraSource, quick_test_source
from bung_labeler.core.image_adjust import apply_adjustments
from bung_labeler.core.storage import (
    DATA_DIR,
    EXPORT_DIR,
    Recipe,
    capture_folder,
    image_label_json_path,
    list_recipes,
    load_annotations,
    save_annotations,
    save_capture,
    save_recipe,
    load_camera_settings,
    save_camera_settings,
    load_class_config,
    save_class_config,
    class_names_from_config,
    infer_role_and_layout,
)
from bung_labeler.core.yolo_export import export_recipe_yolo, export_all_recipes_yolo, export_recipe_obb, export_all_recipes_obb
from bung_labeler.ui.canvas import ImageCanvas


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Bung labeling tool v0.9.42")
        self.resize(1450, 850)
        self.setMinimumSize(1000, 650)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)

        self.camera = CameraSource()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self._last_preview_status_t = 0.0
        self._preview_frame_counter = 0
        self._preview_fps_t0 = time.perf_counter()
        self._preview_fps = 0.0
        # Last camera frame sequence processed by the display timer. Used to skip
        # re-decoding/re-painting an unchanged frame (see _on_timer).
        self._last_frame_seq = None
        self.last_raw = None
        self.last_adjusted = None
        self.current_image_path: Path | None = None
        self.camera_settings = load_camera_settings()
        self.class_config = load_class_config()
        self.class_names = class_names_from_config(self.class_config)
        self._test_model = None
        self._test_model_path = ""
        self._model_test_overlay_active = False

        # v0.9.35 performance cache: index the current recipe once and only
        # re-read a sidecar JSON when that specific file changes. This avoids
        # parsing hundreds of JSON files after every box drag/save/review action.
        self._recipe_index_dirty = True
        self._image_paths_cache: list[Path] = []
        self._image_status_cache: dict[str, dict] = {}

        # Labeling-only build: no native training/TensorRT/live-detect workflow.

        self.recipe = Recipe(group="Default", model="Battery_Model")
        save_recipe(self.recipe)

        self.canvas = ImageCanvas()
        self.canvas.boxes_changed.connect(self._update_box_count)

        self.recipe_list = QListWidget()
        self.recipe_list.itemDoubleClicked.connect(self._load_selected_recipe)

        self.image_list = QListWidget()
        self.image_list.itemDoubleClicked.connect(self._load_selected_image)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        self._build_ui()
        self.polish_buttons()
        self._build_menu()
        self._refresh_recipes()
        self._refresh_images()
        self._apply_theme()
        self._class_changed(0)
        self._update_box_count()

    def _build_menu(self) -> None:
        """Build normal drop-down menus instead of dumping every action on the menu bar.

        The previous first pass added actions directly to menuBar(), which made Qt show
        them as a long row of random labels across the top of the window. Keeping the
        actions inside File/Edit/View/Class/Navigate menus preserves shortcuts without
        cluttering the UI.
        """
        menubar = self.menuBar()
        menubar.clear()

        file_menu = menubar.addMenu("File")
        edit_menu = menubar.addMenu("Edit")
        view_menu = menubar.addMenu("View")
        class_menu = menubar.addMenu("Class")
        nav_menu = menubar.addMenu("Navigate")
        capture_menu = menubar.addMenu("Capture")

        open_action = QAction("Open image", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)

        save_action = QAction("Save labels", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_labels)
        file_menu.addAction(save_action)

        delete_action = QAction("Delete selected annotation", self)
        delete_action.setShortcut(QKeySequence.Delete)
        delete_action.triggered.connect(self.canvas.delete_selected)
        edit_menu.addAction(delete_action)

        delete_image_action = QAction("Delete captured image", self)
        delete_image_action.setShortcut("Shift+Delete")
        delete_image_action.triggered.connect(self.delete_selected_image)
        edit_menu.addAction(delete_image_action)

        zoom_in_action = QAction("Zoom in", self)
        zoom_in_action.setShortcut(QKeySequence.ZoomIn)
        zoom_in_action.triggered.connect(self.canvas.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom out", self)
        zoom_out_action.setShortcut(QKeySequence.ZoomOut)
        zoom_out_action.triggered.connect(self.canvas.zoom_out)
        view_menu.addAction(zoom_out_action)

        fit_action = QAction("Fit image", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.triggered.connect(self.canvas.fit_to_window)
        view_menu.addAction(fit_action)

        refresh_index_action = QAction("Refresh recipe index", self)
        refresh_index_action.setShortcut("Ctrl+F5")
        refresh_index_action.triggered.connect(lambda: self._refresh_images(force=True))
        view_menu.addAction(refresh_index_action)

        battery_action = QAction("Class: battery", self)
        battery_action.setShortcut("B")
        battery_action.triggered.connect(lambda: self.set_class_by_name("battery"))
        class_menu.addAction(battery_action)

        bung_action = QAction("Class: bung", self)
        bung_action.setShortcut("U")
        bung_action.triggered.connect(lambda: self.set_class_by_name("bung"))
        class_menu.addAction(bung_action)

        retainer_action = QAction("Class: retainer", self)
        retainer_action.setShortcut("R")
        retainer_action.triggered.connect(lambda: self.set_class_by_name("retainer"))
        class_menu.addAction(retainer_action)

        next_action = QAction("Next image", self)
        next_action.setShortcut("N")
        next_action.triggered.connect(self.next_image)
        nav_menu.addAction(next_action)

        prev_action = QAction("Previous image", self)
        prev_action.setShortcut("P")
        prev_action.triggered.connect(self.previous_image)
        nav_menu.addAction(prev_action)

        unreviewed_action = QAction("Find next unreviewed", self)
        unreviewed_action.setShortcut("Ctrl+U")
        unreviewed_action.triggered.connect(self.find_next_unreviewed_image)
        nav_menu.addAction(unreviewed_action)

        mark_reviewed_action = QAction("Mark current reviewed", self)
        mark_reviewed_action.setShortcut("Ctrl+Shift+R")
        mark_reviewed_action.triggered.connect(self.mark_current_reviewed)
        nav_menu.addAction(mark_reviewed_action)

        force_review_action = QAction("Force review current", self)
        force_review_action.setShortcut("Ctrl+Shift+F")
        force_review_action.triggered.connect(self.force_mark_current_reviewed)
        nav_menu.addAction(force_review_action)

        capture_action = QAction("Capture adjusted", self)
        capture_action.setShortcut("C")
        capture_action.triggered.connect(lambda: self.capture_frame(save_adjusted=True))
        capture_menu.addAction(capture_action)

    def _scroll_panel(self, widget: QWidget, min_width: int, preferred_width: int) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(min_width)
        scroll.resize(preferred_width, scroll.height())
        widget.setMinimumWidth(min_width - 22)
        scroll.setWidget(widget)
        return scroll

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = self._left_panel()
        # Keep the right rail scroll-safe so Linux/Qt themes do not visually
        # stack or clip buttons on shorter windows.
        right = self._scroll_panel(self._right_panel(), min_width=360, preferred_width=380)
        # v0.9.18: v0.9.17 overcorrected and made the entire left rail too wide.
        # Keep enough room for the capture tab, but let the image canvas stay dominant.
        left.setMinimumWidth(390)
        left.setMaximumWidth(520)
        right.setMinimumWidth(360)
        right.setMaximumWidth(450)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(left)
        splitter.addWidget(self.canvas)
        splitter.addWidget(right)
        splitter.setSizes([410, 800, 380])
        self.setCentralWidget(splitter)

    def _left_panel(self) -> QWidget:
        tabs = QTabWidget()
        tabs.addTab(self._recipe_tab(), "Recipe / SKU")
        tabs.addTab(self._capture_tab(), "Live Capture")
        tabs.addTab(self._adjust_tab(), "Contrast")
        tabs.addTab(self._model_test_tab(), "Test Models")
        tabs.addTab(self._help_tab(), "Instructions")
        # Training, TensorRT, and live inference are intentionally hidden in this
        # labeling-only build. Use Docker later after the dataset is ready.
        return tabs



    def _model_test_tab(self) -> QWidget:
        """Model sandbox: run one trained OBB model on one image.

        This tab intentionally does not change saved labels or live inspection state. It is
        just a verification tool so the user can confirm battery and bung detections
        from the same model before using rotation-aware count testing.
        """
        w = QWidget()
        layout = QVBoxLayout(w)

        title = QLabel("Test trained model / Count Test only")
        title.setStyleSheet("font-size: 10pt; font-weight: 700; color: #bfdbfe;")
        title.setWordWrap(True)
        layout.addWidget(title)

        help_text = QLabel(
            "Load your trained BungVision OBB model, select a test image, then run the model or run a count test."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        form_box = QGroupBox("Model files")
        form = QVBoxLayout(form_box)

        self.test_model_edit = QLineEdit()
        self.test_model_edit.setPlaceholderText("BungVision OBB best.pt or .engine")
        browse_model = QPushButton("Model...")
        browse_model.clicked.connect(self.browse_test_model)
        row = QHBoxLayout(); row.addWidget(self.test_model_edit); row.addWidget(browse_model)
        form.addWidget(QLabel("OBB model")); form.addLayout(row)

        self.test_image_edit = QLineEdit()
        self.test_image_edit.setPlaceholderText("test image path")
        browse_img = QPushButton("Image...")
        browse_img.clicked.connect(self.browse_test_image)
        use_current = QPushButton("Use Current")
        use_current.clicked.connect(self.use_current_test_image)
        row = QHBoxLayout(); row.addWidget(self.test_image_edit); row.addWidget(browse_img); row.addWidget(use_current)
        form.addWidget(QLabel("Test image")); form.addLayout(row)


        layout.addWidget(form_box)

        settings_box = QGroupBox("Inference settings")
        settings = QFormLayout(settings_box)
        self.test_imgsz_spin = QSpinBox(); self.test_imgsz_spin.setRange(320, 2048); self.test_imgsz_spin.setSingleStep(32); self.test_imgsz_spin.setValue(736)
        self.test_conf_spin = QDoubleSpinBox(); self.test_conf_spin.setRange(0.01, 0.99); self.test_conf_spin.setSingleStep(0.05); self.test_conf_spin.setValue(0.45)
        self.test_device_edit = QLineEdit("0")
        self.test_device_edit.setPlaceholderText("0, cpu, cuda:0")
        settings.addRow("Image size", self.test_imgsz_spin)
        settings.addRow("Confidence", self.test_conf_spin)
        settings.addRow("Device", self.test_device_edit)
        self.test_hide_saved_labels_check = QCheckBox("Hide saved labels while testing")
        self.test_hide_saved_labels_check.setChecked(True)
        self.test_hide_saved_labels_check.setToolTip("Hides existing/manual labels on the canvas during model testing without deleting them.")
        settings.addRow("Display", self.test_hide_saved_labels_check)

        self.count_required_spin = QLineEdit(str(self.recipe.expected_bungs))
        self.count_required_spin.setObjectName("countRequiredEdit")
        self.count_required_spin.setValidator(QIntValidator(1, 99, self))
        self.count_required_spin.setMaxLength(2)
        self.count_required_spin.setFixedWidth(48)
        self.count_required_spin.setPlaceholderText("6")
        self.count_required_spin.editingFinished.connect(self._sync_required_count_from_text)
        self.battery_class_filter_edit = QLineEdit("battery,0")
        self.battery_class_filter_edit.setToolTip("Comma-separated detection-model battery class names or IDs. Default assumes single OBB model class 0 is battery.")
        self.count_class_filter_edit = QLineEdit("bung,1")
        self.count_class_filter_edit.setToolTip("Comma-separated detection-model bung class names or IDs. Partial names work, e.g. bung matches bungs/rubber_bung. Default assumes class 1 is bung.")
        settings.addRow("Required count", self.count_required_spin)
        settings.addRow("Battery class", self.battery_class_filter_edit)
        settings.addRow("Count class", self.count_class_filter_edit)
        layout.addWidget(settings_box)

        run_btn = QPushButton("Run Model")
        run_btn.clicked.connect(self.run_model_test)
        count_btn = QPushButton("Run Count")
        count_btn.setToolTip("Runs the OBB model and counts bung centers whose centers fall inside each detected battery OBB polygon.")
        count_btn.clicked.connect(self.run_count_test)
        clear_btn = QPushButton("Clear Overlay")
        clear_btn.setToolTip("Clears the model-test overlay and hides saved labels on the canvas. It does not delete or save labels.")
        clear_btn.clicked.connect(self.clear_model_test_overlay)
        show_labels_btn = QPushButton("Show Labels")
        show_labels_btn.setToolTip("Shows saved/manual labels again. This does not affect model-test overlays.")
        show_labels_btn.clicked.connect(self.show_saved_annotations)

        run_grid = QGridLayout()
        run_grid.setHorizontalSpacing(8)
        run_grid.setVerticalSpacing(8)
        test_buttons = (run_btn, count_btn, clear_btn, show_labels_btn)
        for i, btn in enumerate(test_buttons):
            btn.setMinimumHeight(32)
            btn.setMinimumWidth(0)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            run_grid.addWidget(btn, i // 2, i % 2)
        layout.addLayout(run_grid)

        self.test_results_text = QTextEdit()
        self.test_results_text.setReadOnly(True)
        self.test_results_text.setMinimumHeight(180)
        self.test_results_text.setPlainText(
            "Step-by-step:\n"
            "1. Load the BungVision OBB best.pt or .engine model.\n"
            "2. Choose a saved/captured test image.\n"
            "3. Click Run Model or Run Count.\n\n"
            "Expected result: blue battery polygons, green bung polygons/centers, and a text summary."
        )
        layout.addWidget(self.test_results_text)
        layout.addStretch(1)
        return w

    def _help_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        title = QLabel("BungVision Label Studio — built-in workflow guide")
        title.setStyleSheet("font-size: 12pt; font-weight: 700; color: #bfdbfe;")
        title.setWordWrap(True)
        layout.addWidget(title)

        guide = QTextEdit()
        guide.setReadOnly(True)
        guide.setPlainText(
            "Purpose\n"
            "This tool now separates YOLO training labels from future inspection recipes.\n\n"
            "YOLO labels teach the model what objects look like.\n"
            "Recipes, which will come next, teach BungVision where bungs should be for a SKU.\n\n"
            "Recommended first-pass workflow\n"
            "1. Create or load a Group / Model recipe.\n"
            "2. Capture or open images.\n"
            "3. Select class 'battery' and tool 'OBB'.\n"
            "4. Drag a rough rectangle around the visible battery/lid region, then move the corner handles to match the rotated battery.\n"
            "5. Select class 'bung' and tool 'OBB', then draw/adjust each bung as a rotated four-corner object.\n"
            "6. Add any needed custom classes from the right-side Custom Classes panel.\n"
            "7. Save labels.\n"
            "8. Export the OBB dataset for YOLO obb training.\n\n"
            "BungVision import review\n"
            "- Imported BungVision JSON labels that do not contain a review marker show as REVIEW in the captured-image list.\n"
            "- Use Find Unreviewed or Show only needs review to work through them.\n"
            "- Clicking Save, Save + Next, or Mark Reviewed writes the reviewed marker into the sidecar JSON.\n- Export defaults to reviewed images only so unreviewed BungVision imports do not go into training by accident.\n\n"
            "OBB controls\n"
            "- Right-click or Ctrl-click selects an annotation.\n"
            "- Arrow keys nudge the selected object. Shift+Arrow nudges 10 pixels.\n"
            "- Mouse wheel zooms. Middle-drag or Alt-drag pans.\n\n"
            "Custom labels\n"
            "Use the Custom Classes box on the right side to add labels like terminal, label, cap, barcode, or date_code.\n"
            "Class IDs should stay stable after you train a model. Rename classes instead of reordering them.\n\n"
            "Export rules\n"
            "- OBB export writes YOLO oriented labels: class x1 y1 x2 y2 x3 y3 x4 y4.\n"
            "- Legacy box labels are converted to zero-rotation OBB rectangles during OBB export so old images remain usable.\n"
            "- Detect export remains available only as a compatibility fallback.\n"
        )
        layout.addWidget(guide)
        return w

    def _recipe_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        form_box = QGroupBox("Grouped Recipe")
        form = QFormLayout(form_box)
        self.group_edit = QLineEdit(self.recipe.group)
        self.group_edit.setMaximumWidth(180)
        self.group_edit.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.model_edit = QLineEdit(self.recipe.model)
        self.model_edit.setMaximumWidth(180)
        self.model_edit.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.expected_spin = QLineEdit(str(self.recipe.expected_bungs))
        self.expected_spin.setObjectName("expectedBungsEdit")
        self.expected_spin.setValidator(QIntValidator(1, 99, self))
        self.expected_spin.setMaxLength(2)
        self.expected_spin.setFixedWidth(48)
        self.expected_spin.setPlaceholderText("6")
        self.expected_spin.editingFinished.connect(self._sync_expected_bungs_from_text)
        self.notes_edit = QTextEdit()
        self.notes_edit.setFixedHeight(54)
        self.notes_edit.setPlaceholderText("Notes, lighting setup, lens height, battery family, etc.")
        form.addRow("Group", self.group_edit)
        form.addRow("Model", self.model_edit)
        form.addRow("Expected bungs", self.expected_spin)
        form.addRow("Notes", self.notes_edit)

        save_btn = QPushButton("Save Recipe")
        save_btn.clicked.connect(self.save_recipe_from_ui)
        load_btn = QPushButton("Load Selected Recipe")
        load_btn.clicked.connect(self._load_selected_recipe)

        layout.addWidget(form_box)
        layout.addWidget(save_btn)
        layout.addWidget(load_btn)
        layout.addWidget(QLabel("Recipes"))
        layout.addWidget(self.recipe_list)
        return w

    def _capture_tab(self) -> QWidget:
        """Live capture tab.

        v0.9.19 keeps the capture controls compact and readable.
        Resolution fields are plain manual-entry boxes with no spinner arrows,
        defaulting to the Basler 5MP resolution used for BungVision testing.
        """
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer_layout.addWidget(scroll)

        w = QWidget()
        scroll.setWidget(w)
        layout = QVBoxLayout(w)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        cam_box = QGroupBox("Camera / Stream")
        cam_layout = QVBoxLayout(cam_box)
        cam_layout.setContentsMargins(10, 10, 10, 10)
        cam_layout.setSpacing(8)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["Auto", "V4L2", "GStreamer", "FFmpeg", "Basler/Pylon"])
        self.backend_combo.setCurrentText(str(self.camera_settings.get("camera_backend", "V4L2")))
        self.backend_combo.currentTextChanged.connect(self._on_camera_backend_changed)
        self.backend_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        form.addRow("Backend", self.backend_combo)

        self.source_edit = QLineEdit(str(self.camera_settings.get("camera_source", "0")))
        self.source_edit.setPlaceholderText("0, /dev/video0, video.mp4, rtsp://, or Basler serial")
        self.source_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        form.addRow("Source", self.source_edit)

        self.width_spin = QLineEdit(str(int(self.camera_settings.get("width", 2592) or 2592)))
        self.width_spin.setValidator(QIntValidator(0, 8192, self))
        self.width_spin.setPlaceholderText("2592")
        self.height_spin = QLineEdit(str(int(self.camera_settings.get("height", 1944) or 1944)))
        self.height_spin.setValidator(QIntValidator(0, 8192, self))
        self.height_spin.setPlaceholderText("1944")
        self.fps_spin = QLineEdit(str(int(self.camera_settings.get("fps", 0) or 0)))
        self.fps_spin.setValidator(QIntValidator(0, 240, self))
        self.fps_spin.setPlaceholderText("Default")
        self.preview_scale_combo = QComboBox()
        self.preview_scale_combo.addItems(["Full", "1/2", "1/3", "1/4"])
        self.preview_scale_combo.setCurrentText(str(self.camera_settings.get("preview_scale", "1/2") or "1/2"))
        self.exposure_auto_check = QCheckBox("Auto exposure")
        self.exposure_auto_check.setChecked(bool(self.camera_settings.get("exposure_auto", True)))
        self.exposure_auto_check.stateChanged.connect(self._on_exposure_auto_changed)
        self.exposure_us_edit = QLineEdit(str(int(self.camera_settings.get("exposure_us", 0) or 0)))
        self.exposure_us_edit.setValidator(QIntValidator(0, 10000000, self))
        self.exposure_us_edit.setPlaceholderText("Manual us")
        self.exposure_us_edit.setAlignment(Qt.AlignCenter)
        self.exposure_us_edit.setFixedWidth(92)
        self.apply_exposure_btn = QPushButton("Apply Exposure")
        self.apply_exposure_btn.clicked.connect(self.apply_exposure_to_camera)

        # Keep manual camera-format fields compact. These are QLineEdit boxes
        # on purpose: the operator types exact resolutions instead of clicking
        # spinner arrows.
        for edit in (self.width_spin, self.height_spin):
            edit.setFixedWidth(72)
            edit.setAlignment(Qt.AlignCenter)
        self.fps_spin.setFixedWidth(58)
        self.fps_spin.setAlignment(Qt.AlignCenter)
        self.preview_scale_combo.setFixedWidth(82)

        format_grid = QGridLayout()
        format_grid.setHorizontalSpacing(8)
        format_grid.setVerticalSpacing(6)
        format_grid.addWidget(QLabel("Width"), 0, 0)
        format_grid.addWidget(self.width_spin, 0, 1)
        format_grid.addWidget(QLabel("Height"), 0, 2)
        format_grid.addWidget(self.height_spin, 0, 3)
        format_grid.addWidget(QLabel("FPS"), 1, 0)
        format_grid.addWidget(self.fps_spin, 1, 1)
        format_grid.addWidget(QLabel("Preview"), 1, 2)
        format_grid.addWidget(self.preview_scale_combo, 1, 3)
        format_grid.setColumnStretch(4, 1)
        form.addRow("Format", format_grid)

        self.apply_exposure_btn.setProperty("compactCaptureButton", True)
        self.apply_exposure_btn.setMinimumHeight(24)
        self.apply_exposure_btn.setMaximumHeight(26)
        self.apply_exposure_btn.setMaximumWidth(150)
        self.apply_exposure_btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        exposure_box = QGroupBox("Exposure")
        exposure_layout = QGridLayout(exposure_box)
        exposure_layout.setHorizontalSpacing(8)
        exposure_layout.setVerticalSpacing(8)
        exposure_layout.addWidget(self.exposure_auto_check, 0, 0, 1, 2)
        exposure_layout.addWidget(QLabel("Manual us"), 1, 0)
        exposure_layout.addWidget(self.exposure_us_edit, 1, 1)
        exposure_layout.addWidget(self.apply_exposure_btn, 2, 0, 1, 2, Qt.AlignLeft)
        exposure_layout.setColumnStretch(1, 1)

        self._on_exposure_auto_changed()
        # The detailed camera controls are edited in a popup to keep this tab compact.

        self.basler_hint_label = QLabel("Basler: Source may be blank or a serial/model filter.")
        self.basler_hint_label.setWordWrap(True)
        self.basler_hint_label.setStyleSheet("color: #94a3b8; font-size: 8pt;")

        opts_box = QGroupBox("Camera Options")
        opts_layout = QVBoxLayout(opts_box)
        opts_layout.setContentsMargins(10, 10, 10, 10)
        opts_layout.setSpacing(4)
        self.force_v4l2_check = QCheckBox("Force V4L2 — use Linux USB-camera backend")
        self.force_v4l2_check.setChecked(bool(self.camera_settings.get("force_v4l2", True)))
        self.low_latency_check = QCheckBox("Low latency — reduce buffering/delay")
        self.low_latency_check.setChecked(bool(self.camera_settings.get("low_latency", True)))
        self.threaded_camera_check = QCheckBox("Threaded reader — smoother preview capture")
        self.threaded_camera_check.setChecked(bool(self.camera_settings.get("threaded_camera", True)))
        self.mjpg_check = QCheckBox("MJPG — request compressed camera stream")
        self.mjpg_check.setChecked(bool(self.camera_settings.get("mjpg", True)))
        self.skip_heavy_live_check = QCheckBox("Skip heavy filters — keep live view faster")
        self.skip_heavy_live_check.setChecked(bool(self.camera_settings.get("skip_heavy_live", True)))
        for widget in [
            self.force_v4l2_check,
            self.low_latency_check,
            self.threaded_camera_check,
            self.mjpg_check,
            self.skip_heavy_live_check,
        ]:
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            opts_layout.addWidget(widget)

        self._camera_settings_hidden = QWidget()
        self._camera_settings_hidden.setVisible(False)
        hidden_layout = QVBoxLayout(self._camera_settings_hidden)
        hidden_layout.setContentsMargins(0, 0, 0, 0)
        hidden_layout.setSpacing(0)
        hidden_layout.addLayout(form)
        hidden_layout.addWidget(exposure_box)
        hidden_layout.addWidget(self.basler_hint_label)
        hidden_layout.addWidget(opts_box)
        cam_layout.addWidget(self._camera_settings_hidden)
        self._on_camera_backend_changed(self.backend_combo.currentText())

        camera_settings_btn = QPushButton("Camera Settings...")
        camera_settings_btn.clicked.connect(self.open_camera_settings_dialog)
        camera_settings_btn.setProperty("compactCaptureButton", True)
        camera_settings_btn.setMinimumHeight(24)
        camera_settings_btn.setMaximumHeight(26)
        self.camera_settings_summary = QLabel()
        self.camera_settings_summary.setWordWrap(True)
        self.camera_settings_summary.setStyleSheet("color: #cbd5e1;")
        cam_layout.addWidget(camera_settings_btn, 0, Qt.AlignLeft)
        cam_layout.addWidget(self.camera_settings_summary)
        self._update_camera_settings_summary()

        self.test_cam_btn = QPushButton("Test")
        self.test_cam_btn.clicked.connect(self.test_camera)
        self.open_cam_btn = QPushButton("Open Preview")
        self.open_cam_btn.clicked.connect(self.open_camera)
        self.close_cam_btn = QPushButton("Stop")
        self.close_cam_btn.clicked.connect(self.close_camera)
        cap_raw = QPushButton("Capture Raw")
        cap_raw.clicked.connect(lambda: self.capture_frame(save_adjusted=False))
        cap_adj = QPushButton("Capture Adjusted")
        cap_adj.clicked.connect(lambda: self.capture_frame(save_adjusted=True))

        control_box = QGroupBox("Actions")
        control_layout = QGridLayout(control_box)
        control_layout.setContentsMargins(8, 8, 8, 8)
        control_layout.setHorizontalSpacing(6)
        control_layout.setVerticalSpacing(4)
        control_buttons = [self.test_cam_btn, self.open_cam_btn, self.close_cam_btn, cap_raw, cap_adj]
        for i, btn in enumerate(control_buttons):
            btn.setProperty("compactCaptureButton", True)
            btn.setMinimumHeight(24)
            btn.setMaximumHeight(26)
            btn.setMinimumWidth(0)
            btn.setMaximumWidth(16777215)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            control_layout.addWidget(btn, i // 2, i % 2)
        control_layout.setColumnStretch(0, 1)
        control_layout.setColumnStretch(1, 1)

        list_header = QLabel("Captured Images")
        list_header.setStyleSheet("font-weight: 700;")

        review_box = QGroupBox("Review Filter")
        review_layout = QVBoxLayout(review_box)
        review_layout.setContentsMargins(8, 8, 8, 8)
        review_layout.setSpacing(4)
        self.show_unreviewed_only_check = QCheckBox("Show only needs review")
        self.show_unreviewed_only_check.setToolTip("Show only images with imported/saved JSON labels that have not been marked reviewed.")
        self.show_unreviewed_only_check.stateChanged.connect(self._refresh_images)
        find_unreviewed = QPushButton("Find Unreviewed")
        find_unreviewed.clicked.connect(self.find_next_unreviewed_image)
        mark_reviewed = QPushButton("Mark Current Reviewed")
        mark_reviewed.clicked.connect(self.mark_current_reviewed)
        force_reviewed = QPushButton("Force Review Current")
        force_reviewed.setToolTip("Use this only when you intentionally want a mismatch image exported, such as a missing-bung/fail example.")
        force_reviewed.clicked.connect(self.force_mark_current_reviewed)
        for btn in (find_unreviewed, mark_reviewed, force_reviewed):
            btn.setProperty("compactCaptureButton", True)
            btn.setMinimumHeight(24)
            btn.setMaximumHeight(26)
            btn.setMinimumWidth(0)
            btn.setMaximumWidth(16777215)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        review_layout.addWidget(self.show_unreviewed_only_check)
        review_grid = QGridLayout()
        review_grid.setHorizontalSpacing(6)
        review_grid.setVerticalSpacing(4)
        for i, btn in enumerate((find_unreviewed, mark_reviewed, force_reviewed)):
            review_grid.addWidget(btn, i // 2, i % 2)
        review_grid.setColumnStretch(0, 1)
        review_grid.setColumnStretch(1, 1)
        review_layout.addLayout(review_grid)

        load_selected = QPushButton("Load Selected")
        load_selected.clicked.connect(self._load_selected_image)
        delete_selected = QPushButton("Delete Image")
        delete_selected.clicked.connect(self.delete_selected_image)
        for btn in (load_selected, delete_selected):
            btn.setProperty("compactCaptureButton", True)
            btn.setMinimumHeight(24)
            btn.setMaximumHeight(26)
            btn.setMinimumWidth(0)
            btn.setMaximumWidth(16777215)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        image_button_row = QHBoxLayout()
        image_button_row.setContentsMargins(0, 0, 0, 0)
        image_button_row.setSpacing(6)
        image_button_row.addWidget(load_selected)
        image_button_row.addWidget(delete_selected)

        layout.addWidget(cam_box)
        layout.addWidget(control_box)
        layout.addWidget(review_box)
        layout.addWidget(list_header)
        layout.addWidget(self.image_list, 1)
        layout.addLayout(image_button_row)
        return outer

    def _slider(self, minv, maxv, val, cb) -> QSlider:
        s = QSlider(Qt.Horizontal)
        s.setRange(minv, maxv)
        s.setValue(val)
        s.valueChanged.connect(cb)
        return s

    def _adjust_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        box = QGroupBox("Non-destructive Preview")
        form = QFormLayout(box)
        self.brightness_slider = self._slider(-100, 100, self.recipe.brightness, self._adjustment_changed)
        self.contrast_slider = self._slider(-100, 100, self.recipe.contrast, self._adjustment_changed)
        self.gamma_slider = self._slider(20, 300, int(self.recipe.gamma * 100), self._adjustment_changed)
        self.sharpen_slider = self._slider(0, 100, self.recipe.sharpen, self._adjustment_changed)
        self.clahe_check = QCheckBox("Enable CLAHE")
        self.clahe_check.setChecked(self.recipe.clahe_enabled)
        self.clahe_check.stateChanged.connect(self._adjustment_changed)
        self.clahe_clip_slider = self._slider(5, 100, int(self.recipe.clahe_clip * 10), self._adjustment_changed)
        self.clahe_grid_slider = self._slider(2, 16, self.recipe.clahe_grid, self._adjustment_changed)
        form.addRow("Brightness", self.brightness_slider)
        form.addRow("Contrast", self.contrast_slider)
        form.addRow("Gamma", self.gamma_slider)
        form.addRow("Sharpen", self.sharpen_slider)
        form.addRow("CLAHE", self.clahe_check)
        form.addRow("CLAHE clip", self.clahe_clip_slider)
        form.addRow("CLAHE grid", self.clahe_grid_slider)

        reset = QPushButton("Reset Adjustments")
        reset.clicked.connect(self.reset_adjustments)
        layout.addWidget(box)
        layout.addWidget(reset)
        layout.addStretch()
        return w


    def _right_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        label_box = QGroupBox("Annotation")
        v = QVBoxLayout(label_box)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(4)
        self.mode_label = QLabel("Mode: Labeling")
        self.mode_label.setWordWrap(True)
        self.guidance_label = QLabel("OBB labels: drag to draw, then adjust the four corner handles.")
        self.guidance_label.setWordWrap(True)
        self.guidance_label.setStyleSheet("color: #bfdbfe; font-weight: 600;")

        self.class_combo = QComboBox()
        self.class_combo.addItems(self.class_names)
        self.class_combo.currentIndexChanged.connect(self._class_changed)
        self.tool_combo = QComboBox()
        self.tool_combo.addItem("OBB / 4-corner", "obb")
        self.tool_combo.addItem("Box fallback", "box")
        self.tool_combo.currentIndexChanged.connect(self._tool_changed)
        self.count_label = QLabel("Battery: 0 / 1   Bungs: 0 / expected 6")
        self.dataset_label = QLabel("Dataset: 0 images, 0 labeled, 0 ready")

        save = QPushButton("Save")
        save.clicked.connect(self.save_labels)
        save_next = QPushButton("Save + Next")
        save_next.clicked.connect(self.save_and_next)
        copy_prev = QPushButton("Copy Prev")
        copy_prev.clicked.connect(self.copy_previous_labels)
        qa_btn = QPushButton("Find Problem")
        qa_btn.clicked.connect(self.find_next_problem_image)
        delete = QPushButton("Delete Box")
        delete.setToolTip("Delete only the selected on-screen box. Click Save when you want to write the change.")
        delete.clicked.connect(self.canvas.delete_selected)
        clear = QPushButton("Clear Boxes")
        clear.setToolTip("Clear the on-screen boxes for this image without deleting or overwriting the saved JSON label file.")
        clear.clicked.connect(self.clear_boxes_unsaved)
        clear_saved = QPushButton("Delete Saved JSON")
        clear_saved.setToolTip("Delete the saved .json labels for this image after confirmation.")
        clear_saved.clicked.connect(self.delete_saved_labels_confirmed)

        zminus = QPushButton("−")
        zminus.clicked.connect(self.canvas.zoom_out)
        zfit = QPushButton("Fit")
        zfit.clicked.connect(self.canvas.fit_to_window)
        zplus = QPushButton("+")
        zplus.clicked.connect(self.canvas.zoom_in)

        right_panel_buttons = (save, save_next, copy_prev, qa_btn, delete, clear, clear_saved, zminus, zfit, zplus)
        for btn in right_panel_buttons:
            btn.setProperty("rightPanelButton", True)
            btn.setMinimumHeight(24)
            btn.setMaximumHeight(26)
            btn.setMinimumWidth(0)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)
        form.addRow("Class", self.class_combo)
        form.addRow("Tool", self.tool_combo)

        def button_row(*buttons: QPushButton) -> QHBoxLayout:
            row = QHBoxLayout()
            row.setSpacing(6)
            row.setContentsMargins(0, 0, 0, 0)
            for b in buttons:
                row.addWidget(b)
            return row

        v.addWidget(self.mode_label)
        v.addWidget(self.guidance_label)
        v.addLayout(form)
        v.addWidget(self.count_label)
        v.addWidget(self.dataset_label)
        v.addLayout(button_row(zminus, zfit, zplus))
        v.addLayout(button_row(save, save_next))
        v.addLayout(button_row(copy_prev, qa_btn))
        v.addLayout(button_row(delete, clear))
        v.addWidget(clear_saved)

        class_box = QGroupBox("Custom Classes")
        cv = QVBoxLayout(class_box)
        cv.setContentsMargins(8, 8, 8, 8)
        cv.setSpacing(4)
        self.class_list_label = QLabel()
        self.class_list_label.setWordWrap(True)
        self.new_class_edit = QLineEdit()
        self.new_class_edit.setPlaceholderText("battery, bung, retainer, terminal, label...")
        self.new_class_tool_combo = QComboBox()
        self.new_class_tool_combo.addItem("OBB / 4-corner", "OBB")
        self.new_class_tool_combo.addItem("Box fallback", "BOX")
        self.new_class_tool_combo.setCurrentIndex(0)
        self.new_class_tool_combo.setToolTip("Default annotation tool for the new class. OBB is the normal BungVision training workflow; Box is only for compatibility/fallback labels.")
        add_class_btn = QPushButton("Add Custom Class")
        add_class_btn.clicked.connect(self.add_custom_class)
        apply_tool_btn = QPushButton("Apply Tool to Selected Class")
        apply_tool_btn.setToolTip("Change the default tool for the class currently selected in the Annotation Class dropdown.")
        apply_tool_btn.clicked.connect(self.apply_selected_class_tool_default)
        for btn in (add_class_btn, apply_tool_btn):
            btn.setProperty("rightPanelButton", True)
            btn.setMinimumHeight(24)
            btn.setMaximumHeight(26)
            btn.setMinimumWidth(0)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        cv.addWidget(self.class_list_label)
        cv.addWidget(QLabel("New class"))
        cv.addWidget(self.new_class_edit)
        cv.addWidget(QLabel("Default tool"))
        cv.addWidget(self.new_class_tool_combo)
        cv.addWidget(add_class_btn)
        cv.addWidget(apply_tool_btn)
        self._refresh_class_list_label()

        export_box = QGroupBox("Export")
        ev = QVBoxLayout(export_box)
        ev.setContentsMargins(8, 8, 8, 8)
        ev.setSpacing(4)
        self.export_task_combo = QComboBox()
        self.export_task_combo.addItem("OBB dataset - batteries + bungs", "obb")
        self.export_task_combo.addItem("Detect boxes dataset - compatibility", "detect")
        self.export_class_mode = QComboBox()
        self.export_class_mode.addItem("Use annotation class names", "label_names")
        self.export_class_mode.addItem("Model battery + model bung", "model_specific")
        self.export_class_mode.addItem("Model battery + generic bung", "battery_model_generic_bung")
        self.export_class_mode.addItem("Generic battery + generic bung", "generic")
        self.export_class_mode.setCurrentIndex(0)
        exp = QPushButton("Export Dataset")
        exp.clicked.connect(self.export_yolo)
        exp_all = QPushButton("Export All")
        exp_all.clicked.connect(self.export_all_yolo)
        for btn in (exp, exp_all):
            btn.setProperty("rightPanelButton", True)
            btn.setMinimumHeight(24)
            btn.setMaximumHeight(26)
            btn.setMinimumWidth(0)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        export_btn_row = QHBoxLayout()
        export_btn_row.setSpacing(6)
        export_btn_row.addWidget(exp)
        export_btn_row.addWidget(exp_all)
        ev.addWidget(QLabel("Export task"))
        ev.addWidget(self.export_task_combo)
        ev.addWidget(QLabel("Class export mode"))
        ev.addWidget(self.export_class_mode)
        ev.addLayout(export_btn_row)
        export_note = QLabel("YOLO OBB export. Reviewed and force-reviewed images only.")
        export_note.setWordWrap(True)
        export_note.setStyleSheet("color: #94a3b8;")
        ev.addWidget(export_note)

        layout.addWidget(label_box)
        layout.addWidget(class_box)
        layout.addWidget(export_box)
        layout.addStretch(1)
        return w


    def _class_changed(self, idx: int) -> None:
        if idx < 0:
            idx = 0
        if idx >= len(self.class_names):
            return
        self.canvas.class_id = idx
        self.canvas.class_name = self.class_names[idx]
        default_tool = self._default_tool_for_class(idx)
        if hasattr(self, "tool_combo"):
            target = "obb" if default_tool == "OBB" else "box"
            tool_idx = self.tool_combo.findData(target)
            if tool_idx >= 0:
                self.tool_combo.blockSignals(True)
                self.tool_combo.setCurrentIndex(tool_idx)
                self.tool_combo.blockSignals(False)
                self.canvas.set_annotation_kind(target)
        if hasattr(self, "guidance_label"):
            if self.canvas.annotation_kind == "obb":
                self.guidance_label.setText("OBB Tool — drag to draw, then adjust the four corner handles.")
            else:
                self.guidance_label.setText("Box fallback — draw a normal axis-aligned YOLO box.")

    def _tool_changed(self, idx: int) -> None:
        kind = self.tool_combo.currentData() if hasattr(self, "tool_combo") else "box"
        self.canvas.set_annotation_kind(kind)
        if hasattr(self, "guidance_label"):
            if kind == "obb":
                self.guidance_label.setText("OBB Tool — drag to draw, then adjust the four corner handles.")
            else:
                self.guidance_label.setText("Box fallback — draw a normal axis-aligned YOLO box.")

    def _default_tool_for_class(self, class_id: int) -> str:
        for c in self.class_config:
            if int(c.get("id", -1)) == int(class_id):
                tool = str(c.get("default_tool", "OBB")).upper()
                return "BOX" if tool == "BOX" else "OBB"
        return "OBB"

    def _refresh_class_combo(self) -> None:
        self.class_names = class_names_from_config(self.class_config)
        if not hasattr(self, "class_combo"):
            return
        current = self.class_combo.currentIndex()
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItems(self.class_names)
        self.class_combo.setCurrentIndex(max(0, min(current, len(self.class_names) - 1)))
        self.class_combo.blockSignals(False)
        self._class_changed(self.class_combo.currentIndex())
        self._refresh_class_list_label()

    def _refresh_class_list_label(self) -> None:
        if not hasattr(self, "class_list_label"):
            return
        rows = []
        for c in sorted(self.class_config, key=lambda x: int(x.get("id", 0))):
            if not c.get("enabled", True):
                continue
            role = str(c.get("role", "custom"))
            tool = str(c.get("default_tool", "OBB")).upper()
            tool = "BOX" if tool == "BOX" else "OBB"
            rows.append(f"{int(c.get('id', 0))}: {c.get('name')} ({role}, {tool})")
        self.class_list_label.setText("\n".join(rows))

    def _add_class_record(self, name: str, role: str, layout: str = "none", default_tool: str = "OBB") -> bool:
        safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name).strip("_") or "custom"
        existing = {str(c.get("name", "")).lower() for c in self.class_config}
        if safe.lower() in existing:
            QMessageBox.information(self, "Class", f"Class already exists: {safe}")
            return False
        next_id = max([int(c.get("id", -1)) for c in self.class_config], default=-1) + 1
        tool = "BOX" if str(default_tool).upper() == "BOX" else "OBB"
        self.class_config.append({
            "id": next_id,
            "name": safe,
            "default_tool": tool,
            "enabled": True,
            "role": role,
            "tool_locked": True,
        })
        save_class_config(self.class_config)
        self._refresh_class_combo()
        try:
            idx = self.class_names.index(safe)
            self.class_combo.setCurrentIndex(idx)
        except ValueError:
            self.class_combo.setCurrentIndex(max(0, self.class_combo.count() - 1))
        self.status.showMessage(f"Added class {next_id}: {safe} ({tool})", 5000)
        return True

    def apply_selected_class_tool_default(self) -> None:
        if not hasattr(self, "class_combo") or not hasattr(self, "new_class_tool_combo"):
            return
        class_id = int(self.class_combo.currentIndex())
        if class_id < 0:
            return
        tool = "BOX" if str(self.new_class_tool_combo.currentData()).upper() == "BOX" else "OBB"
        changed = False
        for c in self.class_config:
            if int(c.get("id", -1)) == class_id:
                c["default_tool"] = tool
                c["tool_locked"] = True
                changed = True
                break
        if not changed:
            QMessageBox.warning(self, "Class Tool", "Could not find the selected class in the class configuration.")
            return
        save_class_config(self.class_config)
        self._refresh_class_combo()
        self.class_combo.setCurrentIndex(class_id)
        self.status.showMessage(f"Set class {class_id} default tool to {tool}", 5000)

    def add_custom_class(self) -> None:
        name = self.new_class_edit.text().strip() if hasattr(self, "new_class_edit") else ""
        if not name:
            QMessageBox.information(self, "Custom Class", "Enter a non-battery class name first, for example terminal, label, cap, barcode, or date_code.")
            return
        safe = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name).strip("_") or "custom"
        inferred_role, _inferred_layout = infer_role_and_layout(safe)
        default_tool = self.new_class_tool_combo.currentData() if hasattr(self, "new_class_tool_combo") else "OBB"
        if self._add_class_record(safe, inferred_role, "none", str(default_tool)):
            self.new_class_edit.clear()


    def set_class_by_name(self, name: str) -> None:
        if not hasattr(self, "class_combo"):
            return
        for i, n in enumerate(self.class_names):
            if n == name:
                self.class_combo.setCurrentIndex(i)
                return

    def _apply_theme(self) -> None:
        checkbox_check = (Path(__file__).resolve().parent / "assets" / "checkbox_check.svg").as_posix()
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #0f172a;
                color: #e5e7eb;
                font-size: 9pt;
            }
            QMenuBar {
                background: #0b1220;
                color: #dbeafe;
                padding: 2px 6px;
                spacing: 12px;
                border-bottom: 1px solid #334155;
            }
            QMenuBar::item {
                background: transparent;
                padding: 4px 10px;
                border-radius: 6px;
            }
            QMenuBar::item:selected { background: #1e293b; }
            QMenu {
                background: #111827;
                color: #e5e7eb;
                border: 1px solid #334155;
                padding: 6px;
            }
            QMenu::item { padding: 7px 28px 7px 18px; border-radius: 5px; }
            QMenu::item:selected { background: #1d4ed8; }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 10px;
                margin-top: 12px;
                padding: 10px 8px 8px 8px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #93c5fd;
            }
            QPushButton {
                min-height: 28px;
                padding: 4px 8px;
                background: #1d4ed8;
                color: white;
                border: 0;
                border-radius: 7px;
                font-weight: 700;
                text-align: center;
            }
            QPushButton[compactCaptureButton="true"], QPushButton[rightPanelButton="true"] {
                min-height: 22px;
                max-height: 26px;
                padding: 2px 7px;
                border-radius: 5px;
            }
            QPushButton:hover { background: #2563eb; }
            QPushButton:pressed { background: #1e40af; }
            QCheckBox {
                spacing: 8px;
                color: #e5e7eb;
                min-height: 24px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #93c5fd;
                border-radius: 4px;
                background: #020617;
            }
            QCheckBox::indicator:hover { border: 2px solid #bfdbfe; }
            QCheckBox::indicator:checked {
                background: #020617;
                border: 2px solid #bfdbfe;
                image: url("__CHECKBOX_CHECK__");
            }
            QCheckBox::indicator:disabled {
                border: 2px solid #475569;
                background: #0f172a;
            }
            QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget {
                background: #111827;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 4px 6px;
                color: #e5e7eb;
                min-height: 24px;
            }
            QLineEdit#expectedBungsEdit, QLineEdit#countRequiredEdit {
                padding: 4px 5px;
                min-height: 24px;
                max-width: 48px;
            }
            QComboBox QAbstractItemView {
                min-height: 120px;
                selection-background-color: #1d4ed8;
            }
            QTextEdit { min-height: 56px; }
            QLabel { padding: 2px 0; }
            QTabWidget::pane {
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 6px;
            }
            QTabBar::tab {
                background: #111827;
                color: #cbd5e1;
                padding: 6px 8px;
                margin-right: 3px;
                border: 1px solid #334155;
                border-bottom: 0;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                min-width: 70px;
            }
            QTabBar::tab:selected { background: #1e293b; color: white; }
            QSlider::groove:horizontal { height: 6px; background: #334155; border-radius: 3px; }
            QSlider::handle:horizontal { width: 18px; background: #60a5fa; margin: -6px 0; border-radius: 9px; }
            QScrollArea { border: 0; }
        """.replace("__CHECKBOX_CHECK__", checkbox_check))

        for widget in self.findChildren(QWidget):
            layout = widget.layout()
            if layout is not None:
                layout.setContentsMargins(6, 6, 6, 6)
                layout.setSpacing(6)

    def _on_exposure_auto_changed(self, *args) -> None:
        manual = not (self.exposure_auto_check.isChecked() if hasattr(self, "exposure_auto_check") else True)
        if hasattr(self, "exposure_us_edit"):
            self.exposure_us_edit.setEnabled(manual)
        if hasattr(self, "apply_exposure_btn"):
            self.apply_exposure_btn.setEnabled(True)

    def apply_exposure_to_camera(self) -> None:
        auto = self.exposure_auto_check.isChecked() if hasattr(self, "exposure_auto_check") else True
        exposure_us = self._int_line_value(self.exposure_us_edit, 0) if hasattr(self, "exposure_us_edit") else 0
        msg = self.camera.set_exposure(auto, exposure_us)
        self.status.showMessage(msg, 8000)
        if not self.camera.is_open():
            QMessageBox.information(self, "Exposure", msg + "\n\nOpen Preview first to apply exposure to the active camera.")

    def _update_camera_settings_summary(self) -> None:
        if not hasattr(self, "camera_settings_summary"):
            return
        backend = self.backend_combo.currentText() if hasattr(self, "backend_combo") else "Auto"
        source = self.source_edit.text().strip() if hasattr(self, "source_edit") else ""
        if not source and backend == "Basler/Pylon":
            source = "Any Basler"
        elif not source:
            source = "0"
        width = self._int_line_value(self.width_spin, 0) if hasattr(self, "width_spin") else 0
        height = self._int_line_value(self.height_spin, 0) if hasattr(self, "height_spin") else 0
        fps = self._int_line_value(self.fps_spin, 0) if hasattr(self, "fps_spin") else 0
        preview = self.preview_scale_combo.currentText() if hasattr(self, "preview_scale_combo") else "1/2"
        exposure = "auto" if (self.exposure_auto_check.isChecked() if hasattr(self, "exposure_auto_check") else True) else f"{self._int_line_value(self.exposure_us_edit, 0)} us"
        size = f"{width}x{height}" if width and height else "default size"
        fps_text = f"{fps} FPS" if fps else "default FPS"
        self.camera_settings_summary.setText(f"{backend} | Source {source} | {size}, {fps_text} | Preview {preview} | Exposure {exposure}")

    def _camera_stream_signature(self) -> tuple:
        """Return settings that require a camera reopen to take effect.

        Preview scale and exposure can be applied without renegotiating the
        stream, but backend/source/format/backend options cannot. Keeping this
        separate prevents the UI from looking like a resolution change applied
        while the live reader is still showing the old camera mode.
        """
        backend = self.backend_combo.currentText() if hasattr(self, "backend_combo") else "Auto"
        is_basler = backend == "Basler/Pylon"
        return (
            backend,
            self.source_edit.text().strip() if hasattr(self, "source_edit") else "0",
            self._int_line_value(self.width_spin, 0) if hasattr(self, "width_spin") else 0,
            self._int_line_value(self.height_spin, 0) if hasattr(self, "height_spin") else 0,
            self._int_line_value(self.fps_spin, 0) if hasattr(self, "fps_spin") else 0,
            False if is_basler else (self.force_v4l2_check.isChecked() if hasattr(self, "force_v4l2_check") else False),
            False if is_basler else (self.mjpg_check.isChecked() if hasattr(self, "mjpg_check") else False),
            self.low_latency_check.isChecked() if hasattr(self, "low_latency_check") else True,
            self.threaded_camera_check.isChecked() if hasattr(self, "threaded_camera_check") else True,
        )

    def open_camera_settings_dialog(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Camera Settings")
        dlg.setMinimumWidth(460)
        dlg_font = self.font()
        if dlg_font.pointSize() <= 0:
            dlg_font.setPointSize(9)
        dlg.setFont(dlg_font)
        layout = QVBoxLayout(dlg)

        form_box = QGroupBox("Camera / Stream")
        form = QFormLayout(form_box)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        backend_combo = QComboBox()
        backend_combo.addItems(["Auto", "V4L2", "GStreamer", "FFmpeg", "Basler/Pylon"])
        backend_combo.setCurrentText(self.backend_combo.currentText())
        source_edit = QLineEdit(self.source_edit.text())
        source_edit.setPlaceholderText("0, /dev/video0, video.mp4, rtsp://, or Basler serial")

        width_edit = QLineEdit(self.width_spin.text())
        width_edit.setValidator(QIntValidator(0, 8192, dlg))
        height_edit = QLineEdit(self.height_spin.text())
        height_edit.setValidator(QIntValidator(0, 8192, dlg))
        fps_edit = QLineEdit(self.fps_spin.text())
        fps_edit.setValidator(QIntValidator(0, 240, dlg))
        preview_combo = QComboBox()
        preview_combo.addItems(["Full", "1/2", "1/3", "1/4"])
        preview_combo.setCurrentText(self.preview_scale_combo.currentText())

        format_grid = QGridLayout()
        format_grid.setHorizontalSpacing(8)
        format_grid.setVerticalSpacing(6)
        for edit in (width_edit, height_edit):
            edit.setFixedWidth(72)
            edit.setAlignment(Qt.AlignCenter)
        fps_edit.setFixedWidth(58)
        fps_edit.setAlignment(Qt.AlignCenter)
        preview_combo.setFixedWidth(82)
        format_grid.addWidget(QLabel("Width"), 0, 0)
        format_grid.addWidget(width_edit, 0, 1)
        format_grid.addWidget(QLabel("Height"), 0, 2)
        format_grid.addWidget(height_edit, 0, 3)
        format_grid.addWidget(QLabel("FPS"), 1, 0)
        format_grid.addWidget(fps_edit, 1, 1)
        format_grid.addWidget(QLabel("Preview"), 1, 2)
        format_grid.addWidget(preview_combo, 1, 3)
        format_grid.setColumnStretch(4, 1)

        form.addRow("Backend", backend_combo)
        form.addRow("Source", source_edit)
        form.addRow("Format", format_grid)
        layout.addWidget(form_box)

        exposure_box = QGroupBox("Exposure")
        exposure_layout = QGridLayout(exposure_box)
        exposure_layout.setHorizontalSpacing(8)
        exposure_layout.setVerticalSpacing(6)
        exposure_auto_check = QCheckBox("Auto exposure")
        exposure_auto_check.setChecked(self.exposure_auto_check.isChecked())
        exposure_us_edit = QLineEdit(self.exposure_us_edit.text())
        exposure_us_edit.setValidator(QIntValidator(0, 10000000, dlg))
        exposure_us_edit.setFixedWidth(92)
        exposure_us_edit.setAlignment(Qt.AlignCenter)
        apply_exposure_btn = QPushButton("Apply Exposure")
        apply_exposure_btn.setProperty("compactCaptureButton", True)
        exposure_layout.addWidget(exposure_auto_check, 0, 0, 1, 2)
        exposure_layout.addWidget(QLabel("Manual us"), 1, 0)
        exposure_layout.addWidget(exposure_us_edit, 1, 1)
        exposure_layout.addWidget(apply_exposure_btn, 2, 0, 1, 2, Qt.AlignLeft)
        exposure_layout.setColumnStretch(1, 1)
        layout.addWidget(exposure_box)

        opts_box = QGroupBox("Camera Options")
        opts_layout = QVBoxLayout(opts_box)
        force_v4l2_check = QCheckBox("Force V4L2 - use Linux USB-camera backend")
        force_v4l2_check.setChecked(self.force_v4l2_check.isChecked())
        low_latency_check = QCheckBox("Low latency - reduce buffering/delay")
        low_latency_check.setChecked(self.low_latency_check.isChecked())
        threaded_camera_check = QCheckBox("Threaded reader - smoother preview capture")
        threaded_camera_check.setChecked(self.threaded_camera_check.isChecked())
        mjpg_check = QCheckBox("MJPG - request compressed camera stream")
        mjpg_check.setChecked(self.mjpg_check.isChecked())
        skip_heavy_live_check = QCheckBox("Skip heavy filters - keep live view faster")
        skip_heavy_live_check.setChecked(self.skip_heavy_live_check.isChecked())
        for widget in (force_v4l2_check, low_latency_check, threaded_camera_check, mjpg_check, skip_heavy_live_check):
            opts_layout.addWidget(widget)
        layout.addWidget(opts_box)

        def sync_enabled(*args) -> None:
            manual = not exposure_auto_check.isChecked()
            exposure_us_edit.setEnabled(manual)
            is_basler = backend_combo.currentText() == "Basler/Pylon"
            force_v4l2_check.setEnabled(not is_basler)
            mjpg_check.setEnabled(not is_basler)

        def apply_to_current(*, save: bool = False, close: bool = True) -> None:
            was_open = self.camera.is_open() if hasattr(self, "camera") else False
            old_stream_sig = self._camera_stream_signature() if was_open else None

            self.backend_combo.setCurrentText(backend_combo.currentText())
            self.source_edit.setText(source_edit.text().strip())
            self.width_spin.setText(width_edit.text().strip())
            self.height_spin.setText(height_edit.text().strip())
            self.fps_spin.setText(fps_edit.text().strip())
            self.preview_scale_combo.setCurrentText(preview_combo.currentText())
            self.exposure_auto_check.setChecked(exposure_auto_check.isChecked())
            self.exposure_us_edit.setText(exposure_us_edit.text().strip())
            self.force_v4l2_check.setChecked(force_v4l2_check.isChecked())
            self.low_latency_check.setChecked(low_latency_check.isChecked())
            self.threaded_camera_check.setChecked(threaded_camera_check.isChecked())
            self.mjpg_check.setChecked(mjpg_check.isChecked())
            self.skip_heavy_live_check.setChecked(skip_heavy_live_check.isChecked())
            self._on_exposure_auto_changed()
            self._on_camera_backend_changed(self.backend_combo.currentText())
            self._update_camera_settings_summary()

            new_stream_sig = self._camera_stream_signature() if was_open else None
            stream_changed = bool(was_open and old_stream_sig != new_stream_sig)

            if save:
                self.camera_settings = {
                    "camera_source": self.source_edit.text().strip(),
                    "camera_backend": self.backend_combo.currentText(),
                    "width": self._int_line_value(self.width_spin, 2592),
                    "height": self._int_line_value(self.height_spin, 1944),
                    "fps": self._int_line_value(self.fps_spin, 0),
                    "preview_scale": self.preview_scale_combo.currentText(),
                    "exposure_auto": self.exposure_auto_check.isChecked(),
                    "exposure_us": self._int_line_value(self.exposure_us_edit, 0),
                    "force_v4l2": self.force_v4l2_check.isChecked(),
                    "low_latency": self.low_latency_check.isChecked(),
                    "threaded_camera": self.threaded_camera_check.isChecked(),
                    "mjpg": self.mjpg_check.isChecked(),
                    "skip_heavy_live": self.skip_heavy_live_check.isChecked(),
                }
                path = save_camera_settings(self.camera_settings)
                self.status.showMessage(f"Saved camera settings: {path}", 5000)
            else:
                self.status.showMessage("Camera settings applied", 4000)

            if stream_changed:
                # Width/height/FPS/backend changes do not affect an already-open
                # capture device until it is reopened. Reopen immediately so the
                # screen reflects the requested resolution instead of continuing
                # to show the stale negotiated mode.
                self.timer.stop()
                self.camera.close()
                self.blank_frame_count = 0
                self.open_camera()

            if close:
                dlg.accept()

        exposure_auto_check.stateChanged.connect(sync_enabled)
        backend_combo.currentTextChanged.connect(sync_enabled)
        apply_exposure_btn.clicked.connect(lambda: (apply_to_current(save=False, close=False), self.apply_exposure_to_camera()))
        sync_enabled()

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        apply_btn = QPushButton("Apply")
        save_btn = QPushButton("Save Settings")
        cancel_btn = QPushButton("Cancel")
        apply_btn.clicked.connect(lambda: apply_to_current(save=False))
        save_btn.clicked.connect(lambda: apply_to_current(save=True))
        cancel_btn.clicked.connect(dlg.reject)
        for btn in (apply_btn, save_btn, cancel_btn):
            btn.setProperty("compactCaptureButton", True)
            button_row.addWidget(btn)
        layout.addLayout(button_row)
        dlg.exec()

    def _int_line_value(self, edit: QLineEdit, default: int = 0) -> int:
        try:
            text = edit.text().strip()
            return int(text) if text else default
        except Exception:
            return default

    def _set_int_line_value(self, edit: QLineEdit, value: int) -> None:
        edit.setText(str(max(1, min(99, int(value)))))

    def _expected_bungs_value(self) -> int:
        value = self._int_line_value(self.expected_spin, getattr(self.recipe, "expected_bungs", 6))
        return max(1, min(99, int(value)))

    def _count_required_value(self) -> int:
        if hasattr(self, "count_required_spin"):
            return max(1, min(99, self._int_line_value(self.count_required_spin, self._expected_bungs_value())))
        return self._expected_bungs_value()

    def _sync_expected_bungs_from_text(self) -> None:
        self._set_int_line_value(self.expected_spin, self._expected_bungs_value())
        if hasattr(self, "count_required_spin") and not self.count_required_spin.text().strip():
            self._set_int_line_value(self.count_required_spin, self._expected_bungs_value())
        self._update_box_count()
        # Expected-count changes reclassify ready/problem images, so refresh the
        # dataset summary explicitly now that _update_box_count no longer does.
        self._update_dataset_summary()

    def _sync_required_count_from_text(self) -> None:
        if hasattr(self, "count_required_spin"):
            self._set_int_line_value(self.count_required_spin, self._count_required_value())

    def _current_recipe_from_ui(self) -> Recipe:
        return Recipe(
            group=self.group_edit.text().strip() or "Default",
            model=self.model_edit.text().strip() or "Battery_Model",
            expected_bungs=self._expected_bungs_value(),
            brightness=self.brightness_slider.value(),
            contrast=self.contrast_slider.value(),
            gamma=self.gamma_slider.value() / 100.0,
            clahe_enabled=self.clahe_check.isChecked(),
            clahe_clip=self.clahe_clip_slider.value() / 10.0,
            clahe_grid=self.clahe_grid_slider.value(),
            sharpen=self.sharpen_slider.value(),
            notes=self.notes_edit.toPlainText(),
        )

    def save_recipe_from_ui(self) -> None:
        old_safe = getattr(self.recipe, "safe_name", "")
        old_expected = getattr(self.recipe, "expected_bungs", None)
        self.recipe = self._current_recipe_from_ui()
        path = save_recipe(self.recipe)
        self._refresh_recipes()
        if old_safe != self.recipe.safe_name:
            self._reset_recipe_image_index()
        elif old_expected != self.recipe.expected_bungs:
            # Expected count changes affect every cached OK/CHECK status.
            self._image_status_cache.clear()
        self._refresh_images()
        self.status.showMessage(f"Saved recipe: {path}", 5000)
        self._update_camera_settings_summary()
        self._update_box_count()

    def _refresh_recipes(self) -> None:
        self.recipe_list.clear()
        for r in list_recipes():
            self.recipe_list.addItem(f"{r.group} / {r.model}")

    def _load_selected_recipe(self) -> None:
        item = self.recipe_list.currentItem()
        if not item:
            return
        text = item.text()
        group, model = [x.strip() for x in text.split("/", 1)]
        for r in list_recipes():
            if r.group == group and r.model == model:
                self.recipe = r
                self.group_edit.setText(r.group)
                self.model_edit.setText(r.model)
                self._set_int_line_value(self.expected_spin, r.expected_bungs)
                if hasattr(self, "count_required_spin"):
                    self._set_int_line_value(self.count_required_spin, r.expected_bungs)
                self.brightness_slider.setValue(r.brightness)
                self.contrast_slider.setValue(r.contrast)
                self.gamma_slider.setValue(int(r.gamma * 100))
                self.clahe_check.setChecked(r.clahe_enabled)
                self.clahe_clip_slider.setValue(int(r.clahe_clip * 10))
                self.clahe_grid_slider.setValue(r.clahe_grid)
                self.sharpen_slider.setValue(r.sharpen)
                self.notes_edit.setText(r.notes)
                self._reset_recipe_image_index()
                self._refresh_images()
                self._update_box_count()
                self.status.showMessage(f"Loaded recipe: {r.group} / {r.model}", 5000)
                return

    def _review_record(self, reason: str = "operator_review", *, force: bool = False, counts: tuple[int, int] | None = None) -> dict:
        record = {
            "reviewed": True,
            "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "reviewed_by": "BungVision Label Studio v0.9.42",
            "source": "bungvision_label_studio",
            "tool": "BungVision Label Studio",
            "reason": reason,
        }
        if force:
            batt, bung = counts if counts is not None else self._current_label_counts()
            expected = self._expected_bungs_value() if hasattr(self, "expected_spin") else self.recipe.expected_bungs
            record.update({
                "forced_review": True,
                "review_status": "forced_reviewed",
                "forced_reason": "quantity_mismatch",
                "battery_count": int(batt),
                "bung_count": int(bung),
                "expected_bungs": int(expected),
                "warning": "Operator force-reviewed this image even though the required quantities did not match.",
            })
        return record

    def _is_label_studio_review_marker(self, review: dict | None) -> bool:
        if not isinstance(review, dict) or not bool(review.get("reviewed", False)):
            return False
        text = " ".join(
            str(review.get(k, ""))
            for k in ("source", "tool", "review_source", "reviewed_by", "reviewer", "app")
        ).lower()
        return "bungvision_label_studio" in text or "bung label studio" in text or "label studio" in text

    def _annotation_reviewed(self, data: dict | None) -> bool:
        """True only for labels explicitly reviewed inside this labeler.

        BungVision runtime/import JSON can contain generic fields such as
        reviewed=true or review_status=ok/pass. Those should not count as
        operator review for training export. Legacy v0.9.28-v0.9.30 Label
        Studio markers are still accepted because they include reviewed_by
        containing "BungVision Label Studio".
        """
        if not data:
            return False
        review = data.get("review") if isinstance(data, dict) else None
        if self._is_label_studio_review_marker(review):
            return True
        if bool(data.get("reviewed", False)):
            top_level_review = {
                "reviewed": True,
                "source": data.get("review_source") or data.get("source") or data.get("origin") or data.get("imported_from"),
                "tool": data.get("review_tool") or data.get("tool") or data.get("app"),
                "reviewed_by": data.get("reviewed_by"),
            }
            return self._is_label_studio_review_marker(top_level_review)
        return False

    def _annotation_force_reviewed(self, data: dict | None) -> bool:
        if not data or not self._annotation_reviewed(data):
            return False
        review = data.get("review") if isinstance(data, dict) else None
        if isinstance(review, dict) and (
            bool(review.get("forced_review", False))
            or bool(review.get("force_reviewed", False))
            or str(review.get("review_status", "")).lower() == "forced_reviewed"
        ):
            return True
        return bool(data.get("forced_review", False) or data.get("force_reviewed", False))

    def _needs_review_for_image(self, path: Path, data: dict | None = None) -> bool:
        if data is None:
            data = load_annotations(path)
        if not data or not data.get("boxes"):
            return False
        return not self._annotation_reviewed(data)

    def _image_counts_from_data(self, data: dict | None) -> tuple[int, int]:
        if not data or not data.get("boxes"):
            return 0, 0
        boxes = [self._normalize_import_box(b) for b in data.get("boxes", [])]
        batt = sum(1 for b in boxes if str(b.get("label", "")).startswith("battery") or int(b.get("class_id", -1)) == 0)
        bung = sum(1 for b in boxes if str(b.get("label", "")).startswith("bung") or int(b.get("class_id", -1)) == 1)
        return batt, bung

    def _counts_match_required(self, batt: int, bung: int) -> bool:
        expected = self._expected_bungs_value() if hasattr(self, "expected_spin") else self.recipe.expected_bungs
        return int(batt) == 1 and int(bung) == int(expected)

    def _current_label_counts(self) -> tuple[int, int]:
        batt = sum(1 for b in self.canvas.boxes if self._box_kind(b) == "battery")
        bung = sum(1 for b in self.canvas.boxes if self._box_kind(b) == "bung")
        return batt, bung

    def _reset_recipe_image_index(self) -> None:
        """Force a full recipe-folder reindex on the next image-list refresh."""
        self._recipe_index_dirty = True
        self._image_paths_cache = []
        self._image_status_cache.clear()

    def _invalidate_image_status(self, path: Path | None) -> None:
        """Drop only one image from the cached review/status table."""
        if not path:
            return
        try:
            self._image_status_cache.pop(str(Path(path).resolve()), None)
        except Exception:
            self._image_status_cache.pop(str(path), None)

    def _json_mtime_ns(self, path: Path) -> tuple[bool, int]:
        json_path = image_label_json_path(path)
        try:
            exists = json_path.exists()
            return exists, json_path.stat().st_mtime_ns if exists else 0
        except Exception:
            return False, 0

    def _get_recipe_image_paths(self, *, force: bool = False) -> list[Path]:
        """Return cached images for the current recipe, newest first."""
        if force or getattr(self, "_recipe_index_dirty", True):
            folder = capture_folder(self.recipe)
            self._image_paths_cache = sorted(folder.glob("*.jpg"), reverse=True)
            valid = {str(p.resolve()) for p in self._image_paths_cache}
            for key in list(self._image_status_cache.keys()):
                if key not in valid:
                    self._image_status_cache.pop(key, None)
            self._recipe_index_dirty = False
        return list(self._image_paths_cache)

    def _cached_image_status(self, path: Path, *, force: bool = False) -> dict:
        """Fast per-image review/count/status lookup.

        A sidecar JSON is parsed only when it is new or its mtime changed.
        The recipe expected quantity is included so OK/CHECK status updates when
        the recipe count changes.
        """
        key = str(Path(path).resolve())
        try:
            image_mtime = Path(path).stat().st_mtime_ns if Path(path).exists() else 0
        except Exception:
            image_mtime = 0
        json_exists, json_mtime = self._json_mtime_ns(path)
        expected = self._expected_bungs_value() if hasattr(self, "expected_spin") else self.recipe.expected_bungs
        cached = self._image_status_cache.get(key)
        if (
            cached
            and not force
            and cached.get("image_mtime") == image_mtime
            and cached.get("json_exists") == json_exists
            and cached.get("json_mtime") == json_mtime
            and cached.get("expected") == int(expected)
        ):
            return cached

        data = None
        if json_exists:
            try:
                data = load_annotations(path)
            except Exception:
                data = None

        batt = bung = 0
        labeled = bool(data and data.get("boxes"))
        reviewed = forced = needs_review = False
        status = "unlabeled"

        if labeled:
            batt, bung = self._image_counts_from_data(data)
            needs_review = self._needs_review_for_image(path, data)
            reviewed = self._annotation_reviewed(data)
            forced = self._annotation_force_reviewed(data)
            if needs_review:
                status = "needs_review"
                prefix = f"🟡 REVIEW {batt}B/{bung}U  "
            elif forced:
                status = "forced"
                prefix = f"⚠ FORCE REVIEW {batt}B/{bung}U  "
            elif batt == 1 and bung == int(expected):
                status = "ready"
                prefix = "✓ REVIEWED OK  "
            else:
                status = "problem"
                prefix = "⚠ REVIEWED CHECK  "
        elif json_exists:
            prefix = "◇ JSON EMPTY  "
        else:
            prefix = "□ NO JSON  "

        entry = {
            "path": path,
            "image_mtime": image_mtime,
            "json_exists": bool(json_exists),
            "json_mtime": json_mtime,
            "expected": int(expected),
            "status": status,
            "prefix": prefix,
            "battery_count": int(batt),
            "bung_count": int(bung),
            "labeled": bool(labeled),
            "reviewed": bool(reviewed),
            "needs_review": bool(needs_review),
            "forced": bool(forced),
        }
        self._image_status_cache[key] = entry
        return entry

    def _refresh_images(self, *args, force: bool = False) -> None:
        # Qt checkbox signals pass an int state; do not treat that as a force refresh.
        if args and isinstance(args[0], bool):
            force = force or bool(args[0])

        review_only = bool(
            hasattr(self, "show_unreviewed_only_check")
            and self.show_unreviewed_only_check.isChecked()
        )
        # Build the visible list and tally the dataset summary in one pass. The
        # summary counts every image in the recipe regardless of the review-only
        # view filter, so it stays correct without a second walk of the cache.
        totals = self._new_summary_totals()
        self.image_list.setUpdatesEnabled(False)
        try:
            self.image_list.clear()
            for p in self._get_recipe_image_paths(force=force):
                entry = self._cached_image_status(p)
                self._accumulate_summary(totals, entry)
                if review_only and not entry.get("needs_review", False):
                    continue
                self.image_list.addItem(entry.get("prefix", "") + p.name)
        finally:
            self.image_list.setUpdatesEnabled(True)
        self._set_dataset_summary_label(totals)

    def _on_camera_backend_changed(self, backend: str) -> None:
        is_basler = backend == "Basler/Pylon"
        if hasattr(self, "source_edit"):
            # Keep Source editable for Basler so a serial/model filter can be typed when needed.
            self.source_edit.setEnabled(True)
            self.source_edit.setPlaceholderText("Optional Basler serial/model" if is_basler else "0, /dev/video0, video.mp4, or rtsp://")
        if hasattr(self, "force_v4l2_check"):
            self.force_v4l2_check.setEnabled(not is_basler)
        if hasattr(self, "mjpg_check"):
            self.mjpg_check.setEnabled(not is_basler)
        if hasattr(self, "basler_hint_label"):
            self.basler_hint_label.setVisible(is_basler)
        if is_basler and hasattr(self, "status"):
            self.status.showMessage("Basler/Pylon selected. Source may be left blank or set to a serial/model filter.", 5000)

    def _parse_source(self):
        """Return the camera source in the form expected by the selected backend.

        For normal OpenCV/V4L2 sources, blank means camera index 0. For
        Basler/Pylon, blank is valid and means "use the first Pylon camera";
        keeping it blank avoids accidentally treating the optional Basler source
        field like a USB /dev/video index.
        """
        backend = self.backend_combo.currentText() if hasattr(self, "backend_combo") else "Auto"
        text = self.source_edit.text().strip()
        if backend == "Basler/Pylon" and not text:
            return ""
        src = text or "0"
        if src.isdigit():
            return int(src)
        return src

    def test_camera(self) -> None:
        """Open the camera briefly and report whether frames are readable."""
        src = self._parse_source()
        width = self._int_line_value(self.width_spin, 0) or None
        height = self._int_line_value(self.height_spin, 0) or None
        backend = self.backend_combo.currentText() if hasattr(self, "backend_combo") else "Auto"
        exposure_auto = self.exposure_auto_check.isChecked() if hasattr(self, "exposure_auto_check") else True
        exposure_us = self._int_line_value(self.exposure_us_edit, 0) if hasattr(self, "exposure_us_edit") else 0
        result = quick_test_source(src, backend=backend, width=width, height=height, exposure_auto=exposure_auto, exposure_us=exposure_us)
        title = "Camera Test Passed" if result.ok else "Camera Test Failed"
        QMessageBox.information(self, title, result.message)
        self.status.showMessage(result.message, 8000)

    def open_camera(self) -> None:
        # Keep the recipe object in sync with the capture fields, but do not
        # call save_recipe_from_ui() here. That method writes "Saved recipe" to
        # the status bar, which hid camera-open failures and made the Open
        # Preview button look like it was wired to Save Recipe.
        self.recipe = self._current_recipe_from_ui()
        src = self._parse_source()
        width = self._int_line_value(self.width_spin, 0) or None
        height = self._int_line_value(self.height_spin, 0) or None
        fps = self._int_line_value(self.fps_spin, 0) or None if hasattr(self, "fps_spin") else None
        backend = self.backend_combo.currentText() if hasattr(self, "backend_combo") else "Auto"
        is_basler = backend == "Basler/Pylon"
        exposure_auto = self.exposure_auto_check.isChecked() if hasattr(self, "exposure_auto_check") else True
        exposure_us = self._int_line_value(self.exposure_us_edit, 0) if hasattr(self, "exposure_us_edit") else 0

        self.status.showMessage(f"Opening {backend} camera preview...", 3000)
        self.blank_frame_count = 0
        if not self.camera.open(
            src,
            width,
            height,
            fps=fps,
            backend=backend,
            low_latency=self.low_latency_check.isChecked() if hasattr(self, "low_latency_check") else True,
            mjpg=self.mjpg_check.isChecked() if hasattr(self, "mjpg_check") else True,
            threaded=self.threaded_camera_check.isChecked() if hasattr(self, "threaded_camera_check") else True,
            force_v4l2=(False if is_basler else (self.force_v4l2_check.isChecked() if hasattr(self, "force_v4l2_check") else False)),
            exposure_auto=exposure_auto,
            exposure_us=exposure_us,
        ):
            QMessageBox.warning(
                self,
                "Camera",
                self.camera.last_result.message
                + "\n\nTry these quick checks:\n"
                + "• Source 0, then 1, then /dev/video0 for OpenCV cameras\n"
                + "• Backend V4L2 for normal USB webcams\n"
                + "• Backend Basler/Pylon for Basler industrial cameras\n"
                + "• Width/Height set to Default\n"
                + "• Basler test: python -c \"from pypylon import pylon; print(pylon.TlFactory.GetInstance().EnumerateDevices())\"",
            )
            return
        # Force the first tick after (re)opening to process a frame.
        self._last_frame_seq = None
        self.timer.start(16)
        self.status.showMessage(self.camera.last_result.message, 8000)

    def close_camera(self) -> None:
        self.timer.stop()
        self.camera.close()
        self.status.showMessage("Live view stopped", 5000)

    def _adjustment_changed(self, *args) -> None:
        self.recipe = self._current_recipe_from_ui()
        if self.last_raw is not None and not self.camera.is_open():
            self.last_adjusted = self._adjust_frame(self.last_raw)
            self.canvas.set_frame(self.last_adjusted)

    def _adjust_frame(self, frame):
        return apply_adjustments(
            frame,
            brightness=self.brightness_slider.value(),
            contrast=self.contrast_slider.value(),
            gamma=self.gamma_slider.value() / 100.0,
            clahe_enabled=self.clahe_check.isChecked(),
            clahe_clip=self.clahe_clip_slider.value() / 10.0,
            clahe_grid=self.clahe_grid_slider.value(),
            sharpen=self.sharpen_slider.value(),
        )

    def _adjust_live_frame(self, frame):
        """Fast preview adjustment path. Keeps live view responsive while preserving full-quality capture."""
        skip_heavy = getattr(self, "skip_heavy_live_check", None)
        if skip_heavy is not None and skip_heavy.isChecked():
            return apply_adjustments(
                frame,
                brightness=self.brightness_slider.value(),
                contrast=self.contrast_slider.value(),
                gamma=self.gamma_slider.value() / 100.0,
                clahe_enabled=False,
                clahe_clip=self.clahe_clip_slider.value() / 10.0,
                clahe_grid=self.clahe_grid_slider.value(),
                sharpen=0,
            )
        return self._adjust_frame(frame)

    def _scale_preview_frame(self, frame):
        combo = getattr(self, "preview_scale_combo", None)
        if combo is None:
            return frame
        value = combo.currentText()
        if value == "Full":
            return frame
        scale = {"1/2": 0.5, "1/3": 1.0 / 3.0, "1/4": 0.25}.get(value, 1.0)
        if scale >= 0.999:
            return frame
        h, w = frame.shape[:2]
        return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    def _on_timer(self) -> None:
        # The display timer runs faster than most cameras deliver frames, so the
        # threaded reader often still holds the same frame as the previous tick.
        # Skip the decode/adjust/scale/repaint pipeline until a new frame arrives.
        if getattr(self.camera, "threaded", False):
            seq = self.camera.frame_seq()
            if seq == self._last_frame_seq and self.last_raw is not None:
                return
        else:
            seq = None

        ok, frame = self.camera.read()
        if not ok or frame is None:
            self.blank_frame_count = getattr(self, "blank_frame_count", 0) + 1
            if self.blank_frame_count in (1, 30, 120):
                self.status.showMessage(
                    f"Camera is open but no frame was read ({self.blank_frame_count} misses). Try Stop/Open, V4L2 backend, source 1, or default resolution.",
                    8000,
                )
            return

        self._last_frame_seq = seq
        self.blank_frame_count = 0
        # Drop stale buffered frames when requested. This makes the display feel current,
        # even if it means skipping intermediate frames.
        if hasattr(self, "low_latency_check") and self.low_latency_check.isChecked():
            self.camera.drain(1)

        self.last_raw = frame
        preview_frame = self._scale_preview_frame(frame)
        self.last_adjusted = self._adjust_live_frame(preview_frame)
        self.canvas.set_frame(self.last_adjusted)

        self._preview_frame_counter += 1
        now_t = time.perf_counter()
        elapsed = now_t - self._preview_fps_t0
        if elapsed >= 1.0:
            self._preview_fps = self._preview_frame_counter / elapsed
            self._preview_frame_counter = 0
            self._preview_fps_t0 = now_t
            cam_fps = self.camera.read_fps() if hasattr(self.camera, "read_fps") else 0.0
            self.status.showMessage(f"Live view: display {self._preview_fps:.1f} FPS, camera read {cam_fps:.1f} FPS", 1200)


    def capture_frame(self, save_adjusted: bool) -> None:
        if self.last_raw is None:
            QMessageBox.information(self, "Capture", "No frame available yet. Open live view first.")
            return
        self.save_recipe_from_ui()
        adjusted = self._adjust_frame(self.last_raw) if save_adjusted else None
        raw_path, adj_path = save_capture(self.recipe, self.last_raw, adjusted)
        self._recipe_index_dirty = True
        self._refresh_images(force=True)
        self.current_image_path = adj_path if adj_path else raw_path
        self.canvas.load_image(self.current_image_path)
        self.canvas.clear_boxes()
        self.status.showMessage(f"Captured: {self.current_image_path.name}", 5000)



    def browse_test_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select BungVision OBB model", str(EXPORT_DIR), "YOLO Model (*.pt *.onnx *.engine);;All files (*.*)")
        if path:
            self.test_model_edit.setText(path)

    def browse_test_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select test image", str(capture_folder(self.recipe)), "Images (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self.test_image_edit.setText(path)
            self._load_image_path(Path(path))

    def use_current_test_image(self) -> None:
        if not self.current_image_path:
            QMessageBox.information(self, "Test Models", "Open or capture an image first, then click Use Current.")
            return
        self.test_image_edit.setText(str(self.current_image_path))

    def clear_model_test_overlay(self) -> None:
        """Remove visual test layers without deleting saved label data."""
        if hasattr(self.canvas, "clear_all_visual_overlays"):
            self.canvas.clear_all_visual_overlays()
        elif hasattr(self.canvas, "clear_model_test_overlays"):
            self.canvas.clear_model_test_overlays()
        self._model_test_overlay_active = False
        self.status.showMessage("Visual overlays cleared; saved labels were not deleted", 4000)

    def show_saved_annotations(self) -> None:
        """Show saved/manual labels again after testing."""
        if hasattr(self.canvas, "set_annotation_visibility"):
            self.canvas.set_annotation_visibility(True)
        self.status.showMessage("Saved labels are visible again", 3000)

    def _model_test_device_arg(self):
        text = self.test_device_edit.text().strip() if hasattr(self, "test_device_edit") else "0"
        if not text:
            return None
        if text.lower() == "cpu":
            return "cpu"
        if text.isdigit():
            return int(text)
        return text

    def _load_test_model(self, model_path: str, model_name: str):
        if not model_path:
            raise RuntimeError(f"Select a {model_name} model first.")
        p = Path(model_path)
        if not p.exists():
            raise RuntimeError(f"{model_name} model not found:\n{p}")

        # Cache the test model so repeated Run Test clicks do not reload .pt files.
        if self._test_model is not None and self._test_model_path == str(p):
            return self._test_model

        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError(
                "Ultralytics is not installed in this Python environment.\n\n"
                "Install it with:\n"
                "pip install ultralytics\n\n"
                f"Original error: {e}"
            )
        model = YOLO(str(p))
        self._test_model = model
        self._test_model_path = str(p)
        return model

    def run_model_test(self) -> None:
        if not hasattr(self, "test_results_text"):
            return
        model_path = self.test_model_edit.text().strip()
        image_text = self.test_image_edit.text().strip()
        if not image_text and self.current_image_path:
            image_text = str(self.current_image_path)
            self.test_image_edit.setText(image_text)
        if not image_text:
            QMessageBox.information(self, "Test Models", "Select a test image first.")
            return
        image_path = Path(image_text)
        if not image_path.exists():
            QMessageBox.warning(self, "Test Models", f"Test image not found:\n{image_path}")
            return

        frame = cv2.imread(str(image_path))
        if frame is None:
            QMessageBox.warning(self, "Test Models", f"Could not read image:\n{image_path}")
            return

        try:
            self.status.showMessage("Loading/running model...", 2000)
            QApplication.processEvents()
            model = self._load_test_model(model_path, "BungVision OBB")

            imgsz = int(self.test_imgsz_spin.value())
            device = self._model_test_device_arg()
            conf = float(self.test_conf_spin.value())

            common_args = {"imgsz": imgsz, "verbose": False}
            if device is not None:
                common_args["device"] = device

            t0 = time.perf_counter()
            results = model.predict(frame, conf=conf, **common_args)
            t1 = time.perf_counter()
        except Exception as e:
            tb = traceback.format_exc()
            self.test_results_text.setPlainText(f"Model test failed:\n{e}\n\n{tb}")
            QMessageBox.warning(self, "Test Models", str(e))
            return

        battery_items, battery_count, angle_lines = self._battery_obb_overlay_items(results)
        if battery_count == 0:
            battery_items, battery_count, angle_lines = self._battery_box_overlay_items(results)
        bung_items, bung_count = self._bung_overlay_items(results)

        # Keep model-test graphics in a separate canvas overlay layer. Do not bake
        # them into the image pixmap, and do not convert them into saved labels.
        # Always start from a clean visual overlay state so repeated tests never stack.
        if hasattr(self.canvas, "clear_model_test_overlays"):
            self.canvas.clear_model_test_overlays()
        if self.current_image_path != image_path:
            self._load_image_path(image_path)
        elif self.last_raw is None:
            self.last_raw = frame
            self.last_adjusted = self._adjust_frame(frame)
            self.canvas.set_frame(self.last_adjusted)
            self.canvas.image_path = image_path
        # For model testing, hide saved/manual labels by default. They are still loaded
        # and saved normally; this only prevents visual stacking over model results.
        hide_saved = True
        if hasattr(self, "test_hide_saved_labels_check"):
            hide_saved = self.test_hide_saved_labels_check.isChecked()
        if hasattr(self.canvas, "set_annotation_visibility"):
            self.canvas.set_annotation_visibility(not hide_saved)
        if hasattr(self.canvas, "set_model_test_overlays"):
            self.canvas.set_model_test_overlays(battery_items + bung_items)
        self.current_image_path = image_path
        self._model_test_overlay_active = True

        summary = []
        summary.append(f"Image: {image_path.name}")
        summary.append(f"Image size: {frame.shape[1]} x {frame.shape[0]}")
        summary.append(f"Battery detections: {battery_count}")
        summary.append(f"Bung detections: {bung_count}")
        summary.append(f"Model time: {(t1 - t0) * 1000:.1f} ms")
        if angle_lines:
            summary.append("")
            summary.append("Battery details:")
            summary.extend(angle_lines)
        summary.append("")
        summary.append("Overlay legend:")
        summary.append("Blue polygon/box = battery detection")
        summary.append("Green polygon/box + center = bung detection. No filled class shapes are used.")
        summary.append("")
        summary.append("This is preview-only. It does not save labels or affect live inspection.")
        self.test_results_text.setPlainText("\n".join(summary))
        self.status.showMessage(f"Model test complete: {battery_count} batteries, {bung_count} bungs", 7000)


    def _run_test_model_on_image(self, image_path: Path):
        model_path = self.test_model_edit.text().strip()
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Could not read image:\n{image_path}")
        model = self._load_test_model(model_path, "BungVision OBB")
        imgsz = int(self.test_imgsz_spin.value())
        device = self._model_test_device_arg()
        conf = float(self.test_conf_spin.value())
        common_args = {"imgsz": imgsz, "verbose": False}
        if device is not None:
            common_args["device"] = device
        t0 = time.perf_counter()
        results = model.predict(frame, conf=conf, **common_args)
        t1 = time.perf_counter()
        return frame, results, t0, t1

    def _current_test_image_path(self) -> Path | None:
        image_text = self.test_image_edit.text().strip() if hasattr(self, "test_image_edit") else ""
        if not image_text and self.current_image_path:
            image_text = str(self.current_image_path)
            self.test_image_edit.setText(image_text)
        if not image_text:
            return None
        return Path(image_text)


    def _model_class_name(self, names, cls_id: int) -> str:
        """Return a stable class name from Ultralytics result/model names.

        Ultralytics can expose names as a dict ({0: 'bung'}) or a list
        (['bung']). Older code only handled dicts, which made Run Count filter
        out valid bungs when names were list-like.
        """
        try:
            if isinstance(names, dict):
                return str(names.get(cls_id, f"class_{cls_id}"))
            if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                return str(names[cls_id])
        except Exception:
            pass
        return f"class_{cls_id}"

    def _normalize_class_token(self, value: str) -> str:
        """Normalize class/filter text for forgiving matching."""
        import re
        return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())

    def _class_filter_match(self, name: str, cls_id: int, names: set[str], ids: set[int]) -> bool:
        """Return True when a detection belongs to a requested model class.

        Single-model OBB testing must not treat every polygon as both a
        battery and a bung.  Matching is forgiving for names but exact for
        numeric IDs:
        - numeric class IDs match exactly
        - exact lowercase names match
        - normalized names match (rubber_bung == rubber bung)
        - partial tokens match (bung matches bungs/rubber_bung/bung_cap)
        """
        if cls_id in ids:
            return True
        lname = str(name).strip().lower()
        nname = self._normalize_class_token(lname)
        for token in names:
            token_l = str(token).strip().lower()
            ntok = self._normalize_class_token(token_l)
            if not ntok:
                continue
            if lname == token_l or nname == ntok:
                return True
            if ntok in nname:
                return True
        return False

    def _count_filter_match(self, name: str, cls_id: int, count_names: set[str], count_ids: set[int]) -> bool:
        """Return True if a detection should be counted as a bung."""
        return self._class_filter_match(name, cls_id, count_names, count_ids)

    def _filter_names_from_edit(self, edit_attr: str, default_text: str) -> set[str]:
        widget = getattr(self, edit_attr, None)
        text = widget.text().strip() if widget is not None else default_text
        names = {part.strip().lower() for part in text.split(",") if part.strip() and not part.strip().isdigit()}
        default_names = {part.strip().lower() for part in default_text.split(",") if part.strip() and not part.strip().isdigit()}
        return names or default_names

    def _filter_ids_from_edit(self, edit_attr: str, default_text: str) -> set[int]:
        widget = getattr(self, edit_attr, None)
        text = widget.text().strip() if widget is not None else default_text
        ids: set[int] = set()
        for part in text.split(","):
            value = part.strip()
            if value.isdigit():
                ids.add(int(value))
        return ids

    def _battery_class_names(self) -> set[str]:
        return self._filter_names_from_edit("battery_class_filter_edit", "battery,0")

    def _battery_class_ids(self) -> set[int]:
        return self._filter_ids_from_edit("battery_class_filter_edit", "battery,0")

    def _battery_filter_match(self, name: str, cls_id: int) -> bool:
        return self._class_filter_match(name, cls_id, self._battery_class_names(), self._battery_class_ids())

    def _count_class_names(self) -> set[str]:
        return self._filter_names_from_edit("count_class_filter_edit", "bung,1")

    def _count_class_ids(self) -> set[int]:
        """Optional numeric class IDs entered in Count class, e.g. bung,1."""
        return self._filter_ids_from_edit("count_class_filter_edit", "bung,1")

    def _point_inside_polygon(self, x: float, y: float, poly: list[list[float]]) -> bool:
        try:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            return cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0
        except Exception:
            return False

    def run_count_test(self) -> None:
        """OBB count test: count bung centers inside each detected battery polygon."""
        if not hasattr(self, "test_results_text"):
            return
        image_path = self._current_test_image_path()
        if image_path is None:
            QMessageBox.information(self, "Count Test", "Select a test image first.")
            return
        if not image_path.exists():
            QMessageBox.warning(self, "Count Test", f"Test image not found:\n{image_path}")
            return
        try:
            self.status.showMessage("Running count test...", 2000)
            QApplication.processEvents()
            frame, results, t0, t1 = self._run_test_model_on_image(image_path)
        except Exception as e:
            tb = traceback.format_exc()
            self.test_results_text.setPlainText(f"Count test failed:\n{e}\n\n{tb}")
            QMessageBox.warning(self, "Count Test", str(e))
            return

        battery_items, battery_count, angle_lines = self._battery_obb_overlay_items(results)
        if battery_count == 0:
            battery_items, battery_count, angle_lines = self._battery_box_overlay_items(results)
        raw_bung_items, raw_bung_count = self._bung_overlay_items(results)
        count_names = self._count_class_names()
        count_ids = self._count_class_ids()
        required = int(self._count_required_value()) if hasattr(self, "count_required_spin") else int(self.recipe.expected_bungs)

        # Filter using the detection model's own class name/id, not the label-project class map.
        # This prevents classes such as positive_terminal/negative_terminal from being counted as bungs.
        bung_dets = []
        ignored_class = []
        for idx, item in enumerate(raw_bung_items):
            name = str(item.get("name", f"class_{item.get('cls_id', -1)}")).strip().lower()
            cls_id = int(item.get("cls_id", -999))
            if self._count_filter_match(name, cls_id, count_names, count_ids):
                det = dict(item)
                det["idx"] = idx
                det["assigned"] = False
                bung_dets.append(det)
            else:
                ignored_class.append(item)

        overlay_items: list[dict] = []
        lines: list[str] = []
        pass_count = 0
        fail_count = 0

        for bnum, batt in enumerate([x for x in battery_items if str(x.get("type", "")).startswith("battery")], start=1):
            poly = batt.get("points", [])
            if not poly and "xyxy" in batt:
                x1, y1, x2, y2 = [float(v) for v in batt.get("xyxy", [0, 0, 0, 0])]
                poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            counted_indices = []
            for det in bung_dets:
                if det.get("assigned"):
                    continue
                cx = float(det.get("cx", 0))
                cy = float(det.get("cy", 0))
                if self._point_inside_polygon(cx, cy, poly):
                    det["assigned"] = True
                    counted_indices.append(det["idx"])
            count = len(counted_indices)
            status = "PASS" if count >= required else "FAIL"
            if status == "PASS":
                pass_count += 1
            else:
                fail_count += 1
            label = f"Battery {bnum}: {count}/{required} {status}"
            batt_item = dict(batt)
            batt_item["label"] = label
            batt_item["status"] = status.lower()
            overlay_items.append(batt_item)
            lines.append(label)

        # Draw bungs as outlines only: green if counted, yellow if outside any battery polygon, gray if ignored by class.
        for det in bung_dets:
            item = dict(det)
            item["status"] = "counted" if det.get("assigned") else "outside"
            item["label"] = f"{det.get('name', 'bung')} {float(det.get('conf', 0.0)):.2f}"
            overlay_items.append(item)
        for det in ignored_class:
            item = dict(det)
            item["status"] = "ignored"
            overlay_items.append(item)

        outside_count = sum(1 for d in bung_dets if not d.get("assigned"))
        if battery_count == 0:
            lines.append("No battery OBB/box detected; count test cannot assign bungs to a battery.")

        if hasattr(self.canvas, "clear_model_test_overlays"):
            self.canvas.clear_model_test_overlays()
        if self.current_image_path != image_path:
            self._load_image_path(image_path)
        elif self.last_raw is None:
            self.last_raw = frame
            self.last_adjusted = self._adjust_frame(frame)
            self.canvas.set_frame(self.last_adjusted)
            self.canvas.image_path = image_path
        if hasattr(self.canvas, "set_annotation_visibility"):
            self.canvas.set_annotation_visibility(False)
        if hasattr(self.canvas, "set_model_test_overlays"):
            self.canvas.set_model_test_overlays(overlay_items)
        self.current_image_path = image_path
        self._model_test_overlay_active = True

        final = "PASS" if battery_count > 0 and fail_count == 0 else "FAIL"
        summary = []
        summary.append(f"Count Test: {final}")
        summary.append(f"Image: {image_path.name}")
        summary.append(f"Battery detections: {battery_count}")
        summary.append(f"Raw detection-model boxes: {raw_bung_count}")
        class_counts = {}
        for item in raw_bung_items:
            key = f"{item.get('name', 'unknown')}[{item.get('cls_id', '?')}]"
            class_counts[key] = class_counts.get(key, 0) + 1
        if class_counts:
            summary.append("Detection class counts: " + ", ".join(f"{k}={v}" for k, v in sorted(class_counts.items())))
        if raw_bung_count > 0 and not bung_dets:
            summary.append("WARNING: No detections matched the Count class filter. Try using the numeric class ID shown above, or a broader name such as 'bung'.")
        counted_filter = ', '.join(sorted(count_names))
        if count_ids:
            counted_filter += (", " if counted_filter else "") + ", ".join(str(x) for x in sorted(count_ids))
        summary.append(f"Counted class filter: {counted_filter}")
        summary.append(f"Required count per battery: {required}")
        summary.append(f"Bung detections outside batteries: {outside_count}")
        summary.append(f"Ignored non-count classes: {len(ignored_class)}")
        summary.append(f"Model time: {(t1 - t0) * 1000:.1f} ms")
        if angle_lines:
            summary.append("")
            summary.append("Battery details:")
            summary.extend(angle_lines)
        if lines:
            summary.append("")
            summary.append("Count details:")
            summary.extend(lines)
        summary.append("")
        summary.append("Overlay legend:")
        summary.append("Blue polygon/box = battery count region")
        summary.append("Green outline = counted bung inside a battery")
        summary.append("Yellow outline = bung detection outside all batteries")
        summary.append("Gray outline = ignored non-count class")
        self.test_results_text.setPlainText("\n".join(summary))
        self.status.showMessage(f"Count test {final}: {pass_count} pass, {fail_count} fail", 7000)
    def _normalize_angle_deg(self, angle: float) -> float:
        """Normalize an image-space angle to [-90, 90) degrees for readable skew."""
        while angle >= 90.0:
            angle -= 180.0
        while angle < -90.0:
            angle += 180.0
        return angle

    def _polygon_long_edge_angle(self, pts: np.ndarray) -> tuple[float | None, float]:
        """Return the angle of the longest edge and its length from a 4-point OBB polygon.

        Ultralytics xywhr angles can appear wrong for long rectangular parts because
        the model/formatter may swap width/height or use a different angle convention.
        For a battery, the useful plant-floor angle is usually the long-edge skew, so
        compute it directly from the drawn polygon.
        """
        if pts is None or len(pts) < 4:
            return None, 0.0
        best_angle = None
        best_len = 0.0
        for i in range(4):
            p1 = pts[i].astype(float)
            p2 = pts[(i + 1) % 4].astype(float)
            dx = float(p2[0] - p1[0])
            dy = float(p2[1] - p1[1])
            length = math.hypot(dx, dy)
            if length > best_len:
                best_len = length
                best_angle = self._normalize_angle_deg(math.degrees(math.atan2(dy, dx)))
        return best_angle, best_len

    def _battery_box_overlay_items(self, results) -> tuple[list[dict], int, list[str]]:
        """Convert normal YOLO detect box results into temporary canvas overlay items."""
        items: list[dict] = []
        lines: list[str] = []
        count = 0
        for r in results or []:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            try:
                xyxy = boxes.xyxy.cpu().numpy()
            except Exception:
                xyxy = []
            try:
                confs = boxes.conf.cpu().numpy()
            except Exception:
                confs = []
            try:
                clss = boxes.cls.cpu().numpy()
            except Exception:
                clss = []
            names = getattr(r, "names", {}) or {}
            for i, b in enumerate(xyxy):
                x1, y1, x2, y2 = [float(v) for v in b]
                cx = float((x1 + x2) / 2); cy = float((y1 + y2) / 2)
                conf = float(confs[i]) if i < len(confs) else 0.0
                cls_id = int(clss[i]) if i < len(clss) else 0
                name = self._model_class_name(names, cls_id)
                if not self._battery_filter_match(name, cls_id):
                    continue
                count += 1
                items.append({
                    "type": "battery_box",
                    "xyxy": [x1, y1, x2, y2],
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "cx": cx,
                    "cy": cy,
                    "conf": conf,
                    "cls_id": cls_id,
                    "name": name,
                    "label": f"{name} {conf:.2f}",
                })
                lines.append(f"{count}. {name} conf={conf:.2f}, box=({int(x1)}, {int(y1)})-({int(x2)}, {int(y2)}), center=({int(cx)}, {int(cy)})")
        return items, count, lines

    def _battery_obb_overlay_items(self, results) -> tuple[list[dict], int, list[str]]:
        """Convert OBB model results into temporary canvas overlay items."""
        items: list[dict] = []
        count = 0
        lines: list[str] = []
        for r in results or []:
            obb = getattr(r, "obb", None)
            if obb is None:
                continue
            try:
                polys = obb.xyxyxyxy.cpu().numpy()
            except Exception:
                polys = []
            try:
                xywhr = obb.xywhr.cpu().numpy()
            except Exception:
                xywhr = []
            try:
                confs = obb.conf.cpu().numpy()
            except Exception:
                confs = []
            try:
                clss = obb.cls.cpu().numpy()
            except Exception:
                clss = []
            names = getattr(r, "names", {}) or {}

            for i, poly in enumerate(polys):
                pts = np.array(poly, dtype=float).reshape(-1, 2)[:4]
                if len(pts) < 4:
                    continue
                cx = float(np.mean(pts[:, 0])); cy = float(np.mean(pts[:, 1]))
                edge_angle_deg, edge_len = self._polygon_long_edge_angle(pts)
                raw_angle_deg = None
                if i < len(xywhr) and len(xywhr[i]) >= 5:
                    raw_angle_deg = self._normalize_angle_deg(math.degrees(float(xywhr[i][4])))
                conf = float(confs[i]) if i < len(confs) else 0.0
                cls_id = int(clss[i]) if i < len(clss) else 0
                name = self._model_class_name(names, cls_id)
                if not self._battery_filter_match(name, cls_id):
                    continue
                count += 1

                label = f"{name} {conf:.2f}"
                if edge_angle_deg is not None:
                    label += f" | long {edge_angle_deg:.1f}°"

                items.append({
                    "type": "battery_obb",
                    "points": [[float(x), float(y)] for x, y in pts],
                    "cx": cx,
                    "cy": cy,
                    "label": label,
                })
                if edge_angle_deg is not None:
                    raw_txt = f", raw_xywhr={raw_angle_deg:.1f} deg" if raw_angle_deg is not None else ""
                    lines.append(f"{count}. {name} conf={conf:.2f}, long-edge angle={edge_angle_deg:.1f} deg{raw_txt}, center=({int(cx)}, {int(cy)})")
                else:
                    lines.append(f"{count}. {name} conf={conf:.2f}, center=({int(cx)}, {int(cy)})")
        return items, count, lines

    def _bung_overlay_items(self, results) -> tuple[list[dict], int]:
        """Convert bung model results into temporary canvas overlay items.

        Supports current YOLO OBB models first, with detect boxes as a fallback
        so older test models can still be previewed.
        """
        items: list[dict] = []
        count = 0
        # The Count-class filter is constant for this call; parse it once instead
        # of rebuilding the name/id sets for every detection.
        count_names = self._count_class_names()
        count_ids = self._count_class_ids()
        for r in results or []:
            names = getattr(r, "names", {}) or {}

            obb = getattr(r, "obb", None)
            if obb is not None:
                obb_count_for_result = 0
                try:
                    polys = obb.xyxyxyxy.cpu().numpy()
                except Exception:
                    polys = []
                try:
                    confs = obb.conf.cpu().numpy()
                except Exception:
                    confs = []
                try:
                    clss = obb.cls.cpu().numpy()
                except Exception:
                    clss = []
                obb_seen_for_result = len(polys) > 0
                for i, poly in enumerate(polys):
                    pts = np.array(poly, dtype=float).reshape(-1, 2)[:4]
                    if len(pts) < 4:
                        continue
                    cx = float(np.mean(pts[:, 0])); cy = float(np.mean(pts[:, 1]))
                    conf = float(confs[i]) if i < len(confs) else 0.0
                    cls_id = int(clss[i]) if i < len(clss) else 0
                    name = self._model_class_name(names, cls_id)
                    if not self._count_filter_match(name, cls_id, count_names, count_ids):
                        continue
                    items.append({
                        "type": "bung_obb",
                        "points": [[float(x), float(y)] for x, y in pts],
                        "cx": cx,
                        "cy": cy,
                        "conf": conf,
                        "cls_id": cls_id,
                        "name": name,
                        "label": f"{name} {conf:.2f}",
                    })
                    count += 1
                    obb_count_for_result += 1
                if obb_seen_for_result:
                    continue

            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            try:
                xyxy = boxes.xyxy.cpu().numpy()
            except Exception:
                xyxy = []
            try:
                confs = boxes.conf.cpu().numpy()
            except Exception:
                confs = []
            try:
                clss = boxes.cls.cpu().numpy()
            except Exception:
                clss = []
            for i, b in enumerate(xyxy):
                x1, y1, x2, y2 = [float(v) for v in b]
                cx = float((x1 + x2) / 2); cy = float((y1 + y2) / 2)
                conf = float(confs[i]) if i < len(confs) else 0.0
                cls_id = int(clss[i]) if i < len(clss) else 0
                name = self._model_class_name(names, cls_id)
                if not self._count_filter_match(name, cls_id, count_names, count_ids):
                    continue
                items.append({
                    "type": "bung_box",
                    "xyxy": [x1, y1, x2, y2],
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "cx": cx,
                    "cy": cy,
                    "conf": conf,
                    "cls_id": cls_id,
                    "name": name,
                    "label": f"{name} {conf:.2f}",
                })
                count += 1
        return items, count

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open image", str(capture_folder(self.recipe)), "Images (*.jpg *.jpeg *.png *.bmp)")
        if path:
            self._load_image_path(Path(path))

    def _load_selected_image(self) -> None:
        item = self.image_list.currentItem()
        if not item:
            return
        name = self._image_name_from_list_item(item.text()) if hasattr(self, "_image_name_from_list_item") else item.text()
        self._load_image_path(capture_folder(self.recipe) / name)

    def _load_image_path(self, path: Path) -> None:
        self.close_camera()
        if not self.canvas.load_image(path):
            QMessageBox.warning(self, "Image", "Could not load image.")
            return
        self.current_image_path = path
        self.last_raw = cv2.imread(str(path))
        self.last_adjusted = self._adjust_frame(self.last_raw)
        self.canvas.set_frame(self.last_adjusted)
        # Keep labels tied to the selected image dimensions/path, even after preview adjustments.
        self.canvas.image_path = path
        data = load_annotations(path)
        if data:
            self.canvas.set_boxes_from_dicts(data.get("boxes", []))
        else:
            self.canvas.clear_boxes()
        if hasattr(self.canvas, "clear_model_test_overlays"):
            self.canvas.clear_model_test_overlays()
        if hasattr(self.canvas, "set_annotation_visibility"):
            self.canvas.set_annotation_visibility(True)
        self._model_test_overlay_active = False
        self.status.showMessage(f"Loaded image: {path.name}", 5000)


    def _image_name_from_list_item(self, text: str) -> str:
        # Current list format uses readable status prefixes like
        # "✓ JSON OK  image.jpg" / "□ NO JSON  image.jpg". Older v0.9.x
        # builds used a simple two-character prefix. Support both so saved
        # projects remain navigable after upgrades.
        if "  " in text:
            return text.split("  ", 1)[1]
        if text[:2] in ("✓ ", "⚠ ", "□ ", "◇ "):
            return text[2:]
        return text

    def delete_selected_image(self) -> None:
        item = self.image_list.currentItem() if hasattr(self, "image_list") else None
        path = None

        if item:
            name = self._image_name_from_list_item(item.text())
            path = capture_folder(self.recipe) / name
        elif self.current_image_path:
            path = self.current_image_path

        if path is None:
            QMessageBox.information(self, "Delete Image", "Select a captured image first.")
            return

        if not path.exists():
            QMessageBox.information(self, "Delete Image", f"Image does not exist:\n{path}")
            self._refresh_images()
            return

        related = [path]
        label_path = image_label_json_path(path)
        if label_path.exists():
            related.append(label_path)

        # If deleting a raw image, include the matching adjusted image and label.
        if not path.stem.endswith("_adjusted"):
            adjusted = path.with_name(path.stem + "_adjusted" + path.suffix)
            if adjusted.exists():
                related.append(adjusted)
                adj_label = image_label_json_path(adjusted)
                if adj_label.exists():
                    related.append(adj_label)

        # If deleting an adjusted image, keep the raw unless explicitly selected separately.
        unique = []
        seen = set()
        for p in related:
            if p not in seen:
                unique.append(p)
                seen.add(p)

        msg = "Delete selected captured image?"
        msg += "\n\nFiles to delete:\n" + "\n".join(p.name for p in unique)

        reply = QMessageBox.question(
            self,
            "Delete Captured Image",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        deleted = []
        for p in unique:
            try:
                if p.exists():
                    p.unlink()
                    deleted.append(p.name)
            except Exception as e:
                QMessageBox.warning(self, "Delete Image", f"Could not delete:\n{p}\n\n{e}")
                return

        if self.current_image_path in unique:
            self.current_image_path = None
            self.last_raw = None
            self.last_adjusted = None
            self.canvas.clear_boxes()
            self.canvas.pixmap = None
            self.canvas.update()

        self._recipe_index_dirty = True
        for p in unique:
            self._invalidate_image_status(p)
        self._refresh_images(force=True)
        self.status.showMessage("Deleted: " + ", ".join(deleted), 6000)

    def copy_previous_labels(self) -> None:
        if not self.current_image_path:
            QMessageBox.information(self, "Labels", "Open or capture an image before copying labels.")
            return
        current_json = image_label_json_path(self.current_image_path)
        label_dir = current_json.parent
        candidates = [p for p in sorted(label_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if p != current_json]
        for p in candidates:
            try:
                import json
                data = json.loads(p.read_text(encoding="utf-8"))
                boxes = [self._normalize_import_box(b) for b in data.get("boxes", [])]
                if boxes:
                    self.canvas.set_boxes_from_dicts(boxes)
                    self.status.showMessage(f"Copied labels from: {p.name}", 5000)
                    self._update_box_count()
                    return
            except Exception:
                continue
        QMessageBox.information(self, "Labels", "No previous saved label file was found for this recipe.")

    def _boxes_for_review(self) -> list[dict]:
        boxes = [b.to_dict() for b in self.canvas.boxes]
        data = load_annotations(self.current_image_path) if self.current_image_path else None
        if not boxes and data and data.get("boxes"):
            boxes = [self._normalize_import_box(b) for b in data.get("boxes", [])]
        return boxes

    def _counts_from_box_dicts(self, boxes: list[dict]) -> tuple[int, int]:
        batt = 0
        bung = 0
        for raw in boxes:
            b = self._normalize_import_box(raw)
            label = str(b.get("label", ""))
            class_id = int(b.get("class_id", -1))
            if label.startswith("battery") or class_id == 0:
                batt += 1
            elif label.startswith("bung") or class_id == 1:
                bung += 1
        return batt, bung

    def mark_current_reviewed(self) -> None:
        if not self.current_image_path:
            QMessageBox.information(self, "Review", "Open or capture an image before marking it reviewed.")
            return
        boxes = self._boxes_for_review()
        if not boxes:
            QMessageBox.information(self, "Review", "This image has no labels to review yet.")
            return
        batt, bung = self._counts_from_box_dicts(boxes)
        expected = self._expected_bungs_value() if hasattr(self, "expected_spin") else self.recipe.expected_bungs
        if not self._counts_match_required(batt, bung):
            QMessageBox.information(
                self,
                "Review",
                f"This image does not match the recipe quantities.\n\n"
                f"Battery: {batt} / 1\n"
                f"Bungs: {bung} / expected {expected}\n\n"
                "Use Force Review Current only if this is intentional, such as a missing-bung/fail training example.",
            )
            return
        path = save_annotations(
            self.current_image_path,
            self.canvas.image_w,
            self.canvas.image_h,
            boxes,
            self.class_names,
            review=self._review_record("manual_mark_reviewed"),
        )
        self._invalidate_image_status(self.current_image_path)
        self.status.showMessage(f"Marked reviewed: {path.name}", 5000)
        self._refresh_images()
        self._update_dataset_summary()

    def force_mark_current_reviewed(self) -> None:
        if not self.current_image_path:
            QMessageBox.information(self, "Force Review", "Open or capture an image before force-reviewing it.")
            return
        boxes = self._boxes_for_review()
        if not boxes:
            QMessageBox.information(self, "Force Review", "This image has no labels to review yet.")
            return
        batt, bung = self._counts_from_box_dicts(boxes)
        expected = self._expected_bungs_value() if hasattr(self, "expected_spin") else self.recipe.expected_bungs
        if self._counts_match_required(batt, bung):
            path = save_annotations(
                self.current_image_path,
                self.canvas.image_w,
                self.canvas.image_h,
                boxes,
                self.class_names,
                review=self._review_record("manual_mark_reviewed"),
            )
            self.status.showMessage(f"Marked reviewed: {path.name}", 5000)
            self._refresh_images()
            self._update_dataset_summary()
            return
        reply = QMessageBox.question(
            self,
            "Force Review Quantity Mismatch",
            f"Force this image to reviewed even though the required quantities do not match?\n\n"
            f"Battery: {batt} / 1\n"
            f"Bungs: {bung} / expected {expected}\n\n"
            "This image will be included in reviewed-only export and training. Use this for intentional fail/missing-bung examples.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        path = save_annotations(
            self.current_image_path,
            self.canvas.image_w,
            self.canvas.image_h,
            boxes,
            self.class_names,
            review=self._review_record("force_review_quantity_mismatch", force=True, counts=(batt, bung)),
        )
        self._invalidate_image_status(self.current_image_path)
        self.status.showMessage(f"Force-reviewed mismatch image: {path.name}", 7000)
        self._refresh_images()
        self._update_dataset_summary()

    def find_next_unreviewed_image(self) -> None:
        images = self._get_recipe_image_paths()
        if not images:
            QMessageBox.information(self, "Review", "No captured/imported images found for this recipe.")
            return
        start = self._current_image_index()
        order = list(range(max(0, start + 1), len(images))) + list(range(0, max(0, start + 1)))
        for idx in order:
            entry = self._cached_image_status(images[idx])
            if entry.get("needs_review", False):
                batt = int(entry.get("battery_count", 0))
                bung = int(entry.get("bung_count", 0))
                self._load_image_path(images[idx])
                self.status.showMessage(f"Review: loaded unreviewed image ({batt} battery, {bung} bungs)", 6000)
                return
        QMessageBox.information(self, "Review", "No unreviewed labeled images found for this recipe.")

    def save_labels(self) -> None:
        if not self.current_image_path:
            QMessageBox.information(self, "Labels", "Open or capture an image before saving labels.")
            return
        boxes = [b.to_dict() for b in self.canvas.boxes]
        batt, bung = self._counts_from_box_dicts(boxes)
        expected = self._expected_bungs_value() if hasattr(self, "expected_spin") else self.recipe.expected_bungs
        review = self._review_record("save_labels") if self._counts_match_required(batt, bung) else None
        path = save_annotations(
            self.current_image_path,
            self.canvas.image_w,
            self.canvas.image_h,
            boxes,
            self.class_names,
            review=review,
            clear_review=(review is None),
        )
        self._invalidate_image_status(self.current_image_path)
        if review is None:
            self.status.showMessage(
                f"Saved labels only; not reviewed because counts are {batt} battery, {bung}/{expected} bungs. Use Force Review if intentional.",
                8000,
            )
        else:
            self.status.showMessage(f"Saved labels and marked reviewed: {path}", 5000)
        self._update_dataset_summary()
        self._refresh_images()

    def polish_buttons(self) -> None:
        """Prevent clipped button text on Linux/Qt themes without breaking compact panels."""
        for btn in self.findChildren(QPushButton):
            if btn.property("compactCaptureButton"):
                btn.setMinimumHeight(24)
                btn.setMaximumHeight(26)
                btn.setMinimumWidth(0)
                if btn.maximumWidth() > 16777214:
                    btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                else:
                    btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
                continue
            if btn.property("rightPanelButton"):
                btn.setMinimumHeight(24)
                btn.setMaximumHeight(26)
                btn.setMinimumWidth(0)
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                continue
            text_len = max(6, len(btn.text()))
            btn.setMinimumHeight(28)
            btn.setMinimumWidth(min(118, max(54, text_len * 6 + 16)))
            btn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        for combo in self.findChildren(QComboBox):
            combo.setMaxVisibleItems(12)
            combo.view().setMinimumHeight(120)
            combo.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
            combo.setMinimumHeight(26)

    def _simple_label_from_text(self, label: str, class_id: int = -1) -> tuple[str, int]:
        label_l = str(label or "").lower()
        if label_l == "battery" or label_l.startswith("battery_") or int(class_id) == 0:
            return "battery", 0
        if label_l == "bung" or label_l.startswith("bung_") or int(class_id) == 1:
            return "bung", 1
        if label_l == "retainer" or label_l.startswith("retainer_") or int(class_id) == 2:
            return "retainer", 2
        return str(label or ""), int(class_id)

    def _normalize_import_box(self, box: dict) -> dict:
        """Normalize BungVision runtime JSON boxes to the editor's simple labels."""
        original_label = str(box.get("label", "") or "")
        original_class_id = int(box.get("class_id", -1))
        label, class_id = self._simple_label_from_text(original_label, original_class_id)

        normalized = dict(box)
        normalized["label"] = label
        normalized["class_id"] = class_id

        if "source_label" not in normalized and original_label != label:
            normalized["source_label"] = original_label
        if "source_class_id" not in normalized and original_class_id != class_id:
            normalized["source_class_id"] = original_class_id

        return normalized

    def _box_kind(self, box) -> str:
        label = getattr(box, "label", "") or ""
        class_id = int(getattr(box, "class_id", -1))
        if str(label).startswith("battery") or class_id == 0:
            return "battery"
        if str(label).startswith("bung") or class_id == 1:
            return "bung"
        if str(label).startswith("retainer") or class_id == 2:
            return "retainer"
        return str(label)

    def _update_box_count(self) -> None:
        battery_count = sum(1 for b in self.canvas.boxes if self._box_kind(b) == "battery")
        bung_count = sum(1 for b in self.canvas.boxes if self._box_kind(b) == "bung")
        expected = self._expected_bungs_value() if hasattr(self, "expected_spin") else self.recipe.expected_bungs
        state = "OK" if battery_count == 1 and bung_count == expected else "CHECK"
        if hasattr(self, "count_label"):
            self.count_label.setText(f"Battery: {battery_count} / 1   Bungs: {bung_count} / expected {expected}  [{state}]")
        # Editing on-screen boxes does not change the on-disk dataset, so the
        # summary is refreshed by save/review/delete/capture and recipe changes
        # instead of walking every sidecar on each box draw/nudge.

    def clear_boxes_unsaved(self) -> None:
        """Clear the editable canvas only; never overwrite or delete saved JSON."""
        if not self.canvas.boxes:
            self.status.showMessage("No on-screen boxes to clear", 3000)
            return
        self.canvas.clear_boxes()
        self._update_box_count()
        self.status.showMessage("On-screen boxes cleared. Saved JSON was not changed; click Save to overwrite it.", 6000)

    def delete_saved_labels_confirmed(self) -> None:
        if not self.current_image_path:
            QMessageBox.information(self, "Delete Saved JSON", "Open or capture an image first.")
            return
        label_path = image_label_json_path(self.current_image_path)
        if not label_path.exists():
            self.status.showMessage("No saved JSON exists for this image", 3000)
            return
        reply = QMessageBox.question(
            self,
            "Delete Saved JSON",
            f"Delete the saved label JSON for this image?\n\n{label_path.name}\n\n"
            "This does not delete the image. It will remove the file-list JSON indicator until you save labels again.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        try:
            label_path.unlink()
        except Exception as exc:
            QMessageBox.warning(self, "Delete Saved JSON", f"Could not delete saved JSON:\n{exc}")
            return
        self.canvas.clear_boxes()
        self._update_box_count()
        self._invalidate_image_status(self.current_image_path)
        self._refresh_images()
        self.status.showMessage("Saved JSON deleted for current image", 5000)

    # Backward-compatible slot name used by older builds/actions.
    def clear_labels_confirmed(self) -> None:
        self.delete_saved_labels_confirmed()

    def _image_status(self, path: Path) -> tuple[str, int, int]:
        entry = self._cached_image_status(path)
        status = entry.get("status", "unlabeled")
        # For QA/problem search, unreviewed labeled images are still problems.
        if status == "needs_review":
            status = "problem"
        return status, int(entry.get("battery_count", 0)), int(entry.get("bung_count", 0))

    @staticmethod
    def _new_summary_totals() -> dict:
        return {"total": 0, "labeled": 0, "ready": 0, "forced": 0, "problems": 0, "needs_review": 0}

    def _accumulate_summary(self, totals: dict, entry: dict) -> None:
        """Fold one cached image-status entry into the running dataset totals."""
        totals["total"] += 1
        status = entry.get("status", "unlabeled")
        if entry.get("labeled", False):
            totals["labeled"] += 1
        if status == "ready":
            totals["ready"] += 1
        elif status == "forced":
            totals["forced"] += 1
        elif status == "problem":
            totals["problems"] += 1
        elif status == "needs_review":
            totals["needs_review"] += 1
            totals["problems"] += 1

    def _set_dataset_summary_label(self, totals: dict) -> None:
        if not hasattr(self, "dataset_label"):
            return
        self.dataset_label.setText(
            f"Dataset: {totals['total']} images, {totals['labeled']} labeled, "
            f"{totals['ready']} ready, {totals['forced']} forced, "
            f"{totals['problems']} problems, {totals['needs_review']} needs review"
        )

    def _update_dataset_summary(self) -> None:
        if not hasattr(self, "dataset_label"):
            return
        totals = self._new_summary_totals()
        for p in self._get_recipe_image_paths():
            self._accumulate_summary(totals, self._cached_image_status(p))
        self._set_dataset_summary_label(totals)


    def _current_image_index(self) -> int:
        if not self.current_image_path:
            return -1
        images = self._get_recipe_image_paths()
        try:
            return images.index(self.current_image_path)
        except ValueError:
            return -1

    def _load_image_by_index(self, idx: int) -> None:
        images = self._get_recipe_image_paths()
        if not images:
            QMessageBox.information(self, "Images", "No captured images found for this recipe.")
            return
        idx = max(0, min(len(images) - 1, idx))
        self._load_image_path(images[idx])

    def next_image(self) -> None:
        idx = self._current_image_index()
        if idx < 0:
            self._load_image_by_index(0)
        else:
            self._load_image_by_index(idx + 1)

    def previous_image(self) -> None:
        idx = self._current_image_index()
        if idx < 0:
            self._load_image_by_index(0)
        else:
            self._load_image_by_index(idx - 1)

    def save_and_next(self) -> None:
        self.save_labels()
        self.next_image()

    def find_next_problem_image(self) -> None:
        images = self._get_recipe_image_paths()
        if not images:
            QMessageBox.information(self, "QA", "No captured images found for this recipe.")
            return
        start = self._current_image_index()
        order = list(range(max(0, start + 1), len(images))) + list(range(0, max(0, start + 1)))
        for idx in order:
            status, batt, bung = self._image_status(images[idx])
            if status not in ("ready", "forced"):
                self._load_image_path(images[idx])
                self.status.showMessage(f"QA: {status} image loaded ({batt} battery, {bung} bungs)", 6000)
                return
        QMessageBox.information(self, "QA", "No problem images found. All reviewed/force-reviewed images are handled.")

    def reset_adjustments(self) -> None:
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.gamma_slider.setValue(100)
        self.sharpen_slider.setValue(0)
        self.clahe_check.setChecked(False)
        self.clahe_clip_slider.setValue(20)
        self.clahe_grid_slider.setValue(8)

    def _export_mode(self) -> str:
        return self.export_class_mode.currentData() if hasattr(self, "export_class_mode") else "model_specific"

    def _export_task(self) -> str:
        return self.export_task_combo.currentData() if hasattr(self, "export_task_combo") else "obb"

    def _export_reviewed_only(self) -> bool:
        # Reviewed-only export is intentionally hardcoded. There is no UI option to include unreviewed imports.
        return True

    def export_yolo(self) -> None:
        self.save_recipe_from_ui()
        mode = self._export_mode()
        task = self._export_task()
        reviewed_only = self._export_reviewed_only()
        try:
            if task == "obb":
                out = export_recipe_obb(self.recipe.safe_name, class_mode=mode, reviewed_only=reviewed_only)
                train_hint = "yolo obb train model=yolo11s-obb.pt data=data.yaml ..."
            else:
                out = export_recipe_yolo(self.recipe.safe_name, class_mode=mode, reviewed_only=reviewed_only)
                train_hint = "yolo detect train model=yolo11s.pt data=data.yaml ..."
        except Exception as e:
            QMessageBox.warning(self, "Export", str(e))
            return
        data_yaml = out / "data.yaml"
        QMessageBox.information(
            self,
            "Export complete",
            f"YOLO dataset exported to:\n{out}\n\nTraining file:\n{data_yaml}\n\nTask:\n{task}\nClass mode:\n{mode}\nReview filter:\nreviewed only\n\nSuggested command:\n{train_hint}"
        )
        self.status.showMessage(f"Exported YOLO {task} dataset: {out}", 8000)


    def export_all_yolo(self) -> None:
        self.save_recipe_from_ui()
        mode = self._export_mode()
        task = self._export_task()
        reviewed_only = self._export_reviewed_only()
        try:
            if task == "obb":
                out = export_all_recipes_obb(class_mode=mode, reviewed_only=reviewed_only)
            else:
                out = export_all_recipes_yolo(class_mode=mode, reviewed_only=reviewed_only)
        except Exception as e:
            QMessageBox.warning(self, "Export All Recipes", str(e))
            return
        data_yaml = out / "data.yaml"
        manifest = out / "manifest.csv"
        QMessageBox.information(
            self,
            "Export All complete",
            f"Combined YOLO dataset exported to:\n{out}\n\nTraining file:\n{data_yaml}\nManifest:\n{manifest}\n\nTask:\n{task}\nClass mode:\n{mode}\nReview filter:\nreviewed only"
        )
        self.status.showMessage(f"Exported combined YOLO {task} dataset: {out}", 8000)


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
