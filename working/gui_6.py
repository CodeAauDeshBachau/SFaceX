import sys
import cv2
import numpy as np
import time
import random
import os
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QProgressBar,
    QTextEdit,
    QGroupBox,
    QGridLayout,
    QCheckBox,
)
from PySide6.QtCore import QTimer, QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap, QFont, QPalette, QColor
from enroll import enroll


class FaceDetectionThread(QThread):
    """
    Enhanced face detection thread with motion-based ROI detection,
    KCF tracking, and CLAHE preprocessing for improved performance.
    """

    frame_ready = Signal(np.ndarray)
    debug_frame_ready = Signal(np.ndarray)  # For showing debug info
    face_detected = Signal(bool)
    face_captured_signal = Signal(str)
    status_update = Signal(str)
    instruction_update = Signal(str)
    tracking_info = Signal(str)  # New signal for tracking information

    def __init__(self):
        super().__init__()

        # --- State Management ---
        self.running = False
        self.phase = (
            "IDLE"  # Can be IDLE, DIRECTIONS, CALIBRATING, ENROLLING, COMPLETED
        )
        self.save_face = False

        # --- Directional Capture ---
        self.capture_count = 0
        self.direction_captures = 0
        self.current_direction_index = 0
        self.current_expression_index = 0
        self.max_expressions = 3
        self.directions = [
            ("Look Straight", ["Neutral", "Smile", "Slight Frown"]),
            ("Look Up", ["Eyes Up", "Mouth Slightly Open", "Neutral"]),
            ("Look Down", ["Chin Down", "Neutral", "Eyes on a low point"]),
            (
                "Look Right",
                ["Turn head slightly right", "Neutral", "Smile to the right"],
            ),
            ("Look Left", ["Turn head slightly left", "Neutral", "Smile to the left"]),
        ]

        # --- Calibration Capture ---
        self.calibration_count = 0
        self.MAX_CALIBRATION_IMAGES = 15
        self.stable_tracking_counter = 0
        self.STABLE_THRESHOLD = 10

        # --- Enhanced Detection Parameters ---
        self.frontalFaceCascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.COLOR_FRONTAL = (247, 173, 62)  # BGR color for rectangle
        self.COLOR_TRACKING = (255, 0, 0)  # Blue for tracking rectangle
        self.COLOR_ROI = (0, 255, 255)  # Yellow for ROI rectangles
        self.RECT_THICKNESS = 2
        self.scale = 0.7

        # --- Motion Detection & Tracking ---
        self.back_sub = None
        self.face_tracker = None
        self.tracker_active = False
        self.frame_count = 0
        self.ROI_INTERVAL = 10  # Frames between full-frame detection
        self.last_motion_rois = []
        self.last_contours = []

        # --- CLAHE for lighting adaptation ---
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # --- Debug options ---
        self.show_debug = False
        self.show_motion_analysis = False

        # --- Ensure Directories Exist ---
        os.makedirs("./data", exist_ok=True)
        for direction, _ in self.directions:
            os.makedirs(f"./data/{direction.replace(' ', '_')}", exist_ok=True)
        os.makedirs("./positive", exist_ok=True)

    def preprocess_image(self, gray):
        """Apply CLAHE for lighting robustness"""
        if self.show_debug:
            return self.clahe.apply(gray)
        return gray

    def detect_faces_with_roi(self, gray, frame, rois=None):
        """Enhanced face detection with ROI optimization"""
        faces = []

        if rois is None or len(rois) == 0:
            # Full frame detection
            faces = self.frontalFaceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(50, 50),  # Reduced for better detection
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
        else:
            # ROI-based detection (faster)
            for x, y, w, h in rois:
                # Ensure ROI is within frame bounds
                roi_x = max(0, x)
                roi_y = max(0, y)
                roi_w = min(w, gray.shape[1] - roi_x)
                roi_h = min(h, gray.shape[0] - roi_y)

                if roi_w > 0 and roi_h > 0:
                    roi_gray = gray[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
                    detected = self.frontalFaceCascade.detectMultiScale(
                        roi_gray,
                        scaleFactor=1.2,
                        minNeighbors=5,
                        minSize=(40, 40),
                        flags=cv2.CASCADE_SCALE_IMAGE,
                    )
                    # Convert ROI coordinates back to full frame
                    for dx, dy, dw, dh in detected:
                        faces.append((roi_x + dx, roi_y + dy, dw, dh))

        return faces

    def draw_debug_info(self, frame, faces, motion_rois, contours):
        """Draw debug information on frame"""
        debug_frame = frame.copy()

        # Draw faces in frontal color
        for x, y, w, h in faces:
            cv2.rectangle(
                debug_frame,
                (x, y),
                (x + w, y + h),
                self.COLOR_FRONTAL,
                self.RECT_THICKNESS,
            )
            # Add face label
            cv2.putText(
                debug_frame,
                "FACE",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.COLOR_FRONTAL,
                1,
            )

        # Draw motion ROIs in yellow
        for x, y, w, h in motion_rois:
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), self.COLOR_ROI, 1)

        # Draw contours in red
        if len(contours) > 0:
            cv2.drawContours(debug_frame, contours, -1, (0, 0, 255), 1)

        # Draw tracking rectangle if active
        if self.tracker_active and self.face_tracker is not None:
            success, bbox = self.face_tracker.update(debug_frame)
            if success:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(
                    debug_frame, (x, y), (x + w, y + h), self.COLOR_TRACKING, 3
                )
                cv2.putText(
                    debug_frame,
                    "TRACKING",
                    (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.COLOR_TRACKING,
                    2,
                )

        return debug_frame

    def get_current_instruction(self):
        """Returns the appropriate instruction text based on the current phase."""
        if self.phase == "DIRECTIONS":
            if self.current_direction_index < len(self.directions):
                direction, expressions = self.directions[self.current_direction_index]
                expression = expressions[
                    self.current_expression_index % len(expressions)
                ]
                return f"{direction}\n({expression})"
        elif self.phase == "CALIBRATING":
            return f"Calibration: Hold Steady\nSlight pose variations are welcome.\n({self.calibration_count}/{self.MAX_CALIBRATION_IMAGES})"
        elif self.phase == "ENROLLING":
            return "Enrolling user data...\nPlease wait."
        elif self.phase == "COMPLETED":
            return "All captures complete!\nEnrollment finished."
        return "Press 'Start Detection'"

    def run(self):
        """Enhanced main thread loop with motion detection and tracking."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_update.emit("Error: Camera not accessible.")
            self.running = False
            return

        # Initialize background subtractor
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=50, detectShadows=False
        )

        self.status_update.emit("Camera initialized with enhanced detection - Ready")
        self.instruction_update.emit(self.get_current_instruction())

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.status_update.emit("Error: Could not read from camera")
                break

            # Resize for faster processing
            frame_small = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

            # Apply CLAHE preprocessing if enabled
            if self.show_debug:
                gray = self.preprocess_image(gray)

            faces_detected = False
            motion_rois = []
            contours = []

            # --- Tracking Mode ---
            if self.tracker_active and self.face_tracker is not None:
                success, bbox = self.face_tracker.update(frame_small)
                if success:
                    x, y, w, h = map(int, bbox)
                    # Draw tracking rectangle
                    cv2.rectangle(
                        frame_small, (x, y), (x + w, y + h), self.COLOR_TRACKING, 3
                    )

                    faces_detected = True
                    self.stable_tracking_counter += 1

                    # Update tracking info
                    self.tracking_info.emit(
                        f"Tracking: Stable for {self.stable_tracking_counter} frames"
                    )

                    # Handle phase-specific capture logic
                    if self.phase == "DIRECTIONS" and self.save_face:
                        # Scale coordinates back to original frame
                        orig_x = int(x / self.scale)
                        orig_y = int(y / self.scale)
                        orig_w = int(w / self.scale)
                        orig_h = int(h / self.scale)
                        face_rect = (orig_x, orig_y, orig_w, orig_h)
                        self.capture_and_save_face(frame, face_rect, "DIRECTIONS")
                        self.save_face = False

                    elif self.phase == "CALIBRATING":
                        if self.stable_tracking_counter >= self.STABLE_THRESHOLD:
                            # Scale coordinates back to original frame
                            orig_x = int(x / self.scale)
                            orig_y = int(y / self.scale)
                            orig_w = int(w / self.scale)
                            orig_h = int(h / self.scale)
                            face_rect = (orig_x, orig_y, orig_w, orig_h)
                            self.capture_and_save_face(frame, face_rect, "CALIBRATING")
                            self.stable_tracking_counter = 0

                else:
                    # Tracking failed, reset
                    self.face_tracker = None
                    self.tracker_active = False
                    self.stable_tracking_counter = 0
                    self.tracking_info.emit(
                        "Tracking lost - Switching to detection mode"
                    )

            # --- Detection Mode ---
            if not self.tracker_active:
                # Motion analysis for ROI optimization
                fg_mask = self.back_sub.apply(gray)
                _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
                thresh = cv2.erode(thresh, None, iterations=1)
                thresh = cv2.dilate(thresh, None, iterations=2)

                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                motion_rois = [
                    cv2.boundingRect(cnt)
                    for cnt in contours
                    if cv2.contourArea(cnt) > 300
                ]

                self.last_motion_rois = motion_rois
                self.last_contours = contours

                # Decide detection strategy
                use_full_frame = (
                    self.frame_count % self.ROI_INTERVAL == 0 or len(motion_rois) == 0
                )
                rois = None if use_full_frame else motion_rois

                # Detect faces
                faces = self.detect_faces_with_roi(gray, frame_small, rois)

                if len(faces) > 0:
                    faces_detected = True
                    # Initialize tracker with the first detected face
                    face_rect = faces[0]
                    x, y, w, h = face_rect

                    # Initialize KCF tracker
                    self.face_tracker = cv2.legacy.TrackerKCF_create()
                    self.face_tracker.init(frame_small, (x, y, w, h))
                    self.tracker_active = True
                    self.stable_tracking_counter = 0

                    # Draw detection rectangle
                    cv2.rectangle(
                        frame_small,
                        (x, y),
                        (x + w, y + h),
                        self.COLOR_FRONTAL,
                        self.RECT_THICKNESS,
                    )

                    self.tracking_info.emit("Face detected - Tracking initialized")
                else:
                    self.tracking_info.emit("Scanning for faces...")

            # Emit face detection status
            self.face_detected.emit(faces_detected)

            # Create debug frame if requested
            if self.show_debug:
                debug_frame = self.draw_debug_info(
                    frame_small, [], motion_rois, contours
                )
                self.debug_frame_ready.emit(debug_frame)

            # Emit the main frame
            display_frame = cv2.flip(frame_small, 1)
            self.frame_ready.emit(display_frame)

            self.frame_count += 1
            self.msleep(30)

        self.cap.release()
        self.status_update.emit("Camera disconnected")

    def capture_and_save_face(self, frame, face_rect, capture_type):
        """Enhanced face capture with better error handling"""
        x, y, w, h = face_rect

        # Ensure coordinates are within frame bounds
        height, width = frame.shape[:2]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)

        if w <= 0 or h <= 0:
            self.status_update.emit("Error: Invalid face crop dimensions.")
            return

        face_crop = frame[y : y + h, x : x + w]

        if face_crop.size == 0:
            self.status_update.emit("Error: Failed to crop face.")
            return

        timestamp = time.strftime("%Y%m%d-%H%M%S")

        if capture_type == "DIRECTIONS":
            # Save 2 images per click for directional capture
            for i in range(2):
                direction_name = self.directions[self.current_direction_index][
                    0
                ].replace(" ", "_")
                filename = f"./data/{direction_name}/face_{direction_name}_{self.direction_captures + 1}_{timestamp}_{i}.jpg"
                cv2.imwrite(filename, face_crop)
                self.face_captured_signal.emit(filename)
                self.status_update.emit(f"Captured: {os.path.basename(filename)}")
                self.direction_captures += 1
                time.sleep(0.1)

            self.capture_count += 1
            self.current_expression_index += 1
            self.instruction_update.emit(self.get_current_instruction())

            if self.direction_captures >= 6:
                self.status_update.emit(
                    f"Direction '{self.directions[self.current_direction_index][0]}' complete!"
                )
                self.current_direction_index += 1
                self.direction_captures = 0
                self.capture_count = 0
                self.current_expression_index = 0

                if self.current_direction_index >= len(self.directions):
                    self.phase = "CALIBRATING"
                    self.status_update.emit(
                        "Directional capture finished. Starting calibration."
                    )

                self.instruction_update.emit(self.get_current_instruction())

        elif capture_type == "CALIBRATING":
            if self.calibration_count < self.MAX_CALIBRATION_IMAGES:
                filename = f"./positive/calib_face_{self.calibration_count + 1}_{timestamp}.jpg"
                cv2.imwrite(filename, face_crop)
                self.face_captured_signal.emit(filename)
                self.calibration_count += 1
                self.status_update.emit(
                    f"Calibration image {self.calibration_count}/{self.MAX_CALIBRATION_IMAGES} captured."
                )
                self.instruction_update.emit(self.get_current_instruction())

                if self.calibration_count >= self.MAX_CALIBRATION_IMAGES:
                    self.phase = "ENROLLING"
                    self.status_update.emit("Calibration complete. Enrolling...")
                    self.instruction_update.emit(self.get_current_instruction())
                    try:
                        enroll()
                        self.status_update.emit(
                            "Enrollment process completed successfully!"
                        )
                        self.phase = "COMPLETED"
                    except Exception as e:
                        self.status_update.emit(f"Enrollment Error: {e}")
                        self.phase = "IDLE"
                    self.instruction_update.emit(self.get_current_instruction())

    def start_detection(self):
        """Starts the enhanced face detection thread."""
        self.phase = "DIRECTIONS"
        self.running = True
        self.frame_count = 0
        self.tracker_active = False
        self.face_tracker = None
        self.start()

    def stop_detection(self):
        """Stops the face detection thread."""
        self.running = False
        self.wait()
        self.phase = "IDLE"
        self.tracker_active = False
        self.face_tracker = None

    def request_capture(self):
        """Called by the UI to request a manual capture during DIRECTIONS phase."""
        if self.phase == "DIRECTIONS" and self.direction_captures < 6:
            self.save_face = True

    def toggle_debug(self, enabled):
        """Toggle debug mode for enhanced visualization."""
        self.show_debug = enabled

    def toggle_motion_analysis(self, enabled):
        """Toggle motion analysis display."""
        self.show_motion_analysis = enabled


class SFaceXMainWindow(QMainWindow):
    """Enhanced main application window with debug options."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SFaceX - Enhanced Face Authentication System v3.0")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(self.get_stylesheet())

        self.detection_thread = FaceDetectionThread()
        self.detection_thread.frame_ready.connect(self.update_frame)
        self.detection_thread.debug_frame_ready.connect(self.update_debug_frame)
        self.detection_thread.face_detected.connect(self.update_face_status)
        self.detection_thread.face_captured_signal.connect(self.on_face_captured)
        self.detection_thread.status_update.connect(self.update_status)
        self.detection_thread.instruction_update.connect(self.update_instruction)
        # self.detection_thread.tracking_info.connect(self.update_tracking_info)

        self.setup_ui()
        self.face_detected = False

    def get_stylesheet(self):
        """Enhanced CSS stylesheet for the application."""
        return """
        QMainWindow { background-color: #1e1e1e; color: #ffffff; }
        QLabel { color: #ffffff; font-size: 14px; }
        QPushButton {
            background-color: #3d3d3d; border: 2px solid #555555; border-radius: 8px;
            color: #ffffff; font-size: 14px; font-weight: bold; padding: 10px 20px; min-width: 120px;
        }
        QPushButton:hover { background-color: #4d4d4d; border-color: #777777; }
        QPushButton:pressed { background-color: #2d2d2d; }
        QPushButton:disabled { background-color: #2a2a2a; color: #666666; border-color: #444444; }
        QPushButton#startButton { background-color: #2d5a2d; border-color: #4a8c4a; }
        QPushButton#startButton:hover { background-color: #3d6a3d; }
        QPushButton#stopButton { background-color: #5a2d2d; border-color: #8c4a4a; }
        QPushButton#stopButton:hover { background-color: #6a3d3d; }
        QPushButton#captureButton { background-color: #2d4a5a; border-color: #4a7a8c; }
        QPushButton#captureButton:hover { background-color: #3d5a6a; }
        QCheckBox {
            color: #ffffff; font-size: 12px; spacing: 5px;
        }
        QCheckBox::indicator {
            width: 18px; height: 18px;
        }
        QCheckBox::indicator:unchecked {
            border: 2px solid #555555; background-color: #2d2d2d; border-radius: 3px;
        }
        QCheckBox::indicator:checked {
            border: 2px solid #4a8c4a; background-color: #2d5a2d; border-radius: 3px;
        }
        QGroupBox {
            font-size: 16px; font-weight: bold; border: 2px solid #555555;
            border-radius: 10px; margin-top: 1ex; padding-top: 10px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
        QFrame#videoFrame { border: 3px solid #555555; border-radius: 10px; background-color: #2d2d2d; }
        QLabel#videoLabel { background-color: #1a1a1a; border-radius: 8px; }
        QTextEdit {
            background-color: #2d2d2d; border: 2px solid #555555; border-radius: 8px;
            color: #ffffff; font-family: 'Consolas', 'Monaco', monospace; font-size: 12px;
        }
        QLabel#titleLabel { font-size: 24px; font-weight: bold; color: #4a8c4a; padding: 10px; }
        QLabel#statusLabel { font-size: 16px; padding: 5px; border-radius: 5px; background-color: #3d3d3d; }
        QLabel#instructionLabel {
            font-size: 20px; font-weight: bold; color: #f0a500; padding: 15px;
            border-radius: 8px; background-color: #333333; text-align: center;
        }
        QLabel#trackingLabel {
            font-size: 14px; color: #87ceeb; padding: 5px;
            border-radius: 5px; background-color: #2d3d4d;
        }
        """

    def setup_ui(self):
        """Enhanced UI setup with debug options."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel (Video and Controls) ---
        left_panel = QVBoxLayout()
        title_label = QLabel("SFaceX Enhanced Detection")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(title_label)

        self.instruction_label = QLabel("Press 'Start Detection'")
        self.instruction_label.setObjectName("instructionLabel")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setWordWrap(True)
        left_panel.addWidget(self.instruction_label)

        # Tracking info label
        # self.tracking_label = QLabel("Tracking: Ready")
        # self.tracking_label.setObjectName("trackingLabel")
        # self.tracking_label.setAlignment(Qt.AlignCenter)
        # left_panel.addWidget(self.tracking_label)

        video_group = QGroupBox("Live Video Feed")
        video_layout = QVBoxLayout(video_group)
        self.video_label = QLabel("Camera Feed\nClick 'Start Detection' to begin")
        self.video_label.setObjectName("videoLabel")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)
        left_panel.addWidget(video_group)

        # Debug options
        debug_group = QGroupBox("Debug Options")
        debug_layout = QVBoxLayout(debug_group)

        self.debug_checkbox = QCheckBox("Enable CLAHE & Enhanced Visualization")
        self.debug_checkbox.toggled.connect(self.toggle_debug)
        debug_layout.addWidget(self.debug_checkbox)

        self.motion_checkbox = QCheckBox("Show Motion Analysis")
        self.motion_checkbox.toggled.connect(self.toggle_motion_analysis)
        debug_layout.addWidget(self.motion_checkbox)

        left_panel.addWidget(debug_group)

        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)
        self.start_button = QPushButton("Start Detection")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button = QPushButton("Stop Detection")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        self.capture_button = QPushButton("Capture (0/3)")
        self.capture_button.setObjectName("captureButton")
        self.capture_button.clicked.connect(self.capture_face)
        self.capture_button.setEnabled(False)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.capture_button)
        left_panel.addWidget(controls_group)

        # --- Right Panel (Status and Logs) ---
        right_panel = QVBoxLayout()
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("System Ready")
        self.status_label.setObjectName("statusLabel")
        status_layout.addWidget(self.status_label)
        self.face_status_label = QLabel("No Face Detected")
        self.face_status_label.setAlignment(Qt.AlignCenter)
        self.face_status_label.setStyleSheet(
            "QLabel { background-color: #5a2d2d; border: 2px solid #8c4a4a; border-radius: 8px; padding: 10px; font-size: 16px; font-weight: bold; }"
        )
        status_layout.addWidget(self.face_status_label)
        right_panel.addWidget(status_group)

        log_group = QGroupBox("Activity Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        right_panel.addWidget(log_group)

        captured_images_group = QGroupBox("Captured Images")
        captured_images_layout = QVBoxLayout(captured_images_group)
        self.captured_images_grid = QGridLayout()
        captured_images_layout.addLayout(self.captured_images_grid)
        right_panel.addWidget(captured_images_group)
        right_panel.addStretch()

        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        self.add_log("SFaceX Enhanced System Initialized")
        self.add_log("Features: Motion-based ROI, KCF Tracking, CLAHE Preprocessing")

    def toggle_debug(self, enabled):
        """Toggle debug mode in detection thread."""
        self.detection_thread.toggle_debug(enabled)
        self.add_log(f"Debug mode: {'Enabled' if enabled else 'Disabled'}")

    def toggle_motion_analysis(self, enabled):
        """Toggle motion analysis display."""
        self.detection_thread.toggle_motion_analysis(enabled)
        self.add_log(f"Motion analysis: {'Enabled' if enabled else 'Disabled'}")

    # def update_tracking_info(self, info):
    #     """Update tracking information display."""
    #     self.tracking_label.setText(f"Tracking: {info}")

    def start_detection(self):
        """Handler for the 'Start Detection' button."""
        self.clear_thumbnails()
        self.detection_thread.start_detection()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.add_log("Enhanced face detection started")
        self.add_log("Using: Background subtraction + KCF tracking")

    def stop_detection(self):
        """Handler for the 'Stop Detection' button."""
        self.detection_thread.stop_detection()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.video_label.setText("Camera Feed\nClick 'Start Detection' to begin")
        self.update_face_status(False)
        self.tracking_label.setText("Tracking: Ready")
        self.add_log("Face detection stopped")

    def capture_face(self):
        """Handler for the manual 'Capture' button."""
        if self.face_detected:
            self.detection_thread.request_capture()
            self.add_log("Face capture requested")
            self.capture_button.setText(
                f"Capture ({self.detection_thread.direction_captures + 1}/6)"
            )
        else:
            self.add_log("No face detected - cannot capture")

    def update_frame(self, frame):
        """Update the main video display with the current frame."""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            frame.data, width, height, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def update_debug_frame(self, frame):
        """Update debug frame display if debug mode is enabled."""
        # This could be used for a separate debug window if needed
        pass

    def update_face_status(self, detected):
        """Update the face detection status indicator."""
        self.face_detected = detected
        if detected:
            self.face_status_label.setText("✓ Face Detected")
            self.face_status_label.setStyleSheet(
                "QLabel { background-color: #2d5a2d; border: 2px solid #4a8c4a; border-radius: 8px; padding: 10px; font-size: 16px; font-weight: bold; }"
            )
            if self.detection_thread.phase == "DIRECTIONS":
                self.capture_button.setEnabled(True)
        else:
            self.face_status_label.setText("✗ No Face Detected")
            self.face_status_label.setStyleSheet(
                "QLabel { background-color: #5a2d2d; border: 2px solid #8c4a4a; border-radius: 8px; padding: 10px; font-size: 16px; font-weight: bold; }"
            )
            self.capture_button.setEnabled(False)

    def update_status(self, message):
        """Update the system status label."""
        self.status_label.setText(message)
        self.add_log(message)

    def update_instruction(self, instruction):
        """Update the instruction label."""
        self.instruction_label.setText(instruction)

    def on_face_captured(self, filename):
        """Handle when a face is captured and saved."""
        self.add_log(f"Face saved: {os.path.basename(filename)}")
        self.add_thumbnail(filename)

        # Update capture button text
        if self.detection_thread.phase == "DIRECTIONS":
            remaining = 6 - self.detection_thread.direction_captures
            self.capture_button.setText(
                f"Capture ({self.detection_thread.direction_captures}/6)"
            )
            if remaining <= 0:
                self.capture_button.setEnabled(False)

    def add_log(self, message):
        """Add a timestamped message to the activity log."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def add_thumbnail(self, filename):
        """Add a thumbnail of the captured image to the grid."""
        if os.path.exists(filename):
            # Load and resize image for thumbnail
            image = cv2.imread(filename)
            if image is not None:
                # Resize to thumbnail size
                thumbnail = cv2.resize(image, (80, 80))
                height, width, channel = thumbnail.shape
                bytes_per_line = 3 * width
                q_image = QImage(
                    thumbnail.data, width, height, bytes_per_line, QImage.Format_RGB888
                ).rgbSwapped()
                pixmap = QPixmap.fromImage(q_image)

                # Create thumbnail label
                thumbnail_label = QLabel()
                thumbnail_label.setPixmap(pixmap)
                thumbnail_label.setFixedSize(85, 85)
                thumbnail_label.setStyleSheet(
                    "border: 2px solid #555555; border-radius: 5px;"
                )

                # Add to grid
                row = len(self.thumbnails) // 4
                col = len(self.thumbnails) % 4
                self.captured_images_grid.addWidget(thumbnail_label, row, col)

                # Store reference
                if not hasattr(self, "thumbnails"):
                    self.thumbnails = []
                self.thumbnails.append(thumbnail_label)

    def clear_thumbnails(self):
        """Clear all thumbnail images from the grid."""
        if hasattr(self, "thumbnails"):
            for thumbnail in self.thumbnails:
                thumbnail.deleteLater()
            self.thumbnails.clear()

        # Clear the grid layout
        for i in reversed(range(self.captured_images_grid.count())):
            child = self.captured_images_grid.itemAt(i)
            if child:
                widget = child.widget()
                if widget:
                    widget.deleteLater()

    def closeEvent(self, event):
        """Handle application close event."""
        if self.detection_thread.running:
            self.detection_thread.stop_detection()
        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("SFaceX Enhanced")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("SFaceX Team")

    # Apply dark theme
    app.setStyle("Fusion")

    # Create and show main window
    window = SFaceXMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
