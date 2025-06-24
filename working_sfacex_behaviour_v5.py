import sys
import os
import json
import time
import cv2
import numpy as np
import statistics
import math
from operator import attrgetter
from datetime import datetime

# from PySide6.QtWidgets import QMessageBox, QPushButton
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QFont, QIcon, QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QStackedWidget,
    QGridLayout,
    QMessageBox,
)
import threading
from PySide6.QtCore import Signal

# Assuming sfacex_to_be_checked.py contains the LBPChiSquareAuthenticator
# and is in the same directory or accessible via PYTHONPATH.
from sfacex_to_be_checked import LBPChiSquareAuthenticator, get_uniform_lbp_mapping
from behaviour_learning import BehavioralAuthenticationSystem

# ==============================================================================
#  START: Enhanced Face Detection and Processing Class
# ==============================================================================


class FaceDetectionProcessor:
    """
    Handles face detection, tracking, ROI extraction, and preprocessing.
    This class remains unchanged as per the request.
    """

    def __init__(self):
        # --- Haar Cascade Setup ---
        self.frontal_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # --- Detection Parameters ---
        self.COLOR_FRONTAL = (247, 173, 62)  # BGR color for rectangle
        self.RECT_THICKNESS = 2
        self.detection_scale = 0.7  # Scale down frame for faster detection

        # --- Face Tracking Parameters ---
        self.stable_tracking_counter = 0
        self.STABLE_THRESHOLD = 5  # Frames of stability needed for authentication
        self.last_face_rect = None
        self.face_stability_tolerance = 30  # Pixels tolerance for face movement

        # --- Quality Assessment ---
        self.min_face_size = (80, 80)  # Minimum face size for processing
        self.max_face_size = (400, 400)  # Maximum face size for processing

    def detect_faces(self, frame):
        """
        Detect faces using Haar Cascade with optimizations
        Returns: list of face rectangles [(x, y, w, h), ...]
        """
        # Scale down for faster detection
        frame_small = cv2.resize(
            frame, (0, 0), fx=self.detection_scale, fy=self.detection_scale
        )
        gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces_small = self.frontal_face_cascade.detectMultiScale(
            gray_small, scaleFactor=1.1, minNeighbors=4, flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Scale coordinates back to original frame size
        faces = []
        for x_small, y_small, w_small, h_small in faces_small:
            x = int(x_small / self.detection_scale)
            y = int(y_small / self.detection_scale)
            w = int(w_small / self.detection_scale)
            h = int(h_small / self.detection_scale)
            faces.append((x, y, w, h))

        return faces

    def is_face_stable(self, current_face_rect):
        """
        Check if face position is stable for reliable authentication
        """
        if self.last_face_rect is None:
            self.last_face_rect = current_face_rect
            self.stable_tracking_counter = 1
            return False

        # Calculate center points
        curr_cx = current_face_rect[0] + current_face_rect[2] // 2
        curr_cy = current_face_rect[1] + current_face_rect[3] // 2
        last_cx = self.last_face_rect[0] + self.last_face_rect[2] // 2
        last_cy = self.last_face_rect[1] + self.last_face_rect[3] // 2

        # Calculate distance between centers
        distance = math.sqrt((curr_cx - last_cx) ** 2 + (curr_cy - last_cy) ** 2)

        if distance <= self.face_stability_tolerance:
            self.stable_tracking_counter += 1
        else:
            self.stable_tracking_counter = 1

        self.last_face_rect = current_face_rect
        return self.stable_tracking_counter >= self.STABLE_THRESHOLD

    def assess_face_quality(self, face_rect, frame_shape):
        """
        Assess if the detected face is suitable for authentication
        """
        x, y, w, h = face_rect
        frame_h, frame_w = frame_shape[:2]

        # Check face size
        if w < self.min_face_size[0] or h < self.min_face_size[1]:
            return False, "Face too small"
        if w > self.max_face_size[0] or h > self.max_face_size[1]:
            return False, "Face too large"

        # Check if face is within frame bounds
        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            return False, "Face partially outside frame"

        # Check face aspect ratio (should be roughly square)
        aspect_ratio = w / h
        if aspect_ratio < 0.7 or aspect_ratio > 1.4:
            return False, "Invalid face aspect ratio"

        return True, "Good quality"

    def extract_face_roi(self, frame, face_rect, padding=0.0):
        """
        Extract and preprocess face Region of Interest (ROI)
        """
        x, y, w, h = face_rect
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        x_pad = max(0, x - pad_w)
        y_pad = max(0, y - pad_h)
        w_pad = min(frame.shape[1] - x_pad, w + 2 * pad_w)
        h_pad = min(frame.shape[0] - y_pad, h + 2 * pad_h)
        face_roi = frame[y_pad : y_pad + h_pad, x_pad : x_pad + w_pad]
        return face_roi if face_roi.size > 0 else None

    def draw_face_rectangles(self, frame, faces, stable_face_index=-1):
        """
        Draw rectangles around detected faces
        """
        display_frame = frame.copy()
        for i, (x, y, w, h) in enumerate(faces):
            color, thickness = (
                ((0, 255, 0), self.RECT_THICKNESS + 2)
                if i == stable_face_index
                else (self.COLOR_FRONTAL, self.RECT_THICKNESS)
            )
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
            if i == stable_face_index:
                cv2.putText(
                    display_frame,
                    "STABLE",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
        return display_frame

    def reset_tracking(self):
        """Reset face tracking state"""
        self.stable_tracking_counter = 0
        self.last_face_rect = None


# ==============================================================================
#  END: Enhanced Face Detection and Processing Class
# ==============================================================================


class StartScreen(QWidget):
    """
    The initial screen that predicts the user's unlock method preference.
    """

    # Signals to notify the main app to switch screens
    prediction_ready = Signal(str)

    def __init__(self, behavior_model):
        super().__init__()
        self.behavior_model = behavior_model
        self.predicted_method = "FACE"  # Default prediction

        self.setObjectName("StartScreen")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(30)

        self.title_label = QLabel("Behavioral Authentication")
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel("Analyzing user behavior...")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.start_button = QPushButton("Start Face Unlock Simulation")
        self.start_button.setObjectName("StartButton")
        self.start_button.setCursor(Qt.PointingHandCursor)
        self.start_button.setDisabled(True)  # Disabled until prediction is made
        self.start_button.clicked.connect(self.start_simulation)

        layout.addWidget(self.title_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.start_button)

        # Use a timer to run the prediction shortly after the UI is shown
        QTimer.singleShot(500, self.run_prediction)

    def run_prediction(self):
        """
        Runs the behavior prediction model and updates the UI.
        """
        # Running in a separate thread to avoid freezing the GUI
        threading.Thread(target=self._get_prediction, daemon=True).start()

    def _get_prediction(self):
        """Worker function for prediction."""
        try:
            recommendation = self.behavior_model.get_recommendation()
            self.predicted_method = recommendation["recommended_method"]
            confidence = recommendation["confidence"]

            # Since this is a worker thread, we can't update UI directly.
            # We'll use a signal or QTimer to schedule the UI update on the main thread.
            QTimer.singleShot(
                0,
                lambda: self.update_ui_after_prediction(
                    self.predicted_method, confidence
                ),
            )
        except Exception as e:
            print(f"Prediction error: {e}")
            QTimer.singleShot(
                0, lambda: self.update_ui_after_prediction("FACE", 0, error=True)
            )

    def update_ui_after_prediction(self, method, confidence, error=False):
        """Updates the UI on the main thread with the prediction result."""
        if error:
            self.status_label.setText("Could not predict. Defaulting to Face Unlock.")
        else:
            self.status_label.setText(
                f"Prediction: You will likely use <b>{method}</b> (Confidence: {confidence:.2f}%)"
            )
        self.start_button.setEnabled(True)

    def start_simulation(self):
        """
        Emits a signal to the main application to switch to the predicted screen.
        """
        self.prediction_ready.emit(self.predicted_method)


class FaceScanScreen(QWidget):
    def __init__(self, switch_to_pin, switch_to_home, authenticator, behavior_model):
        super().__init__()
        self.switch_to_pin = switch_to_pin
        self.switch_to_home = switch_to_home
        self.authenticator = authenticator
        self.behavior_model = behavior_model

        # Camera state management
        self.camera = None
        self.camera_ready = False
        self.camera_init_thread = None
        self.is_scanning = False

        self.face_processor = FaceDetectionProcessor()
        self.current_frame = None
        self.current_faces = []
        self.stable_face_rect = None
        self.authentication_in_progress = False
        self.last_auth_time = 0
        self.last_auth_result = None

        self.setup_ui()

        # Start camera initialization in background when screen is created
        self.pre_initialize_camera()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(30)

        self.camera_label = QLabel("Camera Off")
        self.camera_label.setObjectName("CameraLabel")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setStyleSheet("background-color: #000; border-radius: 10px;")

        self.status_label = QLabel("Position your face in the frame")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Inter", 16))

        self.start_scan_button = QPushButton("Start Scan")
        self.start_scan_button.setObjectName("StartScanButton")
        self.start_scan_button.setCursor(Qt.PointingHandCursor)
        self.start_scan_button.clicked.connect(self.start_scan_sequence)

        self.pin_button = QPushButton("Use PIN Instead")
        self.pin_button.setObjectName("SecondaryButton")
        self.pin_button.setCursor(Qt.PointingHandCursor)
        self.pin_button.clicked.connect(self.switch_to_pin)

        layout.addWidget(self.camera_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.start_scan_button)
        layout.addWidget(self.pin_button)

        # Timers
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_frame)
        self.auth_timer = QTimer(self)
        self.auth_timer.timeout.connect(self.run_face_authentication)

    def pre_initialize_camera(self):
        """Initialize camera in background thread"""
        if self.camera_init_thread and self.camera_init_thread.is_alive():
            return

        self.camera_init_thread = threading.Thread(
            target=self._init_camera_worker, daemon=True
        )
        self.camera_init_thread.start()

    def _init_camera_worker(self):
        """Worker thread to initialize camera"""
        try:
            print("DEBUG: Pre-initializing camera...")
            camera = cv2.VideoCapture(0)

            if camera.isOpened():
                # Set camera properties
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(
                    cv2.CAP_PROP_BUFFERSIZE, 1
                )  # Reduce buffer for faster response
                camera.set(cv2.CAP_PROP_FPS, 30)

                # Read a dummy frame to fully initialize
                ret, _ = camera.read()
                if ret:
                    self.camera = camera
                    self.camera_ready = True
                    print("DEBUG: Camera pre-initialized successfully")

                    # Update UI on main thread
                    QTimer.singleShot(
                        0, lambda: self.update_camera_status("Camera Ready")
                    )
                else:
                    camera.release()
                    print(
                        "DEBUG: Camera pre-initialization failed - could not read frame"
                    )
            else:
                print("DEBUG: Camera pre-initialization failed - could not open")

        except Exception as e:
            print(f"DEBUG: Camera pre-initialization error: {e}")

    def update_camera_status(self, status):
        """Update camera status on main thread"""
        if self.camera_ready and not self.is_scanning:
            self.status_label.setText("Ready to scan - click 'Start Scan'")

    def start_scan_sequence(self):
        """Start scanning - now much faster since camera is pre-initialized"""
        print("DEBUG: Starting face scan sequence...")

        # Check authentication enrollment
        if (
            not hasattr(self.authenticator, "threshold")
            or self.authenticator.threshold is None
        ):
            self.status_label.setText("System not enrolled. Use PIN.")
            return

        self.start_scan_button.hide()

        if self.camera_ready and self.camera and self.camera.isOpened():
            # Camera is ready - start immediately!
            print("DEBUG: Camera already ready, starting scan...")
            self.start_scanning()
        else:
            # Camera not ready yet - show loading and wait
            self.status_label.setText("Initializing camera...")
            self.wait_for_camera_and_start()

    def wait_for_camera_and_start(self):
        """Wait for camera to be ready and then start scanning"""
        if self.camera_ready and self.camera and self.camera.isOpened():
            self.start_scanning()
        else:
            # Check again in 100ms
            QTimer.singleShot(100, self.wait_for_camera_and_start)

    def start_scanning(self):
        """Actually start the scanning process"""
        self.is_scanning = True
        self.face_processor.reset_tracking()
        self.camera_timer.start(33)  # ~30 FPS
        self.auth_timer.start(1000)  # Auth every 1 second
        self.status_label.setText("Looking for face...")
        print("DEBUG: Face scanning started")

    def stop_scan(self):
        """Stop scanning but keep camera initialized for faster restart"""
        print("DEBUG: Stopping face scan...")
        self.is_scanning = False
        self.camera_timer.stop()
        self.auth_timer.stop()

        # DON'T release camera - keep it for faster restart
        # self.camera.release()  # <-- Remove this line

        self.face_processor.reset_tracking()
        self.authentication_in_progress = False
        self.last_auth_result = None

        # Reset UI
        self.camera_label.setText("Camera Off")
        self.camera_label.setStyleSheet("background-color: #000; border-radius: 10px;")
        self.status_label.setText("Ready to scan - click 'Start Scan'")
        self.start_scan_button.show()

    def release_camera(self):
        """Actually release camera resources (call this when leaving screen)"""
        if self.camera:
            self.camera.release()
            self.camera = None
            self.camera_ready = False
            print("DEBUG: Camera released")

    def update_frame(self):
        if not self.is_scanning or not self.camera:
            return

        ret, frame = self.camera.read()
        if not ret:
            self.status_label.setText("Error reading frame")
            return

        self.current_frame = frame.copy()
        faces = self.face_processor.detect_faces(frame)
        self.current_faces = faces

        stable_face_index = -1
        self.stable_face_rect = None

        if faces:
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            is_good_quality, quality_msg = self.face_processor.assess_face_quality(
                largest_face, frame.shape
            )

            if is_good_quality:
                if self.face_processor.is_face_stable(largest_face):
                    self.stable_face_rect = largest_face
                    stable_face_index = faces.index(largest_face)
                    self.status_label.setText("Face stable, authenticating...")
                else:
                    self.status_label.setText("Hold still...")
            else:
                self.status_label.setText(f"Quality issue: {quality_msg}")
                self.face_processor.reset_tracking()
        else:
            self.status_label.setText("No face detected")
            self.face_processor.reset_tracking()

        display_frame = self.face_processor.draw_face_rectangles(
            frame, faces, stable_face_index
        )
        display_frame = cv2.flip(display_frame, 1)
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.camera_label.setPixmap(
            pixmap.scaled(
                self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def run_face_authentication(self):
        current_time = time.time()
        if (
            not self.is_scanning
            or self.stable_face_rect is None
            or self.authentication_in_progress
            or (current_time - self.last_auth_time) < 2.0
        ):
            return

        print("DEBUG: Attempting face authentication...")
        self.authentication_in_progress = True
        self.last_auth_time = current_time

        try:
            face_roi = self.face_processor.extract_face_roi(
                self.current_frame, self.stable_face_rect
            )
            if face_roi is None:
                raise ValueError("Failed to extract face ROI")

            result, distance = self.authenticator.classify_template(face_roi)
            print(f"DEBUG: Authentication result: {result}, distance: {distance}")

            self.last_auth_result = result
            # --- BUGFIX: Store last result in authenticator for later check ---
            self.authenticator.last_auth_result = result

            if result == "enrolled_person":
                self.authenticator.unlock_type = "FACE"
                self.status_label.setText("Face Recognized! Unlocking...")
                self.stop_scan()
                self.behavior_model.record_unlock_attempt("FACE")
                QTimer.singleShot(1000, self.switch_to_home)
            else:
                self.status_label.setText(f"Face not recognized. Retrying...")

        except Exception as e:
            print(f"DEBUG: Authentication error: {e}")
            self.status_label.setText("Authentication error. Retrying...")
            self.last_auth_result = "error_extracting_features"
        finally:
            self.authentication_in_progress = False


class PinScreen(QWidget):
    # This class remains largely unchanged.
    def __init__(self, switch_to_face, switch_to_home, authenticator, behavior_model):
        super().__init__()
        self.switch_to_face = switch_to_face
        self.switch_to_home = switch_to_home
        self.authenticator = authenticator
        self.behavior_model = behavior_model
        self.pin = ""

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setSpacing(20)

        title = QLabel("Enter PIN")
        title.setObjectName("StatusLabel")
        title.setFont(QFont("Inter", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)

        self.pin_display = QLabel()
        self.pin_display.setObjectName("PinDisplay")
        self.pin_display.setAlignment(Qt.AlignCenter)
        self.pin_display.setMinimumHeight(50)
        self.update_pin_display()

        grid_layout = QGridLayout()
        grid_layout.setSpacing(15)

        buttons = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "Face", "0", "<"]
        positions = [(i, j) for i in range(4) for j in range(3)]

        for position, value in zip(positions, buttons):
            button = QPushButton(value)
            button.setCursor(Qt.PointingHandCursor)
            button.setFixedSize(80, 80)
            if value.isdigit():
                button.setObjectName("PinButton")
                button.clicked.connect(lambda _, v=value: self.add_digit(v))
            elif value == "Face":
                button.setObjectName("SecondaryButton")
                button.clicked.connect(self.switch_to_face)
            elif value == "<":
                button.setObjectName("SecondaryButton")
                button.clicked.connect(self.backspace)
            grid_layout.addWidget(button, *position)

        main_layout.addWidget(title)
        main_layout.addWidget(self.pin_display)
        main_layout.addLayout(grid_layout)

    def add_digit(self, digit):
        if len(self.pin) < 6:
            self.pin += digit
            self.update_pin_display()
            if len(self.pin) == 6:
                self.check_pin()

    def backspace(self):
        self.pin = self.pin[:-1]
        self.update_pin_display()

    def reset_pin(self):
        self.pin = ""
        self.update_pin_display()

    def update_pin_display(self):
        self.pin_display.setText("●" * len(self.pin) + "○" * (6 - len(self.pin)))

    def check_pin(self):
        correct_pin = self.authenticator.pin
        if correct_pin is None:
            self.pin_display.setText("No PIN set")
            QTimer.singleShot(1500, self.reset_pin)
            return

        if self.pin == correct_pin:
            print("PIN correct.")
            self.authenticator.unlock_type = "PIN"
            self.pin_display.setText("PIN Accepted")
            self.behavior_model.record_unlock_attempt("PIN")
            self.switch_to_home()
        else:
            print("PIN incorrect.")
            self.pin_display.setText("Incorrect PIN")
            QTimer.singleShot(1000, self.reset_pin)


class HomeScreen(QWidget):
    def __init__(self, lock_system, authenticator):
        super().__init__()
        self.setObjectName("HomeScreen")
        self.lock_system = lock_system
        self.authenticator = authenticator

        # --- BUGFIX: Removed the QMessageBox from __init__ ---
        # It will now be triggered by MainWindow when this screen is shown.

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        welcome_label = QLabel("Welcome Home!")
        welcome_label.setObjectName("StatusLabel")
        welcome_label.setFont(QFont("Inter", 28, QFont.Bold))

        info_label = QLabel("The system is now unlocked.")
        info_label.setFont(QFont("Inter", 14))

        lock_button = QPushButton("Lock System")
        lock_button.setObjectName("SecondaryButton")
        lock_button.clicked.connect(self.lock_system)
        lock_button.setCursor(Qt.PointingHandCursor)

        layout.addWidget(welcome_label)
        layout.addWidget(info_label)
        layout.addSpacing(30)
        layout.addWidget(lock_button)

    def prompt_for_enrollment(self):
        """
        --- BUGFIX: New method to show the enrollment prompt. ---
        This is called conditionally when the home screen is displayed.
        """
        if (
            self.authenticator.unlock_type == "PIN"
            and self.authenticator.last_auth_result == "not_enrolled_person_near_miss"
        ):

            print("DEBUG: Prompting user to enroll face.")
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Enroll New Face?")
            msg_box.setText(
                "Your face was not fully recognized, but it was a close match. Would you like to add this face to your profile to improve future logins?"
            )
            yes_button = msg_box.addButton("Yes, Add Face", QMessageBox.AcceptRole)
            no_button = msg_box.addButton("No, Thanks", QMessageBox.RejectRole)
            msg_box.setIcon(QMessageBox.Question)
            msg_box.exec()

            if msg_box.clickedButton() == yes_button:
                print("DEBUG: User chose to enroll.")
                self.authenticator.adapt_model_with_feedback(True)

            # Reset flags to prevent the dialog from showing again
            self.authenticator.last_auth_result = None
            self.authenticator.unlock_type = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimized Face Unlock System")
        self.setGeometry(100, 100, 400, 700)

        # Initialize models
        self.authenticator = LBPChiSquareAuthenticator()
        self.behavior_model = BehavioralAuthenticationSystem()

        # Create central widget and stacked widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.stacked_widget = QStackedWidget()

        # Create all screens ONCE
        self.start_screen = StartScreen(self.behavior_model)
        self.face_scan_screen = FaceScanScreen(
            self.show_pin_screen,
            self.show_home_screen,
            self.authenticator,
            self.behavior_model,
        )
        self.pin_screen = PinScreen(
            self.show_face_scan_screen,
            self.show_home_screen,
            self.authenticator,
            self.behavior_model,
        )
        self.home_screen = HomeScreen(self.show_face_scan_screen, self.authenticator)

        # Add screens to stack
        self.stacked_widget.addWidget(self.start_screen)
        self.stacked_widget.addWidget(self.face_scan_screen)
        self.stacked_widget.addWidget(self.pin_screen)
        self.stacked_widget.addWidget(self.home_screen)

        # Setup main layout
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.addWidget(self.stacked_widget)

        # Connect signals
        self.stacked_widget.currentChanged.connect(self.on_screen_changed)
        self.start_screen.prediction_ready.connect(self.handle_prediction_ready)

        # Decide initial screen based on behavior data
        MIN_BEHAVIOR_DATA_COUNT = 10
        if self.behavior_model.get_data_count() < MIN_BEHAVIOR_DATA_COUNT:
            print("DEBUG: Not enough behavior data. Starting with Face Scan.")
            self.show_face_scan_screen()
        else:
            print("DEBUG: Enough behavior data. Starting with prediction screen.")
            self.show_start_screen()

    # Remove the duplicate setup_ui method entirely!

    def show_start_screen(self):
        self.face_scan_screen.stop_scan()
        self.pin_screen.reset_pin()
        self.stacked_widget.setCurrentWidget(self.start_screen)

    def show_face_scan_screen(self):
        self.pin_screen.reset_pin()
        self.stacked_widget.setCurrentWidget(self.face_scan_screen)
        # Pre-initialize camera when showing face screen
        self.face_scan_screen.pre_initialize_camera()

    def show_pin_screen(self):
        self.face_scan_screen.stop_scan()  # Stop scanning but keep camera ready
        self.stacked_widget.setCurrentWidget(self.pin_screen)

    def show_home_screen(self):
        self.face_scan_screen.stop_scan()
        self.pin_screen.reset_pin()
        self.stacked_widget.setCurrentWidget(self.home_screen)

    def handle_prediction_ready(self, method):
        """Handle prediction result from start screen"""
        if method == "PIN":
            self.show_pin_screen()
        else:
            self.show_face_scan_screen()

    def on_screen_changed(self, index):
        """Handle screen transitions - NO automatic camera start"""
        widget = self.stacked_widget.widget(index)
        if widget == self.home_screen:
            # Only handle enrollment prompt on home screen
            self.home_screen.prompt_for_enrollment()
        # Remove the automatic camera start from here!

    def closeEvent(self, event):
        """Properly release camera on exit"""
        self.face_scan_screen.release_camera()  # Use new method
        event.accept()


def set_stylesheet(app):
    # Stylesheet remains unchanged.
    app.setStyleSheet(
        """
        QMainWindow, QWidget { background-color: #1a202c; color: #e2e8f0; font-family: Inter, sans-serif; }
        #StatusLabel { color: #a0aec0; }
        #PinDisplay { font-size: 36px; font-weight: bold; color: #e2e8f0; letter-spacing: 5px; }
        QPushButton#PinButton { font-size: 24px; font-weight: bold; color: #e2e8f0; background-color: #2d3748; border: none; border-radius: 40px; }
        QPushButton#PinButton:hover { background-color: #4a5568; }
        QPushButton#PinButton:pressed { background-color: #718096; }
        QPushButton#SecondaryButton { font-size: 14px; color: #a0aec0; background-color: transparent; border: 1px solid #4a5568; border-radius: 15px; padding: 10px 20px; }
        QPushButton#SecondaryButton:hover { background-color: #2d3748; color: #e2e8f0; }
        QPushButton#SecondaryButton:pressed { background-color: #4a5568; }
        QMessageBox { background-color: #2d3748; }
        QMessageBox QLabel { color: #e2e8f0; font-size: 14px; }
        QMessageBox QPushButton { font-size: 12px; color: #e2e8f0; background-color: #4a5568; border-radius: 10px; padding: 8px 16px; min-width: 80px;}
        QMessageBox QPushButton:hover { background-color: #718096; }
        QPushButton#StartScanButton {
        font-size: 14px;
        color: #e2e8f0;
        background-color: #4299E1; /* Blue color */
        border: none;
        border-radius: 12px;
        padding: 8px 16px;
        max-width: 150px;
        margin-left: auto;
        margin-right: auto;
        }
        QPushButton#StartScanButton:hover { background-color: #3182CE; }
        QPushButton#StartScanButton:pressed { background-color: #2B6CB0; }
        QLabel#TitleLabel { font-size: 24px; font-weight: bold; color: #e2e8f0; }
        QLabel#CameraLabel { border: 2px solid #4A5568; border-radius: 10px; }
        QStackedWidget { border: none; }
        QGridLayout { margin: 20px; }
        QVBoxLayout { margin: 20px; }
        QHBoxLayout { margin: 20px; }
        QScrollBar:vertical { border: none; background: #2d3748; width: 10px; }
        QScrollBar::handle:vertical { background: #4a5568; min-height: 20px; border-radius: 5px; }
        QScrollBar::handle:vertical:hover { background: #718096; }
        """
    )


# Application entry point: launches the behavioral authentication GUI.
# if __name__ == "__main__":
if __name__ == "__main__":
    # Ensure the LBP mapping table exists before starting the app
    # If this is slow, it will still delay startup, but it's a one-time cost.
    get_uniform_lbp_mapping()

    app = QApplication(sys.argv)
    set_stylesheet(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
