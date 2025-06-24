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
)
from PySide6.QtCore import QTimer, QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap, QFont, QPalette, QColor
from enroll import enroll


class FaceDetectionThread(QThread):
    """
    Handles camera interaction, face detection, and the multi-stage capture process.
    This thread manages the logic for capturing directional images, calibration images,
    and triggering the final enrollment.
    """

    frame_ready = Signal(np.ndarray)
    face_detected = Signal(bool)
    face_captured_signal = Signal(str)
    status_update = Signal(str)
    instruction_update = Signal(str)

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
        self.max_expressions = 3  # Number of expressions per direction
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
        self.STABLE_THRESHOLD = 10  # Frames of stability needed to auto-capture

        # --- OpenCV Setup ---
        self.frontalFaceCascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.COLOR_FRONTAL = (247, 173, 62)  # BGR color for rectangle
        self.RECT_THICKNESS = 2
        self.scale = 0.7  # Scale down frame for faster detection

        # --- Ensure Directories Exist ---
        os.makedirs("./data", exist_ok=True)
        for direction, _ in self.directions:
            os.makedirs(f"./data/{direction.replace(' ', '_')}", exist_ok=True)
        os.makedirs("./positive", exist_ok=True)

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
        """Main thread loop for video capture and processing."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_update.emit("Error: Camera not accessible.")
            self.running = False
            return

        self.status_update.emit("Camera initialized - Ready for face detection")
        self.instruction_update.emit(self.get_current_instruction())

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.status_update.emit("Error: Could not read from camera")
                break

            frame_small = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            faces = self.frontalFaceCascade.detectMultiScale(gray, 1.1, 4)

            # Draw rectangles around detected faces
            for x, y, w, h in faces:
                cv2.rectangle(
                    frame_small,
                    (x, y),
                    (x + w, y + h),
                    self.COLOR_FRONTAL,
                    self.RECT_THICKNESS,
                )

            if len(faces) > 0:
                self.face_detected.emit(True)
                # --- Phase-based Logic ---
                if self.phase == "DIRECTIONS" and self.save_face:
                    self.capture_and_save_face(frame, faces[0], "DIRECTIONS")
                    self.save_face = False
                elif self.phase == "CALIBRATING":
                    self.stable_tracking_counter += 1
                    if self.stable_tracking_counter > self.STABLE_THRESHOLD:
                        self.capture_and_save_face(frame, faces[0], "CALIBRATING")
                        self.stable_tracking_counter = 0  # Reset after capture
            else:
                self.face_detected.emit(False)
                self.stable_tracking_counter = (
                    0  # Reset stability counter if face is lost
                )

            display_frame = cv2.flip(frame_small, 1)
            self.frame_ready.emit(display_frame)

            self.msleep(30)  # Control loop speed

        self.cap.release()
        self.status_update.emit("Camera disconnected")

    def capture_and_save_face(self, frame, face_rect, capture_type):
        """Crops and saves a face image for either DIRECTIONS or CALIBRATION phase."""
        x_small, y_small, w_small, h_small = face_rect
        x = int(x_small / self.scale)
        y = int(y_small / self.scale)
        w = int(w_small / self.scale)
        h = int(h_small / self.scale)
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
                time.sleep(0.1)  # Small delay for a slightly different image

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
            # Save 1 image automatically for calibration
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
                        self.phase = "IDLE"  # Reset on error
                    self.instruction_update.emit(self.get_current_instruction())

    def start_detection(self):
        """Starts the face detection thread."""
        self.phase = "DIRECTIONS"  # Start with the first phase
        self.running = True
        self.start()

    def stop_detection(self):
        """Stops the face detection thread."""
        self.running = False
        self.wait()
        self.phase = "IDLE"

    def request_capture(self):
        """Called by the UI to request a manual capture during DIRECTIONS phase."""
        if self.phase == "DIRECTIONS" and self.direction_captures < 6:
            self.save_face = True


class SFaceXMainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SFaceX - Face Authentication System v2.1 (Calibration)")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(self.get_stylesheet())

        self.detection_thread = FaceDetectionThread()
        self.detection_thread.frame_ready.connect(self.update_frame)
        self.detection_thread.face_detected.connect(self.update_face_status)
        self.detection_thread.face_captured_signal.connect(self.on_face_captured)
        self.detection_thread.status_update.connect(self.update_status)
        self.detection_thread.instruction_update.connect(self.update_instruction)

        self.setup_ui()
        self.face_detected = False

    def get_stylesheet(self):
        """Returns the CSS stylesheet for the application."""
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
        """

    def setup_ui(self):
        """Sets up the main user interface layout and widgets."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel (Video and Controls) ---
        left_panel = QVBoxLayout()
        title_label = QLabel("SFaceX Calibration")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(title_label)

        self.instruction_label = QLabel("Press 'Start Detection'")
        self.instruction_label.setObjectName("instructionLabel")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setWordWrap(True)
        left_panel.addWidget(self.instruction_label)

        video_group = QGroupBox("Live Video Feed")
        video_layout = QVBoxLayout(video_group)
        self.video_label = QLabel("Camera Feed\nClick 'Start Detection' to begin")
        self.video_label.setObjectName("videoLabel")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)
        left_panel.addWidget(video_group)

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
        self.add_log("SFaceX System Initialized")

    def start_detection(self):
        """Handler for the 'Start Detection' button."""
        self.clear_thumbnails()
        self.detection_thread.start_detection()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.add_log("Face detection started")

    def stop_detection(self):
        """Handler for the 'Stop Detection' button."""
        self.detection_thread.stop_detection()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.capture_button.setEnabled(False)
        self.video_label.setText("Camera Feed\nClick 'Start Detection' to begin")
        self.update_face_status(False)
        self.add_log("Face detection stopped")

    def capture_face(self):
        """Handler for the manual 'Capture' button."""
        if self.face_detected:
            self.detection_thread.request_capture()
            self.add_log("Face capture requested")
            self.capture_button.setText(
                f"Capture ({self.detection_thread.capture_count + 1}/3)"
            )
            if self.detection_thread.capture_count >= 2:
                self.capture_button.setEnabled(False)

    def update_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)

    def update_face_status(self, detected):
        """Updates UI elements based on whether a face is detected."""
        self.face_detected = detected

        # Enable capture button only during the DIRECTIONS phase if a face is detected
        can_capture_manually = (
            detected
            and self.detection_thread.phase == "DIRECTIONS"
            and self.detection_thread.capture_count < 3
        )
        self.capture_button.setEnabled(can_capture_manually)

        if detected:
            self.face_status_label.setText("Face Detected ✓")
            self.face_status_label.setStyleSheet(
                "QLabel { background-color: #2d5a2d; border: 2px solid #4a8c4a; border-radius: 8px; padding: 10px; font-size: 16px; font-weight: bold; }"
            )
        else:
            self.face_status_label.setText("No Face Detected")
            self.face_status_label.setStyleSheet(
                "QLabel { background-color: #5a2d2d; border: 2px solid #8c4a4a; border-radius: 8px; padding: 10px; font-size: 16px; font-weight: bold; }"
            )

    def update_status(self, message):
        """Updates the main status label and logs the message."""
        self.status_label.setText(message)
        self.add_log(message)

    def on_face_captured(self, filename):
        """Creates and displays a thumbnail for a newly captured image."""
        self.add_log(f"Image saved: {os.path.basename(filename)}")
        pixmap = QPixmap(filename)
        thumbnail = QLabel()
        thumbnail.setPixmap(
            pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        thumbnail.setToolTip(filename)

        # Place thumbnail in the grid based on capture phase
        if self.detection_thread.phase == "DIRECTIONS":
            row = self.detection_thread.current_direction_index
            col = self.detection_thread.direction_captures - 1
            if col < 6:
                self.captured_images_grid.addWidget(thumbnail, row, col)
        elif (
            self.detection_thread.phase == "CALIBRATING"
            or self.detection_thread.phase == "ENROLLING"
        ):
            row = len(
                self.detection_thread.directions
            )  # Place calibration images on a new row
            col = self.detection_thread.calibration_count - 1
            if col < self.detection_thread.MAX_CALIBRATION_IMAGES:
                self.captured_images_grid.addWidget(thumbnail, row, col)

    def update_instruction(self, instruction):
        """Updates the main instruction label for the user."""
        self.instruction_label.setText(instruction)

        # Reset manual capture button text when a new direction starts
        if self.detection_thread.phase == "DIRECTIONS":
            self.capture_button.setText(
                f"Capture ({self.detection_thread.capture_count}/3)"
            )

        # Disable manual capture button if not in DIRECTIONS phase or process is complete
        is_manual_phase = self.detection_thread.phase == "DIRECTIONS"
        self.capture_button.setEnabled(is_manual_phase)

        if "complete" in instruction.lower():
            self.stop_button.setEnabled(False)
            self.start_button.setEnabled(True)

    def add_log(self, message):
        """Appends a timestamped message to the activity log."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )

    def clear_thumbnails(self):
        """Removes all captured image thumbnails from the grid."""
        for i in reversed(range(self.captured_images_grid.count())):
            self.captured_images_grid.itemAt(i).widget().setParent(None)

    def closeEvent(self, event):
        """Ensures the detection thread is stopped when closing the window."""
        if self.detection_thread.isRunning():
            self.detection_thread.stop_detection()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SFaceXMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # A placeholder for the enroll function if it's not in the same directory
    # In your actual use, you would have your `enroll.py` file.
    if not os.path.exists("enroll.py"):
        with open("enroll.py", "w") as f:
            f.write("import time\n")
            f.write("def enroll():\n")
            f.write("    print('Starting enrollment process...')\n")
            f.write("    # This is a placeholder for your actual enrollment logic\n")
            f.write("    # It might involve training a model on the captured images\n")
            f.write("    time.sleep(5) # Simulate a long process\n")
            f.write("    print('Enrollment process finished.')\n")
    main()
