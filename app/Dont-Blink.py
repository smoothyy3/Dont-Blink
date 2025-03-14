from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QMessageBox, QFileDialog, QCheckBox)
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import cv2
import sys
import os
import math
import datetime
from ultralytics import YOLO
import requests
import shutil
import subprocess
import time
from natsort import natsorted
from packaging import version

# show avalable cameras to user
def list_cameras():
    index = 0
    available_cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            break
        available_cameras.append(str(index))
        cap.release()
        index += 1
    return available_cameras

#---------------------------------------------------------------#
# Camera processing
#---------------------------------------------------------------#
class YOLOProcessingThread(QThread):
    finished_signal = pyqtSignal()
    
    def __init__(self, cap, output_folder):
        super().__init__()
        self.cap = cap
        self.output_folder = output_folder
        self.running = True
    
    def run(self):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(BASE_DIR, "weights", "best.pt")
        model = YOLO(weights_path)
        
        if self.cap.get(cv2.CAP_PROP_FPS) == 30:
            fSkip = 7.5
        else:
            fSkip = math.ceil(self.cap.get(cv2.CAP_PROP_FPS)/4)
        
        prevX = 1
        lastWasTheSame = False
        wiggleRoom = 0.003
        frame_skip = fSkip
        frame_count = 0
        photo_taken = False
        
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # extract dimensions once on first frame
            if frame_count == 0:
                hImg, wImg, _ = frame.shape

            if frame_count % frame_skip == 0:
                results = model.predict(frame, device="cpu")
                frame_skip = fSkip
                
                for r in results:
                    for box in r.boxes:
                        x, y, w, h = box.xywh[0].tolist() # Get the x, y, w, h coordinates.
                        xNorm, yNorm, wNorm, hNorm =  x/wImg, y/hImg, w/wImg, h/hImg
                        
                        if abs(xNorm - prevX) < wiggleRoom:
                            prevX = xNorm
                            if not lastWasTheSame:
                                lastWasTheSame = True
                            elif not photo_taken:
                                filename = os.path.join(self.output_folder, f"frame_{frame_count}.jpg")
                                cv2.imwrite(filename, frame)
                                photo_taken = True
                        else:
                            lastWasTheSame = False
                            photo_taken = False
                        
                        if prevX > xNorm:
                            prevX = xNorm
            frame_count += 1
        
        self.finished_signal.emit()
    
    def stop(self):
        self.running = False


class YOLOVideoProcessingThread(QThread):
    finished_signal = pyqtSignal()
    
    def __init__(self, video_path, output_folder):
        super().__init__()
        self.video_path = video_path
        self.output_folder = output_folder
        self.running = True
    
    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(BASE_DIR, "weights", "best.pt")
        model = YOLO(weights_path)
        
        if self.cap.get(cv2.CAP_PROP_FPS) == 30:
            fSkip = 7.5
        else:
            fSkip = math.ceil(self.cap.get(cv2.CAP_PROP_FPS)/4)
        
        prevX = 1
        lastWasTheSame = False
        wiggleRoom = 0.003
        frame_skip = fSkip
        frame_count = 0
        photo_taken = False
        
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            # extract dimensions once on first frame
            if frame_count == 0:
                hImg, wImg, _ = frame.shape

            if frame_count % frame_skip == 0:
                results = model.predict(frame, device="cpu")
                frame_skip = fSkip
                
                for r in results:
                    for box in r.boxes:
                        x, y, w, h = box.xywh[0].tolist() # Get the x, y, w, h coordinates.
                        xNorm, yNorm, wNorm, hNorm =  x/wImg, y/hImg, w/wImg, h/hImg
                        
                        if abs(xNorm - prevX) < wiggleRoom:
                            prevX = xNorm
                            if not lastWasTheSame:
                                lastWasTheSame = True
                            elif not photo_taken:
                                filename = os.path.join(self.output_folder, f"frame_{frame_count}.jpg")
                                cv2.imwrite(filename, frame)
                                photo_taken = True
                        else:
                            lastWasTheSame = False
                            photo_taken = False
                        
                        if prevX > xNorm:
                            prevX = xNorm
            frame_count += 1
        
        self.finished_signal.emit()
    
    def stop(self):
        self.running = False

#---------------------------------------------------------------#
# Camera App
#---------------------------------------------------------------#
class CameraApp(QWidget):
    def __init__(self):
        super().__init__()

        # Set up main layout
        self.layout = QVBoxLayout()

        # Load External Stylesheet
        self.load_stylesheet("style.qss")

        # Set Fixed Window Size
        self.setFixedSize(624, 600)

        # Initialize UI Components
        self.create_title_bar()
        self.create_input_selection()
        self.create_camera_selection()
        self.create_buttons()

        # Initialize variables
        self.initialize_variables()

        # Set main layout
        self.setLayout(self.layout)

        self.update_input_selection()

    #---------------------------------------------------------------#
    # Load inital variables

    def initialize_variables(self):
        """Initialize essential variables and configurations."""
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_preview)
        self.output_folder = ""
        self.current_session_folder = ""
        self.processing_thread = None
        self.video_processing_thread = None

        # Load YOLO Model
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        weights_path = os.path.join(BASE_DIR, "weights", "best.pt")
        self.model = YOLO(weights_path)

        # Adjust UI Elements' Sizes
        self.camera_selection.setFixedWidth(50)
        self.select_button.setFixedWidth(250)
        self.select_folder_button.setFixedWidth(250)
        self.start_button.setFixedWidth(250)
        self.stop_button.setFixedWidth(250)
        self.timelapse_button.setFixedWidth(250)

    # Load the stylesheet
    def load_stylesheet(self, filename):
        """Loads an external QSS file and applies it to the app."""
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        qss_path = os.path.join(BASE_DIR, filename)

        if os.path.exists(qss_path):
            with open(qss_path, "r") as f:
                self.setStyleSheet(f.read())
        else:
            print(f"⚠️ Stylesheet file '{filename}' not found!")

    def create_title_bar(self):
        """Creates the top title bar with update button."""
        title_layout = QHBoxLayout()

        # Get Version Number
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS  # PyInstaller temp extraction folder
        else:
            base_path = os.path.dirname(__file__)  # Running from source

        current_version_path = os.path.join(base_path, "version.txt")

        current_version = "Unknown"
        if os.path.exists(current_version_path):
            with open(current_version_path, "r") as f:
                current_version = f.read().strip()

        # Set window title with version
        self.setWindowTitle(f"Dont-Blink v{current_version}")

        title_layout.addStretch()  # Pushes the update button to the right

        # Update Button
        self.update_button = QPushButton("Check for Update")
        self.update_button.setFixedSize(170, 30)
        self.update_button.setObjectName("update_button")  # Apply custom QSS
        self.update_button.clicked.connect(self.check_for_updates)
        title_layout.addWidget(self.update_button)

        # Create a title container
        title_container = QWidget()
        title_container.setLayout(title_layout)
        self.layout.addWidget(title_container)

    def create_input_selection(self):
        """Creates the input selection dropdown for Webcam/MP4."""
        self.input_selection = QComboBox()
        self.input_selection.addItems(["Webcam", "MP4 File"])
        self.input_selection.currentIndexChanged.connect(self.update_input_selection)
        self.layout.addWidget(self.input_selection)

    def create_camera_selection(self):
        """Creates the webcam/video selection UI."""
        self.input_method_layout = QHBoxLayout()

        # Camera selection (Initially hidden)
        self.camera_selection = QComboBox()
        self.available_cameras = list_cameras()
        if not self.available_cameras:
            self.available_cameras.append("No cameras found")
        self.camera_selection.addItems(self.available_cameras)
        self.camera_selection.setVisible(False)
        self.input_method_layout.addWidget(self.camera_selection)

        # Confirm Camera button (Initially hidden)
        self.select_button = QPushButton("Confirm Camera")
        self.select_button.clicked.connect(self.select_camera)
        self.select_button.setVisible(False)
        self.input_method_layout.addWidget(self.select_button)

        # Select Video File button (Initially hidden)
        self.select_video_button = QPushButton("Select MP4 File")
        self.select_video_button.clicked.connect(self.select_video_file)
        self.select_video_button.setVisible(False)
        self.input_method_layout.addWidget(self.select_video_button)

        # Add layout to the main UI
        self.layout.addLayout(self.input_method_layout)

        # ----- Preview & Detection Checkbox -----
        preview_layout = QHBoxLayout()

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        video_path = os.path.join(BASE_DIR, "assets", "Video.png")

        self.camera_preview_label = QLabel(self)
        self.camera_preview_label.setPixmap(QPixmap(video_path))
        self.camera_preview_label.setFixedSize(426, 240)
        preview_layout.addWidget(self.camera_preview_label)

        self.check_detection = QCheckBox("Preview detection")
        preview_layout.addWidget(self.check_detection)

        # Container for Preview & Checkbox
        preview_container = QWidget()
        preview_container.setLayout(preview_layout)
        preview_container.setMaximumWidth(600)  # Can be adjusted
        self.layout.addWidget(preview_container)

    def create_buttons(self):
        """Creates action buttons like Start, Stop, Output Folder, and Timelapse."""
        
        # ----- Start & Stop Buttons -----
        button_layout = QHBoxLayout()

        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        button_container = QWidget()
        button_container.setLayout(button_layout)
        button_container.setMaximumWidth(525)
        self.layout.addWidget(button_container)

        # ----- Output Folder Button -----
        output_layout = QHBoxLayout()

        self.select_folder_button = QPushButton("Select Output Folder")
        self.select_folder_button.clicked.connect(self.select_output_folder)
        output_layout.addWidget(self.select_folder_button)

        self.output_folder_label = QLabel("Output Folder:")
        output_layout.addWidget(self.output_folder_label)

        output_container = QWidget()
        output_container.setLayout(output_layout)
        output_container.setMaximumWidth(600)
        self.layout.addWidget(output_container)

        # ----- Timelapse Button -----
        timelapse_layout = QHBoxLayout()

        self.timelapse_button = QPushButton("Create Timelapse")
        self.timelapse_button.clicked.connect(self.create_timelapse)
        timelapse_layout.addWidget(self.timelapse_button)

        self.status_label = QLabel("Status:")
        timelapse_layout.addWidget(self.status_label)

        timelapse_container = QWidget()
        timelapse_container.setLayout(timelapse_layout)
        timelapse_container.setMaximumWidth(600)
        self.layout.addWidget(timelapse_container)

    #---------------------------------------------------------------#
    # Input Types
    def update_input_selection(self):
        """Show the correct UI elements based on input selection"""
        selected_input = self.input_selection.currentText()

        if selected_input == "Webcam":
            self.camera_selection.setVisible(True)
            self.select_button.setVisible(True)
            self.select_video_button.setVisible(False)
        else:
            self.camera_selection.setVisible(False)
            self.select_button.setVisible(False)
            self.select_video_button.setVisible(True)

    def select_video_file(self):
        """Let the user choose an MP4 file as input."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4)")
        if file_path:
            self.video_file = file_path
            QMessageBox.information(self, "Video Selected", f"Using video file: {os.path.basename(file_path)}")
            self.cap = cv2.VideoCapture(self.video_file)
            self.is_video = True
            self.timer.start(50)

    def select_camera(self):
        if self.cap is not None:
            self.cap.release()
        self.selected_camera = int(self.camera_selection.currentText())
        self.cap = cv2.VideoCapture(self.selected_camera)
        self.is_video = False
        self.timer.start(50)
    
    #---------------------------------------------------------------#
    # Processing & Updates

    def start_processing(self):
        if not self.output_folder:
            QMessageBox.warning(self, "No Output Folder", "Please select an output folder before starting processing.")
            return
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_session_folder = os.path.join(self.output_folder, f"session_{timestamp}")
        os.makedirs(self.current_session_folder, exist_ok=True)

        self.input_type = self.input_selection.currentText()

        if self.input_type == "Webcam":
            selected_text = self.camera_selection.currentText()
            self.selected_camera = int(selected_text)
            self.cap = cv2.VideoCapture(self.selected_camera)

        elif self.input_type == "MP4 File":
            if not hasattr(self, 'video_file') or not self.video_file:
                QMessageBox.warning(self, "No Video Selected", "Please select a video file before starting processing.")
                return
            self.cap = cv2.VideoCapture(self.video_file)  # Open video file
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Could not open the selected video file.")
                return
            ret, frame = self.cap.read()
            if not ret:
                QMessageBox.critical(self, "Error", "Could not read the first frame of the video.")
                return
            print(f"🎥 First frame read successfully: {frame.shape}")

        self.processing_thread = YOLOProcessingThread(self.cap, self.current_session_folder)
        self.video_processing_thread = YOLOVideoProcessingThread(self.cap, self.current_session_folder)
        self.processing_thread.finished_signal.connect(self.processing_finished)
        self.video_processing_thread.finished_signal.connect(self.processing_finished)
        self.processing_thread.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.check_detection.setChecked(False)
        self.check_detection.setEnabled(False)

    def stop_processing(self):
        if self.processing_thread:
            self.processing_thread.stop()
        self.processing_finished()
    
    def processing_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.check_detection.setEnabled(True)

    #---------------------------------------------------------------#
    # Updater

    def check_for_updates(self):
        latest_version_url = "https://smoothyy3.github.io/Dont-Blink/latest_version.txt"  # Your GitHub Pages URL

        # Determine if running from a bundled PyInstaller EXE
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS  # PyInstaller temp extraction folder
        else:
            base_path = os.path.dirname(__file__)  # Running from source

        current_version_path = os.path.join(base_path, "version.txt")

        try:
            # Read the current version (If it doesn’t exist, default to "0.0")
            if not os.path.exists(current_version_path):
                current_version = "0.0"
            else:
                with open(current_version_path, "r") as f:
                    current_version = f.read().strip()

            # Fetch the latest version info
            response = requests.get(latest_version_url)
            response.raise_for_status()

            latest_version_info = response.text.split("\n")
            latest_version = latest_version_info[0].strip()
            download_url = latest_version_info[1].strip()  # URL of new .exe

            # Use proper version comparison
            if version.parse(latest_version) > version.parse(current_version):
                reply = QMessageBox.question(self, "Update Available", 
                                            f"A new version ({latest_version}) is available. Do you want to update?",
                                            QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.download_and_replace(download_url, latest_version)
            else:
                QMessageBox.information(self, "No Updates", "You have the latest version.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to check for updates: {e}")

    def download_and_replace(self, url, latest_version):
        """Downloads the new version and replaces the running executable safely."""
        
        # Paths
        current_exe = sys.executable
        temp_exe = os.path.join(os.path.dirname(current_exe), "Dont-Blink-Temp.exe")
        backup_exe = os.path.join(os.path.dirname(current_exe), "Dont-Blink-Old.exe")
        update_script = os.path.join(os.path.dirname(current_exe), "update_script.bat")

        try:
            # Download the new exe
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(temp_exe, 'wb') as f:
                    shutil.copyfileobj(r.raw, f)

            QMessageBox.information(self, "Update Ready", "Update downloaded! Please Restart")

            # Create an update script that waits before replacing the exe
            with open(update_script, "w") as f:
                f.write(f"""@echo off
                echo Updating Dont-Blink...
                timeout /t 5 /nobreak > NUL
                del /F /Q "%~dp0Dont-Blink.exe"
                move /Y "%~dp0Dont-Blink-Temp.exe" "%~dp0Dont-Blink.exe"
                timeout /t 2 /nobreak > NUL
                del "%~f0"
                """)

            # Run the update script and exit
            subprocess.Popen(update_script, shell=True)
            sys.exit(0)

        except Exception as e:
            QMessageBox.critical(self, "Update Failed", f"Could not download update: {e}")
    
    #---------------------------------------------------------------#
    # Camera/ Video Preview
    def update_preview(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()

            if self.is_video and not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
                ret, frame = self.cap.read()

            if ret:
                if self.check_detection.isChecked():
                    results = self.model(frame, conf=0.5)
                    for r in results:
                        for box in r.boxes:
                            x, y, w, h = box.xywh[0].tolist()
                            x_min, y_min = int(x - w / 2), int(y - h / 2)
                            x_max, y_max = int(x + w / 2), int(y + h / 2)
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            cv2.putText(frame, f"Conf: {box.conf[0]:.2f}", (x_min, y_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg).scaled(426, 240)
                self.camera_preview_label.setPixmap(pixmap)
        else:
            self.timer.stop() 
    
    #---------------------------------------------------------------#
    # Select Ouputfolder
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_folder_label.setText(f"Output Folder: {os.path.basename(os.path.dirname(self.output_folder))}/{os.path.basename(self.output_folder)}")

    #---------------------------------------------------------------#
    # Create Timelapse
    def create_timelapse(self):
        if not self.current_session_folder:
            QMessageBox.warning(self, "No Session Folder", "No session folder found. Start YOLO processing first.")
            return

        frames_path = self.current_session_folder
        video_path = os.path.join(frames_path, "timelapse.mp4")
        FPS = 15

        images = [img for img in os.listdir(frames_path) if img.endswith(".jpg")]
        images = natsorted(images)

        if not images:
            QMessageBox.warning(self, "No Images", "No images found in the session folder to create a timelapse.")
            return

        first_frame = cv2.imread(os.path.join(frames_path, images[0]))
        if first_frame is None:
            QMessageBox.warning(self, "Error", "Could not read the first image.")
            return

        h, w, _  = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))

        for image in images:
            frame = cv2.imread(os.path.join(frames_path, image))
            if frame is None:
                print(f"Warning: Skipping unreadable image {image}")
                continue
            out.write(frame)

        out.release()
        self.status_label.setText(f"Video saved in Output Folder as: timelapse.mp4")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(BASE_DIR,"assets","icon.png")
    app.setWindowIcon(QIcon(icon_path))
    
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())