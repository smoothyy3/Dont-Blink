from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
                             QComboBox, QMessageBox, QFileDialog, QCheckBox)
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QTimer, QThread, pyqtSignal

import cv2
import sys
import os
import math
import datetime
from ultralytics import YOLO
from natsort import natsorted



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


class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dont-Blink")
        self.setGeometry(100, 100, 640, 480)


        # Setzt das QSS für das gesamte Fenster
        self.setStyleSheet("""
        QWidget {
            background-color: #ffffff;
            font-family: "Segoe UI", sans-serif;
            color: #333;
            border-radius: 10px;
        }

        QLabel {
            font-size: 14px;
            font-weight: bold;
            color: #444;
        }

        QComboBox {
            background-color: #ffffff;
            border: 1px solid #d1d1d1;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 14px;
        }

        QComboBox::drop-down {
            border-radius: 5px;
        }

        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: 1px solid #4CAF50;
            padding: 10px 20px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
        }

        QPushButton:hover {
            background-color: #45a049;
        }

        QPushButton:pressed {
            background-color: #388e3c;
        }

        QPushButton:disabled {
            background-color: #dcdcdc;
            color: #a5a5a5;
        }

        QCheckBox {
            font-size: 14px;
            padding: 5px 10px;
            color: #444;
        }

        QCheckBox::indicator {
            width: 15px;
            height: 15px;
        }

        QCheckBox::indicator:checked {
            background-color: #4CAF50;
            border-radius: 5px;
        }

        QCheckBox::indicator:unchecked {
            background-color: #dcdcdc;
            border-radius: 5px;
        }

        QLineEdit {
            background-color: #ffffff;
            border: 1px solid #d1d1d1;
            padding: 5px;
            border-radius: 5px;
            font-size: 14px;
        }

        QLabel#camera_preview_label {
            background-color: #222222;
            border-radius: 5px;
        }

        QScrollArea {
            background-color: #f5f5f5;
            border-radius: 10px;
            border: 1px solid #d1d1d1;
        }

        QScrollBar {
            width: 10px;
            background-color: #f5f5f5;
        }

        QScrollBar::handle {
            background-color: #cccccc;
            border-radius: 5px;
        }

        QScrollBar::handle:hover {
            background-color: #aaaaaa;
        }

        QScrollBar::add-line, QScrollBar::sub-line {
            background-color: #f5f5f5;
        }

        QScrollBar::up-arrow, QScrollBar::down-arrow {
            background-color: #f5f5f5;
        }
        """) 
        self.layout = QVBoxLayout()

        self.label = QLabel("Select Camera:")
        self.layout.addWidget(self.label)

        # Erstelle ein horizontales Layout für ComboBox und Button
        camera_layout = QHBoxLayout()

        self.camera_selection = QComboBox()
        self.available_cameras = list_cameras()
        if not self.available_cameras:
            self.available_cameras.append("No cameras found")
        self.camera_selection.addItems(self.available_cameras)
        camera_layout.addWidget(self.camera_selection)

        self.select_button = QPushButton("Confirm Camera")
        self.select_button.clicked.connect(self.select_camera)
        camera_layout.addWidget(self.select_button)

        # Container für Kamera-Auswahl
        camera_container = QWidget()
        camera_container.setLayout(camera_layout)
        camera_container.setMaximumWidth(325)
        self.layout.addWidget(camera_container)

        # Erstelle ein horizontales Layout für Preview und Checkbox
        preview_layout = QHBoxLayout()
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        video_path = os.path.join(BASE_DIR,"Video.png")
        self.camera_preview_label = QLabel(self)
        self.camera_preview_label.setPixmap(QPixmap(video_path))
        self.camera_preview_label.setFixedSize(426, 240)
        preview_layout.addWidget(self.camera_preview_label)

        self.check_detection = QCheckBox("Preview detection")
        preview_layout.addWidget(self.check_detection)

        # Container für Preview + Checkbox
        preview_container = QWidget()
        preview_container.setLayout(preview_layout)
        preview_container.setMaximumWidth(600)  # Kann angepasst werden
        self.layout.addWidget(preview_container)

        # Erstelle ein horizontales Layout für Start und Stop Button
        button_layout = QHBoxLayout()

        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        # Container für Start + Stop Button
        button_container = QWidget()
        button_container.setLayout(button_layout)
        button_container.setMaximumWidth(525)  # Kann angepasst werden
        self.layout.addWidget(button_container)

        # Container für Output Folder Button
        output_layout = QHBoxLayout()
        self.select_folder_button = QPushButton("Select Output Folder")
        self.select_folder_button.clicked.connect(self.select_output_folder)
        output_layout.addWidget(self.select_folder_button)
        
        self.output_folder_label = QLabel("Current Output Folder:")
        output_layout.addWidget(self.output_folder_label)

        output_container = QWidget()
        output_container.setLayout(output_layout)
        output_container.setMaximumWidth(600)  # Kann angepasst werden
        self.layout.addWidget(output_container)

        # Container für Timelapse Button
        timelapse_layout = QHBoxLayout()
        self.timelapse_button = QPushButton("Create Timelapse")
        self.timelapse_button.clicked.connect(self.create_timelapse)
        timelapse_layout.addWidget(self.timelapse_button)

        self.status_label = QLabel("Status:")
        timelapse_layout.addWidget(self.status_label)

        timelapse_container = QWidget()
        timelapse_container.setLayout(timelapse_layout)
        timelapse_container.setMaximumWidth(400)  # Kann angepasst werden
        self.layout.addWidget(timelapse_container)


        self.setLayout(self.layout)
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_preview)
        self.output_folder = ""
        self.current_session_folder = ""
        self.processing_thread = None
        self.model = YOLO(os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "best.pt"))

        
        self.camera_selection.setFixedWidth(50)
        self.select_button.setFixedWidth(250)
        self.select_folder_button.setFixedWidth(250)
        self.start_button.setFixedWidth(250)
        self.stop_button.setFixedWidth(250)
        self.timelapse_button.setFixedWidth(250)
        
    
    def select_camera(self):
        if self.cap is not None:
            self.cap.release()
        
        selected_text = self.camera_selection.currentText()
        if selected_text == "No cameras found":
            QMessageBox.critical(self, "Error", "No cameras detected.")
            return

        self.selected_camera = int(selected_text)
        self.cap = cv2.VideoCapture(self.selected_camera)
        
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open selected camera.")
            return

        self.timer.start(50)
    
    def update_preview(self):
        if self.cap is not None and self.cap.isOpened():
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
    
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
    
    def start_processing(self):
        if not self.output_folder:
            QMessageBox.warning(self, "No Output Folder", "Please select an output folder before starting processing.")
            return
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_session_folder = os.path.join(self.output_folder, f"session_{timestamp}")
        os.makedirs(self.current_session_folder, exist_ok=True)
        self.processing_thread = YOLOProcessingThread(self.cap, self.current_session_folder)
        self.processing_thread.finished_signal.connect(self.processing_finished)
        self.processing_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.check_detection.setChecked(False)
        self.check_detection.setEnabled(False)
        QMessageBox.information(self, "Processing Started", f"Images will be saved in: {self.current_session_folder}")
    
    def stop_processing(self):
        if self.processing_thread:
            self.processing_thread.stop()
        self.processing_finished()
    
    def processing_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.check_detection.setEnabled(True)
    
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

        h, w,  = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))

        for image in images:
            frame = cv2.imread(os.path.join(frames_path, image))
            if frame is None:
                print(f"Warning: Skipping unreadable image {image}")
                continue
            out.write(frame)

        out.release()
        QMessageBox.information(self, "Timelapse Created", f"Timelapse video saved at: {video_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(BASE_DIR,"icon.png")
    app.setWindowIcon(QIcon(icon_path))
    
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())




