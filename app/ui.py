from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, 
                             QComboBox, QMessageBox, QFileDialog)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
import cv2
import sys
import os
import math
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
                results = model.predict(frame, device = "cuda")
                frame_skip = fSkip
            
            if frame_count % frame_skip == 0:
                results = model.predict(frame, device="cuda")
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
        self.setWindowTitle("Camera Selection & Timelapse Processing")
        self.setGeometry(100, 100, 600, 400)
        
        self.layout = QVBoxLayout()
        
        self.label = QLabel("Select Camera:")
        self.layout.addWidget(self.label)
        
        self.camera_selection = QComboBox()
        self.available_cameras = list_cameras()
        if not self.available_cameras:
            self.available_cameras.append("No cameras found")
        self.camera_selection.addItems(self.available_cameras)
        self.layout.addWidget(self.camera_selection)
        
        self.select_button = QPushButton("Confirm Camera")
        self.select_button.clicked.connect(self.select_camera)
        self.layout.addWidget(self.select_button)
        
        self.camera_preview_label = QLabel("Camera preview here")
        self.camera_preview_label.setFixedSize(320, 240)
        self.layout.addWidget(self.camera_preview_label)
        
        self.select_folder_button = QPushButton("Select Output Folder")
        self.select_folder_button.clicked.connect(self.select_output_folder)
        self.layout.addWidget(self.select_folder_button)
        
        self.start_button = QPushButton("Start YOLO Processing")
        self.start_button.clicked.connect(self.start_processing)
        self.layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        self.layout.addWidget(self.stop_button)
        
        self.timelapse_button = QPushButton("Create Timelapse")
        self.timelapse_button.clicked.connect(self.create_timelapse)
        self.layout.addWidget(self.timelapse_button)
        
        self.setLayout(self.layout)
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_preview)
        self.output_folder = ""
        self.processing_thread = None
    
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
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg).scaled(320, 240)
                self.camera_preview_label.setPixmap(pixmap)
        else:
            self.timer.stop()
    
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder = folder
    
    def start_processing(self):
        if not self.output_folder or self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self, "Warning", "Select camera and output folder before processing.")
            return
        
        self.processing_thread = YOLOProcessingThread(self.cap, self.output_folder)
        self.processing_thread.finished_signal.connect(self.processing_finished)
        self.processing_thread.start()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
    
    def stop_processing(self):
        if self.processing_thread:
            self.processing_thread.stop()
        self.processing_finished()
    
    def processing_finished(self):
        QMessageBox.information(self, "Processing Done", "Timelapse Processing finished! Images saved in output folder.")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def create_timelapse(self):
        # (Timelapse creation logic remains unchanged)
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
