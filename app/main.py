import os
import cv2
import math
from ultralytics import YOLO
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture

class CameraApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)  # Zugriff auf die Kamera (0 für die Standardkamera)
        self.model = YOLO("path_to_your_weights")  # Lade das YOLO-Modell
        self.outputFolder = "output_frames"  # Ordner für gespeicherte Frames
        if not os.path.exists(self.outputFolder):
            os.makedirs(self.outputFolder)
        
        self.image = Image()  # Kivy-Image-Widget zur Anzeige des Bildes
        self.button = Button(text="Start YOLO Detection", size_hint=(None, None), size=(200, 50))  # Button zum Starten der Erkennung
        self.button.bind(on_press=self.start_detection)
        
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.image)
        layout.add_widget(self.button)
        
        # Setup für FPS und Frame-Skipping
        self.frame_count = 0
        self.prevX = 1
        self.lastWasTheSame = False
        self.wiggleRoom = 0.003
        self.fSkip = 7.5 if self.capture.get(cv2.CAP_PROP_FPS) == 30 else math.ceil(self.capture.get(cv2.CAP_PROP_FPS)/4)
        self.frame_skip = self.fSkip
        
        Clock.schedule_interval(self.update, 1.0 / 30.0)  # 30 FPS für die Bildaktualisierung
        
        return layout

    def start_detection(self, instance):
        # Hier kannst du die YOLO-Erkennung aktivieren, wenn es ein Button-Press gibt
        print("YOLO detection started!")
    
    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        # Initialisiere die Bilddimensionen einmalig
        if self.frame_count == 0:
            self.hImg, self.wImg, _ = frame.shape

        if self.frame_count % self.frame_skip == 0:
            # Führe YOLO-Erkennung durch
            results = self.model.predict(frame)

            # Durchlaufe die Detektionen und verarbeite sie
            for r in results:
                for box in r.boxes:
                    x, y, w, h = box.xywh[0].tolist()  # Hole die x, y, w, h Koordinaten
                    xNorm, yNorm, wNorm, hNorm = x/self.wImg, y/self.hImg, w/self.wImg, h/self.hImg

                    if abs(xNorm - self.prevX) < self.wiggleRoom:
                        if not self.lastWasTheSame:
                            self.lastWasTheSame = True
                        else:
                            self.lastWasTheSame = False
                            filename = os.path.join(self.outputFolder, f"frame_{self.frame_count}.jpg")
                            cv2.imwrite(filename, frame)
                            self.frame_skip = 100  # Setze das Frame-Skipping hoch, um die Performance zu verbessern

                    if self.prevX > xNorm:
                        self.prevX = xNorm

            # Zeige das Bild im Kivy-Widget an
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(frame.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            self.image.texture = texture

        self.frame_count += 1

    def on_stop(self):
        # Schließe die Kamera, wenn die App gestoppt wird
        self.capture.release()

if __name__ == '__main__':
    CameraApp().run()
