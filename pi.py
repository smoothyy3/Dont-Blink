import tensorflow.lite as tflite
import cv2
import numpy as np
from picamera2 import Picamera2
import time

interpreter = tflite.Interpreter(model_path="printhead_detector.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# access pi cam
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration(main={"size": (224, 224)}))
picam2.start()

frame_interval = 0.37
last_capture_time = time.time()

while True:
    current_time = time.time()
    
    if current_time - last_capture_time >= frame_interval:
        frame = picam2.capture_array()
        frame_resized = cv2.resize(frame, (224, 224))
        input_data = np.expand_dims(frame_resized, axis=0).astype(np.float32) / 255.0

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])

        # if out of frame take pic
        if np.argmax(prediction) == 1:
            filename = f"timelapse/{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
        
        last_capture_time = current_time