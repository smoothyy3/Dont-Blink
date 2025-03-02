import time
import tensorflow.lite as tflite
import cv2
import numpy as np

interpreter = tflite.Interpreter(model_path="printhead_detector.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = cv2.imread("test_image.jpg")
image = cv2.resize(image, (224, 224))
input_data = np.expand_dims(image, axis=0).astype(np.float32) / 255.0

start_time = time.time()

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])

end_time = time.time()
print(f"Inferenzzeit: {end_time - start_time:.4f} Sekunden")