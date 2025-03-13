import cv2
import os

video_path = "/Users/jonas/Desktop/private-git/Dont-Blink/dataset/raw_vid/02.03.25/02.03.25.MP4"
output_folder = "/Users/jonas/Desktop/private-git/Dont-Blink/dataset/raw_img/02.03.25"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_skip = 11

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip == 0:
        filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(filename, frame)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"Frames saved every {frame_skip} frames.")