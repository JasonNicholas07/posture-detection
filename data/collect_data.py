import cv2                          
import mediapipe as mp              
from mediapipe.tasks import python  
from mediapipe.tasks.python import vision
import csv      
import os      
import numpy as np
import urllib.request

model_path = 'pose_landmarker_lite.task'

if not os.path.exists(model_path):
    print("Downloading MediaPipe..")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, model_path)
    print("MediaPipe Downloaded\n")

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    num_poses=1                         
)
detector = vision.PoseLandmarker.create_from_options(options)

csv_file = 'dataset_postur_people.csv'

if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = []
        for i in range(1, 34): # 33 points
            header += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']
        header.append('class')
        writer.writerow(header)

POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), 
                    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), 
                    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), 
                    (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), 
                    (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]

def draw_custom_landmarks(image, landmarks):
    h, w, _ = image.shape

    for connection in POSE_CONNECTIONS:
        idx1, idx2 = connection
        if idx1 < len(landmarks) and idx2 < len(landmarks):
            lm1, lm2 = landmarks[idx1], landmarks[idx2]
            cx1, cy1 = int(lm1.x * w), int(lm1.y * h)
            cx2, cy2 = int(lm2.x * w), int(lm2.y * h)
            cv2.line(image, (cx1, cy1), (cx2, cy2), (255, 255, 255), 2)

    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 4, (0, 255, 0), -1)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("--- Data Gathering ---")
print("Tekan '1' -> NORMAL (Good)")
print("Tekan '2' -> FORWARD (Forward Lean)")
print("Tekan '3' -> BACK (Backward Slouch)")
print("Tekan 'q' -> Exit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed: Can't read frame.")
        break
        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    detection_result = detector.detect(mp_image)
    annotated_image = frame.copy()
    
    if detection_result.pose_landmarks:
        pose_landmarks = detection_result.pose_landmarks[0]
        
        draw_custom_landmarks(annotated_image, pose_landmarks)
        
        pose_row = []
        for landmark in pose_landmarks:
            pose_row.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        key = cv2.waitKey(1) & 0xFF
        class_name = None
        
        if key == ord('1'):
            class_name = 'Normal'
            print("Normal Saved!")
        elif key == ord('2'):
            class_name = 'Forward'
            print("FORWARD Saved!")
        elif key == ord('3'):
            class_name = 'Back'
            print("BACK Saved!")
            
        if class_name is not None:
            pose_row.append(class_name)
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(pose_row)

    cv2.imshow('MediaPipe Tasks', annotated_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nDone. Saved to file dataset_postur_people.csv")
