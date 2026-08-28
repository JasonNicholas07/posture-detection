import cv2
import numpy as np
import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv("data/dataset_postur_more.csv")

start_idx = 4492
window_size = 10
transition_sequence = df.iloc[start_idx : start_idx + window_size]

target_landmarks_1_based = [1, 8, 9, 12, 13, 14, 15, 16, 17, 25, 26, 27, 28]

cols_to_extract = []
for idx in target_landmarks_1_based:
    cols_to_extract.extend([f'x{idx}', f'y{idx}'])

# Extract raw values and reshape to (Frames, Landmarks, Coordinates)
raw_coordinates = transition_sequence[cols_to_extract].values
history_landmarks = raw_coordinates.reshape(window_size, 13, 2)

print("Data Shape:", history_landmarks.shape) 
print("Class Timeline:", transition_sequence['class'].tolist())

def generate_evolution_map(history_landmarks, frame_width=1280, frame_height=720):
    
    canvas = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
    spatial_connections = [
        (0, 1), (0, 2),        # Head
        (3, 4),                # Shoulder to Shoulder
        (3, 5), (4, 6),        # Shoulders to Elbows
        (5, 7), (6, 8),        # Elbows to Wrists
        (3, 9), (4, 10),       # Shoulders to Hips
        (9, 10),               # Hip to Hip
        (9, 11), (10, 12)      # Hips to Knees
    ]

    for t in range(len(history_landmarks) - 1):
        current_frame = history_landmarks[t]
        next_frame = history_landmarks[t + 1]
        
        for joint_idx in range(13):
            pt1 = (int(current_frame[joint_idx][0] * frame_width), int(current_frame[joint_idx][1] * frame_height))
            pt2 = (int(next_frame[joint_idx][0] * frame_width), int(next_frame[joint_idx][1] * frame_height))
            
            cv2.arrowedLine(canvas, pt1, pt2, (255, 0, 0), 2, tipLength=0.3)

    # 4. Draw Spatial Skeletons (Green Lines & Red Dots)
    for t in range(len(history_landmarks)):
        current_frame = history_landmarks[t]
        
        # Green skeletal connections
        for (i, j) in spatial_connections:
            pt1 = (int(current_frame[i][0] * frame_width), int(current_frame[i][1] * frame_height))
            pt2 = (int(current_frame[j][0] * frame_width), int(current_frame[j][1] * frame_height))
            cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)
            
        # Red joint nodes
        for joint_idx in range(13):
            pt = (int(current_frame[joint_idx][0] * frame_width), int(current_frame[joint_idx][1] * frame_height))
            cv2.circle(canvas, pt, 4, (0, 0, 255), -1)

    return canvas

output_map = generate_evolution_map(history_landmarks)
cv2.imwrite("spatio_temporal_report_figure.jpg", output_map)