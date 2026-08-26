import sys
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
import joblib
import urllib.request
import os

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout


class TemporalSmoother:
    def __init__(self, window: int = 10):
        self.window  = window
        self.history = []

    def update(self, prediction: int) -> int:
        self.history.append(prediction)
        if len(self.history) > self.window:
            self.history.pop(0)
        counts = np.bincount(self.history)
        return int(np.argmax(counts))

    def reset(self):
        self.history = []


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    
    shoulder_mid_x = (df['x12'] + df['x13']) / 2
    shoulder_mid_y = (df['y12'] + df['y13']) / 2

    feat['nose_to_shoulder_mid_x'] = df['x1'] - shoulder_mid_x
    feat['nose_to_shoulder_mid_y'] = df['y1'] - shoulder_mid_y
    feat['nose_to_shoulder_mid_dist'] = np.sqrt(
        feat['nose_to_shoulder_mid_x']**2 + feat['nose_to_shoulder_mid_y']**2
    )

    feat['shoulder_width'] = np.sqrt(
        (df['x12'] - df['x13'])**2 + (df['y12'] - df['y13'])**2
    )

    feat['left_ear_shoulder_y_diff']  = df['y8']  - df['y12']
    feat['right_ear_shoulder_y_diff'] = df['y9']  - df['y13']
    feat['ear_shoulder_y_asymmetry']  = (
        feat['left_ear_shoulder_y_diff'] - feat['right_ear_shoulder_y_diff']
    )

    feat['nose_left_ear_x_diff']  = df['x1'] - df['x8']
    feat['nose_right_ear_x_diff'] = df['x1'] - df['x9']
    feat['eye_level_diff'] = df['y3'] - df['y6']

    feat['relative_nose_z'] = df['z1'] - ((df['z12'] + df['z13']) / 2)
    feat['normalized_nose_z'] = feat['relative_nose_z'] / (feat['shoulder_width'] + 0.0001)
    
    feat['neck_forward_angle'] = np.degrees(
        np.arctan2(
            np.abs(feat['relative_nose_z']), 
            np.abs(feat['nose_to_shoulder_mid_y']) + 0.0001
        )
    )

    LANDMARK_COUNT = 13
    v_cols = [f'v{i}' for i in range(1, LANDMARK_COUNT + 1)]
    if all(col in df.columns for col in v_cols):
        feat['mean_visibility'] = df[v_cols].mean(axis=1)
        feat['min_visibility']  = df[v_cols].min(axis=1)

    return feat


# UI HELPERS (Original PosturePal Drawing Functions)
LANDMARK_COUNT = 13
UPPER_CONNECTIONS = [
    (0, 1), (0, 4),                 
    (1, 2), (2, 3),                 
    (4, 5), (5, 6),                 
    (7, 8),                         
    (9, 10),                        
    (11, 12),                       
]

CLASS_COLORS = {
    'Normal':  (34,  197,  94),     
    'Forward': (239, 68,   68),     
    'Back':    (234, 179,  8),      
}
DEFAULT_COLOR = (156, 163, 175)     

def get_color(class_name: str):
    return CLASS_COLORS.get(class_name, DEFAULT_COLOR)

def draw_skeleton(image, landmarks):
    h, w, _ = image.shape
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks[:LANDMARK_COUNT]]

    for (a, b) in UPPER_CONNECTIONS:
        if a < LANDMARK_COUNT and b < LANDMARK_COUNT:
            cv2.line(image, pts[a], pts[b], (200, 200, 200), 2, cv2.LINE_AA)

    for i, (cx, cy) in enumerate(pts):
        vis = landmarks[i].visibility
        alpha = max(0.3, min(1.0, vis))
        color = tuple(int(c * alpha) for c in (255, 230, 80))
        cv2.circle(image, (cx, cy), 6, color, -1, cv2.LINE_AA)
        cv2.circle(image, (cx, cy), 6, (80, 80, 80), 1, cv2.LINE_AA)

def draw_status_panel(image, pred_class: str, proba: np.ndarray, raw_class: str, smoothed_class: str, threshold, le):
    h, w = image.shape[:2]
    panel_w, panel_h = 320, 175
    margin = 16

    overlay = image.copy()
    cv2.rectangle(overlay, (margin, margin),
                (margin + panel_w, margin + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)

    color = get_color(pred_class)
    cv2.rectangle(image, (margin, margin),
                (margin + panel_w, margin + 44), color, -1)
    cv2.putText(image, f"STATUS: {pred_class}",
                (margin + 10, margin + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    bar_x      = margin + 10
    bar_y_start = margin + 58
    bar_max_w  = panel_w - 20
    bar_h      = 16

    for i, cls in enumerate(le.classes_):
        prob   = proba[i]
        bcolor = get_color(cls)
        y      = bar_y_start + i * 34

        cv2.putText(image, f"{cls}", (bar_x, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.rectangle(image, (bar_x + 75, y),
                    (bar_x + 75 + bar_max_w - 75, y + bar_h),
                    (60, 60, 60), -1)
        filled = int((bar_max_w - 75) * prob)
        cv2.rectangle(image, (bar_x + 75, y),
                    (bar_x + 75 + filled, y + bar_h),
                    bcolor, -1)
        cv2.putText(image, f"{prob * 100:.1f}%",
                    (bar_x + 75 + (bar_max_w - 75) + 4, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    footer_y = margin + panel_h - 10
    cv2.putText(image,
                f"raw: {raw_class}  smoothed: {smoothed_class}  thr: {threshold}",
                (margin + 6, footer_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA)

def draw_no_pose(image):
    cv2.putText(image, "No pose detected", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2, cv2.LINE_AA)


# Desktop Pet Companion Window with Dynamic Image States & Chat Bubbles
class DesktopPetWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint | 
            Qt.WindowStaysOnTopHint | 
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # 1. Chat Bubble Label
        self.bubble_label = QLabel(self)
        self.bubble_label.setAlignment(Qt.AlignCenter)
        self.bubble_label.setStyleSheet("""
            background-color: #ffffff;
            color: #dc2626;
            border: 3px solid #dc2626;
            border-radius: 12px;
            padding: 12px;
            font-weight: bold;
            font-family: sans-serif;
            font-size: 16px;
        """)
        self.bubble_label.hide()

        # 2. Shrimp Image Label
        self.image_label = QLabel(self)
        
        # Load all posture images dynamically
        self.pixmaps = {}
        image_files = {
            'Normal': 'assets/udang_normal.png',
            'Forward': 'assets/udang_forward.png',
            'Back': 'assets/udang_backward.png'
        }

        for state, filename in image_files.items():
            if os.path.exists(filename):
                base_px = QPixmap(filename)
                self.pixmaps[state] = base_px.scaledToWidth(180, Qt.SmoothTransformation)
            else:
                # Fallback handler if any specific state image is missing
                print(f"Warning: '{filename}' not found!")

        # Set default initial image (Normal) if available
        if 'Normal' in self.pixmaps:
            self.image_label.setPixmap(self.pixmaps['Normal'])
        else:
            self.image_label.setText("Images missing!")

        layout.addWidget(self.bubble_label)
        layout.addWidget(self.image_label)
        self.setLayout(layout)

        self.move(100, 100)
        self.old_pos = None

    def update_state(self, posture_class: str):
        # 1. Update the shrimp sprite image based on the detected class
        if posture_class in self.pixmaps:
            self.image_label.setPixmap(self.pixmaps[posture_class])

        # 2. Control warning chat bubble visibility and text
        if posture_class != "Normal":
            self.bubble_label.setText(f"⚠️ Fix Posture!\n({posture_class})")
            self.bubble_label.show()
        else:
            self.bubble_label.hide()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.old_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        if self.old_pos is not None:
            delta = event.globalPos() - self.old_pos
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.old_pos = event.globalPos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.old_pos = None


if __name__ == '__main__':
    app = QApplication(sys.argv)

    pet_window = DesktopPetWidget()
    pet_window.show()

    model_path = 'pose_landmarker_lite.task'
    if not os.path.exists(model_path):
        print("Mengambil model MediaPipe dari Google APIs...")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
        urllib.request.urlretrieve(url, model_path)
        print("Model berhasil diunduh!\n")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        num_poses=1
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    print("Memuat model XGBoost...")
    model_data      = joblib.load('posture_xgboost_v1.pkl')
    model           = model_data['model']
    le              = model_data['encoder']
    raw_features    = model_data['raw_features']
    normal_idx      = model_data['normal_idx']
    threshold       = model_data['normal_threshold']

    n_classes   = len(le.classes_)
    smoother    = TemporalSmoother(window=10)
    print(f"Classes: {list(le.classes_)}  |  Normal threshold: {threshold}")

    def predict_with_threshold(proba_1d: np.ndarray) -> int:
        if proba_1d[normal_idx] >= threshold:
            return normal_idx
        mask = np.ones(n_classes, dtype=bool)
        mask[normal_idx] = False
        return int(np.argmax(proba_1d * mask))

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Can't open camera")

    print("Press q to exit.")

    timer = QTimer()
    def process_frame():
        ret, frame = cap.read()
        if not ret:
            print("Failed reading frame...")
            return

        rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result     = detector.detect(mp_image)

        annotated  = frame.copy()

        if result.pose_landmarks:
            pose_landmarks = result.pose_landmarks[0]
            draw_skeleton(annotated, pose_landmarks)

            row = []
            for i in range(LANDMARK_COUNT):
                lm = pose_landmarks[i]
                row.extend([lm.x, lm.y, lm.z, lm.visibility])

            landmark_df = pd.DataFrame([row], columns=raw_features)
            X_live = build_features(landmark_df)

            proba       = model.predict_proba(X_live)[0]
            raw_pred    = predict_with_threshold(proba)
            smooth_pred = smoother.update(raw_pred)

            raw_class     = le.inverse_transform([raw_pred])[0]
            smooth_class  = le.inverse_transform([smooth_pred])[0]

            draw_status_panel(annotated, smooth_class, proba, raw_class, smooth_class, threshold, le)

            # Update the desktop pet image and chat bubble based on classification
            pet_window.update_state(smooth_class)
        else:
            smoother.reset()
            draw_no_pose(annotated)
            pet_window.update_state("Normal")

        cv2.imshow('PosturePal v2', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    timer.timeout.connect(process_frame)
    timer.start(30)

    sys.exit(app.exec_())