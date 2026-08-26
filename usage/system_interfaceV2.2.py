import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
import numpy as np
import joblib
import urllib.request
import os
import customtkinter as ctk
from PIL import Image

# ==========================================
# 1. CORE LOGIC (From your provided code)
# ==========================================

class TemporalSmoother:
    def __init__(self, window: int = 10):
        self.window  = window
        self.history = []

    def update(self, prediction: int) -> int:
        self.history.append(prediction)
        if len(self.history) > self.window:
            self.history.pop(0
            )
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

# --- Load Models ---
model_path = 'pose_landmarker_lite.task'
if not os.path.exists(model_path):
    print("Mengambil model MediaPipe dari Google APIs...")
    url = ("https://storage.googleapis.com/mediapipe-models/"
"pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
    urllib.request.urlretrieve(url, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    num_poses=1
)
detector = vision.PoseLandmarker.create_from_options(options)

model_data      = joblib.load('posture_xgboost_v1.pkl')
model           = model_data['model']
le              = model_data['encoder']
raw_features    = model_data['raw_features']
normal_idx      = model_data['normal_idx']
threshold       = model_data['normal_threshold']
n_classes       = len(le.classes_)

def predict_with_threshold(proba_1d: np.ndarray) -> int:
    if proba_1d[normal_idx] >= threshold:
        return normal_idx
    mask = np.ones(n_classes, dtype=bool)
    mask[normal_idx] = False
    return int(np.argmax(proba_1d * mask))

# ==========================================
# 2. CUSTOMTKINTER UI SETUP
# ==========================================

# Theme and colors
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

CLASS_COLORS = {
    'Normal':  "#22c55e", # Green
    'Forward': "#ef4444", # Red
    'Back':    "#eab308", # Yellow
}

LANDMARK_COUNT = 13
UPPER_CONNECTIONS = [
    (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),
    (7, 8), (9, 10), (11, 12),
]

class PostureApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("PosturePal v2")
        self.geometry("1050x600")
        self.minsize(900, 500)
        
        self.smoother = TemporalSmoother(window=10)
        
        # --- Layout Grid ---
        self.grid_columnconfigure(0, weight=1) # Video gets extra space
        self.grid_columnconfigure(1, weight=0) # Sidebar is fixed width
        self.grid_rowconfigure(0, weight=1)
        
        # --- Left Column: Video Feed ---
        self.video_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # --- Right Column: Dashboard Sidebar ---
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=15)
        self.sidebar.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        self.sidebar.grid_propagate(False) # Keep width fixed
        
        # Title
        self.title_label = ctk.CTkLabel(self.sidebar, text="Posture Dashboard", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=(25, 20), padx=20, anchor="w")
        
        # Status Card
        self.status_card = ctk.CTkFrame(self.sidebar, fg_color="#1e1e1e", corner_radius=10)
        self.status_card.pack(fill="x", padx=20, pady=(0, 20))
    
        
        self.status_header = ctk.CTkLabel(self.status_card, text="CURRENT POSTURE", text_color="gray", font=ctk.CTkFont(size=12))
        self.status_header.pack(pady=(15, 0))
        
        self.status_value = ctk.CTkLabel(self.status_card, text="Waiting...", font=ctk.CTkFont(size=32, weight="bold"))
        self.status_value.pack(pady=(0, 15))

        # Probabilities Card
        self.prob_card = ctk.CTkFrame(self.sidebar, fg_color="#1e1e1e", corner_radius=10)
        self.prob_card.pack(fill="x", padx=20, pady=(0, 20))
        
        self.prob_header = ctk.CTkLabel(self.prob_card, text="PROBABILITIES", text_color="gray", font=ctk.CTkFont(size=12))
        self.prob_header.pack(pady=(15, 10), anchor="w", padx=15)
        
        # Dynamic Progress Bars based on classes
        self.progress_bars = {}
        self.prob_labels = {}
        for cls in le.classes_:
            row_frame = ctk.CTkFrame(self.prob_card, fg_color="transparent")
            row_frame.pack(fill="x", padx=15, pady=5)
            
            lbl = ctk.CTkLabel(row_frame, text=f"{cls}", width=60, anchor="w")
            lbl.pack(side="left")
            
            pb = ctk.CTkProgressBar(row_frame, progress_color=CLASS_COLORS.get(cls, "gray"), height=10)
            pb.set(0)
            pb.pack(side="left", fill="x", expand=True, padx=(10, 10))
            
            pct_lbl = ctk.CTkLabel(row_frame, text="0%", width=35, anchor="e")
            pct_lbl.pack(side="right")
            
            self.progress_bars[cls] = pb
            self.prob_labels[cls] = pct_lbl
            
        ctk.CTkFrame(self.prob_card, height=10, fg_color="transparent").pack() # Bottom padding
        
        # Debug / Raw info
        self.debug_label = ctk.CTkLabel(self.sidebar, text="Raw: -- | Smoothed: --", text_color="gray", font=ctk.CTkFont(size=11))
        self.debug_label.pack(side="bottom", pady=20)

        # --- Initialize Camera ---
        self.cap = cv2.VideoCapture(0)
        self.update_camera()

    def draw_skeleton_on_cv2(self, image, landmarks):
        """Draws aesthetic skeletal lines on the raw OpenCV frame before UI conversion"""
        h, w, _ = image.shape
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks[:LANDMARK_COUNT]]

        # Draw lines
        for (a, b) in UPPER_CONNECTIONS:
            if a < LANDMARK_COUNT and b < LANDMARK_COUNT:
                cv2.line(image, pts[a], pts[b], (150, 150, 150), 2, cv2.LINE_AA)

        # Draw points
        for cx, cy in pts:
            cv2.circle(image, (cx, cy), 5, (255, 200, 50), -1, cv2.LINE_AA)

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            # 1. Process Frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = detector.detect(mp_image)

            if result.pose_landmarks:
                pose_landmarks = result.pose_landmarks[0]
                
                # Draw skeleton directly on the video feed
                self.draw_skeleton_on_cv2(rgb_frame, pose_landmarks)

                # Build features
                row = []
                for i in range(LANDMARK_COUNT):
                    lm = pose_landmarks[i]
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])

                landmark_df = pd.DataFrame([row], columns=raw_features)
                X_live = build_features(landmark_df)

                # Predict
                proba = model.predict_proba(X_live)[0]
                raw_pred = predict_with_threshold(proba)
                smooth_pred = self.smoother.update(raw_pred)

                raw_class = le.inverse_transform([raw_pred])[0]
                smooth_class = le.inverse_transform([smooth_pred])[0]

                # 2. Update UI Dashboard
                self.status_value.configure(
                    text=smooth_class.upper(), 
                    text_color=CLASS_COLORS.get(smooth_class, "white")
                )
                self.debug_label.configure(text=f"Raw: {raw_class} | Smoothed: {smooth_class}")

                # Update Progress Bars
                for i, cls in enumerate(le.classes_):
                    p = proba[i]
                    self.progress_bars[cls].set(p)
                    self.prob_labels[cls].configure(text=f"{int(p*100)}%")

            else:
                self.smoother.reset()
                self.status_value.configure(text="NO POSE", text_color="gray")
                for cls in le.classes_:
                    self.progress_bars[cls].set(0)
                    self.prob_labels[cls].configure(text="0%")

            # 3. Render Video to CustomTkinter
            # Calculate dynamic size to fit the window while maintaining aspect ratio
            frame_h, frame_w = rgb_frame.shape[:2]
            label_w = self.video_frame.winfo_width()
            label_h = self.video_frame.winfo_height()
            
            if label_w > 10 and label_h > 10: # Ensure window is initialized
                scale = min(label_w/frame_w, label_h/frame_h)
                new_w, new_h = int(frame_w * scale), int(frame_h * scale)
                
                pil_image = Image.fromarray(rgb_frame)
                # CTkImage handles HighDPI scaling much better than PhotoImage
                ctk_img = ctk.CTkImage(light_image=pil_image, size=(new_w, new_h))
                self.video_label.configure(image=ctk_img)
                self.video_label.image = ctk_img

        # Loop at ~60fps
        self.after(16, self.update_camera)

    def on_closing(self):
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = PostureApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing) # Clean up camera on exit
    app.mainloop()