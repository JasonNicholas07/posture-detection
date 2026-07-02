
# Posture Classification - XGBoost

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import optuna
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Posture Classification Training")


# 1. define
DATASET_PATH        = 'dataset_postur_v2_clean.csv'
MODEL_OUTPUT_PATH   = 'posture_xgboost_v2.pkl'
N_OPTUNA_TRIALS     = 60
N_CV_FOLDS          = 5
TEST_SIZE           = 0.2
RANDOM_STATE        = 67
NORMAL_THRESHOLD = 0.55
BACK_THRESHOLD = 0.95


# 2. LOAD
print("\nLoading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"Rows: {len(df):}   |   Columns: {len(df.columns)}")
print(f"Class distribution:\n{df['class'].value_counts().to_string()}")


# 3. FEATURES
LANDMARK_COUNT = 13   # upper body landmarks 1–13

raw_features = []
for i in range(1, LANDMARK_COUNT + 1):
    raw_features.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])


def compute_angle(p1, p2, p3):
    # Angle at p2 formed by p1–p2–p3.
    # p1/p2/p3 are (x, y) or (x, y, z) arrays, shape (N, dims).
    # Returns angle in degrees, shape (N,).
    
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.einsum('ij,ij->i', v1, v2) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
    )
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Menggunakan DataFrame kosong agar semua koordinat X, Y, Z mentah
    # otomatis terbuang (Feature Selection)
    feat = pd.DataFrame(index=df.index)

    
    # 1. REFERENSI TITIK TENGAH (Pusat Tubuh)
    
    shoulder_mid_x = (df['x12'] + df['x13']) / 2
    shoulder_mid_y = (df['y12'] + df['y13']) / 2

    
    # 2. FITUR JARAK DAN POSISI (Sumbu X & Y)
    
    # Shoulder midpoint vs nose (forward/back lean indicator) 
    feat['nose_to_shoulder_mid_x'] = df['x1'] - shoulder_mid_x
    feat['nose_to_shoulder_mid_y'] = df['y1'] - shoulder_mid_y
    feat['nose_to_shoulder_mid_dist'] = np.sqrt(
        feat['nose_to_shoulder_mid_x']**2 + feat['nose_to_shoulder_mid_y']**2
    )

    # Shoulder width (lateral lean / slouch) 
    feat['shoulder_width'] = np.sqrt(
        (df['x12'] - df['x13'])**2 + (df['y12'] - df['y13'])**2
    )

    # Ear–shoulder alignment (head tilt/forward head) 
    feat['left_ear_shoulder_y_diff']  = df['y8']  - df['y12']
    feat['right_ear_shoulder_y_diff'] = df['y9']  - df['y13']
    feat['ear_shoulder_y_asymmetry']  = (
        feat['left_ear_shoulder_y_diff'] - feat['right_ear_shoulder_y_diff']
    )

    # Nose–ear horizontal offset (head forward lean) 
    feat['nose_left_ear_x_diff']  = df['x1'] - df['x8']
    feat['nose_right_ear_x_diff'] = df['x1'] - df['x9']

    # Eye level asymmetry (head tilt left/right) 
    feat['eye_level_diff'] = df['y3'] - df['y6']

    
    # 3. FITUR KEDALAMAN (Sumbu Z) & SUDUT RELATIF
    
    # Menggantikan nose_z mentah dengan jarak relatif Hidung terhadap Bahu
    feat['relative_nose_z'] = df['z1'] - ((df['z12'] + df['z13']) / 2)
    
    # Normalisasi agar model kebal terhadap jarak kamera
    feat['normalized_nose_z'] = feat['relative_nose_z'] / (feat['shoulder_width'] + 0.0001)
    
    # Menghitung sudut derajat leher (sangat kuat untuk mendeteksi Forward Head Posture)
    feat['neck_forward_angle'] = np.degrees(
        np.arctan2(
            np.abs(feat['relative_nose_z']), 
            np.abs(feat['nose_to_shoulder_mid_y']) + 0.0001
        )
    )

    
    # 4. STATISTIK VISIBILITAS (Mencegah Halusinasi Kamera)
    LANDMARK_COUNT = 13
    v_cols = [f'v{i}' for i in range(1, LANDMARK_COUNT + 1)]
    
    # Validasi apakah kolom 'v' ada (karena kamera live kadang mengabaikannya)
    if all(col in df.columns for col in v_cols):
        feat['mean_visibility'] = df[v_cols].mean(axis=1)
        feat['min_visibility']  = df[v_cols].min(axis=1)

    return feat


print("\nEngineering features...")
X_full = build_features(df)
print(f"Raw features:        {len(raw_features)}")
print(f"Engineered features: {X_full.shape[1] - len(raw_features)}")
print(f"Total features:      {X_full.shape[1]}")

feature_names = list(X_full.columns)


# 4. LABEL ENCODING + SPLIT
print("\nEncoding labels and splitting data...")
le = LabelEncoder()
y_encoded = le.fit_transform(df['class'])
print(f"    Classes: {list(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)
print(f"    Train: {len(X_train):,}   |   Test: {len(X_test):,}")

normal_idx = int(le.transform(['Normal'])[0])
back_idx = le.transform(['Back'])[0]
print(f"    'Normal' class index: {normal_idx}")
print(f"    'Back' class index: {back_idx}")


# 5. CLASS WEIGHTS
def make_sample_weights(y, normal_idx, normal_multiplier=1.0):
    classes  = np.unique(y)
    cw       = compute_class_weight('balanced', classes=classes, y=y)
    wdict    = dict(zip(classes, cw))
    wdict[normal_idx] *= normal_multiplier
    return np.array([wdict[c] for c in y])


# 6. OPTUNA — CV-BASED 
print(f"\nBayesian Optimization with Optuna ({N_OPTUNA_TRIALS} trials, {N_CV_FOLDS}-fold CV)...")

skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

def objective(trial):
    params = {
        'n_estimators':       trial.suggest_int('n_estimators', 100, 500),
        'learning_rate':      trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth':          trial.suggest_int('max_depth', 3, 10),
        'subsample':          trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':   trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma':              trial.suggest_float('gamma', 0.0, 5.0),
        'min_child_weight':   trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha':          trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':         trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'random_state':       RANDOM_STATE,
        'eval_metric':        'mlogloss',
        'use_label_encoder':  False,
    }
    # Tune the Normal weight multiplier
    normal_multiplier = trial.suggest_float('normal_multiplier', 0.8, 2.5)

    cv_scores = []
    for fold_train_idx, fold_val_idx in skf.split(X_train, y_train):
        X_fold_tr, X_fold_val = X_train.iloc[fold_train_idx], X_train.iloc[fold_val_idx]
        y_fold_tr, y_fold_val = y_train[fold_train_idx],     y_train[fold_val_idx]

        sw = make_sample_weights(y_fold_tr, normal_idx, normal_multiplier)

        model = XGBClassifier(**params)
        model.fit(X_fold_tr, y_fold_tr, sample_weight=sw)

        preds    = model.predict(X_fold_val)
        cv_scores.append(accuracy_score(y_fold_val, preds))

    return np.mean(cv_scores)


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

print("\n" + "=" * 50)
print("  Best Hyperparameters (Optuna CV)")
print("=" * 50)
for k, v in study.best_params.items():
    print(f"  {k:<25}: {v}")
print(f"\n  Best CV accuracy: {study.best_value * 100:.2f}%")
print("=" * 50)


# 7. Training(full train set)
print("\nFinal training on full train set...")
best = study.best_params.copy()
normal_multiplier_best = best.pop('normal_multiplier')

best['random_state']      = RANDOM_STATE
best['eval_metric']       = 'mlogloss'
best['use_label_encoder'] = False

sw_train_final = make_sample_weights(y_train, normal_idx, normal_multiplier_best)

final_model = XGBClassifier(**best)
final_model.fit(X_train, y_train, sample_weight=sw_train_final)
print(f"  Best normal_multiplier: {normal_multiplier_best:.3f}")


# 8. THRESHOLD-BASED PREDICTION FUNCTION
def predict_with_threshold(model, X, normal_idx, back_idx, normal_thresh=0.7, back_thresh=0.8):

    proba      = model.predict_proba(X)                   # (N, n_classes)
    n_classes  = proba.shape[1]
    predictions = []

    for p in proba:
        if p[back_idx] >= back_thresh:
            predictions.append(back_idx)
        elif p[normal_idx] >= normal_thresh:
            predictions.append(normal_idx)
        else:
            # Pick best class that is NEITHER Normal NOR Back
            mask = np.ones(n_classes, dtype=bool)
            mask[normal_idx] = False
            mask[back_idx] = False
            best_other = np.argmax(p * mask)
            predictions.append(int(best_other))

    return np.array(predictions)

# 9. EVALUATION
print("\nEvaluation on held-out test set...")

# Standard argmax prediction
y_pred_raw = final_model.predict(X_test)

# Threshold-based prediction
y_pred_thresh = predict_with_threshold(
    model=final_model, 
    X=X_test, 
    normal_idx=normal_idx, 
    back_idx=back_idx, 
    normal_thresh=NORMAL_THRESHOLD, 
    back_thresh=BACK_THRESHOLD
)

acc_raw    = accuracy_score(y_test, y_pred_raw)
acc_thresh = accuracy_score(y_test, y_pred_thresh)

print(f"\n  Accuracy (argmax):              {acc_raw    * 100:.2f}%")
print(f"  Accuracy (threshold={NORMAL_THRESHOLD}):    {acc_thresh * 100:.2f}%")

print("\n Classification Report (argmax)")
print(classification_report(y_test, y_pred_raw, target_names=le.classes_))

print(f"   Classification Report (threshold={NORMAL_THRESHOLD}) ")
print(classification_report(y_test, y_pred_thresh, target_names=le.classes_))

print("  Confusion Matrix (threshold):")
cm = confusion_matrix(y_test, y_pred_thresh)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=[f'pred_{c}' for c in le.classes_])
print(cm_df.to_string())


# 10. FEATURE IMPORTANCE
print("\nTop 15 Feature Importances:")
importances = final_model.feature_importances_
fi_df = pd.DataFrame({
    'feature':    feature_names,
    'importance': importances
}).sort_values('importance', ascending=False).head(15)
for _, row in fi_df.iterrows():
    bar = '█' * int(row['importance'] * 200)
    print(f"  {row['feature']:<35} {row['importance']:.4f}  {bar}")


# 11. TEMPORAL SMOOTHER 
class TemporalSmoother:

    # Smooths predictions over a rolling window using majority vote.
    # Prevents flickering between classes on single noisy frames.
    # Usage (real-time loop):
    #     smoother = TemporalSmoother(window=10)
    #     stable_class = smoother.update(raw_prediction)

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


# 12. SAVE
export_data = {
    'model':              final_model,
    'encoder':            le,
    'features':           feature_names,
    'raw_features':       raw_features,
    'normal_idx':         normal_idx,
    'normal_threshold':   NORMAL_THRESHOLD,
    'build_features_fn':  build_features,   # save the fn reference for inference
    'TemporalSmoother':   TemporalSmoother,
}
joblib.dump(export_data, MODEL_OUTPUT_PATH)
print(f"\nModel saved → '{MODEL_OUTPUT_PATH}'")