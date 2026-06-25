# Posture AI - Real-Time Posture Tracker

Aplikasi pelacak postur tubuh cerdas berbasis *Computer Vision* dan *Machine Learning*. Aplikasi ini secara *real-time* menganalisis tangkapan *webcam* untuk mendeteksi apakah postur pengguna saat duduk di depan komputer berada dalam kondisi tegak (Normal), membungkuk ke depan (Forward), atau bersandar (Back).

Proyek ini dibangun menggunakan **Google MediaPipe** untuk ekstraksi titik tubuh *(pose estimation)* dan **XGBoost** sebagai otak *Machine Learning* yang telah dioptimasi menggunakan **Bayesian Optimization (Optuna)**.

---

## ✨ Fitur Utama

- **👁️ Real-Time Inference:** Memproses tangkapan *webcam* tanpa *lag* menggunakan arsitektur MediaPipe Tasks API.
- **🧠 Advanced Feature Engineering:** Model tidak hanya menelan data mentah, tetapi menggunakan ekstraksi anatomi spesifik seperti:
  - *Nose-to-Shoulder Distance* (Mendeteksi *Forward Head Posture* / Leher Kura-kura).
  - *Ear-Shoulder Asymmetry* (Mendeteksi kemiringan kepala).
  - *Shoulder Width* (Mendeteksi postur membungkuk/menyusut).
- **🎯 Smart Thresholding:** Menggunakan ambang batas kustom (Normal: 70%, Back: 80%) untuk mencegah *flickering* dan memastikan AI sangat yakin sebelum mengubah status postur.
- **⏱️ Temporal Smoothing & Timer:** Mencegah perubahan status yang terlalu cepat (berkedip) dan memberikan peringatan visual jika postur pengguna buruk selama lebih dari 10 detik.
- **🔄 Active Learning Feedback:** Pengguna dapat menekan tombol `f` saat aplikasi berjalan untuk memberikan koreksi langsung kepada AI. Data ini akan disimpan di `dataset_feedback.csv` untuk melatih ulang model agar semakin pintar beradaptasi dengan kebiasaan unik pengguna.

---

## 🛠️ Tech Stack

- **Python 3.x**
- **Computer Vision:** OpenCV, MediaPipe (Google APIs)
- **Machine Learning:** XGBoost, Scikit-Learn
- **Hyperparameter Tuning:** Optuna (Bayesian Optimization)
- **Data Analysis & Preprocessing:** Pandas, NumPy, Seaborn, Matplotlib

---

## 📂 Struktur Proyek

- `system_interface.py` : Skrip utama untuk menjalankan aplikasi *webcam live tracking*.
- `train_final.py` : Skrip pelatihan model XGBoost menggunakan dataset bersih dan Optuna.
- `utils.py` : Modul bantuan yang berisi fungsi kustom (seperti `TemporalSmoother` dan `build_features`).
- `eda_advanced.py` : Skrip untuk Exploratory Data Analysis (Heatmap, Scatter 2D, Feature Importance).
- `preprocess_outliers.py` : Skrip pembersihan data otomatis menggunakan metode IQR (Interquartile Range) berbasis kelas.
- `posture_xgboost_production.pkl` : File *pickle* berisi model terlatih dan *Label Encoder*. (Pastikan ini sesuai ukuran file di GitHub).

---

## 🚀 Cara Instalasi dan Penggunaan

1. **Clone repository ini**
   ```bash
   git clone [https://github.com/UsernameKamu/posture-pal-ai.git](https://github.com/UsernameKamu/posture-pal-ai.git)
   cd posture-pal-ai
