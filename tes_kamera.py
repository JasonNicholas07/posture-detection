import cv2

print("Memulai tes kamera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("GAGAL: Kamera tidak terdeteksi sejak awal.")
else:
    print("Kamera terdeteksi! Membaca frame video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("GAGAL: Tidak bisa menangkap gambar dari kamera.")
        break
        
    cv2.imshow("Tes Kamera Murni", frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Tes selesai.")