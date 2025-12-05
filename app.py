# app.py - ĐIỂM DANH REALTIME TỪ CAMERA – NHẬN DIỆN N KHUÔN MẶT CÙNG LÚC (2025)
import cv2
import numpy as np
import pickle
import sqlite3
import os
from datetime import datetime
from insightface.app import FaceAnalysis

# === FIX DATABASE BỊ KHÓA ===
def fix_db_if_broken(db_path="database.db"):
    if os.path.exists(db_path):
        try:
            test_conn = sqlite3.connect(db_path, timeout=1)
            test_conn.execute("SELECT 1 FROM attendance LIMIT 1")
            test_conn.close()
        except:
            backup = f"database_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            try:
                os.rename(db_path, backup)
                print(f"DB bị khóa → đổi tên thành {backup}")
            except:
                print("Không thể đổi tên DB")

fix_db_if_broken()

# Load model InsightFace (GPU onboard)
app_face = FaceAnalysis(name='buffalo_l', providers=['DirectMLExecutionProvider', 'CPUExecutionProvider'])
app_face.prepare(ctx_id=0, det_size=(640, 640))

# Load embeddings
with open('embeddings/embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
known_embeddings = np.array(data['embeddings']) if data['embeddings'] else np.array([])
known_names = data['names']
known_ids = data['ids']

print(f"Đã load {len(known_ids)} sinh viên từ embeddings")

# Khởi tạo DB
conn = sqlite3.connect('database.db')
conn.execute('''CREATE TABLE IF NOT EXISTS attendance
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              student_id TEXT, name TEXT, date TEXT, time_in TEXT, time_out TEXT)''')
conn.commit()
conn.close()

def mark_attendance(student_id, name):
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")
    conn = sqlite3.connect('database.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM attendance WHERE student_id=? AND date=?", (student_id, today))
    if cur.fetchone() is None:
        cur.execute("INSERT INTO attendance (student_id, name, date, time_in) VALUES (?, ?, ?, ?)",
                    (student_id, name, today, now))
        conn.commit()
        print(f"[{now}] ĐIỂM DANH: {name} ({student_id})")
    conn.close()

# Mở camera với nhiều backend để đảm bảo hoạt động
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("Hệ thống điểm danh đang chạy... Nhấn Q để thoát")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi camera – đang thử mở lại...")
        cap.release()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        continue

    faces = app_face.get(frame)
    recognized_this_frame = False

    for face in faces:
        bbox = face.bbox.astype(int)
        emb = face.embedding

        if len(known_embeddings) == 0 or emb is None:
            continue

        # Tính độ tương đồng
        sims = np.dot(known_embeddings, emb) / (np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(emb))
        max_sim = np.max(sims) if len(sims) > 0 else 0

        if max_sim > 0.55:
            idx = np.argmax(sims)
            sid = known_ids[idx]
            name = known_names[idx]
            confidence = round(max_sim * 100, 1)

            # GHI ĐIỂM DANH
            mark_attendance(sid, name)

            # VẼ KHUNG + TÊN + % TỰ TIN (ĐẸP HƠN)
            text = f"{name} ({sid}) {confidence}%"
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 4)
            cv2.putText(frame, text, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            recognized_this_frame = True

    # Hiển thị thông báo trên frame nếu có người được nhận diện
    if recognized_this_frame:
        cv2.putText(frame, "DANG NHAN DIEN & DIEM DANH...", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    cv2.imshow("DIEM DANH KHUON MAT - NHAN DIEN N NGUOI CUNG LUC", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Đã thoát hệ thống điểm danh.")