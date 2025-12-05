# web_new/app.py - BẢN CUỐI CÙNG HOÀN HẢO NHẤT (2025-12-04)
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify, Response
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.utils
import json
import os
import pickle
import subprocess
from datetime import datetime
from werkzeug.utils import secure_filename
from models import get_db

# === NHẬN DIỆN KHUÔN MẶT TRONG WEB ===
from insightface.app import FaceAnalysis
import cv2
import numpy as np

# Khởi động model InsightFace (chỉ 1 lần khi web chạy)
face_app = FaceAnalysis(name='buffalo_l', providers=['DirectMLExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load embeddings toàn cục
EMBEDDINGS_PATH = "../embeddings/embeddings.pkl"
known_embeddings = []
known_names = []
known_ids = []

def load_embeddings():
    global known_embeddings, known_names, known_ids
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            with open(EMBEDDINGS_PATH, 'rb') as f:
                data = pickle.load(f)
            known_embeddings = np.array(data['embeddings']) if data['embeddings'] else np.array([])
            known_names = data['names']
            known_ids = data['ids']
            print(f"Đã load {len(known_ids)} sinh viên từ embeddings")
        except Exception as e:
            print(f"Lỗi load embeddings: {e}")
            known_embeddings, known_names, known_ids = np.array([]), [], []

load_embeddings()  # Load ngay khi khởi động

app = Flask(__name__)
app.secret_key = 'smart-attendance-2025-key-fix'

# Cấu hình upload
UPLOAD_TEMP = 'static/uploads'
DATASET_PATH = '../dataset'
os.makedirs(UPLOAD_TEMP, exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id): self.id = id

@login_manager.user_loader
def load_user(user_id): return User(user_id)

# === HÀM HỖ TRỢ ===
def get_total_students():
    return len(known_ids)

# FIX LỖI DATABASE BỊ KHÓA (an toàn tuyệt đối)
def fix_db_if_broken(db_path="../database.db"):
    if os.path.exists(db_path):
        try:
            test_conn = sqlite3.connect(db_path, timeout=1)
            test_conn.execute("SELECT 1 FROM attendance LIMIT 1")
            test_conn.close()
        except:
            backup = f"../database_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            try:
                os.rename(db_path, backup)
                print(f"DB bị khóa → đổi tên thành {backup}")
            except:
                print("Không thể đổi tên DB")


# === TÍNH NĂNG MỚI: CAMERA TRỰC TIẾP TRÊN WEB – NHẬN DIỆN N KHUÔN MẶT CÙNG LÚC (2025) ===
def gen_camera_frames():
    # FIX: Thử nhiều backend để camera luôn mở được
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        success, frame = cap.read()
        if not success:
            cap.release()
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            continue

        faces = face_app.get(frame)
        recognized_list = []  # Danh sách người được nhận diện trong frame này

        for face in faces:
            bbox = face.bbox.astype(int)
            emb = face.embedding

            if len(known_embeddings) == 0 or emb is None:
                continue

            # Tính độ tương đồng với tất cả embeddings
            sims = np.dot(known_embeddings, emb) / (np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(emb))
            max_sim = np.max(sims) if len(sims) > 0 else 0

            if max_sim > 0.55:  # Ngưỡng nhận diện
                idx = np.argmax(sims)
                sid = known_ids[idx]
                name = known_names[idx]
                confidence = round(max_sim * 100, 1)

                # GHI ĐIỂM DANH (chỉ 1 lần/ngày)
                today = datetime.now().strftime("%Y-%m-%d")
                now = datetime.now().strftime("%H:%M:%S")
                conn = sqlite3.connect('../database.db')
                cur = conn.cursor()
                cur.execute("SELECT * FROM attendance WHERE student_id=? AND date=?", (sid, today))
                if cur.fetchone() is None:
                    cur.execute("INSERT INTO attendance (student_id, name, date, time_in) VALUES (?, ?, ?, ?)",
                                (sid, name, today, now))
                    conn.commit()
                    print(f"[WEBCAM] ĐIỂM DANH: {name} ({sid}) - {confidence}% - {now}")
                conn.close()

                # VẼ KHUNG + TÊN + % TỰ TIN (đẹp hơn)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 4)
                text = f"{name} ({sid}) {confidence}%"
                cv2.putText(frame, text, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

                # Thêm vào danh sách để gửi thông báo JS
                recognized_list.append(f"{name} ({sid})")

        # GỬI THÔNG BÁO CHO TẤT CẢ NGƯỜI ĐƯỢC NHẬN DIỆN TRONG FRAME
        if recognized_list:
            names_text = " | ".join(recognized_list)
            js_notify = f"<script>window.parent.postMessage({{'type':'success', 'names':'{names_text}'}}, '*');</script>"
            yield (b'--frame\r\nContent-Type: text/html\r\n\r\n' + js_notify.encode() + b'\r\n')

        # Gửi frame ảnh
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
@app.route('/camera')
@login_required
def camera_page():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# === CÁC ROUTE CŨ CỦA BẠN (GIỮ NGUYÊN 100%) ===
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
        conn.close()
        if user and user['password'] == password:
            login_user(User(user['id']))
            return redirect(url_for('dashboard'))
        flash('Sai tên đăng nhập hoặc mật khẩu!', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def dashboard():
    fix_db_if_broken()
    conn = get_db()
    dates = [r['date'] for r in conn.execute("SELECT DISTINCT date FROM attendance ORDER BY date DESC").fetchall()]
    today = datetime.now().strftime("%Y-%m-%d")
    present_today = conn.execute("SELECT COUNT(*) FROM attendance WHERE date=?", (today,)).fetchone()[0] or 0
    total_students = get_total_students()
    conn.close()
    return render_template('dashboard.html',
                           dates=dates,
                           today=today,
                           present_today=present_today,
                           total=total_students)

@app.route('/realtime')
@login_required
def realtime():
    return render_template('realtime.html')

@app.route('/api/realtime')
def api_realtime():
    today = datetime.now().strftime("%Y-%m-%d")
    conn = get_db()
    rows = conn.execute("SELECT student_id, name, time_in FROM attendance WHERE date=? ORDER BY time_in DESC", (today,)).fetchall()
    conn.close()
    return jsonify([dict(row) for row in rows])

@app.route('/attendance/<date>')
@login_required
def attendance_date(date):
    fix_db_if_broken()
    conn = get_db()
    df = pd.read_sql_query("SELECT student_id AS 'Mã SV', name AS 'Họ tên', time_in AS 'Giờ vào', time_out AS 'Giờ ra' FROM attendance WHERE date=? ORDER BY time_in", conn, params=(date,))
    conn.close()
    table_html = df.to_html(classes='table table-striped', index=False, escape=False)
    return render_template('attendance.html', table=table_html, date=date)

@app.route('/stats')
@login_required
def stats():
    fix_db_if_broken()
    conn = get_db()
    df = pd.read_sql_query("SELECT student_id, name, date FROM attendance", conn)
    conn.close()
    if df.empty:
        flash('Chưa có dữ liệu điểm danh để thống kê!', 'warning')
        return redirect(url_for('dashboard'))
    stats_df = df.groupby(['student_id', 'name']).size().reset_index(name='Số buổi có mặt').sort_values('Số buổi có mặt', ascending=False)
    fig = px.bar(stats_df.head(15), x='name', y='Số buổi có mặt', color='Số buổi có mặt',
                 title='Top 15 sinh viên chuyên cần nhất', color_continuous_scale='Viridis')
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    table_html = stats_df.to_html(classes='table table-striped', index=False, escape=False)
    return render_template('stats.html', table=table_html, graph_json=graph_json)

@app.route('/export/<date>')
@login_required
def export(date):
    fix_db_if_broken()
    conn = get_db()
    df = pd.read_sql_query("SELECT student_id AS 'Mã SV', name AS 'Họ tên', time_in AS 'Giờ vào', time_out AS 'Giờ ra' FROM attendance WHERE date=?", conn, params=(date,))
    conn.close()
    filename = f"DiemDanh_{date}.xlsx"
    df.to_excel(filename, index=False)
    return send_file(filename, as_attachment=True, download_name=filename)

# === QUẢN LÝ SINH VIÊN & TRAIN - ĐÃ CÓ TÌM KIẾM ===
@app.route('/manage')
@login_required
def manage_students():
    query = request.args.get('q', '').strip().lower()
    students = []

    if os.path.exists(DATASET_PATH):
        for folder in os.listdir(DATASET_PATH):
            folder_path = os.path.join(DATASET_PATH, folder)
            if not os.path.isdir(folder_path):
                continue
            parts = folder.split("_", 1)
            if len(parts) != 2:
                continue
            mssv, name_raw = parts
            name = name_raw.replace("_", " ")

            # TÌM KIẾM THEO MSSV HOẶC TÊN
            if query and query not in mssv.lower() and query not in name.lower():
                continue

            photos = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if photos:
                first_photo = f"/static/dataset/{folder}/{photos[0]}"
                students.append({
                    'mssv': mssv,
                    'name': name,
                    'photo': first_photo,
                    'photo_count': len(photos)
                })

    students.sort(key=lambda x: x['mssv'])
    return render_template('manage_students.html', students=students, total=len(students))

@app.route('/add_student', methods=['GET', 'POST'])
@login_required
def add_student():
    if request.method == 'POST':
        mssv = request.form['mssv'].strip()
        name = request.form['name'].strip().replace(" ", "_")
        photos = request.files.getlist('photos')
        if not mssv or not name or not photos:
            flash('Vui lòng điền đầy đủ!', 'danger')
            return redirect('/add_student')
        folder_name = f"{mssv}_{name}"
        student_path = os.path.join(DATASET_PATH, folder_name)
        os.makedirs(student_path, exist_ok=True)
        saved = 0
        for photo in photos:
            if photo and photo.filename:
                filename = secure_filename(photo.filename)
                photo.save(os.path.join(student_path, filename))
                saved += 1
        flash(f'Đã thêm: {name.replace("_", " ")} ({mssv}) – {saved} ảnh', 'success')
        return redirect('/manage')
    return render_template('add_student.html')


@app.route('/retrain', methods=['POST'])
@login_required
def retrain():
    flash('Đang khởi động huấn luyện AI... (có thể mất 10-60 giây)', 'info')

    try:
        # ĐƯỜNG DẪN TUYỆT ĐỐI → KHÔNG BAO GIỜ LỖI TRÊN WINDOWS!
        train_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train_embeddings.py'))

        if not os.path.exists(train_script):
            flash('KHÔNG TÌM THẤY FILE train_embeddings.py ở thư mục gốc!', 'danger')
            return redirect('/manage')

        print(f"Đang chạy file: {train_script}")

        result = subprocess.run(
            ['python', train_script],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.dirname(train_script)  # Chạy đúng từ thư mục gốc
        )

        # TẢI LẠI EMBEDDINGS
        load_embeddings()

        if result.returncode == 0:
            flash('HUẤN LUYỆN AI THÀNH CÔNG! Hệ thống đã cập nhật toàn bộ sinh viên mới!', 'success')
        else:
            error = result.stderr.strip() or result.stdout.strip() or "Lỗi không xác định"
            flash(f'Thuật toán lỗi: {error[:200]}', 'danger')
            print(f"TRAIN LỖI: {error}")

    except FileNotFoundError:
        flash('Không tìm thấy Python hoặc file train_embeddings.py!', 'danger')
    except subprocess.TimeoutExpired:
        flash('Train quá lâu (timeout)! Vui lòng kiểm tra dữ liệu ảnh.', 'danger')
    except Exception as e:
        flash(f'Lỗi hệ thống: {str(e)}', 'danger')

    return redirect('/manage')

# === CHẠY WEB ===
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)