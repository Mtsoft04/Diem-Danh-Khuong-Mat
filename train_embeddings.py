import os
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from datetime import datetime

# === KHỞI TẠO MODEL INSIGHTFACE (TỐI ƯU GPU ONBOARD) ===
print("Đang khởi động model InsightFace Buffalo_l (có thể mất 5-15 giây lần đầu)...")
app = FaceAnalysis(
    name='buffalo_l',
    providers=['DirectMLExecutionProvider', 'CPUExecutionProvider']
)
app.prepare(ctx_id=0, det_size=(640, 640))
print("Model đã sẵn sàng!\n")

# === CẤU HÌNH ===
dataset_path = "dataset"
output_path = "embeddings/embeddings.pkl"
os.makedirs("embeddings", exist_ok=True)

known_embeddings = []
known_names = []
known_ids = []

print(f"Đang quét thư mục: {os.path.abspath(dataset_path)}")
print("-" * 70)

if not os.path.exists(dataset_path):
    print(f"LỖI: Không tìm thấy thư mục '{dataset_path}'!")
    print("   Vui lòng tạo thư mục 'dataset' và đặt ảnh theo định dạng:")
    print("   dataset/MSSV_TenSinhVien/anh1.jpg")
    exit()

total_folders = 0
success_count = 0
fail_count = 0

for student_folder in os.listdir(dataset_path):
    student_path = os.path.join(dataset_path, student_folder)
    if not os.path.isdir(student_path):
        continue

    total_folders += 1
    parts = student_folder.split("_", 1)
    if len(parts) < 2:
        print(f"SKIP: Tên thư mục sai định dạng → {student_folder}")
        fail_count += 1
        continue

    student_id = parts[0].strip()
    student_name = parts[1].replace("_", " ").strip()

    # Tìm ảnh trong thư mục
    img_files = [f for f in os.listdir(student_path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

    if not img_files:
        print(f"KHÔNG CÓ ẢNH: {student_folder} → bỏ qua")
        fail_count += 1
        continue

    embedded = False
    for img_file in img_files:
        img_path = os.path.join(student_path, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"   Không đọc được ảnh: {img_file}")
            continue

        faces = app.get(img)
        if len(faces) == 0:
            continue  # Thử ảnh tiếp theo
        elif len(faces) > 1:
            print(f"   Có {len(faces)} khuôn mặt → dùng khuôn mặt đầu tiên: {img_file}")

        # Lấy embedding từ khuôn mặt đầu tiên
        embedding = faces[0].embedding
        known_embeddings.append(embedding)
        known_names.append(student_name)
        known_ids.append(student_id)

        print(f"THÀNH CÔNG → {student_id} - {student_name} ({len(img_files)} ảnh)")
        success_count += 1
        embedded = True
        break  # Chỉ cần 1 ảnh tốt là đủ

    if not embedded:
        print(f"THẤT BẠI → Không tìm thấy khuôn mặt hợp lệ: {student_folder}")
        fail_count += 1

print("-" * 70)
print(f"HOÀN TẤT HUẤN LUYỆN!")
print(f"   • Thành công: {success_count} sinh viên")
print(f"   • Thất bại: {fail_count} sinh viên")
print(f"   • Tổng cộng: {len(known_embeddings)} embeddings được tạo")

# === LƯU FILE ===
data = {
    "embeddings": known_embeddings,
    "names": known_names,
    "ids": known_ids,
    "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_students": len(known_ids)
}

with open(output_path, 'wb') as f:
    pickle.dump(data, f)

print(f"\nĐã lưu vào: {os.path.abspath(output_path)}")
print("Bây giờ bạn có thể chạy camera hoặc web để điểm danh!")