# Ứng Dụng Điểm Danh Bằng Khuôn Mặt (Face Attendance System) 📸
# 1. Giới Thiệu Chung
Face Attendance System là một ứng dụng điểm danh tự động dựa trên công nghệ Nhận dạng Khuôn mặt (Face Recognition). Dự án được thiết kế và phát triển bởi MTSoft.
Ứng dụng sử dụng các thuật toán Xử lý ảnh và Thị giác máy tính để nhận diện và xác minh danh tính người dùng thông qua camera, đảm bảo quá trình điểm danh nhanh chóng và chính xác.
👉 Dự án này là mã nguồn mở (Open Source), được chia sẻ rộng rãi nhằm mục đích học tập, nghiên cứu và tham khảo. Chúng tôi hoan nghênh mọi đóng góp để cải thiện ứng dụng!
________________________________________
# 2. Các Công Nghệ Chính
Công Nghệ	Mô Tả
Python	Ngôn ngữ lập trình chính.
OpenCV	Thư viện hàng đầu cho Xử lý ảnh và Thị giác máy tính.
face_recognition	Thư viện mạnh mẽ dựa trên dlib để nhận dạng khuôn mặt.
SQLite	Hệ quản trị cơ sở dữ liệu để lưu trữ thông tin người dùng và dữ liệu điểm danh.
Web Framework	(Cần điền vào nếu có, ví dụ: Flask/Django) để xây dựng giao diện web.
________________________________________
# 3. Lý Thuyết Chuyên Môn
# 3.1. Xử Lý Ảnh (Image Processing)
Xử lý ảnh là thao tác biến đổi ảnh đầu vào để tạo ra ảnh đã được cải tiến hoặc trích xuất thông tin.
•	Mục tiêu: Tiền xử lý dữ liệu ảnh cho Thị giác Máy tính (ví dụ: chuyển ảnh màu sang ảnh xám Grayscale, giảm nhiễu, tăng cường độ tương phản).
•	Kỹ thuật chính: Lọc không gian (Spatial Filtering), Biến đổi màu.
# 3.2. Thị Giác Máy Tính (Computer Vision - CV)
Thị giác máy tính là lĩnh vực cho phép máy tính "hiểu" và "diễn giải" nội dung của hình ảnh và video.
•	Ứng dụng trong dự án:
o	Phát hiện Khuôn mặt (Face Detection): Khoanh vùng vị trí khuôn mặt trong khung hình.
o	Nhận dạng Khuôn mặt (Face Recognition): So sánh đặc trưng khuôn mặt được phát hiện với dữ liệu đã lưu trữ để xác định danh tính.
# 3.3. Cơ chế Nhận dạng (Face Recognition Pipeline)
Quá trình nhận dạng thường bao gồm:
1.	Phát hiện và Căn chỉnh: Tìm vị trí khuôn mặt và chuẩn hóa góc nhìn, kích thước.
2.	Mã hóa Đặc trưng (Encoding): Chuyển khuôn mặt thành một vector số học (thường là 128 chiều), gọi là Face Embeddings, đại diện cho đặc điểm sinh học.
3.	So sánh: Tính khoảng cách (ví dụ: Euclidean Distance) giữa embedding mới và các embeddings trong database để tìm ra người khớp nhất.
________________________________________
# 4. Hướng Dẫn Cài Đặt 🛠️
# 4.1. Cài Đặt Cơ Sở Dữ Liệu SQLite
•	Dự án sử dụng SQLite, một CSDL không cần cài đặt máy chủ riêng.
•	Hành động: Đảm bảo file database (ví dụ: database.db) nằm đúng trong thư mục gốc của dự án.
# 4.2. Cài Đặt Thư Viện Python
Sử dụng PyCharm (hoặc terminal) để tạo môi trường ảo và cài đặt các thư viện cần thiết cho Xử lý ảnh và Thị giác máy tính.
Chạy lệnh sau:
Bash
pip install opencv-python
pip install face-recognition
pip install numpy
pip install pandas
pip install Flask 
# Thêm các thư viện khác nếu cần (ví dụ: dlib, Pillow)
Thư Viện	Mục Đích Chuyên Môn
opencv-python	Xử lý ảnh, video và tiền xử lý dữ liệu.
face-recognition	Thực hiện nhận dạng khuôn mặt dựa trên thuật toán Dlib.
numpy	Xử lý mảng và ma trận, nền tảng cho các phép toán Xử lý ảnh.
Flask	(Nếu dùng) Xây dựng ứng dụng web cho giao diện.
# 4.3. Chạy Ứng Dụng
Sau khi cài đặt môi trường hoàn tất:
1.	Điều hướng đến folder Web_new.
2.	Chạy file chính của ứng dụng bằng lệnh:
Bash
python Web_new/app.py
3.	Mở trình duyệt web và truy cập vào địa chỉ hiển thị trên terminal (thường là http://127.0.0.1:5000/) để bắt đầu điểm danh.
________________________________________
# 5. Đóng Góp
Chúng tôi khuyến khích các nhà phát triển đóng góp vào dự án mã nguồn mở này. Vui lòng tạo Pull Request hoặc gửi Issue nếu bạn phát hiện lỗi hoặc có đề xuất tính năng mới.

