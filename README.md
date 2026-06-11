# 🛡️ Ứng dụng Phát hiện Giao dịch Gian lận (Fraud Detection Web App)

Ứng dụng web được chuyển đổi từ mô hình thử nghiệm học máy sang ứng dụng trực quan hóa thời gian thực bằng nền tảng framework **Streamlit**. Hệ thống hỗ trợ đắc lực cho các chuyên viên rủi ro ngân hàng và fintech thực hiện giám sát phân tích hành vi và chẩn đoán sớm rủi ro gian lận giao dịch.

---

## 🧭 Kiến trúc Mô hình Học máy sử dụng
* **Thuật toán chính:** `RandomForestClassifier` (Rừng cây quyết định ngẫu nhiên độc lập tối ưu phân loại).
* **Tiền xử lý dữ liệu:** Chuẩn hóa trung bình và phương sai bằng `StandardScaler`.
* **Kỹ thuật xử lý mất cân bằng lớp:** Sử dụng thuật toán cân bằng nâng cao **SMOTE** tự động tái cấu trúc phân phối lớp thiểu số (Nhãn 1 - Gian lận) tránh hiện tượng mô hình bị lệch thiên kiến dự báo.

---

## 🛠️ Hướng dẫn cài đặt và khởi chạy ứng dụng

### Bước 1: Khởi tạo và thiết lập môi trường máy ảo bảo mật (Khuyến nghị)
```bash
# Tạo môi trường ảo python
python -m venv venv

# Kích hoạt môi trường ảo (Hệ điều hành Windows)
venv\Scripts\activate

# Kích hoạt môi trường ảo (Hệ điều hành macOS/Linux)
source venv/bin/activate
