import streamlit as tf
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import io

# ==============================================================================
# CẤU HÌNH TRANG ĐẦU TIÊN
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Gian lận Giao dịch",
    page_icon="🛡️"
)

# ==============================================================================
# CÁC HÀM CACHE DÙNG CHUNG
# ==============================================================================
@st.cache_data
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ bytes để đảm bảo tính hashable cho cache streamlit"""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        return None

# ==============================================================================
# THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải dữ liệu huấn luyện
    uploaded_file = st.file_uploader(
        "Tải lên dữ liệu huấn luyện (CSV/XLSX)", 
        type=["csv", "xlsx"],
        help="Chọn tệp dữ liệu có cấu trúc tương tự dataset1.csv (Chứa các biến X_1 đến X_14 và cột mục tiêu 'default')"
    )
    
    st.divider()
    
    # 2. Tham số mô hình
    st.subheader("Tham số mô hình AI")
    st.caption("Thuật toán: RandomForestClassifier")
    
    n_estimators = st.slider(
        "Số lượng cây (n_estimators)", 
        min_value=10, 
        max_value=300, 
        value=100, 
        step=10,
        help="Số lượng cây quyết định trong rừng độc lập."
    )
    
    max_depth = st.slider(
        "Độ sâu tối đa (max_depth)", 
        min_value=2, 
        max_value=50, 
        value=15, 
        step=1,
        help="Độ sâu tối đa của mỗi cây quyết định. Giúp kiểm soát quá khớp (overfitting)."
    )
    
    random_state = st.number_input(
        "Trạng thái ngẫu nhiên (random_state)", 
        value=42, 
        step=1,
        help="Đảm bảo kết quả huấn luyện có thể tái lập giống nhau giữa các lần chạy."
    )
    
    # Cấu hình SMOTE & Test Size nâng cao trong Expander
    with st.expander("⚙️ Cấu hình nâng cao"):
        use_smote = st.checkbox(
            "Áp dụng SMOTE xử lý mất cân bằng", 
            value=True,
            help="Tự động sinh mẫu nhân tạo cho nhóm thiểu số (gian lận) để cải thiện độ chính xác."
        )
        test_size = st.slider(
            "Tỷ lệ tập kiểm tra (Test size)", 
            min_value=0.1, 
            max_value=0.5, 
            value=0.3, 
            step=0.05,
            help="Tỷ lệ phân chia dữ liệu cho việc đánh giá mô hình."
        )

    st.divider()
    
    # 3. Nút kích hoạt hành động hành trình huấn luyện
    btn_train = st.button(
        "🚀 Huấn luyện mô hình", 
        type="primary", 
        use_container_width=True,
        help="Bấm để bắt đầu phân tách dữ liệu, tiền xử lý và xây dựng mô hình AI."
    )

# ==============================================================================
# THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG
# ==============================================================================
st.title("🛡️ Hệ thống Phát hiện Giao dịch Gian lận")
st.caption("Ứng dụng học máy phân tích hành vi giao dịch tài chính, tự động nhận diện và cảnh báo sớm các dấu hiệu rủi ro gian lận dựa trên nền tảng thuật toán Random Forest.")

if uploaded_file is None:
    st.info("💡 Vui lòng tải tệp dữ liệu mẫu `.csv` hoặc `.xlsx` tại thanh điều hướng bên trái (Sidebar) để kích hoạt ứng dụng.")
    st.stop()
else:
    # Đọc dữ liệu đã upload thông qua hàm cache
    file_bytes = uploaded_file.read()
    df_raw = load_data(file_bytes, uploaded_file.name)
    
    if df_raw is None:
        st.error("Tệp dữ liệu không hợp lệ. Vui lòng kiểm tra lại định dạng.")
        st.stop()
        
    st.caption(f"📁 Đang dùng tệp dữ liệu: **{uploaded_file.name}**")

st.divider()

# Khai báo các cột tính năng bắt buộc dựa trên cấu trúc mô hình
feature_cols = [f'X_{i}' for i in range(1, 15)]
target_col = 'default'

# Kiểm tra tính toàn vẹn của cấu trúc Schema dữ liệu
if not all(col in df_raw.columns for col in feature_cols + [target_col]):
    st.error(f"❌ Cấu trúc tệp dữ liệu không tương thích! Yêu cầu phải chứa đầy đủ các cột: {', '.join(feature_cols)} và cột mục tiêu '{target_col}'.")
    st.stop()

# ==============================================================================
# KHỐI XỬ LÝ CHÍNH: HUẤN LUYỆN MÔ HÌNH (Lưu Session State)
# ==============================================================================
if btn_train:
    with st.spinner("⏳ Hệ thống đang xử lý dữ liệu và huấn luyện mô hình, vui lòng đợi..."):
        X = df_raw[feature_cols]
        y = df_raw[target_col]
        
        # Phân tách tập Train - Test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        # Chuẩn hóa dữ liệu StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Áp dụng kỹ thuật cân bằng lớp SMOTE nếu được bật
        if use_smote:
            smote = SMOTE(random_state=random_state)
            X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        else:
            X_train_res, y_train_res = X_train_scaled, y_train
            
        # Khởi tạo và khớp mô hình học máy
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state,
            n_jobs=-1
