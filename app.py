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
# LỆNH STREAMLIT ĐẦU TIÊN: CẤU HÌNH TRANG CHỦ
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
    """Nạp dữ liệu từ định dạng bytes để đảm bảo tính hashable cho cơ chế cache"""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi nạp tệp dữ liệu: {e}")
        return None

# ==============================================================================
# THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH BIẾN VÀ THAM SỐ
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải tập tin dữ liệu
    uploaded_file = st.file_uploader(
        "Tải lên dữ liệu huấn luyện mẫu (CSV/XLSX)", 
        type=["csv", "xlsx"],
        help="Chọn tệp có cấu trúc tương thích với hệ thống (Gồm 14 cột tính năng X_1 đến X_14 và biến mục tiêu 'default')."
    )
    
    st.divider()
    
    # 2. Định hình tham số siêu mô hình AI
    st.subheader("Tham số mô hình AI")
    st.caption("Thuật toán sử dụng: **RandomForestClassifier**")
    
    n_estimators = st.slider(
        "Số lượng cây quyết định (n_estimators)", 
        min_value=10, 
        max_value=300, 
        value=100, 
        step=10,
        help="Số lượng cây quyết định độc lập được xây dựng trong mô hình rừng."
    )
    
    max_depth = st.slider(
        "Độ sâu tối đa của cây (max_depth)", 
        min_value=2, 
        max_value=50, 
        value=15, 
        step=1,
        help="Giới hạn độ sâu phân nhánh tối đa của mỗi cây nhằm kiểm soát rủi ro quá khớp (overfitting)."
    )
    
    random_state = st.number_input(
        "Trạng thái ngẫu nhiên (random_state)", 
        value=42, 
        step=1,
        help="Đảm bảo tính nhất quán và khả năng tái lập kết quả phân tích giống nhau giữa các lần huấn luyện."
    )
    
    # Mở rộng cấu hình nâng cao phục vụ tiền xử lý cấu trúc
    with st.expander("⚙️ Thiết lập nâng cao"):
        use_smote = st.checkbox(
            "Áp dụng thuật toán SMOTE", 
            value=True,
            help="Tự động cân bằng lớp dữ liệu bằng cách sinh mẫu nhân tạo cho nhóm giao dịch gian lận (thiếu số)."
        )
        test_size = st.slider(
            "Tỷ lệ phân chia tập kiểm tra (Test size)", 
            min_value=0.1, 
            max_value=0.5, 
            value=0.3, 
            step=0.05,
            help="Tỷ lệ phân tách phần trăm dữ liệu dùng riêng cho mục đích thẩm định và đánh giá mô hình."
        )

    st.divider()
    
    # 3. Kích hoạt huấn luyện duy nhất một lần
    btn_train = st.button(
        "🚀 Khởi chạy huấn luyện", 
        type="primary", 
        use_container_width=True,
        help="Nhấp chọn để kích hoạt tiến trình làm sạch, tiền xử lý và fit dữ liệu vào mô hình học máy."
    )

# ==============================================================================
# THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG VÀ KIỂM TRA TRẠNG THÁI RỖNG
# ==============================================================================
# Thay đổi màu tiêu đề sang màu đỏ bằng tính năng định dạng màu chữ của Streamlit
st.title("🛡️ :red[Ứng dụng Phát hiện Giao dịch Gian lận Tài chính]")
st.caption("Giải pháp tích hợp công nghệ Học máy hỗ trợ nhận diện, chấm điểm rủi ro và ngăn ngừa sớm hành vi gian lận giao dịch trực tuyến dựa trên nền tảng phân tích thuộc tính số học.")

if uploaded_file is None:
    st.info("💡 Hệ thống đang chờ dữ liệu đầu vào. Vui lòng tải lên tệp dữ liệu mẫu ở bảng điều khiển Sidebar bên trái để kích hoạt.")
    st.stop()
else:
    file_bytes = uploaded_file.read()
    df_raw = load_data(file_bytes, uploaded_file.name)
    
    if df_raw is None:
        st.error("❌ Định dạng tệp dữ liệu không hợp lệ. Vui lòng kiểm tra lại cấu trúc.")
        st.stop()
        
    st.caption(f"📁 Đang kết nối nguồn dữ liệu: **{uploaded_file.name}**")

st.divider()

# Xác lập tập biến cố định từ tài liệu phân tích notebook
feature_cols = [f'X_{i}' for i in range(1, 15)]
target_col = 'default'

# Thẩm định Schema kiểm tra tính toàn vẹn cột thuộc tính
if not all(col in df_raw.columns for col in feature_cols + [target_col]):
    st.error(f"❌ Cấu trúc bảng dữ liệu thiếu hụt thuộc tính bắt buộc! Tệp tải lên phải bao gồm đầy đủ 14 biến số (`X_1` - `X_14`) và cột phân loại mục tiêu `{target_col}`.")
    st.stop()

# ==============================================================================
# KHỐI XỬ LÝ CHÍNH: HUẤN LUYỆN VÀ LƯU TRỮ TRẠNG THÁI (SESSION STATE)
# ==============================================================================
if btn_train:
    with st.spinner("⏳ Hệ thống đang thực hiện chuẩn hóa dữ liệu và huấn luyện mô hình học máy, vui lòng đợi..."):
        X = df_raw[feature_cols]
        y = df_raw[target_col]
        
        # 1. Phân tách tập dữ liệu Train/Test có giữ nguyên tỷ lệ cấu trúc phân phối lớp
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 2. Xây dựng bộ chuẩn hóa thang đo thuộc tính StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 3. Kỹ thuật tiền xử lý mất cân bằng phân phối
        if use_smote:
            smote = SMOTE(random_state=random_state)
            X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
        else:
            X_train_res = X_train_scaled
            y_train_res = y_train
            
        # 4. Huấn luyện thực tế bộ phân loại ngẫu nhiên
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train_res, y_train_res)
        
        # 5. Đánh giá chất lượng và xuất ra xác suất rủi ro dự báo
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Đóng gói và lưu chuyển bền vững sang bộ nhớ tạm Session State phục vụ đa tab
        st.session_state['model_fitted'] = model
        st.session_state['scaler'] = scaler
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred
        st.session_state['y_prob'] = y_prob
        st.session_state['feature_importances'] = model.feature_importances_
        
    st.success("🎉 Quy trình huấn luyện mô hình hoàn tất thành công! Vui lòng kiểm tra các phân tích tại Tab chức năng bên dưới.")

# ==============================================================================
# PHÂN VÙNG NỘI DUNG CHÍNH QUA GIAO DIỆN TABS
# ==============================================================================
tab_overview, tab_viz, tab_metrics, tab_inference = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa biến", 
    "🎯 Đánh giá kiểm định", 
    "🔮 Mô phỏng chẩn đoán"
])

# ------------------------------------------------------------------------------
# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# ------------------------------------------------------------------------------
with tab_overview:
    st.subheader("Cấu trúc thông số tệp dữ liệu thô")
    
    file_size_mb = len(file_bytes) / (1024 * 1024)
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Tổng số lượng bản ghi (Dòng)", f"{df_raw.shape[0]:,}")
    col_m2.metric("Số trường thông tin (Cột)", f"{df_raw.shape[1]}")
    col_m3.metric("Kích thước vật lý tệp tin", f"{file_size_mb:.2f} MB")
    
    st.subheader("👀 Xem trước cấu trúc 5 hàng bản ghi đầu tiên")
    st.dataframe(df_raw.head(5), use_container_width=True)
    
    st.subheader("📋 Thống kê mô tả đặc trưng phân phối toán học (Chỉ hiển thị các biến đưa vào mô hình)")
    st.dataframe(df_raw[feature_cols].describe().T, use_container_width=True)

# ------------------------------------------------------------------------------
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# ------------------------------------------------------------------------------
with tab_viz:
    st.subheader("Biểu đồ phân tích hành vi đặc trưng dữ liệu")
    
    # Tính toán cơ cấu tỷ lệ nhãn mục tiêu
    class_counts = df_raw[target_col].value_counts().reset_index()
    class_counts.columns = ['Trạng thái', 'Số lượng']
    class_counts['Trạng thái'] = class_counts['Trạng thái'].map({0: 'Bình thường (0)', 1: 'Gian lận (1)'})
    
    # Thiết lập phân bổ bố cục lưới biểu đồ cân đối 2x2
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    with row1_col1:
        fig_target = px.bar(
            class_counts, x='Trạng thái', y='Số lượng',
            color='Trạng thái', title="Tỷ lệ phân phối biến mục tiêu (Mất cân bằng lớp)",
            color_discrete_sequence=['#1f77b4', '#d62728'], text_auto=True
        )
        st.plotly_chart(fig_target, use_container_width=True)
        
    with row1_col2:
        fig_x1 = px.histogram(
            df_raw, x='X_1', color=target_col, barmode='overlay',
            title="Biểu đồ phân phối mật độ Đặc trưng X_1",
            color_discrete_map={0: '#1f77b4', 1: '#d62728'}, marginal="box"
        )
        st.plotly_chart(fig_x1, use_container_width=True)
        
    with row2_col1:
        fig_x5 = px.histogram(
            df_raw, x='X_5', color=target_col, barmode='overlay',
            title="Biểu đồ phân phối mật độ Đặc trưng X_5",
            color_discrete_map={0: '#1f77b4', 1: '#d62728'}, marginal="box"
        )
        st.plotly_chart(fig_x5, use_container_width=True)
        
    with row2_col2:
        # Sử dụng phương pháp lấy mẫu tối ưu tránh trễ kết xuất giao diện nặng
        df_sample = df_raw.sample(n=min(2500, len(df_raw)), random_state=42)
        fig_scatter = px.scatter(
            df_sample, x='X_1', y='X_2', color=target_col,
            title="Tương quan không gian phân tán giữa thuộc tính X_1 và X_2",
            color_discrete_map={0: '#1f77b4', 1: '#d62728'}, opacity=0.6
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------------------------------------------------------------
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH"
# ------------------------------------------------------------------------------
with tab_metrics:
    if 'model_fitted' not in st.session_state:
        st.info("📢 Chưa ghi nhận dữ liệu mô hình khả dụng trong phiên làm việc hiện tại. Vui lòng thiết lập tham số và nhấn nút 'Khởi chạy huấn luyện' tại Sidebar.")
    else:
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        y_prob = st.session_state['y_prob']
        feature_importances = st.session_state['feature_importances']
        
        st.subheader("🎯 Chỉ số đo lường hiệu năng mô hình trên tập kiểm thử độc lập (Test Set)")
        
        # Biên soạn cấu trúc báo cáo chỉ số
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Thiết kế khối hiển thị thông số Metrics dạng thẻ
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Độ chính xác chung (Accuracy)", f"{report['accuracy']:.4f}")
        m_col2.metric("Độ chuẩn xác lớp rủi ro (Precision)", f"{report['1']['precision']:.4f}")
        m_col3.metric("Tỷ lệ bao phủ/Bắt sót (Recall)", f"{report['1']['recall']:.4f}")
        m_col4.metric("Điểm F1-Score (F-Measure)", f"{report['1']['f1-score']:.4f}")
        
        st.divider()
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("**Ma trận phân loại chi tiết (Confusion Matrix Heatmap):**")
            fig_cm = px.imshow(
                cm.tolist(),
                x=['Dự đoán Thường (0)', 'Dự đoán Gian lận (1)'],
                y=['Thực tế Thường (0)', 'Thực tế Gian lận (1)'],
                text_auto=True, color_continuous_scale='Blues',
                labels=dict(x="Nhãn hệ thống dự báo", y="Nhãn kiểm tra thực tế", color="Số lượng")
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with res_col2:
            st.markdown(f"**Đường cong đặc tính hoạt động phân tách (ROC Curve - AUC: {roc_auc:.4f}):**")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'Random Forest (AUC={roc_auc:.3f})', line=dict(color='darkorange', width=2)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Phân bổ ngẫu nhiên', line=dict(dash='dash', color='navy')))
            fig_roc.update_layout(xaxis_title='Tỷ lệ dương tính giả (FPR)', yaxis_title='Tỷ lệ dương tính thật (TPR)', margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_roc, use_container_width=True)
            
        st.divider()
        
        # Đánh giá trọng số đóng góp của tập tính năng đầu vào
        st.subheader("🔍 Biểu đồ trọng số đóng góp đặc trưng (Feature Importances)")
        df_importance = pd.DataFrame({
            'Thuộc tính biến': feature_cols,
            'Trọng số quyết định': feature_importances
        }).sort_values(by='Trọng số quyết định', ascending=True)
        
        fig_imp = px.bar(
            df_importance, x='Trọng số quyết định', y='Thuộc tính biến', 
            orientation='h', color='Trọng số quyết định', color_continuous_scale='Viridis',
            title='Thứ tự các thuộc tính có tầm ảnh hưởng lớn nhất đến quyết định phân loại rủi ro'
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ------------------------------------------------------------------------------
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH" (INFERENCE)
# ------------------------------------------------------------------------------
with tab_inference:
    if 'model_fitted' not in st.session_state:
        st.info("📢 Tính năng chẩn đoán đang bị khóa. Vui lòng thực hiện thao tác huấn luyện mô hình thành công trước khi chấm điểm rủi ro.")
    else:
        model = st.session_state['model_fitted']
        scaler = st.session_state['scaler']
        
        st.subheader("🔮 Chẩn đoán và Thẩm định xác suất gian lận trực tuyến")
        
        mode = st.radio(
            "Lựa chọn hình thức kiểm tra giao dịch đầu vào:",
            ["👉 Kiểm tra đơn lẻ thủ công (Nhập tay)", "📂 Thẩm định danh sách hàng loạt (Tải file danh sách mới)"],
            horizontal=True
        )
        
        # CHẾ ĐỘ CHẨN ĐOÁN 1: NHẬP FORM THỦ CÔNG
        if mode == "👉 Kiểm tra đơn lẻ thủ công (Nhập tay)":
            st.markdown("##### Nhập thông số thuộc tính kỹ thuật của giao dịch cần kiểm tra:")
            
            with st.form("single_predict_form"):
                cols = st.columns(4)
                input_values = {}
                
                for idx, col_name in enumerate(feature_cols):
                    # Thiết lập phân bổ giá trị mặc định tối ưu theo giá trị trung vị mẫu
                    median_val = float(df_raw[col_name].median())
                    min_val = float(df_raw[col_name].min())
                    max_val = float(df_raw[col_name].max())
                    
                    with cols[idx % 4]:
                        input_values[col_name] = st.number_input(
                            f"Giá trị trường {col_name}", 
                            min_value=min_val - abs(min_val)*2,
                            max_value=max_value + abs(max_value)*2,
                            value=median_val,
                            format="%.6f"
                        )
                
                submit_predict = st.form_submit_button("🛡️ Khởi chạy phân tích rủi ro", type="primary")
                
            if submit_predict:
                df_single = pd.DataFrame([input_values])
                # Áp dụng bộ tiền xử lý chuẩn hóa StandardScaler đồng bộ mẫu huấn luyện
                df_single_scaled = scaler.transform(df_single)
                
                pred_class = model.predict(df_single_scaled)[0]
                pred_prob = model.predict_proba(df_single_scaled)[0][1]
                
                st.markdown("### Kết quả kết luận kiểm tra từ hệ thống:")
                if pred_class == 1:
                    st.error(f"🚨 CẢNH BÁO: Giao dịch có nguy cơ **GIAN LẬN** cao! (Chỉ số xác suất rủi ro tích tụ: **{pred_prob*100:.2f}%**)")
                else:
                    st.success(f"✅ AN TOÀN: Giao dịch được xác định **BÌNH THƯỜNG**. (Xác suất rủi ro thấp: **{pred_prob*100:.2f}%**)")
                    
        # CHẾ ĐỘ CHẨN ĐOÁN 2: BATCH INFERENCE HÀNG LOẠT QUA TỆP TIN
        else:
            st.markdown("##### Tải lên tệp danh sách các hồ sơ giao dịch tài chính mới cần xử lý:")
            st.caption("⚠️ Yêu cầu bắt buộc: Tập tin định dạng Excel hoặc CSV phải cấu trúc chuẩn xác 14 cột thuộc tính từ `X_1` đến `X_14`.")
            
            batch_file = st.file_uploader("Nạp tệp kiểm tra hàng loạt (X_new)", type=["csv", "xlsx"])
            
            if batch_file is not None:
                if batch_file.name.endswith('.csv'):
                    df_batch = pd.read_csv(batch_file)
                else:
                    df_batch = pd.read_excel(batch_file)
                    
                # Kiểm tra tính đồng bộ cấu trúc Schema
                if not all(col in df_batch.columns for col in feature_cols):
                    st.error("❌ Cấu trúc tệp dữ liệu không tương thích. Vui lòng đảm bảo tệp chứa đầy đủ tên các cột từ X_1 đến X_14.")
                else:
                    X_batch = df_batch[feature_cols]
                    X_batch_scaled = scaler.transform(X_batch)
                    
                    # Tiến hành quét chấm điểm đồng loạt qua bộ phân loại
                    batch_preds = model.predict(X_batch_scaled)
                    batch_probs = model.predict_proba(X_batch_scaled)[:, 1]
                    
                    df_res = df_batch.copy()
                    df_res['Dự đoán phân loại'] = batch_preds
                    df_res['Xác suất rủi ro gian lận'] = batch_probs
                    df_res['Kết luận thẩm định'] = df_res['Dự đoán phân loại'].map({0: 'Bình thường', 1: 'Cảnh báo gian lận'})
                    
                    fraud_total = int(np.sum(batch_preds == 1))
                    total_records = len(df_res)
                    
                    col_b1, col_b2 = st.columns(2)
                    col_b1.metric("Tổng số lượng giao dịch đã quét xử lý", f"{total_records:,}")
                    col_b2.metric("Số vụ giao dịch phát hiện dấu hiệu gian lận", f"{fraud_total:,}", delta=f"{fraud_total/total_records*100:.2f}% tỷ lệ rủi ro", delta_color="inverse")
                    
                    st.markdown("**Bảng tổng hợp chi tiết kết quả phân tích hệ thống:**")
                    st.dataframe(df_res, use_container_width=True)
                    
                    # Kết xuất bộ đệm xuất file tải xuống cho điều tra viên
                    csv_buffer = io.StringIO()
                    df_res.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                    csv_data_bytes = csv_buffer.getvalue().encode('utf-8-sig')
                    
                    st.download_button(
                        label="📥 Tải xuống báo cáo kết quả thẩm định tổng thể (.CSV)",
                        data=csv_data_bytes,
                        file_name="Bao_cao_kiem_tra_gian_lan_dong_loat.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
