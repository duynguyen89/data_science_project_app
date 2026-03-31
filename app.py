import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Required for xgboost pipeline unpickling
def convert_to_string(x):
    return x.astype(str)

from backend import (
    load_and_clean_data, 
    lambdas, 
    predict_price, 
    get_anomaly_model,
    get_recommendation_system,
    recommend_houses,
    get_clustering_model
)

# Config
st.set_page_config(page_title="Hệ Sinh Thái Bất Động Sản", layout="wide")

# Custom CSS for Premium Design
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .reportview-container {background: #f8f9fa;}
    h1, h2, h3 {color: #2c3e50; font-family: 'Inter', sans-serif;}
    .stButton>button {
        background-image: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        color: white; border: none; border-radius: 8px; padding: 10px 24px;
        transition: transform 0.2s, box-shadow 0.2s;
        font-weight: 600;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0,242,254,0.4);
    }
    .metric-card {
        background: white; padding: 20px; border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05); text-align: center;
        border-top: 4px solid #00f2fe; margin-bottom: 20px;
    }
    .metric-value {font-size: 28px; font-weight: bold; color: #1abc9c;}
    .stSelectbox label, .stNumberInput label, .stTextInput label {
        font-weight: 600; color: #34495e;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def get_data():
    return load_and_clean_data()

df = get_data()

try:
    st.image("banner_nhatot.png", use_container_width=True)
except:
    st.title("🏡 Hệ Sinh Thái Bất Động Sản Nâng Cao")

menu = ["Trang chủ", "Dự báo giá nhà", "Phát hiện bất thường", "Gợi ý nhà tương tự", "Phân cụm thị trường"]
choice = st.sidebar.selectbox('🔍 Tính năng chính', menu)

if choice == 'Trang chủ':
    st.markdown("## Chào mừng bạn đến với Hệ sinh thái Bất động sản AI")
    st.markdown("""
    Nền tảng này ứng dụng các mô hình học máy tiên tiến để giúp bạn:
    - 📈 **Dự báo giá nhà**: Định giá bất động sản chính xác với XGBoost.
    - 🚨 **Phát hiện bất thường**: Nhận diện các tin đăng giả hoặc giá ảo bằng Isolation Forest.
    - 💡 **Gợi ý nhà tương tự**: Tìm kiếm các bất động sản phù hợp dựa trên nội dung (Content-based Filtering).
    - 📊 **Phân cụm thị trường**: Hiểu rõ các phân khúc bất động sản trên thị trường với thuật toán K-Means.
    
    *Dữ liệu được xử lý tự động từ hàng nghìn tin đăng bất động sản tại TP.HCM.*
    """)
    st.dataframe(df.head(10))

elif choice == 'Dự báo giá nhà':
    st.markdown("## 📈 Dự báo Giá Bất Động Sản")
    st.write("Nhập thông tin chi tiết của căn nhà để nhận định giá từ mô hình AI.")
    
    df_cat = df.copy()
    
    col1, col2 = st.columns(2)
    with col1:
        dien_tich_dat = st.number_input("Diện tích đất (m2)", min_value=1.0, value=50.0)
        dien_tich_su_dung = st.number_input("Diện tích sử dụng (m2)", min_value=1.0, value=60.0)
        chieu_ngang = st.number_input("Chiều ngang (m)", min_value=1.0, value=4.0)
        tong_so_tang = st.number_input("Tổng số tầng", min_value=1, value=2)
        so_phong_ngu = st.number_input("Số phòng ngủ", min_value=1, value=2)
    
    with col2:
        loai_hinh = st.selectbox("Loại hình", df['loai_hinh'].dropna().unique().tolist() + ['Khác'])
        giay_to_phap_ly = st.selectbox("Giấy tờ pháp lý", df['giay_to_phap_ly'].dropna().unique().tolist() + ['Khác'])
        tinh_trang_noi_that = st.selectbox("Tình trạng nội thất", df['tinh_trang_noi_that'].dropna().unique().tolist() + ['Khác'])
        huong_cua_chinh = st.selectbox("Hướng cửa chính", df['huong_cua_chinh'].dropna().unique().tolist() + ['Khác'])
        dac_diem = st.selectbox("Đặc điểm", df['dac_diem'].dropna().unique().tolist() + ['Khác'])
        dia_chi_cu = st.selectbox("Địa chỉ (Phường/Quận)", df['dia_chi_cu'].dropna().unique().tolist() + ['Khác'])
        dia_chi_moi = st.selectbox("Quận/Huyện mới", df['dia_chi_moi'].dropna().unique().tolist() + ['Khác'])

    if st.button("🔮 Dự Báo Giá Ngay"):
        features = {
            'dien_tich_dat': dien_tich_dat,
            'dien_tich_su_dung': dien_tich_su_dung,
            'chieu_ngang': chieu_ngang,
            'tong_so_tang': tong_so_tang,
            'so_phong_ngu': so_phong_ngu,
            'loai_hinh': loai_hinh,
            'giay_to_phap_ly': giay_to_phap_ly,
            'tinh_trang_noi_that': tinh_trang_noi_that,
            'huong_cua_chinh': huong_cua_chinh,
            'dac_diem': dac_diem,
            'dia_chi_cu': dia_chi_cu,
            'dia_chi_moi': dia_chi_moi
        }
        with st.spinner("Mô hình XGBoost đang tính toán..."):
            try:
                price = predict_price(features, lambdas)
                st.markdown(f'<div class="metric-card"><p>Giá dự kiến của căn nhà</p><div class="metric-value">{price:,.2f} Tỷ VNĐ</div></div>', unsafe_allow_html=True)
                st.success("Dự báo thành công dựa trên phân tích tương quan của hàng nghìn tin đăng.")
            except Exception as e:
                st.error(f"Đã xảy ra lỗi khi chạy mô hình: {e}")

elif choice == 'Phát hiện bất thường':
    st.markdown("## 🚨 Phát hiện Tin đăng Bất thường")
    st.write("Sử dụng mô hình `IsolationForest` (được lưu tự động dạng `model_anomaly_detection_IsolationForest.pkl`) để đánh giá tin đăng theo diện tích, giá bán, số phòng ngủ, chiều ngang.")
    
    isf, stats_dict = get_anomaly_model(df)
    
    tab_xgb, tab_single, tab_multi = st.tabs(["🔥 XGBoost Residuals", "📝 Kiểm Tra Chuyên Sâu (1 Tin)", "📁 Kiểm Tra Hàng Loạt (File CSV)"])
    
    def explain_anomaly_extended(dtsd, price, ngang, phong, dtd, tang, loai_hinh, phap_ly, noi_that, dac_diem, huong, dia_chi_cu, dia_chi_moi, anomaly_pred):
        reasons = []
        price_per_m2 = price / dtsd if dtsd > 0 else 0
        local_price = stats_dict.get('local_price_per_m2', {}).get(dia_chi_cu, stats_dict['median_price_per_m2'])
        type_area = stats_dict.get('type_median_area', {}).get(loai_hinh, stats_dict['median_dien_tich'])

        # 1. Price checks
        if price_per_m2 < (local_price * 0.3):
            reasons.append(f"- **Chênh lệch giá khu vực**: Giá khu vực {dia_chi_cu} thường khoảng {local_price:.2f} Tỷ/m2, nhưng tin này chỉ {price_per_m2:.2f} Tỷ/m2. Cảnh báo mồi nhử ảo thu thập SĐT.")
        
        # 2. Structural checks (Area vs Floors)
        expected_usability = dtd * tang * 1.2 # slightly generous for balconies etc.
        if dtsd > expected_usability and tang > 0:
            reasons.append(f"- **Diện tích phi lý (Phóng đại)**: Diện tích sàn sử dụng ({dtsd} m2) lớn bất thường so với ({dtd} m2 Đất x {tang} Lầu).")
            
        if dtsd < (dtd * tang * 0.2) and tang > 0 and dtd > 10:
            reasons.append(f"- **Diện tích phi lý (Thu nhỏ)**: Có {dtd} m2 đất và {tang} lầu nhưng diện tích sử dụng chỉ vỏn vẹn {dtsd} m2, báo hiệu thông tin nhập sai.")
            
        if dtsd < (phong * 5) and phong > 0:
            reasons.append(f"- **Định dạng không gian phi lý**: Không thể nhồi nhét {phong} phòng ngủ trong không gian chỉ có {dtsd} m2 diện tích sử dụng.")

        # 3. Categorical checks
        if loai_hinh == 'Biệt thự, nhà liền kề':
            if price < 3.0:
                reasons.append(f"- **Phân khúc cao cấp ảo**: Nhãn dán là Biệt thự nhưng rao bán với giá {price} Tỷ (mức giá không tưởng đối với phân khúc này), cảnh báo mồi nhử câu view ảo.")
            if dtd < 40:
                reasons.append(f"- **Phân loại sai lệch**: Đăng là '{loai_hinh}' nhưng diện tích đất lại siêu nhỏ ({dtd} m2, trong khi biệt thự thực tế thường > {max(40, type_area)} m2).")
        
        if loai_hinh == 'Nhà ngõ, hẻm' and ngang > 10 and dtd < 50:
            reasons.append(f"- **Sai lệch kiến trúc ngõ hẻm**: Nhà trong hẻm nhưng chiều ngang lại dài hơn chiều sâu rất nhiều một cách vô lý.")
            
        if phap_ly == 'Đã có sổ' and price < (local_price * dtsd * 0.3):
            reasons.append(f"- **Pháp lý ảo**: Căn nhà báo 'Đã có sổ' nhưng khai báo giá 'phá giá' thị trường, thường là đánh tráo sổ (nhà chỗ khác nhưng đăng sổ giả).")
            
        if pd.notna(noi_that) and 'Cao cấp' in str(noi_that) and price < (local_price * dtsd * 0.5):
            reasons.append(f"- **Tin đăng câu view**: Cam kết nội thất '{noi_that}' nhưng giá chỉ bằng nửa giá nhà thô trung bình.")

        if phong > (tang * 4) and tang > 0:
            reasons.append(f"- **Số phòng thiết kế phi lý**: Có {phong} phòng nhưng chỉ có {tang} tầng, tỷ lệ vách ngăn không gian quá tải đối với nhà phố.")

        if anomaly_pred == -1 and not reasons:
            reasons.append("- **Tổ hợp cực đoan thuật toán**: Thuật toán Máy học Isolation Forest phát hiện bộ số liệu này bị cô lập hoàn toàn khỏi tập dữ liệu 10,000+ căn nhà của thị trường gốc.")

        # FINAL DECISION
        if anomaly_pred == 1 and not reasons:
            st.success("✅ Phân tích: Tin đăng nằm trong vùng bình thường của thị trường.")
            return

        st.error("⚠️ Phân tích: ĐÂY LÀ TIN ĐĂNG BẤT THƯỜNG (Outlier)")
        if anomaly_pred == 1 and len(reasons) > 0:
            st.warning("🤖 *Mô hình AI IsolationForest nhận diện bình thường do thông số chưa đủ cực đoan, NHƯNG Hệ chuyên gia (Domain Rules) phát hiện lỗi kĩ thuật nhập liệu/tin ảo:*")
        else:
            st.markdown("### 🧠 Sự kiện Phi lý (Mồi nhử/Tin ảo trên 'Nhà Tốt')")

        for r in reasons:
            st.markdown(r)
            
        st.markdown(f"#### Biểu đồ Trực quan hóa Đa chiều")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Price and Area vs Market
        cats1 = ["Giá/m2 so với KV", 'Diện tích Đất', 'Số Tầng']
        median_1 = [local_price, type_area, 2.5]
        house_1 = [price_per_m2, dtd, tang]
        
        x = np.arange(len(cats1))
        width = 0.35
        axes[0].bar(x - width/2, np.log1p(median_1), width, label='Mặt bằng chung (Trung vị)', color='#bdc3c7')
        axes[0].bar(x + width/2, np.log1p(house_1), width, label='Tin đăng hiện tại', color='#e74c3c')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(cats1)
        axes[0].set_ylabel('Giá trị Log (Log1p Scale)')
        axes[0].set_title(f"So sánh các chỉ số tại {dia_chi_cu}")
        axes[0].legend()
        
        # Plot 2: Scatter Highlight
        df_sub = df.dropna(subset=['dien_tich_su_dung', 'gia_ban'])
        sns.scatterplot(data=df_sub, x='dien_tich_su_dung', y='gia_ban', color='#3498db', alpha=0.3, ax=axes[1])
        axes[1].scatter([dtsd], [price], color='red', s=200, edgecolors='black', label='Điểm dị thường (Tin của bạn)')
        axes[1].set_xlabel("Diện tích sử dụng (m2)")
        axes[1].set_ylabel("Giá bán (Tỷ VNĐ)")
        axes[1].set_title("Vị trí của tin đăng này trên thị trường")
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)

    with tab_xgb:
        st.markdown("### 🔥 Phát Hiện Bất Thường Dựa Trên Độ Lệch Khỏi Kỳ Vọng (XGBoost Residuals)")
        st.write("Phương pháp này ứng dụng mô hình dự đoán giá cực mạnh XGBoost để định giá toàn bộ thị trường. Nếu một căn nhà có **Giá Bán Thực Tế** lệch quá lớn (Z-score > 3) so với **Giá Trị Dự Đoán (Kỳ Vọng)** của XGBoost, nó sẽ bị đánh dấu là Giao dịch Bất thường (Quá đắt hoặc Quá rẻ một cách phi lý).")
        
        try:
            df_xgb_out = pd.read_csv("anomaly_detection_from_xgboost_prediction.csv")
            
            # Tính toán lại Z-Score độ lệch y như notebook 1
            df_xgb_out['residual'] = df_xgb_out['gia_ban'] - df_xgb_out['final_prediction']
            df_xgb_out['avg_res'] = df_xgb_out.groupby('loai_hinh')['residual'].transform('mean')
            df_xgb_out['std_res'] = df_xgb_out.groupby('loai_hinh')['residual'].transform('std')
            df_xgb_out['residual_z'] = (df_xgb_out['residual'] - df_xgb_out['avg_res']) / (df_xgb_out['std_res'] + 1e-9)
            df_xgb_out['is_anomaly'] = np.where(df_xgb_out['residual_z'].abs() > 3, "Bất thường", "Bình thường")
            
            c1, c2 = st.columns([1, 1])
            with c1:
                st.info(f"Tổng số tin đăng phân tích: **{len(df_xgb_out)}**")
            with c2:
                st.error(f"Phát hiện **{len(df_xgb_out[df_xgb_out['is_anomaly'] == 'Bất thường'])}** giao dịch có dấu hiệu thao túng giá (Lệch cực chuẩn).")
                
            fig_x, ax_x = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                data=df_xgb_out,
                x='final_prediction',
                y='gia_ban',
                hue='is_anomaly',
                palette={'Bình thường': 'gray', 'Bất thường': 'red'},
                alpha=0.6,
                s=70,
                ax=ax_x
            )
            # Đường biên lý tưởng
            max_val = max(df_xgb_out['gia_ban'].max(), df_xgb_out['final_prediction'].max())
            ax_x.plot([0, max_val], [0, max_val], '--', color='blue', alpha=0.5, label='Đường giá lý tưởng (Y=X)')
            ax_x.set_title("Biểu Đồ Phân Đố Mức Giá: Thực Tế vs Dự Đoán (XGBoost)")
            ax_x.set_xlabel("Giá Dự Đoán Kỳ Vọng (Tỷ VNĐ)")
            ax_x.set_ylabel("Giá Bán Thực Tế Đang Đăng (Tỷ VNĐ)")
            ax_x.legend()
            st.pyplot(fig_x)
            
            st.markdown("#### Bảng Kê Tin Đăng Vi Phạm Nghiêm Trọng (Z-Score > 3)")
            df_violates = df_xgb_out[df_xgb_out['is_anomaly'] == 'Bất thường'].copy()
            df_violates['Độ lệch giá'] = df_violates['residual'].round(2).astype(str) + " Tỷ"
            st.dataframe(df_violates[['tieu_de', 'loai_hinh', 'gia_ban', 'final_prediction', 'Độ lệch giá', 'residual_z']].sort_values('residual_z', ascending=False), use_container_width=True)
            
        except Exception as e:
            st.warning(f"Chưa có tệp `anomaly_detection_from_xgboost_prediction.csv` hoặc xảy ra lỗi ({e}). Vui lòng chạy Notebook Project 1 trước.")

    with tab_single:
        col1, col2 = st.columns(2)
        with col1:
            test_dtsd = st.number_input("Diện tích SD (m2)", min_value=1.0, value=60.0, key="dt1")
            test_dtd = st.number_input("Diện tích Đất (m2)", min_value=1.0, value=50.0, key="dt2")
            test_ngang = st.number_input("Chiều ngang (m)", min_value=1.0, value=4.0, key="ng1")
            test_tang = st.number_input("Tổng số tầng", min_value=1, value=2, key="tg1")
            test_phong = st.number_input("Số phòng ngủ", min_value=1, value=2, key="np1")
            test_price = st.number_input("Mức giá (Tỷ VNĐ)", min_value=0.1, value=5.0, key="pr1")
            
        with col2:
            test_lh = st.selectbox("Loại hình", df['loai_hinh'].dropna().unique().tolist() + ['Khác'], key="s1")
            test_pl = st.selectbox("Giấy tờ pháp lý", df['giay_to_phap_ly'].dropna().unique().tolist() + ['Khác'], key="s2")
            test_nt = st.selectbox("Tình trạng nội thất", df['tinh_trang_noi_that'].dropna().unique().tolist() + ['Khác'], key="s3")
            test_dd = st.selectbox("Đặc điểm", df['dac_diem'].dropna().unique().tolist() + ['Khác'], key="s4")
            test_huong = st.selectbox("Hướng cửa chính", df['huong_cua_chinh'].dropna().unique().tolist() + ['Khác'], key="s5")
            test_dc1 = st.selectbox("Địa chỉ cũ (Phường/Quận)", df['dia_chi_cu'].dropna().unique().tolist() + ['Khác'], key="s6")
            test_dc2 = st.selectbox("Địa chỉ mới (Phường)", df['dia_chi_moi'].dropna().unique().tolist() + ['Khác'], key="s7")
            
        if st.button("🔍 Kiểm Tra Tin Đăng"):
            pred = isf.predict([[test_dtsd, test_price, test_phong, test_ngang, test_dtd, test_tang]])
            explain_anomaly_extended(
                test_dtsd, test_price, test_ngang, test_phong, test_dtd, test_tang, 
                test_lh, test_pl, test_nt, test_dd, test_huong, test_dc1, test_dc2, pred[0]
            )

    with tab_multi:
        req_cols = ["dien_tich_su_dung", "gia_ban", "so_phong_ngu", "chieu_ngang", "dien_tich_dat", "tong_so_tang"]
        st.write(f"Tải lên tệp CSV bao gồm các cột `{req_cols}`.")
        uploaded_file = st.file_uploader("Chọn file CSV", type=["csv"], key="uf1")
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            if all(c in batch_df.columns for c in req_cols):
                # Làm sạch dữ liệu rác (Ví dụ: "18 m²", "5,5 Tỷ") TRỰC TIẾP trên dataframe gốc để logic kiểm tra phía sau cũng nhận được số thực
                for col in req_cols:
                    if batch_df[col].dtype == object:
                        batch_df[col] = batch_df[col].astype(str).str.replace(r'[^0-9,.]', '', regex=True).str.replace(',', '.')
                        batch_df[col] = pd.to_numeric(batch_df[col], errors='coerce')
                
                # Điền NA cho các cột lỗi và trích xuất vector X
                batch_df[req_cols] = batch_df[req_cols].fillna(0)
                X_batch = batch_df[req_cols]
                ml_preds = isf.predict(X_batch)
                
                # Expert system overlay for hard rules
                def final_anomaly_decision(row, pred):
                    if row['dien_tich_su_dung'] < (row['so_phong_ngu'] * 5): return -1
                    if 'loai_hinh' in row and row['loai_hinh'] == 'Biệt thự, nhà liền kề' and row['gia_ban'] < 3.0: return -1
                    if 'tong_so_tang' in row and row['tong_so_tang'] > 0:
                        if row['dien_tich_su_dung'] > (row['dien_tich_dat'] * row['tong_so_tang'] * 1.2): return -1
                        if row['dien_tich_su_dung'] < (row['dien_tich_dat'] * row['tong_so_tang'] * 0.2) and row['dien_tich_dat'] > 10: return -1
                        if row['so_phong_ngu'] > (row['tong_so_tang'] * 4): return -1
                    return pred

                batch_df['Anomaly_Pred'] = [final_anomaly_decision(row, p) for (_, row), p in zip(batch_df.iterrows(), ml_preds)]
                
                def styler(val):
                    color = '#ffe6e6' if val == -1 else ''
                    return f'background-color: {color}'

                anomalies = batch_df[batch_df['Anomaly_Pred'] == -1]
                st.warning(f"Đã phát hiện {len(anomalies)} tin đăng bất thường trên tổng số {len(batch_df)} dòng.")
                st.dataframe(batch_df.style.applymap(styler, subset=['Anomaly_Pred']))
            else:
                st.error(f"Tệp CSV phải chứa đủ các cột số: {req_cols}")

    st.markdown("### Phân tích Insight trên Tập Test (20% Thị trường)")
    st.write("Dựa theo mô hình IsolationForest được huấn luyện trên tập Train (80%) độc lập.")
    req_cols = ["dien_tich_su_dung", "gia_ban", "so_phong_ngu", "chieu_ngang", "dien_tich_dat", "tong_so_tang"]
    df_anomaly = df.dropna(subset=["dien_tich_su_dung", "gia_ban"]).copy()
    for col in req_cols:
        if col in df_anomaly.columns:
            df_anomaly[col] = df_anomaly[col].fillna(df_anomaly[col].median())
            
    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(df_anomaly, test_size=0.2, random_state=42)
    
    X_test = test_df[req_cols].fillna(0)
    test_df['Anomaly'] = isf.predict(X_test)
    test_df['Trạng thái'] = test_df['Anomaly'].map({1: 'Bình thường', -1: 'Bất thường'})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=test_df, x='dien_tich_su_dung', y='gia_ban', hue='Trạng thái', palette={ 'Bình thường': '#3498db', 'Bất thường': '#e74c3c'}, alpha=0.6, ax=ax)
    ax.set_title("Phân bổ ranh giới bất thường nhà đất (Tập Test)")
    ax.set_xlabel("Diện tích sử dụng (m2)")
    ax.set_ylabel("Giá bán (Tỷ VNĐ)")
    st.pyplot(fig)


elif choice == 'Gợi ý nhà tương tự':
    st.markdown("## 💡 Gợi ý Nhà Theo Nội Dung (Content-based)")
    st.write("Thuật toán TF-IDF & Cosine Similarity sẽ tìm kiếm các tin đăng có nội dung miêu tả tương tự nhất với căn nhà bạn đang quan tâm.")
    
    df_rec, tfidf_matrix = get_recommendation_system(df)
    
    # User selects a house
    sample_houses = df_rec.head(100)
    house_options = [(row['tieu_de'], row['id']) for idx, row in sample_houses.iterrows()]
    selected_house_id = st.selectbox(
        "Lựa chọn một nhà bạn quan tâm từ danh sách:",
        options=[h[1] for h in house_options],
        format_func=lambda x: [h[0] for h in house_options if h[1] == x][0]
    )
    
    sh = df_rec[df_rec['id'] == selected_house_id].iloc[0]
    st.info(f"**Nhà mục tiêu:** {sh['tieu_de']}\n\n**Mô tả:** {sh['mo_ta'][:200]}...")
    
    if st.button("🌟 Tìm Nhà Tương Tự"):
        recommendations = recommend_houses(df_rec, tfidf_matrix, selected_house_id, top_n=5)
        st.markdown("### Top 5 Nhà Phù Hợp Đi Kèm:")
        for idx, row in recommendations.iterrows():
            with st.expander(f"📌 {row['tieu_de']} - {row['gia_ban']} Tỷ"):
                st.write(f"**Vị trí:** {row['dia_chi_cu']}")
                st.write(f"**Chi tiết:** {row['mo_ta'][:500]}...")

elif choice == 'Phân cụm thị trường':
    st.markdown("## 📊 Phân Cụm Thị Trường (GMM)")
    st.write("Chia thị trường thành các phân khúc đặc trưng (Mô hình Gaussian Mixture Model)")
    
    df_cluster = pd.read_csv("clustering_results_gmm.csv")
    df_cluster = df_cluster.rename(columns={'cluster': 'prediction'})
    
    # 1. Map Clusters to Labels based on Max Price
    max_prices = df_cluster.groupby('prediction')['gia_ban'].max().sort_values()
    sorted_clusters = max_prices.index.tolist()
    labels = ["Bình dân", "Sơ cấp", "Trung cấp", "Cao cấp"]
    label_map = {c: labels[i] if i < len(labels) else f"Phân khúc {i+1}" for i, c in enumerate(sorted_clusters)}
    df_cluster['Nhóm Phân Khúc'] = df_cluster['prediction'].map(label_map)
    
    st.markdown("### 🔍 Phân Loại Phân Khúc Cho Căn Nhà Mới")
    with st.expander("📝 Nhập thông số căn nhà của bạn để xem thuộc phân khúc nào", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            i_dtd = st.number_input("Diện tích đất (m2)", value=50.0, step=1.0, key="c_dtd")
            i_dtsd = st.number_input("Diện tích sử dụng (m2)", value=120.0, step=1.0, key="c_dtsd")
        with c2:
            i_tang = st.number_input("Tổng số tầng", value=2.0, step=1.0, key="c_tang")
        with c3:
            i_gia = st.number_input("Giá dự kiến (Tỷ)", value=5.0, step=0.5, key="c_gia")
            
        if st.button("📐 Xác Định Phân Khúc (Machine Learning)"):
            try:
                # Tải model GMM đã huấn luyện và Scaler tương ứng
                import joblib
                gmm = joblib.load('model_gmm.pkl')
                scaler_gmm = joblib.load('scaler_gmm.pkl')
                
                # Biến ẩn (Do người dùng yêu cầu bỏ nhập liệu nên ta lấy trung vị thị trường để mồi)
                median_ngang = df_cluster['chieu_ngang'].median()
                median_phong = df_cluster['so_phong_ngu'].median()
                
                # Tạo vector input chuẩn theo kích thước huấn luyện 6 Features:
                # 'dien_tich_dat', 'dien_tich_su_dung', 'chieu_ngang', 'tong_so_tang', 'so_phong_ngu', 'gia_ban'
                input_df = pd.DataFrame([{
                    'dien_tich_dat': i_dtd,
                    'dien_tich_su_dung': i_dtsd,
                    'chieu_ngang': median_ngang,
                    'tong_so_tang': i_tang,
                    'so_phong_ngu': median_phong,
                    'gia_ban': i_gia
                }])
                
                # Scale và Predict
                X_scaled = scaler_gmm.transform(input_df)
                cluster_label = gmm.predict(X_scaled)[0]
                assigned_label = label_map.get(cluster_label, f"Cụm {cluster_label}")
                
                st.success(f"🤖 Dựa trên Gaussian Mixture Model (GMM), căn nhà này thuộc phân khúc: **{assigned_label.upper()}**")
                
            except Exception as e:
                st.error(f"Lỗi khi chạy dự báo GMM: {e}. Có thể model_gmm.pkl chưa được tạo từ Notebook.")

    st.markdown("---")
    

    st.markdown("### Giá trị trung bình theo từng phân khúc")
    stats = df_cluster.groupby('Nhóm Phân Khúc')[["dien_tich_dat", "dien_tich_su_dung", "gia_ban"]].mean().reset_index()
    stats['Rank'] = stats['Nhóm Phân Khúc'].map(lambda x: labels.index(x) if x in labels else 99)
    stats = stats.sort_values('Rank').drop(columns=['Rank'])
    st.dataframe(stats.style.highlight_max(axis=0))
    
    st.markdown("### Thống kê Giá theo từng Phân khúc (Cluster)")
    price_stats = df_cluster.groupby('Nhóm Phân Khúc')['gia_ban'].agg(
        Avg_Price='mean',
        Min_Price='min',
        Max_Price='max'
    ).reset_index()
    
    price_stats['Rank'] = price_stats['Nhóm Phân Khúc'].map(lambda x: labels.index(x) if x in labels else 99)
    price_stats = price_stats.sort_values('Rank').drop(columns=['Rank'])
    
    price_stats_melted = price_stats.melt(id_vars='Nhóm Phân Khúc', var_name='Chỉ số giá', value_name='Price (Tỷ VNĐ)')
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    custom_palette = {"Avg_Price": "#42557c", "Min_Price": "#2c867c", "Max_Price": "#6fb96b"}
    
    sns.barplot(
        data=price_stats_melted, 
        x='Nhóm Phân Khúc', 
        y='Price (Tỷ VNĐ)', 
        hue='Chỉ số giá', 
        palette=custom_palette,
        ax=ax2
    )
    
    ax2.set_title("Giá Trung bình, Nhỏ nhất và Lớn nhất theo từng Cụm (Cluster)")
    ax2.set_xlabel("Phân Khúc")
    ax2.set_ylabel("Price (Tỷ VNĐ)")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    st.pyplot(fig2)
    
    st.markdown("### Thống kê Diện tích Đất theo từng Phân khúc (Cluster)")
    area_stats = df_cluster.groupby('Nhóm Phân Khúc')['dien_tich_dat'].agg(
        Avg_Area='mean',
        Min_Area='min',
        Max_Area='max'
    ).reset_index()
    
    area_stats['Rank'] = area_stats['Nhóm Phân Khúc'].map(lambda x: labels.index(x) if x in labels else 99)
    area_stats = area_stats.sort_values('Rank').drop(columns=['Rank'])
    
    area_stats_melted = area_stats.melt(id_vars='Nhóm Phân Khúc', var_name='Chỉ số Diện tích', value_name='Diện tích (m2)')
    
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    # A different color palette for Area: Dark Purple, Deep Orange, Light Orange, for instance
    area_palette = {"Avg_Area": "#6f42c1", "Min_Area": "#fd7e14", "Max_Area": "#ffc107"}
    
    sns.barplot(
        data=area_stats_melted, 
        x='Nhóm Phân Khúc', 
        y='Diện tích (m2)', 
        hue='Chỉ số Diện tích', 
        palette=area_palette,
        ax=ax3
    )
    
    ax3.set_title("Trung bình, Nhỏ nhất và Lớn nhất Diện tích đất theo từng Cụm (Cluster)")
    ax3.set_xlabel("Phân Khúc")
    ax3.set_ylabel("Diện tích Đất (m2)")
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    st.pyplot(fig3)
    
    st.markdown("### Thống kê Diện tích Sử dụng theo từng Phân khúc (Cluster)")
    usage_area_stats = df_cluster.groupby('Nhóm Phân Khúc')['dien_tich_su_dung'].agg(
        Avg_Usage='mean',
        Min_Usage='min',
        Max_Usage='max'
    ).reset_index()
    
    usage_area_stats['Rank'] = usage_area_stats['Nhóm Phân Khúc'].map(lambda x: labels.index(x) if x in labels else 99)
    usage_area_stats = usage_area_stats.sort_values('Rank').drop(columns=['Rank'])
    
    usage_area_stats_melted = usage_area_stats.melt(id_vars='Nhóm Phân Khúc', var_name='Chỉ số Diện tích', value_name='Diện tích sử dụng (m2)')
    
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    # A distinct color palette (e.g. Blues or Reds)
    usage_palette = {"Avg_Usage": "#dc3545", "Min_Usage": "#17a2b8", "Max_Usage": "#20c997"}
    
    sns.barplot(
        data=usage_area_stats_melted, 
        x='Nhóm Phân Khúc', 
        y='Diện tích sử dụng (m2)', 
        hue='Chỉ số Diện tích', 
        palette=usage_palette,
        ax=ax4
    )
    
    ax4.set_title("Trung bình, Nhỏ nhất và Lớn nhất Diện tích Sử dụng theo phân khúc")
    ax4.set_xlabel("Phân Khúc")
    ax4.set_ylabel("Diện tích Sử dụng (m2)")
    ax4.grid(axis='y', linestyle='--', alpha=0.7)
    
    st.pyplot(fig4)
    
    
    st.markdown("### Thống kê Số Tầng theo từng Phân khúc (Cluster)")
    floor_stats = df_cluster.groupby('Nhóm Phân Khúc')['tong_so_tang'].agg(
        Avg_Floor='mean',
        Min_Floor='min',
        Max_Floor='max'
    ).reset_index()
    
    floor_stats['Rank'] = floor_stats['Nhóm Phân Khúc'].map(lambda x: labels.index(x) if x in labels else 99)
    floor_stats = floor_stats.sort_values('Rank').drop(columns=['Rank'])
    
    floor_stats_melted = floor_stats.melt(id_vars='Nhóm Phân Khúc', var_name='Chỉ số Số tầng', value_name='Tổng số tầng')
    
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    floor_palette = {"Avg_Floor": "#28a745", "Min_Floor": "#17a2b8", "Max_Floor": "#343a40"}
    
    sns.barplot(
        data=floor_stats_melted, 
        x='Nhóm Phân Khúc', 
        y='Tổng số tầng', 
        hue='Chỉ số Số tầng', 
        palette=floor_palette,
        ax=ax6
    )
    
    ax6.set_title("Trung bình, Nhỏ nhất và Lớn nhất Số tầng nhà theo phân khúc")
    ax6.set_xlabel("Phân Khúc")
    ax6.set_ylabel("Tổng số Tầng")
    ax6.grid(axis='y', linestyle='--', alpha=0.7)
    
    st.pyplot(fig6)
    
    st.markdown("### Thống kê Số Phòng Ngủ theo từng Phân khúc (Cluster)")
    room_stats = df_cluster.groupby('Nhóm Phân Khúc')['so_phong_ngu'].agg(
        Avg_Room='mean',
        Min_Room='min',
        Max_Room='max'
    ).reset_index()
    
    room_stats['Rank'] = room_stats['Nhóm Phân Khúc'].map(lambda x: labels.index(x) if x in labels else 99)
    room_stats = room_stats.sort_values('Rank').drop(columns=['Rank'])
    
    room_stats_melted = room_stats.melt(id_vars='Nhóm Phân Khúc', var_name='Chỉ số Số phòng', value_name='Số phòng ngủ')
    
    fig7, ax7 = plt.subplots(figsize=(12, 6))
    room_palette = {"Avg_Room": "#6c757d", "Min_Room": "#d63384", "Max_Room": "#6610f2"}
    
    sns.barplot(
        data=room_stats_melted, 
        x='Nhóm Phân Khúc', 
        y='Số phòng ngủ', 
        hue='Chỉ số Số phòng', 
        palette=room_palette,
        ax=ax7
    )
    
    ax7.set_title("Trung bình, Nhỏ nhất và Lớn nhất Số phòng ngủ theo phân khúc")
    ax7.set_xlabel("Phân Khúc")
    ax7.set_ylabel("Số phòng ngủ")
    ax7.grid(axis='y', linestyle='--', alpha=0.7)
    
    st.pyplot(fig7)

