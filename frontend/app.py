import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Required for xgboost pipeline unpickling
def convert_to_string(x):
    return x.astype(str)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.backend import (
    load_and_clean_data, 
    lambdas, 
    predict_price, 
    get_anomaly_model,
    get_recommendation_system,
    recommend_houses,
    get_kmeans_pipeline_model
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
    from PIL import Image
    img = Image.open("images/banner_estate.png")
    w, h = img.size
    target_h = int(w / 3.5) # Tạo tỷ lệ Panorama (rộng/cao = 3.5)
    top = (h - target_h) / 2
    bottom = (h + target_h) / 2
    img_cropped = img.crop((0, top, w, bottom))
    st.image(img_cropped, use_container_width=True)
except Exception as e:
    st.title("🏡 Hệ Sinh Thái Bất Động Sản Nâng Cao")

menu = ["Trang chủ", "Gợi ý nhà tương tự", "Phân cụm thị trường"]
choice = st.sidebar.selectbox('🔍 Tính năng chính', menu)

if choice == 'Trang chủ':
    st.markdown("## Chào mừng bạn đến với Hệ sinh thái Bất động sản AI")
    st.markdown("""
    Nền tảng này ứng dụng các mô hình học máy tiên tiến để giúp bạn:
    - 💡 **Gợi ý nhà tương tự**: Tìm kiếm các bất động sản phù hợp dựa trên nội dung (Content-based Filtering).
    - 📊 **Phân cụm thị trường**: Hiểu rõ các phân khúc bất động sản trên thị trường với thuật toán K-Means.
    """)

    st.markdown("---")
    
    # --- THÔNG TIN NHÓM & PHÂN CÔNG ---
    st.subheader("👥 Thực hiện bởi nhóm 3: Nguyễn Huỳnh Duy - Ngô Thị Phương Yến")
    work_df = pd.DataFrame({
        "Hạng mục công việc": ["GUI - Project 1", "GUI - Project 2"],
        "Người phụ trách": ["Ngô Thị Phương Yến", "Nguyễn Huỳnh Duy"]
    })
    st.table(work_df)

    # --- HƯỚNG DẪN SỬ DỤNG CHI TIẾT ---
    st.subheader("📖 Hướng dẫn sử dụng nhanh")
    
    guide_steps = pd.DataFrame({
        "Menu tính năng": ["🏠 Trang chủ", "💡 Gợi ý nhà tương tự", "📊 Phân cụm thị trường"],
        "Mục đích": [
            "Xem thông tin nhóm và hướng dẫn sử dụng",
            "Tìm kiếm các bất động sản tương đồng dựa trên nội dung mô tả",
            "Khám phá các phân khúc nhà trên thị trường bằng AI (K-Means)"
        ],
        "Cách thực hiện": [
            "Đọc thông tin tổng quan",
            "Chọn 1 bất động sản mục tiêu có sẵn ➡️ Xem các nhà tương đồng",
            "Xem phân tích biểu đồ & Dự báo cụm thị trường"
        ]
    })
    st.table(guide_steps)

elif choice == 'Gợi ý nhà tương tự':
    st.markdown('## 🏘️ Khám Phá Bất Động Sản Tương Đồng')
    st.write('Hệ thống cung cấp hai phương thức tìm kiếm và gợi ý bằng Trí tuệ nhân tạo (AI) giúp bạn đưa ra quyết định tốt nhất.')
    tab1, tab2 = st.tabs(['💡 Tư vấn dựa trên 1 Căn Có Sẵn', '🔎 Tìm kiếm tự do & Lọc thông số'])

    with tab1:
        st.markdown("## 💡 Gợi ý Nhà Theo Nội Dung (Content-based)")
        st.write("Thuật toán TF-IDF & Cosine Similarity sẽ tìm kiếm các tin đăng có nội dung miêu tả tương tự nhất với căn nhà bạn đang quan tâm.")
        
        df_rec, tfidf_matrix = get_recommendation_system(df)
        
        # Auto-heal cache if it holds an outdated dataframe missing 'id'
        if 'id' not in df_rec.columns:
            st.cache_resource.clear()
            df_rec, tfidf_matrix = get_recommendation_system(df)
            st.rerun()
        
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
            
            # --- Kịch bản NER bóc tách từ khóa tương đồng (Dictionary & Regex Logic) ---
            dict_phap_ly = ["chính chủ", "sổ hồng", "sổ đỏ", "sổ riêng", "giấy tờ hợp lệ", "hoàn công", "pháp lý rõ ràng", "sổ mâm xôi", "vuông vức"]
            dict_vi_tri = ["gần chợ", "trường học", "bệnh viện", "siêu thị", "trung tâm", "tiện ích", "công viên", "an ninh", "yên tĩnh", "dân trí", "khu sầm uất", "gần đường"]
            dict_dac_diem = ["hẻm xe hơi", "mặt tiền", "nở hậu", "ban công", "lầu", "sân thượng", "giếng trời", "gara", "cấp 4", "biệt thự", "góc", "hai mặt tiền", "thang máy", "nội thất", "trệt", "lửng", "sân vườn", "nhà mới", "vào ở ngay"]
            
            def extract_fixed_phrases(txt, dictionary):
                txt_lower = str(txt).lower()
                return [p for p in dictionary if p in txt_lower]
                
            def get_overlapping_NER(house1_row, house2_row):
                txt1 = str(house1_row['mo_ta']) + " " + str(house1_row.get('tieu_de', ''))
                txt2 = str(house2_row['mo_ta']) + " " + str(house2_row.get('tieu_de', ''))
                tags = []
                
                # 1. Bắt Tên Riêng Địa Danh in Hoa (Location)
                import re
                cap_word = r"[A-ZÂĂÊÔƠƯĐÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]+"
                pattern = f"(?:{cap_word}\s+){{1,4}}{cap_word}"
                
                pn1 = set(re.findall(pattern, txt1))
                pn2 = set(re.findall(pattern, txt2))
                common_pn = pn1.intersection(pn2)
                
                noise_pn = {"Giá Bán", "Liên Hệ", "Chi Tiết", "Khu Vực", "Bất Động Sản", "Diện Tích", "Phòng Ngủ", "Mặt Tiền", "Sổ Hồng", "Nhà Đất", "Thương Lượng", "Sổ Đỏ", "Giá Rẻ", "Giảm Giá", "Bán Gấp"}
                common_pn = [pn for pn in common_pn if pn.strip() not in noise_pn]
                for pn in list(set(common_pn))[:3]:
                    tags.append((pn.strip(), "#e0f7fa", "#006064")) # Xanh dương nhạt đục
                    
                # 2. Đặc trưng Pháp lý
                p1 = extract_fixed_phrases(txt1, dict_phap_ly)
                p2 = extract_fixed_phrases(txt2, dict_phap_ly)
                for p in list(set(p1).intersection(set(p2))):
                    tags.append((p.capitalize(), "#e8f5e9", "#1b5e20")) # Xanh lá cây
                    
                # 3. Vị trí/Tiện ích
                v1 = extract_fixed_phrases(txt1, dict_vi_tri)
                v2 = extract_fixed_phrases(txt2, dict_vi_tri)
                for v in list(set(v1).intersection(set(v2))):
                    tags.append((v.capitalize(), "#fff8e1", "#f57f17")) # Cam nhạt
                    
                # 4. Đặc điểm cấu trúc
                d1 = extract_fixed_phrases(txt1, dict_dac_diem)
                d2 = extract_fixed_phrases(txt2, dict_dac_diem)
                for d in list(set(d1).intersection(set(d2))):
                    tags.append((d.capitalize(), "#f3e5f5", "#4a148c")) # Tím
                    
                # 5. So sánh cột Giá tiền và Diện tích thuật toán (Không Text)
                try:
                    g1 = float(house1_row.get('gia_ban', -99))
                    g2 = float(house2_row.get('gia_ban', -99))
                    if g1 > 0 and g2 > 0 and abs(g1 - g2) <= 1.0:
                        tags.append((f"Giá tương đồng: {g2} tỷ", "#ffebee", "#b71c1c")) # Đỏ
                except: pass
                
                try:
                    dt1 = float(house1_row.get('dien_tich_dat', -99))
                    dt2 = float(house2_row.get('dien_tich_dat', -99))
                    if dt1 > 0 and dt2 > 0 and abs(dt1 - dt2) <= 10.0:
                        dt_val = int(dt2) if dt2.is_integer() else round(dt2, 1)
                        tags.append((f"Diện tích tương đồng: {dt_val}m²", "#e8eaf6", "#1a237e")) # Xanh Indigo đậm
                except: pass
                
                return tags
            # ------------------------------------------
            
            for idx, row in recommendations.iterrows():
                with st.expander(f"📌 {row['tieu_de']} - {row['gia_ban']} Tỷ"):
                    st.write(f"**Vị trí:** {row['dia_chi_cu']}")
                    st.write(f"**Chi tiết:** {row['mo_ta'][:500]}...")
                    
                    # Trích xuất và render Label từ khóa BĐS theo Cấu Trúc Khối HTML
                    kws_tags = get_overlapping_NER(sh, row)
                    if kws_tags:
                        html_tags = ""
                        for text, bg, color in kws_tags:
                            html_tags += f"<span style='background-color:{bg}; color:{color}; border-radius:12px; padding:4px 12px; margin-right:6px; font-size:13px; font-weight:bold; border: 1px solid {color}50; display:inline-block; margin-top:5px;'>{text}</span>"
                        
                        st.markdown(f"<div style='margin-top: 10px; margin-bottom: 5px; border-top: 1px dashed #ccc; padding-top: 10px;'><b>Điểm chung nổi bật:</b><br>{html_tags}</div>", unsafe_allow_html=True)
    

    with tab2:
        st.markdown("## 🔎 Tìm kiếm nhà theo Từ khóa & Bộ lọc")
        st.write("Nhập các từ khóa phân tách bằng dấu phẩy và sử dụng bộ lọc để khoanh vùng chính xác căn nhà bạn mong muốn. Sau đó, máy học AI sẽ mở rộng phân tích các bất động sản tương đồng bên dưới.")
        
        # Session State để liên kết Text Input và Nút Vệ Tinh
        if "search_kw" not in st.session_state:
            st.session_state.search_kw = ""
            
        def append_kw(kw):
            current = st.session_state.search_kw
            if current:
                if kw.lower() not in current.lower():
                    st.session_state.search_kw = current + f", {kw}"
            else:
                st.session_state.search_kw = kw
    
        # Các Nút Từ Khóa Gợi Ý Nhanh
        st.markdown("**💡 Dán nhanh từ khóa nổi bật vào khung Search:**")
        quick_kws = ["sổ hồng riêng", "chính chủ", "mặt tiền", "nhà ngõ hẻm", "gần chợ", "gần trường học", "gần bệnh viện", "yên tĩnh", "Bình Thạnh", "Gò Vấp", "Phú Nhuận"]
        
        # Render Responsive Layout cho các Nút
        chunk1 = quick_kws[:6]
        chunk2 = quick_kws[6:]
        c1 = st.columns(len(chunk1))
        for i, kw in enumerate(chunk1):
            with c1[i]:
                st.button(kw[:12]+(".." if len(kw)>12 else ""), on_click=append_kw, args=(kw,), help=kw, key=f"qk1_{i}")
                
        c2 = st.columns(len(chunk2))
        for i, kw in enumerate(chunk2):
            with c2[i]:
                st.button(kw[:12]+(".." if len(kw)>12 else ""), on_click=append_kw, args=(kw,), help=kw, key=f"qk2_{i}")
                
        # Ô nhập liệu tự do
        st.text_input("Nhập chuỗi từ khóa (Các tiêu chí phải NGĂN CÁCH nhau bởi Dấu Phẩy):", key="search_kw")
        
        st.markdown("---")
        
        # Kéo bộ dữ liệu chuẩn đã có cột ID từ Hệ thống Gợi ý (để link được với AI Recommendation)
        df_search, tfidf_matrix = get_recommendation_system(df)
        
        # Tủ điều khiển Min-Max
        with st.expander("⚙️ BỘ LỌC THÔNG SỐ (Kéo Trượt & Thu Hẹp Giới Hạn)", expanded=True):
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                try:
                    min_p, max_p = float(df_search['gia_ban'].dropna().min()), float(df_search['gia_ban'].dropna().max())
                    p_range = st.slider("💰 Giá (Tỷ đồng)", min_value=0.0, max_value=float(max_p), value=(0.0, float(max_p)), step=0.5)
                except: p_range = (0.0, 1000.0)
                
                try:
                    min_dt, max_dt = float(df_search['dien_tich_dat'].dropna().min()), float(df_search['dien_tich_dat'].dropna().max())
                    dt_range = st.slider("🏞️ Diện tích đất (m²)", min_value=0.0, max_value=float(max_dt), value=(0.0, float(max_dt)), step=5.0)
                except: dt_range = (0.0, 500.0)
                
                try:
                    min_tt, max_tt = int(df_search['tong_so_tang'].dropna().min()), int(df_search['tong_so_tang'].dropna().max())
                    tt_range = st.slider("🏢 Tổng số tầng", min_value=min_tt, max_value=max_tt, value=(min_tt, max_tt), step=1)
                except: tt_range = (1, 10)
                
            with fcol2:
                try:
                    min_ng, max_ng = float(df_search['chieu_ngang'].dropna().min()), float(df_search['chieu_ngang'].dropna().max())
                    ng_range = st.slider("📏 Chiều ngang mặt tiền (m)", min_value=0.0, max_value=float(max_ng), value=(0.0, float(max_ng)), step=0.5)
                except: ng_range = (0.0, 50.0)
                
                try:
                    min_sd, max_sd = float(df_search['dien_tich_su_dung'].dropna().min()), float(df_search['dien_tich_su_dung'].dropna().max())
                    sd_range = st.slider("🏗️ Diện tích sử dụng Sàn (m²)", min_value=0.0, max_value=float(max_sd), value=(0.0, float(max_sd)), step=5.0)
                except: sd_range = (0.0, 1000.0)
                
                try:
                    min_pn, max_pn = int(df_search['so_phong_ngu'].dropna().min()), int(df_search['so_phong_ngu'].dropna().max())
                    pn_range = st.slider("🛏️ Số phòng ngủ", min_value=min_pn, max_value=max_pn, value=(min_pn, max_pn), step=1)
                except: pn_range = (1, 20)
                
        if st.button("🚀 BẮT ĐẦU TRUY QUÉT DỮ LIỆU", type="primary", use_container_width=True):
            st.session_state.do_search = True
            
        if st.session_state.get('do_search', False):
            with st.spinner("Hệ thống đang rà soát dữ liệu đối chiếu đa điều kiện..."):
                # 1. Base Filter (Numeric limits)
                mask = (df_search['gia_ban'] >= p_range[0]) & (df_search['gia_ban'] <= p_range[1])
                if 'dien_tich_dat' in df_search.columns: mask &= (df_search['dien_tich_dat'] >= dt_range[0]) & (df_search['dien_tich_dat'] <= dt_range[1])
                if 'dien_tich_su_dung' in df_search.columns: mask &= (df_search['dien_tich_su_dung'] >= sd_range[0]) & (df_search['dien_tich_su_dung'] <= sd_range[1])
                if 'chieu_ngang' in df_search.columns: mask &= (df_search['chieu_ngang'] >= ng_range[0]) & (df_search['chieu_ngang'] <= ng_range[1])
                if 'tong_so_tang' in df_search.columns: mask &= (df_search['tong_so_tang'] >= tt_range[0]) & (df_search['tong_so_tang'] <= tt_range[1])
                if 'so_phong_ngu' in df_search.columns: mask &= (df_search['so_phong_ngu'] >= pn_range[0]) & (df_search['so_phong_ngu'] <= pn_range[1])
                
                f_df = df_search[mask].copy()
                
                # 2. Keyword Filter (ALL-AND condition)
                raw_input = st.session_state.search_kw
                tokens = [t.strip().lower() for t in raw_input.split(',')] if raw_input else []
                tokens = [t for t in tokens if t] # Xóa khoảng rỗng
                
                if tokens:
                    def is_match(row):
                        txt = str(row.get('mo_ta', '')) + " " + str(row.get('tieu_de', '')) + " " + str(row.get('dia_chi', ''))
                        txt = txt.lower()
                        for t in tokens:
                            if t not in txt:
                                return False # Chỉ cần 1 rule fail -> rụng
                        return True
                    f_df = f_df[f_df.apply(is_match, axis=1)]
                    
                if len(f_df) == 0:
                    st.warning("⚠️ Rất tiếc, bộ lọc của bạn quá khắt khe, không tìm thấy căn nhà nào thỏa mãn TẤT CẢ các chỉ tiêu. Vui lòng mở rộng thông số slide trượt hoặc giảm bớt từ khóa text.")
                else:
                    st.success(f"✅ Đã tìm thấy **{len(f_df)}** căn nhà vượt qua màng lọc khắt khe của bạn.")
                    
                    st.markdown("### 🏆 Trích Chọn Các Nhà Top Đầu")
                    for idx, row in f_df.head(10).iterrows(): # Tối đa 10 matching để chống đơ máy
                        with st.container():
                            st.markdown(f"#### 🏷️ Nhà: {row['tieu_de']}")
                            st.caption(f"📍 **Vị trí:** {row.get('dia_chi', 'Chưa xác định')}")
                            st.write(f"📐 **Thông số:** D.Tích Đất: {row.get('dien_tich_dat', '-')}m² | D.Tích Sàn: {row.get('dien_tich_su_dung', '-')}m² | Ngang {row.get('chieu_ngang', '-')}m | {row.get('tong_so_tang', '-')} lầu | Giá: **{row['gia_ban']} Tỷ**")
                            st.write(f"📝 **Mô tả:** {str(row.get('mo_ta', ''))[:350]}...")
                            
                            btn_col, _ = st.columns([0.4, 0.6])
                            with btn_col:
                                if st.button(f"✨ Quét AI 5 Căn Tương Tự Nhất", key=f"btn_search_ai_{row['id']}"):
                                    st.session_state[f"show_sim_{row['id']}"] = True
                                    
                            # Nút kích hoạt AI Recommendations (Content-based tf-idf lookup)
                            if st.session_state.get(f"show_sim_{row['id']}", False):
                                st.markdown("---")
                                st.markdown(f"**🤖 AI đề xuất 5 Bất Động Sản mang thiết kế/hoàn cảnh tương tự căn `{row['id']}`:**")
                                try:
                                    recommendations = recommend_houses(df_search, tfidf_matrix, row['id'], top_n=5)
                                    
                                    # Clone logic NER Highlighting từ tab Gợi ý
                                    import re
                                    dict_phap_ly = ["chính chủ", "sổ hồng", "sổ đỏ", "sổ riêng", "giấy tờ hợp lệ", "hoàn công", "pháp lý rõ ràng", "sổ vuông", "vuông vức"]
                                    dict_vi_tri = ["gần chợ", "trường học", "bệnh viện", "siêu thị", "trung tâm", "tiện ích", "công viên", "an ninh", "yên tĩnh", "dân trí", "khu sầm uất", "gần đường"]
                                    dict_dac_diem = ["hẻm xe hơi", "mặt tiền", "nở hậu", "ban công", "lầu", "sân thượng", "giếng trời", "gara", "cấp 4", "biệt thự", "góc", "hai mặt tiền", "thang máy", "nội thất", "trệt", "lửng", "sân vườn", "nhà mới", "vào ở ngay"]
                                    
                                    def extract_fixed(txt, dic): return [p for p in dic if p in str(txt).lower()]
                                    def get_ner(h1, h2):
                                        tags = []
                                        t1 = str(h1['mo_ta']) + " " + str(h1.get('tieu_de', ''))
                                        t2 = str(h2['mo_ta']) + " " + str(h2.get('tieu_de', ''))
                                        cw = r"[A-ZÂĂÊÔƠƯĐÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]+"
                                        pn_inter = set(re.findall(f"(?:{cw}\s+){{1,4}}{cw}", t1)).intersection(set(re.findall(f"(?:{cw}\s+){{1,4}}{cw}", t2)))
                                        for pn in list([p for p in pn_inter if p.strip() not in {"Giá Bán", "Liên Hệ", "Chi Tiết", "Khu Vực", "Bất Động Sản", "Diện Tích", "Phòng Ngủ", "Mặt Tiền", "Sổ Hồng", "Nhà Đất"}])[:3]: tags.append((pn.strip(), "#e0f7fa", "#006064"))
                                        for p in set(extract_fixed(t1, dict_phap_ly)).intersection(extract_fixed(t2, dict_phap_ly)): tags.append((p.capitalize(), "#e8f5e9", "#1b5e20"))
                                        for v in set(extract_fixed(t1, dict_vi_tri)).intersection(extract_fixed(t2, dict_vi_tri)): tags.append((v.capitalize(), "#fff8e1", "#f57f17"))
                                        for d in set(extract_fixed(t1, dict_dac_diem)).intersection(extract_fixed(t2, dict_dac_diem)): tags.append((d.capitalize(), "#f3e5f5", "#4a148c"))
                                        try: 
                                            if float(h2.get('gia_ban', -99)) > 0 and abs(float(h1.get('gia_ban', -99)) - float(h2.get('gia_ban', -99))) <= 1.0: tags.append((f"Giá tương đồng: {h2['gia_ban']} tỷ", "#ffebee", "#b71c1c"))
                                        except: pass
                                        try: 
                                            dt2 = float(h2.get('dien_tich_dat', -99))
                                            if dt2 > 0 and abs(float(h1.get('dien_tich_dat', -99)) - dt2) <= 10.0: dt_val = int(dt2) if dt2.is_integer() else round(dt2, 1); tags.append((f"D.Tích tương đồng: {dt_val}m²", "#e8eaf6", "#1a237e"))
                                        except: pass
                                        return tags
    
                                    for r_idx, r_row in recommendations.iterrows():
                                        with st.expander(f"✨ LỰA CHỌN {r_idx+1}: {r_row['tieu_de']} - {r_row['gia_ban']} Tỷ"):
                                            st.write(f"**Vị trí:** {r_row.get('dia_chi_cu', 'Chưa xác định')}")
                                            st.write(f"**Chi tiết:** {str(r_row['mo_ta'])[:500]}...")
                                            kws = get_ner(row, r_row)
                                            if kws:
                                                html = "".join([f"<span style='background-color:{bg}; color:{color}; border-radius:12px; padding:4px 12px; margin-right:6px; font-size:13px; font-weight:bold; border: 1px solid {color}50; display:inline-block; margin-top:5px;'>{t}</span>" for t, bg, color in kws])
                                                st.markdown(f"<div style='margin-top: 10px; margin-bottom: 5px; border-top: 1px dashed #ccc; padding-top: 10px;'><b>Điểm chung với căn chính:</b><br>{html}</div>", unsafe_allow_html=True)
                                except Exception as e:
                                    st.error(f"Lệnh AI Recomendation bị lỗi thiết lập: {e}")
                            
                            st.markdown("<br><hr style='margin: 10px 0px;'><br>", unsafe_allow_html=True)
    

elif choice == 'Phân cụm thị trường':


    st.markdown("## 📊 Phân Cụm Thị Trường (KMeans)")
    st.write("Phương pháp phân cụm KMeans nhóm các căn nhà có đặc điểm tương đương để phân tích, từ đó xác định căn nhà thuộc phân khúc nào: \"Bình dân, sơ cấp, trung cấp hay cao cấp\" để phục vụ cho các nghiệp vụ business đằng sau")



    try:
        pipeline, df_cluster = get_kmeans_pipeline_model()
        
        # 1. Map Clusters to Labels based on Max Price
        max_prices = df_cluster.groupby('prediction')['gia_ban'].max().sort_values()
        sorted_clusters = max_prices.index.tolist()
        labels = ["Bình dân", "Sơ cấp", "Trung cấp", "Cao cấp"]
        label_map = {c: labels[i] if i < len(labels) else f"Phân khúc {i+1}" for i, c in enumerate(sorted_clusters)}
        df_cluster['Nhóm Phân Khúc'] = df_cluster['prediction'].map(label_map)
        
        tab1, tab2, tab3 = st.tabs(["📊 Các biểu đồ thống kê", "🔍 Dự đoán nhà thuộc cụm nào", "📂 Phân cụm dữ liệu mới (CSV)"])
        with tab2:
            st.markdown("### 🔍 Phân loại phân khúc cho căn nhà mới")
            with st.expander("📝 Nhập thông số căn nhà của bạn để xem thuộc phân khúc nào", expanded=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    i_dtd = st.number_input("Diện tích đất (m2)", value=50.0, step=1.0, key="k_dtd")
                    i_dtsd = st.number_input("Diện tích sử dụng (m2)", value=120.0, step=1.0, key="k_dtsd")
                with c2:
                    i_tang = st.number_input("Tổng số tầng", value=2.0, step=1.0, key="k_tang")
                    i_phong = st.number_input("Số phòng ngủ", value=2.0, step=1.0, key="k_phong")
                with c3:
                    i_gia = st.number_input("Giá dự kiến (Tỷ)", value=5.0, step=0.5, key="k_gia")
                
                # --- TẦNG 1: GLOBAL OUTLIER CHECK ---
            limits = {
                "Diện tích đất": (df_cluster['dien_tich_dat'].min(), df_cluster['dien_tich_dat'].max(), i_dtd, "m2"),
                "Diện tích sử dụng": (df_cluster['dien_tich_su_dung'].min(), df_cluster['dien_tich_su_dung'].max(), i_dtsd, "m2"),
                "Tổng số tầng": (df_cluster['tong_so_tang'].min(), df_cluster['tong_so_tang'].max(), i_tang, "tầng"),
                "Số phòng ngủ": (df_cluster['so_phong_ngu'].min(), df_cluster['so_phong_ngu'].max(), i_phong, "phòng"),
                "Giá dự kiến": (df_cluster['gia_ban'].min(), df_cluster['gia_ban'].max(), i_gia, "Tỷ")
            }

            outliers = []
            for name, (vmin, vmax, val, unit) in limits.items():
                if val < vmin or val > vmax:
                    outliers.append(f"- **{name}**: {val} {unit} (Ngưỡng cho phép: {vmin} - {vmax} {unit})")
            
            if outliers:
                st.warning("⚠️ **Cảnh báo**: Trị số nhập vào nằm ngoài khoảng không gian dữ liệu huấn luyện của hệ thống:\n" + "\n".join(outliers))

            if st.button("📐 Hiện kết quả"):
                if outliers:
                    st.error("🚨 Bất động sản này thuộc trường hợp ngoại lệ, cần dùng các công cụ khác để đánh giá hoặc liên hệ nhà tư vấn/ chuyên gia để thảo luận chi tiết !")
                else:
                    try:
                        median_ngang = df_cluster['chieu_ngang'].median() if 'chieu_ngang' in df_cluster.columns else 4.0
                        median_phong = df_cluster['so_phong_ngu'].median()
                        
                        num_cols = ["dien_tich_dat", "dien_tich_su_dung", "tong_so_tang", "so_phong_ngu", "gia_ban"]
                        
                        input_df = pd.DataFrame([{
                            'dien_tich_dat': i_dtd,
                            'dien_tich_su_dung': i_dtsd,
                            'chieu_ngang': median_ngang,
                            'tong_so_tang': i_tang,
                            'so_phong_ngu': i_phong,
                            'gia_ban': i_gia
                        }])
                        
                        features_order = ['dien_tich_dat', 'dien_tich_su_dung', 'chieu_ngang', 'tong_so_tang', 'so_phong_ngu', 'gia_ban']
                        input_df = input_df[features_order]
                        
                        cluster_label = pipeline.predict(input_df)[0]
                        assigned_label = label_map.get(cluster_label, f"Cụm {cluster_label}")
                        
                        # --- TẦNG 2: STATISTICAL MATCHING ---
                        matched_clusters = set()
                        cluster_groups = df_cluster.groupby('Nhóm Phân Khúc')
                        for c_name, group in cluster_groups:
                            c_dt_min, c_dt_max = group['dien_tich_dat'].min(), group['dien_tich_dat'].max()
                            c_dtsd_min, c_dtsd_max = group['dien_tich_su_dung'].min(), group['dien_tich_su_dung'].max()
                            c_tang_min, c_tang_max = group['tong_so_tang'].min(), group['tong_so_tang'].max()
                            c_phong_min, c_phong_max = group['so_phong_ngu'].min(), group['so_phong_ngu'].max()
                            c_gia_min, c_gia_max = group['gia_ban'].min(), group['gia_ban'].max()
                            
                            if (c_dt_min <= i_dtd <= c_dt_max and
                                c_dtsd_min <= i_dtsd <= c_dtsd_max and
                                c_tang_min <= i_tang <= c_tang_max and
                                c_phong_min <= i_phong <= c_phong_max and
                                c_gia_min <= i_gia <= c_gia_max):
                                matched_clusters.add(c_name)
                        
                        # Hợp nhất Model Core + Matched Statistical logic
                        matched_clusters.add(assigned_label)
                        
                        # Sắp xếp hiển thị nhãn theo thứ tự từ thấp đến cao
                        final_labels = []
                        for l in labels:
                            if l in matched_clusters:
                                final_labels.append(l)
                        for c in matched_clusters:
                            if c not in labels:
                                final_labels.append(c)
                                
                        final_result_str = ", ".join([str(x).upper() for x in final_labels])
                        
                        st.success(f"🤖 Dựa trên ML K-Means và phương pháp thống kê đối chiếu, căn nhà này thuộc phân khúc: **{final_result_str}**")
                        
                        summary_data = []
                        for l in final_labels:
                            group = df_cluster[df_cluster['Nhóm Phân Khúc'] == l]
                            if not group.empty:
                                summary_data.append({
                                    "Phân khúc": l,
                                    "Diện tích đất (m2)": f"{group['dien_tich_dat'].min():.1f} - {group['dien_tich_dat'].max():.1f}",
                                    "Diện tích SD (m2)": f"{group['dien_tich_su_dung'].min():.1f} - {group['dien_tich_su_dung'].max():.1f}",
                                    "Số Tầng": f"{group['tong_so_tang'].min():.0f} - {group['tong_so_tang'].max():.0f}",
                                    "Số Phòng Ngủ": f"{group['so_phong_ngu'].min():.0f} - {group['so_phong_ngu'].max():.0f}",
                                    "Giá bán (Tỷ)": f"{group['gia_ban'].min():.1f} - {group['gia_ban'].max():.1f}"
                                })
                        
                        if summary_data:
                            table_df = pd.DataFrame(summary_data).set_index("Phân khúc")
                            st.markdown(f"**📑 Bảng tra cứu giới hạn các đặc trưng của các phân khúc đã dự đoán:**")
                            st.table(table_df)
                            
                    except Exception as e:
                        st.error(f"Lỗi khi chạy dự báo KMeans: {e}.")
        
        st.markdown("---")
        
        with tab1:
            price_stats = df_cluster.groupby('Nhóm Phân Khúc')['gia_ban'].agg(
                **{'Trung bình':'mean', 'Thấp nhất':'min', 'Cao nhất':'max'}
            ).reset_index()
            price_stats['Rank'] = price_stats['Nhóm Phân Khúc'].map(lambda x: labels.index(x) if x in labels else 99)
            price_stats = price_stats.sort_values('Rank').drop(columns=['Rank'])
            price_stats_melted = price_stats.melt(id_vars='Nhóm Phân Khúc', var_name='', value_name='Price (Tỷ VNĐ)')
        
            import matplotlib.pyplot as plt
            import seaborn as sns
        
            fig_kp, ax_kp = plt.subplots(figsize=(10, 5))
            sns.barplot(data=price_stats_melted, x='Nhóm Phân Khúc', y='Price (Tỷ VNĐ)', hue='', palette="viridis", ax=ax_kp)
            ax_kp.set_title("Biểu đồ phân phối giá theo từng phân khúc")
            ax_kp.set_xlabel("Nhóm phân khúc")
            ax_kp.set_ylabel("Giá (tỷ đồng)")
            for container in ax_kp.containers:
                ax_kp.bar_label(container, fmt='%.1f', padding=3)
            ax_kp.set_ylim(0, ax_kp.get_ylim()[1] * 1.15)
            st.pyplot(fig_kp)
        

            area_stats = df_cluster.groupby('Nhóm Phân Khúc')['dien_tich_dat'].agg(
                **{'Trung bình':'mean', 'Thấp nhất':'min', 'Cao nhất':'max'}
            ).reset_index()
            area_stats['Rank'] = area_stats['Nhóm Phân Khúc'].map(lambda x: labels.index(x) if x in labels else 99)
            area_stats = area_stats.sort_values('Rank').drop(columns=['Rank'])
            area_stats_melted = area_stats.melt(id_vars='Nhóm Phân Khúc', var_name='', value_name='Diện tích (m2)')
        
            fig_ka, ax_ka = plt.subplots(figsize=(10, 5))
            sns.barplot(data=area_stats_melted, x='Nhóm Phân Khúc', y='Diện tích (m2)', hue='', palette="magma", ax=ax_ka)
            ax_ka.set_title("Biểu đồ phân phối diện tích đất theo từng phân khúc")
            ax_ka.set_xlabel("Nhóm phân khúc")
            ax_ka.set_ylabel("Diện tích đất (m2)")
            for container in ax_ka.containers:
                ax_ka.bar_label(container, fmt='%.1f', padding=3)
            ax_ka.set_ylim(0, ax_ka.get_ylim()[1] * 1.15)
            st.pyplot(fig_ka)
        

            usage_area_stats = df_cluster.groupby('Nhóm Phân Khúc')['dien_tich_su_dung'].agg(
                **{'Trung bình':'mean', 'Thấp nhất':'min', 'Cao nhất':'max'}
            ).reset_index()
            usage_area_stats['Rank'] = usage_area_stats['Nhóm Phân Khúc'].map(lambda x: labels.index(x) if x in labels else 99)
            usage_area_stats = usage_area_stats.sort_values('Rank').drop(columns=['Rank'])
            usage_area_stats_melted = usage_area_stats.melt(id_vars='Nhóm Phân Khúc', var_name='', value_name='Diện tích sử dụng (m2)')
        
            fig_ku, ax_ku = plt.subplots(figsize=(10, 5))
            usage_palette = {"Trung bình": "#dc3545", "Thấp nhất": "#17a2b8", "Cao nhất": "#20c997"}
            sns.barplot(data=usage_area_stats_melted, x='Nhóm Phân Khúc', y='Diện tích sử dụng (m2)', hue='', palette=usage_palette, ax=ax_ku)
            ax_ku.set_title("Biểu đồ phân phối diện tích sử dụng theo từng phân khúc")
            ax_ku.set_xlabel("Nhóm phân khúc")
            ax_ku.set_ylabel("Diện tích sử dụng (m2)")
            for container in ax_ku.containers:
                ax_ku.bar_label(container, fmt='%.1f', padding=3)
            ax_ku.set_ylim(0, ax_ku.get_ylim()[1] * 1.15)
            st.pyplot(fig_ku)
        

            floor_stats = df_cluster.groupby('Nhóm Phân Khúc')['tong_so_tang'].agg(
                **{'Trung bình':'mean', 'Thấp nhất':'min', 'Cao nhất':'max'}
            ).reset_index()
            floor_stats['Rank'] = floor_stats['Nhóm Phân Khúc'].map(lambda x: labels.index(x) if x in labels else 99)
            floor_stats = floor_stats.sort_values('Rank').drop(columns=['Rank'])
            floor_stats_melted = floor_stats.melt(id_vars='Nhóm Phân Khúc', var_name='', value_name='Tổng số tầng')
        
            fig_kf, ax_kf = plt.subplots(figsize=(10, 5))
            floor_palette = {"Trung bình": "#28a745", "Thấp nhất": "#17a2b8", "Cao nhất": "#343a40"}
            sns.barplot(data=floor_stats_melted, x='Nhóm Phân Khúc', y='Tổng số tầng', hue='', palette=floor_palette, ax=ax_kf)
            ax_kf.set_title("Biểu đồ phân phối số tầng theo từng phân khúc")
            ax_kf.set_xlabel("Nhóm phân khúc")
            ax_kf.set_ylabel("Tổng số tầng")
            for container in ax_kf.containers:
                ax_kf.bar_label(container, fmt='%.1f', padding=3)
            ax_kf.set_ylim(0, ax_kf.get_ylim()[1] * 1.15)
            st.pyplot(fig_kf)
        

            room_stats = df_cluster.groupby('Nhóm Phân Khúc')['so_phong_ngu'].agg(
                **{'Trung bình':'mean', 'Thấp nhất':'min', 'Cao nhất':'max'}
            ).reset_index()
            room_stats['Rank'] = room_stats['Nhóm Phân Khúc'].map(lambda x: labels.index(x) if x in labels else 99)
            room_stats = room_stats.sort_values('Rank').drop(columns=['Rank'])
            room_stats_melted = room_stats.melt(id_vars='Nhóm Phân Khúc', var_name='', value_name='Số phòng ngủ')
        
            fig_kr, ax_kr = plt.subplots(figsize=(10, 5))
            room_palette = {"Trung bình": "#6c757d", "Thấp nhất": "#d63384", "Cao nhất": "#6610f2"}
            sns.barplot(data=room_stats_melted, x='Nhóm Phân Khúc', y='Số phòng ngủ', hue='', palette=room_palette, ax=ax_kr)
            ax_kr.set_title("Biểu đồ phân phối số phòng ngủ theo từng phân khúc")
            ax_kr.set_xlabel("Nhóm phân khúc")
            ax_kr.set_ylabel("Số phòng ngủ")
            for container in ax_kr.containers:
                ax_kr.bar_label(container, fmt='%.1f', padding=3)
            ax_kr.set_ylim(0, ax_kr.get_ylim()[1] * 1.15)
            st.pyplot(fig_kr)
            
        with tab3:
            st.markdown("### 📂 Tải Lên Tập Dữ Liệu Khảo Sát Mới")
            st.write("Tải lên file dữ liệu (.csv) chứa danh sách các căn nhà để máy học tự động dò tìm số phân khúc tối ưu nhất (hỗ trợ bởi thuật toán Elbow & Silhouette) và lập báo cáo thống kê chuyên sâu độc lập.")
            req_cols = ['dien_tich_dat', 'dien_tich_su_dung', 'tong_so_tang', 'so_phong_ngu', 'gia_ban']
            
            st.markdown("**📥 Danh sách Dữ liệu thử nghiệm**")
            import os
            sample_dir = os.path.join("data", "full_sample_data")
            if os.path.exists(sample_dir):
                sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.csv')]
                if sample_files:
                    with st.expander("Bấm để chọn file CSV tải về", expanded=False):
                        for idx, fname in enumerate(sample_files):
                            fpath = os.path.join(sample_dir, fname)
                            with open(fpath, "rb") as file:
                                st.download_button(
                                    label=f"⬇️ Tải {fname}",
                                    data=file,
                                    file_name=fname,
                                    mime="text/csv",
                                    key=f"dl_smp_{idx}"
                                )
                else:
                    st.info("Chưa có file mẫu nào.")
            else:
                st.error("Không tìm thấy folder full_sample_data.")
            st.caption("👉 **Hướng dẫn:** Lưu 1 file bất kỳ ở trên về máy ➔ Sau đó kéo thả file vừa lưu vào ô Upload bên dưới để trải nghiệm ngay chức năng Phân cụm!")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            empty_df = pd.DataFrame(columns=req_cols)
            csv_empty = empty_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Tải mẫu CSV (Không Data)",
                data=csv_empty,
                file_name="mau_trong_kmeans.csv",
                mime="text/csv",
                key="dl_empty_data"
            )
            st.caption("👉 Nhắp tải mẫu trắng chứa sẵn cấu trúc để tự lập bộ dữ liệu khảo sát riêng. Cuối cùng, upload ngược lên khung bên dưới để sử dụng cho phân cụm dữ liệu mới.")
            
            st.markdown("---")
            uploaded_file = st.file_uploader("Chọn file CSV dữ liệu nhà đất", type=["csv"], key="kmeans_upload")
            
            if uploaded_file is not None:
                try:
                    df_new = pd.read_csv(uploaded_file)
                    st.success("Tải dữ liệu thành công! Bản xem trước dữ liệu:")
                    st.dataframe(df_new.head())
                    
                    # Các cột bắt buộc dùng để chạy chuẩn hóa Standard Scaler
                    missing = [c for c in req_cols if c not in df_new.columns]
                    if missing:
                        st.error(f"🚨 File CSV thiếu các cột bắt buộc: {', '.join(missing)}")
                    else:
                        # Tiền xử lý dữ liệu (Loại dòng rỗng và Tách số khỏi chuỗi)
                        df_new = df_new.dropna(subset=req_cols).copy()
                        X_new = df_new[req_cols].copy()
                        
                        for col in req_cols:
                            if X_new[col].dtype == object or X_new[col].dtype.name == 'string':
                                X_new[col] = X_new[col].astype(str).str.replace(',', '.', regex=False)
                                X_new[col] = X_new[col].str.extract(r'(\d+\.?\d*)', expand=False).astype(float)
                        
                        # Chốt Data thuần số
                        X_new = X_new.dropna()
                        df_new = df_new.loc[X_new.index].copy()
                        
                        # Đồng bộ dữ liệu sạch (số hóa) ngược về df_new để vẽ biểu đồ
                        for col in req_cols:
                            df_new[col] = X_new[col]
                        
                        if len(df_new) < 3:
                            st.error("Dữ liệu sau khi làm sạch không có đủ 3 dòng toàn số học để chạy thuật toán. Vui lòng kiểm tra lại file CSV (VD: cột diện tích, giá).")
                            st.stop()
                        
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.cluster import KMeans
                        from sklearn.metrics import silhouette_score
                        
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X_new)
                        
                        st.markdown("---")
                        st.markdown("### 📐 1. Tìm Số Cụm K Tối Ưu")
                        with st.spinner("Đang chạy thuật toán tối ưu hóa Elbow và Silhouette..."):
                            inertia = []
                            sil_scores = []
                            # Chạy thử K từ 2 đến min(10, số sample)
                            max_k_test = min(11, len(df_new))
                            K_range = range(2, max_k_test)
                            
                            for k in K_range:
                                km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
                                preds = km_test.fit_predict(X_scaled)
                                inertia.append(km_test.inertia_)
                                sil_scores.append(silhouette_score(X_scaled, preds))
                            
                            best_k_sil = K_range[sil_scores.index(max(sil_scores))]
                            
                            import matplotlib.pyplot as plt
                            fig_opt, (ax_el, ax_sil) = plt.subplots(1, 2, figsize=(15, 5))
                            
                            ax_el.plot(K_range, inertia, marker='o', color='blue', linestyle='--')
                            ax_el.set_title("Phương pháp Đường Cong Cùi Chỏ (Elbow)")
                            ax_el.set_xlabel("Số cụm (k)")
                            ax_el.set_ylabel("Mức độ phân tán (Inertia)")
                            ax_el.grid(True, linestyle=':', alpha=0.6)
                            
                            ax_sil.plot(K_range, sil_scores, marker='s', color='orange', linestyle='-')
                            ax_sil.set_title("Phương pháp Hình Bóng (Silhouette Score)")
                            ax_sil.set_xlabel("Số cụm (k)")
                            ax_sil.set_ylabel("Điểm Silhouette")
                            ax_sil.grid(True, linestyle=':', alpha=0.6)
                            
                            st.pyplot(fig_opt)
                            st.info(f"💡 Dựa theo thuật toán tối ưu Silhouette, số cụm lý tưởng nhất được máy học đề xuất phân chia là: **{best_k_sil} cụm**.")
                            
                        st.markdown("---")
                        st.markdown("### 🧠 2. Huấn Luyện Cụm & Báo Cáo Thống Kê")
                        chosen_k = st.number_input("Nhập số lượng phân khúc (Cụm K) bạn muốn chia:", min_value=2, max_value=20, value=int(best_k_sil))
                        
                        if st.button("🚀 Bắt đầu Phân Cụm Dữ liệu tải lên"):
                            with st.spinner(f"Đang tiến hành gom nhóm {len(df_new)} căn nhà thành {chosen_k} cụm..."):
                                km_final = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
                                df_new['Cluster_ID'] = km_final.fit_predict(X_scaled)
                                
                                # Sắp xếp nhãn cụm dựa theo cột giá cao nhất
                                max_p = df_new.groupby('Cluster_ID')['gia_ban'].max().sort_values()
                                ranked_clusters = max_p.index.tolist()
                                
                                label_map_new = {c: f"Cụm {i+1}" for i, c in enumerate(ranked_clusters)}
                                df_new['Nhóm Phân Khúc'] = df_new['Cluster_ID'].map(label_map_new)
                                
                                st.success(f"🎉 Đã huấn luyện thành công! Toàn bộ file dữ liệu đã được gán nhãn thành {chosen_k} cụm và được xếp hạng dựa trên mức giá kịch trần.")
                                
                                # Vẽ 5 biểu đồ thống kê
                                st.markdown("### Các biểu đồ thống kê (dựa trên kết quả phân cụm KMeans với bộ dữ liệu đã upload)")
                                labels_new = [f"Cụm {i+1}" for i in range(chosen_k)]
                                
                                import seaborn as sns
                                
                                # Biểu đồ 1: GIÁ
                                p_stats = df_new.groupby('Nhóm Phân Khúc')['gia_ban'].agg(**{'Trung bình':'mean', 'Thấp nhất':'min', 'Cao nhất':'max'}).reset_index()
                                p_stats['Rank'] = p_stats['Nhóm Phân Khúc'].map(lambda x: labels_new.index(x) if x in labels_new else 99)
                                p_stats = p_stats.sort_values('Rank').drop(columns=['Rank'])
                                p_melt = p_stats.melt(id_vars='Nhóm Phân Khúc', var_name='', value_name='Giá (tỷ đồng)')
                                fig_p, ax_p = plt.subplots(figsize=(10, 5))
                                sns.barplot(data=p_melt, x='Nhóm Phân Khúc', y='Giá (tỷ đồng)', hue='', palette="viridis", ax=ax_p)
                                ax_p.set_title("Biểu đồ phân phối giá theo từng cụm phân khúc")
                                ax_p.set_xlabel("Nhóm phân khúc")
                                ax_p.set_ylabel("Giá (tỷ đồng)")
                                for c in ax_p.containers: ax_p.bar_label(c, fmt='%.1f', padding=3)
                                ax_p.set_ylim(0, ax_p.get_ylim()[1] * 1.15)
                                st.pyplot(fig_p)
                                
                                # Biểu đồ 2: ĐẤT
                                a_stats = df_new.groupby('Nhóm Phân Khúc')['dien_tich_dat'].agg(**{'Trung bình':'mean', 'Thấp nhất':'min', 'Cao nhất':'max'}).reset_index()
                                a_stats['Rank'] = a_stats['Nhóm Phân Khúc'].map(lambda x: labels_new.index(x) if x in labels_new else 99)
                                a_stats = a_stats.sort_values('Rank').drop(columns=['Rank'])
                                a_melt = a_stats.melt(id_vars='Nhóm Phân Khúc', var_name='', value_name='Diện tích (m2)')
                                fig_a, ax_a = plt.subplots(figsize=(10, 5))
                                sns.barplot(data=a_melt, x='Nhóm Phân Khúc', y='Diện tích (m2)', hue='', palette="magma", ax=ax_a)
                                ax_a.set_title("Biểu đồ phân phối diện tích đất theo từng cụm phân khúc")
                                ax_a.set_xlabel("Nhóm phân khúc")
                                ax_a.set_ylabel("Diện tích đất (m2)")
                                for c in ax_a.containers: ax_a.bar_label(c, fmt='%.1f', padding=3)
                                ax_a.set_ylim(0, ax_a.get_ylim()[1] * 1.15)
                                st.pyplot(fig_a)
                                
                                # Biểu đồ 3: DIỆN TÍCH SỬ DỤNG
                                u_stats = df_new.groupby('Nhóm Phân Khúc')['dien_tich_su_dung'].agg(**{'Trung bình':'mean', 'Thấp nhất':'min', 'Cao nhất':'max'}).reset_index()
                                u_stats['Rank'] = u_stats['Nhóm Phân Khúc'].map(lambda x: labels_new.index(x) if x in labels_new else 99)
                                u_stats = u_stats.sort_values('Rank').drop(columns=['Rank'])
                                u_melt = u_stats.melt(id_vars='Nhóm Phân Khúc', var_name='', value_name='Diện tích sử dụng (m2)')
                                fig_u, ax_u = plt.subplots(figsize=(10, 5))
                                sns.barplot(data=u_melt, x='Nhóm Phân Khúc', y='Diện tích sử dụng (m2)', hue='', palette={"Trung bình": "#dc3545", "Thấp nhất": "#17a2b8", "Cao nhất": "#20c997"}, ax=ax_u)
                                ax_u.set_title("Biểu đồ phân phối diện tích sử dụng theo từng cụm phân khúc")
                                ax_u.set_xlabel("Nhóm phân khúc")
                                ax_u.set_ylabel("Diện tích sử dụng (m2)")
                                for c in ax_u.containers: ax_u.bar_label(c, fmt='%.1f', padding=3)
                                ax_u.set_ylim(0, ax_u.get_ylim()[1] * 1.15)
                                st.pyplot(fig_u)
                                
                                # Biểu đồ 4: TẦNG
                                f_stats = df_new.groupby('Nhóm Phân Khúc')['tong_so_tang'].agg(**{'Trung bình':'mean', 'Thấp nhất':'min', 'Cao nhất':'max'}).reset_index()
                                f_stats['Rank'] = f_stats['Nhóm Phân Khúc'].map(lambda x: labels_new.index(x) if x in labels_new else 99)
                                f_stats = f_stats.sort_values('Rank').drop(columns=['Rank'])
                                f_melt = f_stats.melt(id_vars='Nhóm Phân Khúc', var_name='', value_name='Tổng số tầng')
                                fig_f, ax_f = plt.subplots(figsize=(10, 5))
                                sns.barplot(data=f_melt, x='Nhóm Phân Khúc', y='Tổng số tầng', hue='', palette={"Trung bình": "#28a745", "Thấp nhất": "#17a2b8", "Cao nhất": "#343a40"}, ax=ax_f)
                                ax_f.set_title("Biểu đồ phân phối số tầng theo từng cụm phân khúc")
                                ax_f.set_xlabel("Nhóm phân khúc")
                                ax_f.set_ylabel("Tổng số tầng")
                                for c in ax_f.containers: ax_f.bar_label(c, fmt='%.1f', padding=3)
                                ax_f.set_ylim(0, ax_f.get_ylim()[1] * 1.15)
                                st.pyplot(fig_f)
                                
                                # Biểu đồ 5: TỔNG SỐ PHÒNG
                                r_stats = df_new.groupby('Nhóm Phân Khúc')['so_phong_ngu'].agg(**{'Trung bình':'mean', 'Thấp nhất':'min', 'Cao nhất':'max'}).reset_index()
                                r_stats['Rank'] = r_stats['Nhóm Phân Khúc'].map(lambda x: labels_new.index(x) if x in labels_new else 99)
                                r_stats = r_stats.sort_values('Rank').drop(columns=['Rank'])
                                r_melt = r_stats.melt(id_vars='Nhóm Phân Khúc', var_name='', value_name='Số phòng ngủ')
                                fig_r, ax_r = plt.subplots(figsize=(10, 5))
                                sns.barplot(data=r_melt, x='Nhóm Phân Khúc', y='Số phòng ngủ', hue='', palette={"Trung bình": "#6c757d", "Thấp nhất": "#d63384", "Cao nhất": "#6610f2"}, ax=ax_r)
                                ax_r.set_title("Biểu đồ phân phối số phòng ngủ theo từng cụm phân khúc")
                                ax_r.set_xlabel("Nhóm phân khúc")
                                ax_r.set_ylabel("Số phòng ngủ")
                                for c in ax_r.containers: ax_r.bar_label(c, fmt='%.1f', padding=3)
                                ax_r.set_ylim(0, ax_r.get_ylim()[1] * 1.15)
                                st.pyplot(fig_r)
                                
                except Exception as ex:
                    st.error(f"Lỗi khi xử lý file tải lên: {ex}")
    except Exception as e:
        st.error(f"Lỗi hiển thị cụm KMeans: {e}")
        
