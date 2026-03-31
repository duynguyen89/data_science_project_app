<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" />
</div>

<h1 align="center">🏡 PHÂN TÍCH BẤT ĐỘNG SẢN VỚI HỌC MÁY</h1>
<p align="center"><i>Dự án Khoa học Dữ liệu (Data Science Final Project) - Đồ án thực hành thu thập, phân tích và áp dụng Học Máy trên bộ dữ liệu Bất động sản.</i></p>

---

## 🌟 TỔNG QUAN DỰ ÁN
Dự án được xây dựng với mục tiêu giải quyết bài toán phức tạp trên thị trường Bất Động Sản (BĐS): Phân tích đặc điểm, mức giá, phát hiện các tin đăng bất thường và đặc biệt là áp dụng Trí tuệ Nhân tạo (Machine Learning) để phân khúc thị trường và tự động gợi ý nhà ở phù hợp với kỳ vọng của khách hàng.

Web App hoàn chỉnh được xây dựng trên bộ khung **Streamlit** mạnh mẽ, tốc độ cao, hỗ trợ tối đa việc trực quan hóa dữ liệu và cung cấp trải nghiệm sử dụng (User Experience) ưu việt thông qua bảng điều khiển Interactive tích hợp.

## 🚀 TÍNH NĂNG CỐT LÕI (CORE FEATURES)

### 1. 💡 Hệ thống Gợi ý Bất Động Sản (AI Recommendations)
**Sử dụng kỹ thuật NLP & Content-based Filtering (TF-IDF & Cosine Similarity)**
*   **Module Chọn nhà có sẵn (Mục Tiêu):** Từ một căn nhà ban đầu trong cơ sở dữ liệu, AI tính toán sự tương đồng ngữ nghĩa về mặt mô tả, đặc điểm địa lý để chắt lọc top 5 căn giống nhất. 
*   **Module Tìm kiếm Hybrid:** Kết hợp bộ lọc thông số khắt khe (Giá, diện tích, tầng) & từ khóa (hẻm, tiện ích).
*   **Trích xuất từ khóa bằng NER:** Sử dụng cơ chế bóc tách Regex và bộ Dictionary chuyên ngành để Highlight ngay lập tức các Điểm chung của kết quả (Giá, Vị trí, Pháp lý, ...).

### 2. 📊 Phân Cụm Thị Trường (Clustering Analytics)
**Quy hoạch dữ liệu với K-Means & Tối ưu hóa đa chi tiết**
*   **Dự đoán Phân khúc:** Nhập thông tin (Diện tích, Tầng, Số phòng, Giá). Hệ thống check bằng Pipeline K-Means và mô hình thống kê học máy để trả về kết quả phân khúc: *Bình dân, Sơ cấp, Trung cấp hay Cao cấp*.
*   **Hỗ Trợ Upluad Dữ liệu Phân Cụm:** Cho phép tải lên tệp tin dạng file gốc `.csv`. Tự động dùng phương pháp Silhouette & Elbow dò `K` số cụm lý tưởng phù hợp dữ liệu => Phân cụm => Plot hàng loạt 5 Biểu đồ Insights ngay trên Web App.

---

## 📂 KIẾN TRÚC MÃ NGUỒN (PROJECT DIRECTORY)

```text
GUI_Group_3/
├── backend/            # Code core tiền xử lý, gán Model & Pipeline cho Data
│   └── backend.py
├── data/               # Các luồng dữ liệu chuẩn bị sẵn phục vụ ML Trainning
│   ├── cleaned_data/   # Dữ liệu sạch, đã xử lý missing & mã hóa số (TF-IDF dataset)
│   └── full_sample_data/ # File khảo sát mẫu cho tính năng Cụm dữ liệu K-Means 
├── files/              # Asset quản lý Data Cleaning
│   └── vietnamese-stopwords.txt, english-vnmese.txt...
├── frontend/           # Toàn bộ mã nguồn cốt lõi UI tương tác, tích hợp mã nguồn thư viện của Streamlit
│   └── app.py          # FILE KHỞI CHẠY CHÍNH CUẢ APP
├── images/             # Nơi lưu Resource Image, Banner cho Frontend
├── models/             # Các model đã qua Training (.pkl export)
│   ├── model_kmeans_pipeline.pkl
│   ├── model_anomaly_detection_IsolationForest.pkl
│   ├── model_gia_nha_xgboost.pkl
│   └── anomaly_stats.pkl
├── requirements.txt    # Các thư viện phụ thuộc cho mã nguồn
└── README.md           # Hướng dẫn chi tiết
```

---

## 🛠️ CÀI ĐẶT & CHẠY THỬ LÊN MÁY LOCAL

Dự án yêu cầu cài đặt **Python 3.8+**. Để cài thử và chạy hệ thống trên máy cá nhân, bạn chỉ cần theo 3 Bước đơn giản:

**Bước 1: Clone kho lưu trữ máy về thiết bị (Git)**
```sh
git clone https://github.com/[TÊN_TAI_KHOAN]/data_science_project.git
cd data_science_project/GUI_Group_3
```

**Bước 2: Chuẩn bị môi trường và tải các thư viện Dependencies**
```sh
pip install -r requirements.txt
```

**Bước 3: Khởi chạy ứng dụng GUI Streamlit**
```sh
streamlit run frontend/app.py
```
👉 *Chương trình sẽ tự động cấp một máy tính ảo và khởi động hiển thị Dashboard trên nền Web (Thường có Port là `http://localhost:8501`).*

---

## 🧑‍💻 NHÓM TÁC GIẢ THỰC HIỆN KHOA HỌC DỮ LIỆU
**Nhóm 3:**
1. **Nguyễn Huỳnh Duy** - Phụ trách Phát triển GUI Ứng Dụng AI Project 2: Recomendation (Content-based), Phân cụm nhà ở (K-Means)
2. **Ngô Thị Phương Yến** - Phụ trách Phát triển GUI Project 1

*Xin chân thành cảm ơn các thầy cô Giảng Viên TTTH trường ĐHKHTN đã nhiệt tình hướng dẫn, hỗ trợ kiến thức về ML & Data Science, tạo tiền đề để chúng em hoàn thành dự án quy mô này.*
