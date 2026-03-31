# !pip install pyvi
# !pip install gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn._core.typing import default
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')
import joblib
import re, os, math, unicodedata, builtins
from scipy import stats
from scipy.special import inv_boxcox
from scipy.stats import skew

from functools import reduce
from itertools import combinations

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, FunctionTransformer, QuantileTransformer
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from bs4 import BeautifulSoup
from pyvi import ViTokenizer
from pyvi.ViTokenizer import tokenize
from gensim import models as gensim_models
from gensim import corpora, similarities
import gensim.similarities as gensim_similarities

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

import streamlit as st
folder_path = 'Data'
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Loại bỏ trùng lặp
data = data.drop_duplicates()
# Print the shape of the dataset (rows, columns)
print("\nDataset shape:")
data.shape

# Display data types and non-null counts for each column
print("\nData types and missing values:")
data.info()

# Show summary statistics for each column (e.g., mean, std, min, max)
print("\nDescriptive statistics:")
data.describe(include='all')

# Check for missing values
print("\nMissing values per column:")
data.isnull().sum()
print("First rows:")
data.head(1)

missing = data.isnull().sum()
missing_pct = (missing / len(data) * 100).round(2)
missing_df = pd.DataFrame({'Số NaN': missing, 'Tỷ lệ (%)': missing_pct})
missing_df = missing_df[missing_df['Số NaN'] > 0].sort_values('Tỷ lệ (%)', ascending=False)
print('Các cột có giá trị thiếu:')
missing_df
print(f"Số dòng còn lại: {len(data)}")
def standardize_price(price_str):
    if pd.isna(price_str): return None
    price_str = str(price_str).lower().replace(',', '.')
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", price_str)
    if not nums: return None
    val = float(nums[0])
    if 'tỷ' in price_str: return val
    if 'triệu' in price_str: return val / 1000
    return val

# Áp dụng chuẩn hóa giá và diện tích
data['gia_ban'] = data['gia_ban'].apply(standardize_price)

cols_to_convert = ['dien_tich', 'dien_tich_dat', 'so_phong_ngu', 'chieu_ngang', 'tong_so_tang']
for col in cols_to_convert:
    data[col] = data[col].astype(str).str.replace('[^0-9,.]', '', regex=True).str.replace(',', '.')
    data[col] = pd.to_numeric(data[col], errors='coerce')

columns_to_convert = [
    'gia_ban', 'don_gia', 'dien_tich', 'dien_tich_dat', 'dien_tich_su_dung', 'gia_m2', 'so_phong_ngu',
    'so_phong_ve_sinh', 'chieu_ngang', 'chieu_dai'
]

for col_name in columns_to_convert:
    # Chuyển sang chuỗi, loại bỏ các ký tự không phải số, dấu phẩy, dấu chấm
    # Thay dấu phẩy thành dấu chấm
    # Chuyển sang kiểu số (float), các giá trị không hợp lệ sẽ tự động thành NaN (tương đương null)
    data[col_name] = (
        data[col_name]
        .astype(str)
        .str.replace(r'[^0-9,.]', '', regex=True)
        .str.replace(',', '.')
    )

    # Sử dụng pd.to_numeric với errors='coerce' để chuyển chuỗi rỗng hoặc lỗi thành NaN
    data[col_name] = pd.to_numeric(data[col_name], errors='coerce')

# Hiển thị 3 dòng đầu
data[columns_to_convert].head(3)
def clean_invalid_dimensions_pandas(df):
    # Tạo biến tạm để tính toán
    df['dien_tich_tmp'] = np.where(
        df['chieu_ngang'].notna() & df['chieu_dai'].notna(), # Điều kiện (When)
        df['chieu_ngang'] * df['chieu_dai'],                 # Kết quả nếu True
        0                                                    # Kết quả nếu False
    )

    # Điều kiện 1: chieu_ngang > dien_tich
    cond1 = df['chieu_ngang'] > df['dien_tich']

    # Điều kiện 2: sai lệch diện tích > 0 (chỉ kiểm tra khi cả 3 cột not null)
    cond2 = (df['chieu_ngang'].notna()) & \
            (df['chieu_dai'].notna()) & \
            (df['dien_tich'].notna()) & \
            (abs(df['dien_tich'] - df['dien_tich_tmp']) > 0)

    # Nếu vi phạm bất kỳ điều kiện nào (cond1 HOẶC cond2) -> Gán NaN
    df['chieu_ngang'] = np.where(cond1 | cond2, np.nan, df['chieu_ngang'])

    # Loại bỏ cột tạm dien_tich_tmp để làm sạch DataFrame kết quả
    df = df.drop(columns=['dien_tich_tmp'])
    return df

# Sử dụng:
data = clean_invalid_dimensions_pandas(data)
data.info()
# Lọc các cột có kiểu dữ liệu là số (tương đương double, integer trong Spark)
numeric_df = data.select_dtypes(include=['number'])

# Đếm các giá trị bằng 0 trong các cột này
zero_and_negative_count = (numeric_df <= 0).sum()

# Hiển thị kết quả
zero_and_negative_count.to_frame(name='zero_and_negative_count').T
numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

print("Số lượng giá trị Null trong các cột số:")
data[numeric_cols].isnull().sum()
def visualize_correlation(df):
    #
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    corr_df = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Numeric Features')
    plt.show()
visualize_correlation(data)
# Danh sách các cột cần loại bỏ
cols_to_drop = ['don_gia', 'dien_tich', 'so_phong_ve_sinh', 'gia_m2', 'dien_thoai', 'ma_can', 'ten_phan_khu_lo', 'bieu_do_gia']

# Sử dụng drop với axis=1 để xóa cột
# errors='ignore' giúp code không bị lỗi nếu một trong các cột đã bị xóa trước đó
data = data.drop(columns=cols_to_drop, errors='ignore')

# Hiển thị 3 dòng đầu (tương đương .show(3))
data.head(3)
# Trích xuất dia_chi_cu
# Regex: (Phường\s+[^,]+,\s+Quận\s+[^,]+)
data['dia_chi_cu'] = data['dia_chi'].str.extract(r'(Phường\s+[^,]+,\s+Quận\s+[^,]+)', expand=False)

# Trích xuất dia_chi_moi
# Regex: \((Phường\s+[^,]+)
data['dia_chi_moi'] = data['dia_chi'].str.extract(r'\((Phường\s+[^,]+)', expand=False)

# Xử lý các trường hợp không tìm thấy (NaN hoặc chuỗi rỗng)
# Trong Pandas, nếu không khớp Regex, kết quả trả về là NaN (tương đương null)
dia_chi_cols = ['dia_chi_cu', 'dia_chi_moi']
for col_name in dia_chi_cols:
    # Thay thế cả NaN và chuỗi rỗng bằng 'Chưa xác định'
    data[col_name] = data[col_name].replace(['', np.nan], 'Chưa xác định')

# Loại bỏ cột gốc và hiển thị kết quả
data = data.drop(columns=['dia_chi'])

data[['dia_chi_cu', 'dia_chi_moi']].head(3)
# Lọc các dòng có 'dien_tich_dat' là Null
null_data = data[data['dien_tich_dat'].isna()]

# In kết quả dạng bảng
if null_data.empty:
    print("Không có dòng nào bị Null trong cột dien_tich_dat.")
else:
    print(f"Tìm thấy {len(null_data)} dòng bị Null:")
    display(null_data.head(10))
# Xóa null các dòng invalid
data = data.dropna(subset=['dien_tich_dat'])
null_counts_df = data.isnull().sum().to_frame(name='null_counts').T
print("Bảng thống kê số lượng Null sau khi xóa dòng invalid:")
display(null_counts_df)
def extract_floor_count(text):
    if not text or not isinstance(text, str):
        return None

    text_lower = text.lower()

    lau_variants = ['lầu', 'lẩu', 'làu', 'lâù', 'lâu']
    lung_variants = ['lửng', 'lững', 'lừng', 'lữn', 'lưng', 'lủng']

    lau_pattern = '|'.join(lau_variants)
    lung_pattern = '|'.join(lung_variants)

    tang_pattern = r'(\d+)\s*tầng'
    tang_matches = re.findall(tang_pattern, text_lower)

    if tang_matches:
        return int(tang_matches[0])

    floor_pattern = fr'(\d+)\s*({lau_pattern})'
    mezz_pattern = fr'(\d+)\s*({lung_pattern})'

    floor_matches = re.findall(floor_pattern, text_lower)
    mezz_matches = re.findall(mezz_pattern, text_lower)

    floor_numbers = set([match[0] for match in floor_matches])
    mezz_numbers = set([match[0] for match in mezz_matches])

    has_lau = any(variant in text_lower for variant in lau_variants)
    has_lung = any(variant in text_lower for variant in lung_variants)

    if floor_numbers or mezz_numbers or has_lau or has_lung:
        total_floors = 1
        for f in floor_numbers:
            total_floors += int(f)
        for m in mezz_numbers:
            total_floors += int(m)

        if has_lau and not floor_numbers:
            total_floors += 1
        if has_lung and not mezz_numbers:
            total_floors += 1

        return total_floors
    return None

# Tạo 2 cột mới
data['mo_ta_so_tang'] = data['mo_ta'].apply(extract_floor_count)
data['tieu_de_so_tang'] = data['tieu_de'].apply(extract_floor_count)

# Imputing

# Case 1: Nếu tong_so_tang, tieu_de_so_tang, mo_ta_so_tang đều null -> gán bằng 1
# Sử dụng .isna() và .loc để gán giá trị có điều kiện
mask_all_null = data['tong_so_tang'].isna() & \
                data['tieu_de_so_tang'].isna() & \
                data['mo_ta_so_tang'].isna()
data.loc[mask_all_null, 'tong_so_tang'] = 1

# Case 2: Nếu tong_so_tang bị null, lấy giá trị từ mo_ta hoặc tieu_de (Coalesce)
data['tong_so_tang'] = data['tong_so_tang'].fillna(data['mo_ta_so_tang'])
data['tong_so_tang'] = data['tong_so_tang'].fillna(data['tieu_de_so_tang'])

# Hiển thị kết quả
cols_to_show = ["tieu_de", "tieu_de_so_tang", "mo_ta", "mo_ta_so_tang", "tong_so_tang"]
data[cols_to_show].head(10)
count_high_floors = data[data['tong_so_tang'] > 8].shape[0]
print(f"Số lượng căn nhà có trên 8 tầng: {count_high_floors}")
data = data[data['tong_so_tang'] <= 8]
print(f"Số dòng còn lại: {len(data)}")
# Xác định điều kiện cho training_data và predict_data
training_condition = (
    data['dien_tich_su_dung'].notna() &
    data['dien_tich_dat'].notna() &
    data['tong_so_tang'].notna()
)

predict_condition = (
    data['dien_tich_su_dung'].isna() &
    data['dien_tich_dat'].notna() &
    data['tong_so_tang'].notna()
)

# Tạo tập Train và tập Predict
training_data = data[training_condition].copy()
predict_data = data[predict_condition].copy()

print(f"Total rows: {len(data)}")
print(f"Rows in training_data: {len(training_data)}")
print(f"Rows in predict_data: {len(predict_data)}")

# Chuẩn bị Feature (X) và Target (y)
feature_cols = ["dien_tich_dat", "tong_so_tang"]
X_train = training_data[feature_cols]
y_train = training_data["dien_tich_su_dung"]
X_predict = predict_data[feature_cols]

# Khởi tạo và huấn luyện model Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Dự đoán giá trị còn thiếu
if not predict_data.empty:
    y_pred = lr.predict(X_predict)

    # Chặn các giá trị âm (Capping): nếu < 0 thì lấy giá trị của dien_tich_dat
    # Tương đương logic: when(col("prediction") < 0, col("dien_tich_dat"))
    y_pred_capped = np.where(y_pred < 0, predict_data['dien_tich_dat'], y_pred)

    # Gán giá trị dự đoán vào cột dien_tich_su_dung của predict_data
    predict_data['dien_tich_su_dung'] = y_pred_capped

# Gộp dữ liệu lại (Union)
# Những dòng không thỏa mãn cả 2 điều kiện trên (nếu có) cũng cần được giữ lại
other_data = data[~(training_condition | predict_condition)]

data_imputed = pd.concat([training_data, predict_data, other_data], axis=0).reset_index(drop=True)

# Kiểm tra lại số lượng Null
print("\nSố lượng Null còn lại trong cột 'dien_tich_su_dung':")
print(data_imputed['dien_tich_su_dung'].isna().sum())

# Hiển thị mẫu kết quả
print("\nMẫu dữ liệu sau khi điền khuyết:")
display(predict_data[feature_cols + ['dien_tich_su_dung']].head(10))

# Cập nhật lại biến data chính
data = data_imputed
# Tính toán lại chieu_ngang dựa trên dien_tich_dat và chieu_dai
# Điều kiện: diện tích đất và chiều dài có giá trị, nhưng chiều ngang bị thiếu (Null)
mask = (data['dien_tich_dat'].notna()) & \
       (data['chieu_ngang'].isna()) & \
       (data['chieu_dai'].notna()) & \
       (data['chieu_dai'] > 0) & \
       (data['dien_tich_dat'] > 0)

# Áp dụng công thức: chiều ngang = diện tích / chiều dài
data.loc[mask, 'chieu_ngang'] = data['dien_tich_dat'] / data['chieu_dai']

# Đếm số lượng Null ở tất cả các cột và hiển thị dạng bảng (tương đương .show())
null_summary = data.isnull().sum().to_frame(name='null_count').T

print("Bảng thống kê số lượng Null sau khi tính toán chiều ngang:")
display(null_summary)
# --- Bước 1: Tính tỷ lệ ngang / dài trung vị (Median Ratio) ---
# Lọc các hàng có đủ ngang và dài (>0) để tính tỷ lệ mẫu
valid_mask = (data['chieu_ngang'].notna()) & (data['chieu_dai'] > 0)
ratio_ngang_dai = data.loc[valid_mask, 'chieu_ngang'] / data.loc[valid_mask, 'chieu_dai']

# Tính median (tương đương percentile_approx 0.5)
median_ratio = ratio_ngang_dai.median()

# Fallback nếu không tính được median
if pd.isna(median_ratio) or median_ratio <= 0:
    median_ratio = 0.3

print(f"Median Ratio (Ngang/Dài) tính được: {median_ratio:.4f}")

# --- Bước 2: Impute cho các hàng thiếu cả ngang + dài nhưng có dien_tich_dat ---
# Điều kiện: Cả ngang và dài đều Null, nhưng diện tích đất thì có
impute_mask = (data['chieu_ngang'].isna()) & \
              (data['chieu_dai'].isna()) & \
              (data['dien_tich_dat'].notna())

# Tính toán giá trị tạm thời
# Công thức: diện tích = ngang * dài => diện tích = (dài * ratio) * dài => dài = sqrt(diện tích / ratio)
# Ngược lại: ngang = sqrt(diện tích * ratio)
temp_ngang = np.sqrt(data.loc[impute_mask, 'dien_tich_dat'] * median_ratio)
temp_dai = data.loc[impute_mask, 'dien_tich_dat'] / temp_ngang

# Cập nhật vào dataframe gốc (tương đương coalesce)
data.loc[impute_mask, 'chieu_ngang'] = temp_ngang
data.loc[impute_mask, 'chieu_dai'] = temp_dai

# --- Bước 3: Kiểm tra lại kết quả ---
null_summary = data.isnull().sum().to_frame(name='null_count').T
print("\nSố lượng Null sau khi xử lý ngang/dài đồng thời:")
display(null_summary)
# Điền giá trị khuyết cho các cột phân loại bằng dictionary
fill_values = {
    'giay_to_phap_ly': 'Chưa xác định',
    'tinh_trang_noi_that': 'Chưa xác định',
    'dac_diem': 'Hiện trạng khác'
}

data = data.fillna(value=fill_values)

# Kiểm tra lại số lượng Null của tất cả các cột dạng bảng (tương đương .show())
null_summary = data.isnull().sum().to_frame(name='null_count').T

print("Bảng thống kê số lượng Null sau khi fillna:")
display(null_summary)
def impute_with_distribution_pandas(df, col='huong_cua_chinh', random_state=42):
    """
    Điền null bằng random sample theo distribution thực tế sử dụng Pandas & Numpy.
    """
    # Tính toán phân phối xác suất (tương đương value_counts + prob trong Spark)
    # normalize=True sẽ tự động chia cho tổng số lượng non-null
    dist = df[col].value_counts(normalize=True)

    values = dist.index.tolist()
    probs = dist.values.tolist()

    # Xác định các vị trí bị Null
    is_null = df[col].isna()
    null_count = is_null.sum()

    if null_count > 0:
        # Tạo mẫu ngẫu nhiên dựa trên phân phối đã tính
        rng = np.random.default_rng(random_state)
        random_values = rng.choice(values, size=null_count, p=probs)

        # Điền các giá trị ngẫu nhiên vào đúng các vị trí Null
        df.loc[is_null, col] = random_values

    return df

# Thực hiện điền khuyết
data = impute_with_distribution_pandas(data)

# Kiểm tra lại số lượng Null của tất cả các cột
null_summary = data.isnull().sum().to_frame(name='null_count').T

print("Bảng thống kê số lượng Null sau khi điền khuyết theo phân phối:")
display(null_summary)
def sanitize_string(s):
    if s is None or pd.isna(s): # Kiểm tra thêm pd.isna cho chắc chắn trong Pandas
        return None

    s = str(s).lower()

    # Loại bỏ dấu tiếng Việt
    s = unicodedata.normalize('NFKD', s)
    s = "".join([c for c in s if not unicodedata.combining(c)])
    s = s.replace('đ', 'd').replace('Đ', 'D')

    # Thay thế ký tự đặc biệt và khoảng trắng
    s = re.sub(r'[^\w\s]', '_', s)
    s = re.sub(r'[\s,:\\[\]{}]+', '_', s)

    # Dọn dẹp dấu gạch dưới thừa
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')

    return s

# Danh sách các cột phân loại
categorical_cols = [
    'loai_hinh', 'giay_to_phap_ly', 'tinh_trang_noi_that',
    'huong_cua_chinh', 'dac_diem', 'dia_chi_cu', 'dia_chi_moi'
]

# Áp dụng cho các cột trong danh sách bằng .apply()
# Pandas cho phép lọc các cột tồn tại rất dễ dàng
existing_cols = [c for c in categorical_cols if c in data.columns]

for col_name in existing_cols:
    data[col_name] = data[col_name].apply(sanitize_string)

# Hiển thị mẫu dữ liệu sau khi làm sạch
print("Mẫu dữ liệu sau khi lower case, bỏ dấu và ký tự đặc biệt:")
display(data[existing_cols].head(5))
# Danh sách các cột cần giữ lại
selected_columns = ['tieu_de', 'mo_ta', 'dien_tich_dat', 'dien_tich_su_dung', 'chieu_ngang', \
                    'tong_so_tang', 'so_phong_ngu', 'loai_hinh', 'giay_to_phap_ly', \
                    'tinh_trang_noi_that', 'huong_cua_chinh', 'dac_diem', 'dia_chi_cu', 'dia_chi_moi', 'gia_ban']

data = data[selected_columns].copy()
data[data['gia_ban'] == data['gia_ban'].max()]
len(data[data['gia_ban'] > 100])
data[data['gia_ban'] > 100]
data_1 = data[(data['gia_ban'] > 50) & (data['gia_ban'] < 100)]
data_pr2_1 = data.copy()
# Hiển thị cấu trúc dữ liệu và 5 dòng đầu tiên để kiểm tra
print(f"Kích thước dữ liệu sau khi chọn lọc: {data.shape}")
data.head(5)
# Kiểm tra lại số lượng Null của tất cả các cột
null_summary = data.isnull().sum().to_frame(name='null_count').T

print("Bảng thống kê số lượng Null sau khi điền khuyết theo phân phối:")
display(null_summary)
visualize_correlation(data)
def visualize_skewness_pandas(df):

    # Xác định danh sách cột và số lượng
    columns = df.select_dtypes(include=['number']).columns.tolist()
    num_cols = len(columns)
    num_columns_grid = 3 # Số cột trong lưới biểu đồ (layout)
    num_rows_grid = math.ceil(num_cols / num_columns_grid) # Tự động tính số hàng cần thiết

    # Cấu hình kích thước biểu đồ tổng thể
    plt.figure(figsize=(10, 8))

    for i, column in enumerate(columns, start=1):
        plt.subplot(num_rows_grid, num_columns_grid, i)

        # Vẽ histogram kết hợp đường KDE (biểu thị mật độ phân phối)
        sns.histplot(df[column], kde=True)

        plt.title(f"Distribution of {column}", fontsize=12)
        plt.xlabel("") # Ẩn label trục X để biểu đồ thoáng hơn
        plt.ylabel("Frequency")

    # Tối ưu khoảng cách giữa các biểu đồ con
    plt.tight_layout()
    plt.show()
visualize_skewness_pandas(data)
def visualize_outliers_pandas(df):

    # Xác định các cột số (numeric)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Tự động tính toán lưới biểu đồ (layout)
    num_plots = len(numeric_cols)
    num_cols_grid = 3 # Số cột biểu đồ mỗi hàng
    num_rows_grid = math.ceil(num_plots / num_cols_grid)

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 8))

    for i, column in enumerate(numeric_cols):
        plt.subplot(num_rows_grid, num_cols_grid, i + 1)

        # Vẽ Boxplot
        sns.boxplot(y=df[column])

        plt.title(f'Box Plot of {column}', fontsize=12)
        plt.ylabel('')

    # Hiển thị
    plt.tight_layout()
    plt.show()
visualize_outliers_pandas(data)
def trim_outliers(df, columns, lower_percentile=0.004, upper_percentile=0.996):
    """
    Xử lý outlier bằng phương pháp Trimming (Loại bỏ dòng) sử dụng Pandas.
    - columns: Danh sách các cột cần xử lý
    - lower_percentile: Ngưỡng dưới (0.0 đến 1.0)
    - upper_percentile: Ngưỡng trên (0.0 đến 1.0)
    """
    df_initial_count = len(df)

    # Tạo một bản sao để tránh SettingWithCopyWarning
    df_trimmed = df.copy()

    for col_name in columns:
        if col_name not in df_trimmed.columns:
            print(f"Warning: Column '{col_name}' not found in DataFrame. Skipping.")
            continue

        # Ensure the column is numeric before calculating quantiles
        if df_trimmed[col_name].dtype == 'object':
            df_trimmed[col_name] = pd.to_numeric(df_trimmed[col_name], errors='coerce')

        # Drop rows where the current column is NaN, as quantiles cannot be calculated reliably
        df_trimmed_for_quantile = df_trimmed.dropna(subset=[col_name])

        if len(df_trimmed_for_quantile) == 0:
            print(f"Warning: Column '{col_name}' became entirely NaN after numeric conversion. Cannot calculate quantiles.")
            continue

        # Tính toán giá trị tại các ngưỡng bách phân vị
        lower_limit = df_trimmed_for_quantile[col_name].quantile(lower_percentile)
        upper_limit = df_trimmed_for_quantile[col_name].quantile(upper_percentile)

        if pd.isna(lower_limit) or pd.isna(upper_limit):
            print(f"Warning: Quantile limits for '{col_name}' are NaN [{lower_limit} - {upper_limit}]. Skipping trimming for this column.")
            continue

        # Xác định các dòng là outlier để thống kê
        outliers = df_trimmed[(df_trimmed[col_name] < lower_limit) |
                              (df_trimmed[col_name] > upper_limit)]

        outlier_count = len(outliers)
        per_cent = (outlier_count / df_initial_count) * 100

        print(f"Cột {col_name}: Ngưỡng lọc [{lower_limit:.4f} - {upper_limit:.4f}]")
        print(f"Tỉ lệ loại bỏ: {per_cent:.4f}%")

        # Loại bỏ (Filter) các dòng vượt quá ngưỡng
        df_trimmed = df_trimmed[(df_trimmed[col_name] >= lower_limit) &
                                (df_trimmed[col_name] <= upper_limit)]

    print(f"Tổng số dòng sau khi lọc: {len(df_trimmed)} (Giảm {df_initial_count - len(df_trimmed)} dòng)")

    return df_trimmed
# Áp dụng trim outlier
lower_percentile=0.004
upper_percentile=0.8
cols_to_fix = ["dien_tich_su_dung", "chieu_ngang", "gia_ban"]
data = trim_outliers(data, cols_to_fix, lower_percentile, upper_percentile)
visualize_outliers_pandas(data)
data_pr2_2 = data.copy()
def analyze_numerical_skewness_pandas(df, skew_threshold=0.5):
    print("="*80)
    print("NUMERICAL VARIABLES SKEWNESS ANALYSIS (PANDAS)")
    print("="*80)

    # Lấy các cột số
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    results = {}

    for num_col in numeric_cols:
        # Tính skewness (Pandas bỏ qua NaN mặc định)
        skew_val = df[num_col].skew()

        if pd.isna(skew_val):
            skew_level = "N/A"
        elif abs(skew_val) <= 0.5:
            skew_level = "NORMAL (Nearly symmetric)"
        elif abs(skew_val) <= 1.0:
            skew_level = "MODERATE (Moderately skewed)"
        else:
            skew_level = "HIGH (Highly skewed)"

        results[num_col] = {
            "skewness": skew_val,
            "skew_level": skew_level,
            "needs_transform": abs(skew_val) > skew_threshold if not pd.isna(skew_val) else False
        }

        print(f"\n📊 {num_col}:")
        print(f"   Skewness: {skew_val:.4f} ({skew_level})")
        print(f"   Needs Transform: {'✅ YES' if results[num_col]['needs_transform'] else '❌ NO'}")

    return results
def transform_numerical_variables_pandas(df, transform_plan):
    df_transformed = df.copy()
    transformation_log = {}
    best_lambda = 1 # default if normal distribution
    best_lambda_gia_ban = 1 # Initialize best_lambda_gia_ban here with a default value

    print("\n" + "="*80)
    print("APPLYING TRANSFORMATIONS")
    print("="*80)

    for col_name, method in transform_plan.items():
        if method == "none" or col_name not in df.columns:
            continue

        try:
            if method == "log":
                df_transformed[f"{col_name}_log"] = np.log(df[col_name])
                new_col = f"{col_name}_log"

            elif method == "sqrt":
                df_transformed[f"{col_name}_sqrt"] = np.sqrt(df[col_name])
                new_col = f"{col_name}_sqrt"

            elif method == "boxcox":
                df_transformed[f"{col_name}_boxcox"], best_lambda = stats.boxcox(df[col_name])
                new_col = f"{col_name}_boxcox"
                if new_col == 'gia_ban_boxcox':
                    best_lambda_gia_ban = best_lambda

            # Kiểm tra skewness sau khi biến đổi
            skew_after = df_transformed[new_col].skew()
            transformation_log[col_name] = {"method": method, "new_col": new_col, "skew_after": skew_after}
            print(f"✅ Applied {method.upper()} transform to {col_name} → {new_col}")
            print(f"   Skewness after: {skew_after:.4f}")
            print(f"   Best_lambda: {best_lambda}")

        except Exception as e:
            print(f"❌ Error transforming {col_name}: {e}")

    return df_transformed, best_lambda_gia_ban
def complete_skew_transform_pipeline_pandas(df, numerical_skew_threshold=0.5):
    # Phân tích
    skew_results = analyze_numerical_skewness_pandas(df, numerical_skew_threshold)

    # Lập kế hoạch biến đổi
    transform_plan = {}
    for col_name, stats in skew_results.items():
        if stats.get("needs_transform"):
            skew_val = abs(stats["skewness"])
            if skew_val > 2.0: transform_plan[col_name] = "log"
            elif skew_val > 1.0: transform_plan[col_name] = "sqrt"
            elif skew_val > 0.5: transform_plan[col_name] = "boxcox"

    # Initialize df_processed and best_lambda_gia_ban before the conditional block
    df_processed = df.copy()
    # best_lambda_gia_ban = 1

    # Thực thi
    if transform_plan:
        df_processed, best_lambda_gia_ban = transform_numerical_variables_pandas(df, transform_plan)
    else:
        print("No transformations needed.")

    print("\n" + "="*80)
    print(f"Original columns: {len(df.columns)}")
    print(f"New columns: {len(df_processed.columns)}")

    return df_processed, best_lambda_gia_ban

# Sử dụng
df_transformed, best_lambda_gia_ban = complete_skew_transform_pipeline_pandas(data)
print(f"Kích thước dữ liệu sau transform: {df_transformed.shape}")
print(f"Best_lambda_gia_ban: {best_lambda_gia_ban}")
df_transformed.head(5)
visualize_skewness_pandas(df_transformed)
visualize_outliers_pandas(df_transformed)
data_pr2_3 = df_transformed.copy()
df_transformed[['gia_ban']].min()
def get_numerical_for_ml(df):
    """
    Lấy danh sách các cột số cho ML, ưu tiên các cột đã được biến đổi (_log, _sqrt, _boxcox)
    và loại bỏ hoàn toàn các cột liên quan đến 'gia_ban'.
    """
    # Lấy tất cả các cột có kiểu dữ liệu là số (int, float, long, v.v.)
    all_numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Xác định các cột đã được biến đổi bằng Regex
    pattern = r"^(.*)(_log|_sqrt|_boxcox)$"
    transformed_map = {}

    for col in all_numerical_cols:
        match = re.match(pattern, col)
        if match:
            root_name = match.group(1)
            transformed_map[root_name] = col

    # Logic lọc và ưu tiên
    numerical_cols_for_ml = []
    seen_roots = set()

    for col in all_numerical_cols:
        match = re.match(pattern, col)
        # Nếu là cột biến đổi thì lấy root, nếu không thì lấy chính nó
        root = match.group(1) if match else col

        # Kiểm tra nếu root chứa "gia_ban" thì bỏ qua hoàn toàn (Target leakage)
        if "gia_ban" in root:
            continue

        if root not in seen_roots:
            # Nếu gốc này có bản biến đổi trong map, lấy bản biến đổi đó
            if root in transformed_map:
                numerical_cols_for_ml.append(transformed_map[root])
            else:
                # Nếu không có bản biến đổi, lấy tên gốc
                numerical_cols_for_ml.append(root)

            seen_roots.add(root)

    return numerical_cols_for_ml
numerical_features = get_numerical_for_ml(df_transformed)
categorical_features = df_transformed.select_dtypes(include=['object']).drop(['tieu_de', 'mo_ta'], axis=1).columns.tolist()
# Pipeline xử lý số: Điền khuyết (Imputer) -> Chuẩn hóa (RobustScaler)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)))
])
# Helper function to convert to string (replaces lambda for pickling compatibility)
def convert_to_string(x):
    return x.astype(str)

# Redefine categorical_transformer to include a string conversion step
categorical_transformer_fixed = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
    ('to_string', FunctionTransformer(convert_to_string, validate=False)), # Use the named function here
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Redefine preprocessor with the fixed categorical_transformer
preprocessor_fixed = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer_fixed, categorical_features)
    ])
# normalized_model = Pipeline(steps=[('preprocessor', preprocessor_fixed)])
# normalized_matrix = normalized_model.fit_transform(df_transformed)
# # Get the preprocessor from the normalized_model
# preprocessor = normalized_model.named_steps['preprocessor']

# # Get numerical feature names (they remain the same after scaling/imputing)
# # numerical_output_features = numerical_features
# numerical_output_features = [f'{col}_scaled' for col in numerical_features]
# # Get categorical feature names from the OneHotEncoder step
# # Access the 'cat' pipeline within the preprocessor, then the 'onehot' step within that pipeline
# pre_categorical_output_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
# categorical_output_features = [f'{col}_encoded' for col in pre_categorical_output_features]
# # Combine numerical and categorical feature names
# feature_names = list(numerical_output_features) + list(categorical_output_features)

# # Convert the sparse matrix to a dense array and then to a Pandas DataFrame
# normalized_df = pd.DataFrame(normalized_matrix.toarray(), columns=feature_names)

# # Gán lại index cho khớp với df_transformed
# normalized_df.index = df_transformed.index

# # Display the first 5 rows
# normalized_df.head(5)

# # Tách đặc trưng và nhãn
# transformed_target_cols = [col for col in df_transformed.columns if 'gia_ban_' in col]
# if transformed_target_cols:
#     tf_target_col = transformed_target_cols[0]
# else:
#     tf_target_col = 'gia_ban'

# X = normalized_df.copy()
# y = df_transformed[tf_target_col]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(f"Training data count: {len(X_train)}")
# print(f"Testing data count: {len(X_test)}")
# Tách đặc trưng và nhãn
transformed_target_cols = [col for col in df_transformed.columns if 'gia_ban_' in col]
if transformed_target_cols:
    tf_target_col = transformed_target_cols[0]
else:
    tf_target_col = 'gia_ban'

X = df_transformed[numerical_features + categorical_features]
y = df_transformed[tf_target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data count: {len(X_train)}")
print(f"Testing data count: {len(X_test)}")
# models = {
#     "Linear Regression": LinearRegression(),
#     "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=3, min_samples_leaf=1, max_features=0.2, max_samples=1.0, random_state=42),
#     "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
#     "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
#     "SVM (RBF Kernel)": SVR(kernel='rbf', C=1000, epsilon=0.1),
#     "Gradient-Boosted Trees": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42) # default max_depth=3
# }

# results = {}
# for name, model in models.items():
#     #
#     ml_model = model.fit(X_train, y_train)
#     y_pred = ml_model.predict(X_test)

#     # Convert back to original scale
#     match tf_target_col:
#         case 'gia_ban':
#             y_pred_reals = y_pred
#         case 'gia_ban_log':
#             y_pred_reals = np.exp(y_pred)
#         case 'gia_ban_sqrt':
#             y_pred_reals = np.power(y_pred, 2)
#         case 'gia_ban_boxcox':
#             if best_lambda_gia_ban == 0:
#                 y_pred_reals = np.exp(y_pred)
#             else:
#                 y_pred_reals = inv_boxcox(y_pred, best_lambda_gia_ban)

#     r2 = r2_score(y_test, y_pred_reals)
#     mae = mean_absolute_error(y_test, y_pred_reals)
#     rmse = np.sqrt(mean_squared_error(y_test,y_pred_reals))

#     results[name] = {
#         "R2": round(r2, 3),
#         "MAE": round(mae, 3),
#         "RMSE": round(rmse, 3)
#     }

#     print(f"{name} - R2: {r2:.3f}, MAE: {mae:.3f}")

#     # Vẽ biểu đồ
#     plt.figure(figsize=(10, 8))

#     # Vẽ biểu đồ phân tán (Scatter Plot)
#     sns.scatterplot(
#         x=y_test,
#         y=y_pred_reals,
#         alpha=0.6,
#         edgecolor=None,
#         color='steelblue'
#     )

#     # Thêm đường chéo lý tưởng (Ideal Diagonal Line)
#     # Đường này đại diện cho việc dự báo chính xác 100% (Thực tế = Dự báo)
#     min_val = min(y_test.min(), y_pred_reals.min())
#     max_val = max(y_test.max(), y_pred_reals.max())
#     plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Ideal Prediction')

#     # Trang trí biểu đồ
#     plt.xlabel('Giá thực tế (Actual Price)')
#     plt.ylabel('Giá dự báo (Predicted Price)')
#     plt.title('So sánh Giá thực tế vs. Giá dự báo')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.show()

# # Xuất kết quả so sánh
# print("\nBảng so sánh kết quả:")
# print(pd.DataFrame(results).T)
def build_and_evaluate_models(X_train, y_train, X_test, y_test, param_grid=False, transformed_target_col=None, best_lambda_gia_ban=None):

    # Định nghĩa models
    lr   = LinearRegression()

    lre  = ElasticNet(alpha=0.01, l1_ratio=0.0, max_iter =10, random_state=42)

    rf   = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_leaf=1, min_samples_split=3, max_features=0.2, max_samples=1.0, random_state=42)

    xgb  = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42, verbosity=0)

    # lgbm = LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, objective="regression", random_state=42, verbose=-1)
    lgbm = LGBMRegressor(n_estimators=100, random_state=42)

    gbt  = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

    # svm  = SVR(kernel='rbf', C=1000, epsilon=0.1)
    svm  = SVR(kernel='rbf', C=100, epsilon=0.2, gamma=0.01)

    models      = [lr, lre, rf, xgb, lgbm, gbt, svm]

    model_names = ["Linear Regression", "ElasticNet", "Random Forest Regressor",
                   "XGBoost", "LightGBM Regressor", "Gradient-Boosted Trees Regressor", "SVM (RBF Kernel)"]

    # Param grids
    param_grids = [
        {"model__fit_intercept": [True, False]},                                                                                                                 # lr
        {"model__alpha": [0.01, 0.1, 1.0], "model__l1_ratio": [0.0, 0.5, 1.0], "model__max_iter": [10, 100]},                                                    # elasticnet (lr)
        {"model__n_estimators": [20, 30, 40, 50, 100], "model__max_depth": [3, 4, 5, 6, 10, 15, 30, None], "model__min_samples_split": [3],
         "model__min_samples_leaf": [1], "model__max_samples": [0.8, 1.0], "model__max_features": [0.2]},                                                        # rf
        {"model__n_estimators": [20, 30, 40, 50, 100], "model__max_depth": [3, 4, 5, 6, 10, 15, 30], "model__learning_rate": [0.01, 0.1] },                      # xgb
        {"model__n_estimators": [100, 200], "model__max_depth": [3, 4, 5, 6, 10, 15, 30], "model__learning_rate": [0.01, 0.1], "model__num_leaves": [31, 60]},   # lgbm
        {"model__n_estimators": [20, 30, 40, 50, 100], "model__max_depth": [3, 4, 5, 6, 10, 15, 30], "model__learning_rate": [0.01, 0.1]},                       # gbt
        {"model__C": [100, 1000, 10000], "model__epsilon": [0.01, 0.1, 0.2, 0.5], "model__gamma": ['scale', 'auto', 0.01, 0.1], "model__kernel": ['rbf']}        # svm
        ]


    # Inverse transform helper
    def inverse_transform(preds):
        # Convert back to original scale
        if transformed_target_col == 'gia_ban':
            return preds
        elif transformed_target_col == 'gia_ban_log':
            return np.exp(preds)
        elif transformed_target_col == 'gia_ban_sqrt':
            return np.power(preds, 2)
        elif transformed_target_col == 'gia_ban_boxcox':
            if best_lambda_gia_ban == 0:
                return np.exp(preds)
            else:
                return inv_boxcox(preds, best_lambda_gia_ban)
        else:
            return preds

    # Vòng lặp train / evaluate
    results          = {}
    trained_models   = []
    all_predictions  = []

    for i, (model, name) in enumerate(zip(models, model_names)):
        #
        pipe = Pipeline(steps=[("preprocessor", preprocessor_fixed),
                               ("model", model)])

        if param_grid:
            search = GridSearchCV(
                pipe,
                param_grids[i],
                cv=5,
                # scoring="neg_root_mean_squared_error",
                scoring={
                    "r2": "r2",
                    "mae": "neg_mean_absolute_error"
                },
                refit="mae",  # ưu tiên mae thấp nhất
                n_jobs=-1
                # refit="r2",  # ưu tiên R² cao nhất
                # refit=True
            )
            search.fit(X_train, y_train)
            fitted_pipe  = search.best_estimator_
            best_params  = search.best_params_
            print(f"\n[{name}] Best params: {best_params}")
        else:
            pipe.fit(X_train, y_train)
            fitted_pipe = pipe

        trained_models.append(fitted_pipe)

        # Predict + inverse transform
        raw_preds       = fitted_pipe.predict(X_test)
        final_preds     = inverse_transform(raw_preds)
        all_predictions.append(final_preds)

        # Metrics trên original scale
        rmse = builtins.round(np.sqrt(mean_squared_error(y_test, final_preds)), 3)
        mae  = builtins.round(mean_absolute_error(y_test, final_preds), 3)
        r2   = builtins.round(r2_score(y_test, final_preds), 3)

        results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse}

        print(f"\nMetrics for model {name}:")
        print(f"  RMSE      : {rmse}")
        print(f"  MAE       : {mae}")
        print(f"  R-squared : {r2}")

        # Scatter plot actual vs predicted
        pdf = pd.DataFrame({"gia_ban": y_test, "final_prediction": final_preds})
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x="gia_ban", y="final_prediction", data=pdf, alpha=0.6)
        min_val = min(pdf["gia_ban"].min(), pdf["final_prediction"].min())
        max_val = max(pdf["gia_ban"].max(), pdf["final_prediction"].max())
        plt.plot([min_val, max_val], [min_val, max_val],
                 color="red", linestyle="--", linewidth=2)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title(f"Actual vs. Predicted Prices — {name}")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

    # Bảng so sánh
    print("\nBảng so sánh kết quả:")
    print(pd.DataFrame(results).T.sort_values("MAE"))

    if param_grid:
        return trained_models, model_names, all_predictions
    else:
        return trained_models, model_names, all_predictions
# Không tuning
models, names, preds = build_and_evaluate_models(
    X_train, y_train, X_test, y_test,
    param_grid=False,
    transformed_target_col=tf_target_col,
)

# Lấy predicted data
predictions = pd.DataFrame(preds[0], index=y_test.index)
predictions.columns = ["final_prediction"]

# Tạo df_test_final
df_test_final = df_transformed.join(predictions, how='inner')
df_test_final = df_test_final[data.columns.tolist() + predictions.columns.tolist()]

# Hiển thị test data cuối cùng
df_test_final.head(2)
# Lưu mô hình (đã gồm scaler)
model_filename = 'model_gia_nha_xgboost.pkl'
xgb_index = names.index('XGBoost')
model_to_save = models[xgb_index]

joblib.dump(model_to_save, model_filename)

print(f"Đã lưu mô hình vào file: {model_filename}")

# Khi cần tái sử dụng
# Tải mô hình lên
loaded_model = joblib.load('model_gia_nha_xgboost.pkl')

# Dự báo với dữ liệu mới (X_new)
X_new_sample = X_test
prediction_new = loaded_model.predict(X_new_sample)

prediction_new = pd.DataFrame(prediction_new, index=X_new_sample.index)
prediction_new.columns = ["final_prediction"]

df_new_sample_final = data.join(prediction_new, how='inner')
df_new_sample_final = df_new_sample_final[data.columns.tolist() + prediction_new.columns.tolist()]
df_new_sample_final.tail(10)
r2 = r2_score(y_test, prediction_new)
mae = mean_absolute_error(y_test, prediction_new)
rmse = np.sqrt(mean_squared_error(y_test, prediction_new))

print(f"R-squared: {r2}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
X_test
list_loai_hinh = df_transformed['loai_hinh'].unique().tolist()
print(list_loai_hinh)
list_giay_to_phap_ly = df_transformed['giay_to_phap_ly'].unique().tolist()
print(list_giay_to_phap_ly)
list_tinh_trang_noi_that = df_transformed['tinh_trang_noi_that'].unique().tolist()
print(list_tinh_trang_noi_that)
list_huong_cua_chinh = df_transformed['huong_cua_chinh'].unique().tolist()
print(list_huong_cua_chinh)
list_dac_diem = df_transformed['dac_diem'].unique().tolist()
print(list_dac_diem)
list_huong_cua_chinh = df_transformed['huong_cua_chinh'].unique().tolist()
print(list_huong_cua_chinh)
list_dia_chi_cu = df_transformed['dia_chi_cu'].unique().tolist()
print(list_dia_chi_cu)
list_dia_chi_moi = df_transformed['dia_chi_moi'].unique().tolist()
print(list_dia_chi_moi)
# Tạo dữ liệu giả lập

# new_data = {
#     'id_nha': [0, 1, 2, 3, 4],
#     'dien_tich_dat': [80, 41, 59.1, 48, 210],
#     'dien_tich_su_dung': [80, 82, 150, 130, 630],
#     'chieu_ngang': [4.2, 5, 3.3, 8, 12.5],
#     'tong_so_tang': [2, 2, 4, 3, 3],
#     'so_phong_ngu': [5, 3, 4, 4, 11],
#     'loai_hinh': ['nha_ngo_hem', 'nha_ngo_hem', 'nha_ngo_hem', 'nha_mat_pho_mat_tien', 'nha_biet_thu'],
#     'giay_to_phap_ly': ['da_co_so', 'da_co_so', 'da_co_so', 'da_co_so', 'da_co_so'],
#     'tinh_trang_noi_that': ['noi_that_day_du', 'noi_that_day_du', 'hoan_thien_co_ban', 'noi_that_cao_cap', 'noi_that_day_du'],
#     'huong_cua_chinh': ['dong_nam', 'dong_nam', 'tay_bac', 'bac', 'dong_nam'],
#     'dac_diem': ['hem_xe_hoi', 'hem_xe_hoi', 'nha_no_hau', 'nha_no_hau', 'hem_xe_hoi'],
#     'dia_chi_cu': ['phuong_3_quan_go_vap', 'phuong_3_quan_go_vap', 'phuong_7_quan_phu_nhuan', 'phuong_2_quan_phu_nhuan', 'phuong_7_quan_binh_thanh'],
#     'dia_chi_moi': ['phuong_hanh_thong', 'phuong_hanh_thong', 'phuong_cau_kieu', 'phuong_cau_kieu', 'phuong_gia_dinh'],
#     'gia_ban': [9.9, 4.35, 10, 18.9, 25]

# }

new_data = {
    'id_nha': [0, 1, 2, 3, 4],
    'dien_tich_dat': [32, 41, 40, 40, 45],
    'dien_tich_su_dung': [64, 82, 120, 120, 90],
    'chieu_ngang': [3.5, 5, 5, 4, 4.5],
    'tong_so_tang': [1, 2, 3, 3, 2],
    'so_phong_ngu': [3, 3, 4, 4, 3],
    'loai_hinh': ['nha_pho_lien_ke', 'nha_ngo_hem', 'nha_mat_pho_mat_tien', 'nha_ngo_hem', 'nha_ngo_hem'],
    'giay_to_phap_ly': ['da_co_so', 'da_co_so', 'da_co_so', 'da_co_so', 'da_co_so'],
    'tinh_trang_noi_that': ['noi_that_cao_cap', 'noi_that_day_du', 'noi_that_cao_cap', 'noi_that_day_du', 'noi_that_cao_cap'],
    'huong_cua_chinh': ['dong_nam', 'dong_nam', 'dong_nam', 'dong_bac', 'nam'],
    'dac_diem': ['hem_xe_hoi', 'hem_xe_hoi', 'hem_xe_hoi', 'hem_xe_hoi', 'hem_xe_hoi'],
    'dia_chi_cu': ['phuong_12_quan_go_vap', 'phuong_3_quan_go_vap', 'phuong_3_quan_go_vap', 'phuong_13_quan_phu_nhuan', 'phuong_7_quan_binh_thanh'],
    'dia_chi_moi': ['phuong_an_hoi_tay', 'phuong_hanh_thong', 'phuong_hanh_thong', 'phuong_phu_nhuan', 'phuong_binh_thanh'],
    'gia_ban': [5.56, 4.35, 7.6, 6.3, 6.7]

}

# Khởi tạo DataFrame
df_sample = pd.DataFrame(new_data)

# Đặt id_nha làm index (giống như cách bạn đang làm với df_transformed)
df_sample.set_index('id_nha', inplace=True)

print("DataFrame mẫu bất động sản:")
df_sample
df_X_test_new = df_sample.drop(columns=['gia_ban'])

df_X_test_new['dien_tich_su_dung_boxcox'], _ = stats.boxcox(df_X_test_new['dien_tich_su_dung'])
df_X_test_new['chieu_ngang_boxcox'], _ = stats.boxcox(df_X_test_new['chieu_ngang'])
df_X_test_new['tong_so_tang_boxcox'], _ = stats.boxcox(df_X_test_new['tong_so_tang'])
df_X_test_new['so_phong_ngu_sqrt'] = np.sqrt(df_X_test_new['so_phong_ngu'])
df_X_test_new = df_X_test_new[numerical_features + categorical_features]
df_X_test_new
df_y_test_new = df_sample['gia_ban']
df_y_test_new
# Dự báo với dữ liệu mới (df_X_test_new)

df_prediction_new = loaded_model.predict(df_X_test_new)

df_prediction_new = pd.DataFrame(df_prediction_new, index=df_X_test_new.index)
df_prediction_new.columns = ["final_prediction"]

df_new_sample_final = df_sample.join(df_prediction_new, how='inner')
df_new_sample_final = df_new_sample_final[df_sample.columns.tolist() + df_prediction_new.columns.tolist()]
df_new_sample_final
# # Có tuning GridSearchCV
# models, names, preds = build_and_evaluate_models(
#     X_train, y_train, X_test, y_test,
#     param_grid=True,
#     transformed_target_col=tf_target_col,
#     best_lambda_gia_ban=best_lambda_gia_ban,
# )
df_pr1_2 = df_test_final.copy()
df_pr1_2.head(5)
# Trích xuất quận/huyện bằng Regex
# Phép tương đương của F.regexp_extract(..., 1)
df_pr1_2['quan_huyen'] = df_pr1_2['dia_chi_cu'].str.extract(r"((?:quan|huyen)_.*)")[0]

# Nối chuỗi để tạo cột quan_huyen_loai_hinh
# Phép tương đương của F.concat_ws("_", ...)
df_pr1_2['quan_huyen_loai_hinh'] = df_pr1_2['quan_huyen'] + "_" + df_pr1_2['loai_hinh']
def visualize_bar_price(df, groupby_features):
    # Tính toán giá trung bình
    stats_df = df.groupby(groupby_features).agg({"gia_ban": "mean",
                                                 "final_prediction": "mean"}) \
                                           .rename(columns={"gia_ban": "Thực tế",
                                                            "final_prediction": "Dự đoán"}) \
                                           .sort_values(by="Thực tế") \
                                           .reset_index()

    # Tạo cột hiển thị gộp (Dùng để làm trục X cho biểu đồ)
    stats_df['display_name'] = stats_df[groupby_features].astype(str).agg(' - '.join, axis=1)

    # Chuyển sang long-format
    stats_melted = stats_df.melt(
        id_vars=['display_name'],
        value_vars=["Thực tế", "Dự đoán"],
        var_name="Loại giá",
        value_name="Average_Price"
    )

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")

    # Tiêu đề trục X đẹp hơn
    title_x = " - ".join([f.replace("_", " ").capitalize() for f in groupby_features])

    ax = sns.barplot(
        data=stats_melted,
        x="display_name",
        y="Average_Price",
        hue="Loại giá",
        palette="viridis",
        alpha=0.9
    )

    # Định dạng thẩm mỹ
    plt.title(f"So sánh Giá Thực tế vs Dự đoán theo: {title_x}", fontsize=16, pad=20, fontweight='bold')
    plt.ylabel("Giá trung bình (Tỷ VNĐ)", fontsize=12)
    plt.xlabel("") # Để trống vì display_name đã đủ rõ

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(title="Chú thích", loc='upper left', bbox_to_anchor=(1, 1))

    # Hiển thị số liệu trên đầu cột (Option)
    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(format(p.get_height(), '.1f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha = 'center', va = 'center',
                        xytext = (0, 9),
                        textcoords = 'offset points',
                        fontsize=9)

    plt.tight_layout()
    plt.show()
df_histplot_1 = df_pr1_2
groupby_features = ["loai_hinh"]
visualize_bar_price(df_histplot_1, groupby_features)
groupby_features = ["quan_huyen"]
visualize_bar_price(df_histplot_1, groupby_features)
groupby_features = ["quan_huyen_loai_hinh"]
visualize_bar_price(df_histplot_1, groupby_features)
groupby_features = ["tong_so_tang"]
visualize_bar_price(df_histplot_1, groupby_features)
def detect_real_estate_anomalies_by_residual_z(df, groupby_features):
    # Tính sai số (residual)
    df['residual'] = df['gia_ban'] - df['final_prediction']

    # Tính Trung bình và Độ lệch chuẩn của residual theo từng nhóm (Window function)
    # transform() giúp giữ nguyên số lượng dòng của dataframe gốc
    df['avg_res'] = df.groupby(groupby_features)['residual'].transform('mean')
    df['std_res'] = df.groupby(groupby_features)['residual'].transform('std')

    # Tính Z-score của sai số
    # Thêm 1 lượng cực nhỏ (1e-9) vào mẫu số để tránh lỗi chia cho 0 nếu std = 0
    df['residual_z'] = (df['residual'] - df['avg_res']) / (df['std_res'] + 1e-9)

    # Đánh dấu vi phạm (Outliers) nếu |Z| > 3
    df['violate_residual_z'] = np.where(df['residual_z'].abs() > 3, 1, 0)

    return df
def composite_scores(df, groupby_features, methods,
                     threshold_pr_min=None, threshold_pr_max=None,
                     threshold_po_min=None, threshold_po_max=None,
                     feature_cols=None, isf_contamination=0.05):

    df_anomalies = df.copy()

    # Chạy các phương pháp phát hiện dựa trên danh sách keys của methods
    for method in methods.keys():
        if method == "residual_z":
            df_anomalies = detect_real_estate_anomalies_by_residual_z(df_anomalies, groupby_features)
        elif method == "min_max":
            df_anomalies = detect_real_estate_anomalies_by_min_max(df_anomalies, groupby_features, threshold_pr_min, threshold_pr_max)
        elif method == "outside_conf":
            df_anomalies = detect_real_estate_anomalies_by_outside_conf(df_anomalies, groupby_features, threshold_po_min, threshold_po_max)
        elif method == "isolation_forest":
            df_anomalies = detect_real_estate_anomalies_by_isolation_forest(df_anomalies, groupby_features, feature_cols, isf_contamination)

    # Tính điểm tổng hợp (Weighted Score)
    # Khởi tạo cột điểm bằng 0
    df_anomalies['raw_anomaly_score'] = 0.0

    for method, weights in methods.items():
        for col_name, weight in weights.items():
            if col_name in df_anomalies.columns:
                df_anomalies['raw_anomaly_score'] += df_anomalies[col_name] * weight

    # Chuẩn hóa về thang điểm 100
    max_val = df_anomalies['raw_anomaly_score'].max()

    if max_val == 0 or pd.isna(max_val):
        df_anomalies['final_anomaly_score'] = 0.0
    else:
        df_anomalies['final_anomaly_score'] = (df_anomalies['raw_anomaly_score'] / max_val) * 100

    return df_anomalies
def extract_anomalies(df, threshold_outlier_percent):
    # Tạo bản sao để không ảnh hưởng đến DataFrame gốc
    df_anomalies = df.copy()

    # Tính giá trị ngưỡng (ceiling_score)
    # Tương đương với approxQuantile trong Spark
    # threshold_outlier_percent (ví dụ: 1 - 0.05 = 0.95)
    ceiling_score = df_anomalies["final_anomaly_score"].quantile(1 - threshold_outlier_percent)

    # Lọc các dòng có score >= ngưỡng và sắp xếp giảm dần
    list_anomalies = df_anomalies[df_anomalies["final_anomaly_score"] >= ceiling_score] \
                        .sort_values(by="final_anomaly_score", ascending=False)

    return list_anomalies
def report_anomalies(df, groupby_features, methods, methods_features, methods_flags,
                     threshold_outlier_percent, threshold_pr_min=None,
                     threshold_pr_max=None, threshold_po_min=None,
                     threshold_po_max=None, feature_cols=None, isf_contamination=None):

    # Tính toán điểm tổng hợp
    df_anomalies = composite_scores(df, groupby_features, methods,
                                   threshold_pr_min, threshold_pr_max,
                                   threshold_po_min, threshold_po_max,
                                   feature_cols, isf_contamination)

    # Trích xuất anomalies theo percentile
    list_anomalies = extract_anomalies(df_anomalies, threshold_outlier_percent)

    # Tạo điều kiện lọc (Combined Condition): Chỉ lấy những dòng có ít nhất 1 flag vi phạm
    # Trong Pandas, dùng (df[flags] == 1).any(axis=1) thay cho reduce(|)
    combined_condition = (list_anomalies[methods_flags] == 1).any(axis=1)
    list_anomalies = list_anomalies[combined_condition]

    # Xác định các cột cần hiển thị
    # Sử dụng list để gom nhóm các cột tránh trùng lặp
    base_columns = ["gia_ban", "final_prediction"]
    info_columns = ["dien_tich_dat", "dien_tich_su_dung", "chieu_ngang", "tong_so_tang",
                    "so_phong_ngu", "giay_to_phap_ly", "tinh_trang_noi_that", "dac_diem",
                    "dia_chi_cu", "dia_chi_moi"]

    selected_columns = (base_columns +
                        (methods_features if methods_features else []) +
                        methods_flags +
                        ["final_anomaly_score"] +
                        groupby_features +
                        info_columns)

    # Lọc các cột hiện có trong DataFrame để tránh lỗi KeyNotFound
    selected_columns = [c for c in selected_columns if c in list_anomalies.columns]

    # Sắp xếp (OrderBy)
    orderby_features = ["final_anomaly_score"] + groupby_features + (methods_features if methods_features else [])
    # Sắp xếp giảm dần cho toàn bộ các cột tiêu chí
    list_anomalies = list_anomalies[selected_columns].sort_values(
        by=orderby_features,
        ascending=False
    )

    # Hiển thị kết quả
    print(f"Số lượng nhà bất thường phát hiện: {len(list_anomalies)}")
    # Hiển thị 20 dòng đầu tiên tương tự .show() của Spark
    if not list_anomalies.empty:
        print(list_anomalies.head(20).to_string())

    return df_anomalies, list_anomalies
def scatter_highlight_outliers(df_anomalies, list_anomalies, sample_frac=0.1):
    """
    Vẽ biểu đồ Scatter Plot để trực quan hóa dữ liệu bất thường trên nền dữ liệu mẫu.
    """
    # Lấy mẫu dữ liệu nền (Normal Data) để tránh quá tải biểu đồ nếu dữ liệu lớn
    # Nếu dữ liệu nhỏ, có thể đặt sample_frac=1.0
    pdf_sample = df_anomalies.sample(frac=sample_frac, random_state=42)

    # Dữ liệu bất thường đã là Pandas nên không cần chuyển đổi
    pdf_anomalies = list_anomalies

    plt.figure(figsize=(12, 8))

    # Vẽ dữ liệu bình thường (Màu xám, độ trong suốt thấp)
    sns.scatterplot(
        data=pdf_sample,
        x='final_prediction',
        y='gia_ban',
        alpha=0.3,
        label='Dữ liệu mẫu (Normal)',
        color='gray'
    )

    # Highlight dữ liệu bất thường (Màu đỏ, kích thước lớn hơn)
    sns.scatterplot(
        data=pdf_anomalies,
        x='final_prediction',
        y='gia_ban',
        color='red',
        label='Bất thường (Anomalies)',
        s=100,
        edgecolor='black'
    )

    # Vẽ đường chéo lý tưởng (Y = X)
    # Tìm giá trị lớn nhất của cả 2 trục để vẽ đường cân bằng
    all_max = max(
        pdf_sample[['gia_ban', 'final_prediction']].max().max(),
        pdf_anomalies[['gia_ban', 'final_prediction']].max().max()
    )
    plt.plot([0, all_max], [0, all_max], '--', color='blue', alpha=0.5, label='Đường lý tưởng')

    # Định dạng biểu đồ
    plt.title("Phát hiện bất thường: Giá thực tế vs Giá dự đoán", fontsize=15)
    plt.xlabel("Giá dự đoán (Model Prediction)", fontsize=12)
    plt.ylabel("Giá thực tế (Actual Price)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()
# def visualize_anomalies_by_selected_features(df, methods_flags):
#     # Chỉ lấy các cột cần thiết và chuyển về Pandas để nhẹ bộ nhớ
#     # Giả sử is_anomaly_iforest: 1 là bất thường (đỏ), 0 là bình thường (xanh)
#     pdf = df.select(
#         "quan_huyen_loai_hinh",
#         "giay_to_phap_ly",
#         "gia_ban",
#         *methods_flags
#     ).toPandas()

#     # Khởi tạo biểu đồ
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     fig.suptitle('Biểu đồ phân cụm các điểm bất thường', fontsize=13, fontweight='bold')

#     # Định nghĩa bảng màu
#     # 1 (Bất thường) -> Đỏ
#     # 0 (Bình thường) -> Xanh

#     pdf['is_any_anomaly'] = pdf[methods_flags].max(axis=1)
#     pdf_sorted = pdf.sort_values(by='is_any_anomaly', ascending=True)
#     combined_colors = pdf_sorted['is_any_anomaly'].map({1: '#D32F2F', 0: '#4CAF50'})
#     point_sizes = pdf_sorted['is_any_anomaly'].map({1: 30, 0: 15})
#     # Biểu đồ 1: Quận/ huyện - Loại hình vs Giá
#     axes[0].scatter(
#         pdf['quan_huyen_loai_hinh'],
#         pdf['gia_ban'],
#         c=combined_colors,
#         s=point_sizes,
#         alpha=0.5
#     )
#     axes[0].set_title('Quận huyện & loại hình vs Giá\n(đỏ = bất thường)')
#     axes[0].set_xlabel('')
#     axes[0].set_ylabel('Giá bán')
#     axes[0].tick_params(axis='x', rotation=45)
#     for label in axes[0].get_xticklabels():
#         label.set_horizontalalignment('right')

#     # Biểu đồ 2: Diện tích vs Giá
#     axes[1].scatter(
#         pdf['giay_to_phap_ly'],
#         pdf['gia_ban'],
#         c=combined_colors,
#         s=point_sizes,
#         alpha=0.5
#     )
#     axes[1].set_title('Tình trạng pháp lý vs Giá\n(đỏ = bất thường)')
#     axes[1].set_xlabel('')
#     axes[1].set_ylabel('Giá bán')
#     axes[1].tick_params(axis='x', rotation=45)
#     for label in axes[1].get_xticklabels():
#         label.set_horizontalalignment('right')

#     plt.tight_layout()
#     plt.show()

def visualize_anomalies_by_selected_features(df, methods_flags):
    """
    df: Pandas DataFrame
    methods_flags: List các tên cột chứa cờ bất thường (ví dụ: ['is_anomaly_iforest'])
    """
    # 1. Lấy các cột cần thiết (Lọc trực tiếp trên Pandas DataFrame)
    needed_cols = ["quan_huyen_loai_hinh", "giay_to_phap_ly", "gia_ban"] + methods_flags
    pdf = df[needed_cols].copy()

    # 2. Xác định điểm bất thường tổng hợp (nếu một trong các phương pháp báo 1 thì coi là 1)
    pdf['is_any_anomaly'] = pdf[methods_flags].max(axis=1)

    # Sắp xếp để vẽ điểm xanh (bình thường) trước, điểm đỏ (bất thường) đè lên trên cho dễ thấy
    pdf_sorted = pdf.sort_values(by='is_any_anomaly', ascending=True)

    # 3. Định nghĩa bảng màu và kích thước điểm
    # 1 (Bất thường) -> Đỏ (#D32F2F), 0 (Bình thường) -> Xanh (#4CAF50)
    colors = pdf_sorted['is_any_anomaly'].map({1: '#D32F2F', 0: '#4CAF50'})
    sizes = pdf_sorted['is_any_anomaly'].map({1: 30, 0: 15})

    # 4. Khởi tạo biểu đồ
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Phân tích điểm bất thường trong dữ liệu Bất động sản', fontsize=14, fontweight='bold')

    # Biểu đồ 1: Quận huyện/Loại hình vs Giá
    axes[0].scatter(
        pdf_sorted['quan_huyen_loai_hinh'],
        pdf_sorted['gia_ban'],
        c=colors,
        s=sizes,
        alpha=0.6
    )
    axes[0].set_title('Quận huyện & Loại hình vs Giá\n(Đỏ = Bất thường)')
    axes[0].set_ylabel('Giá bán (Tỷ đồng)')
    axes[0].tick_params(axis='x', rotation=45)

    # Biểu đồ 2: Pháp lý vs Giá
    axes[1].scatter(
        pdf_sorted['giay_to_phap_ly'],
        pdf_sorted['gia_ban'],
        c=colors,
        s=sizes,
        alpha=0.6
    )
    axes[1].set_title('Tình trạng pháp lý vs Giá\n(Đỏ = Bất thường)')
    axes[1].set_ylabel('Giá bán (Tỷ đồng)')
    axes[1].tick_params(axis='x', rotation=45)

    # Căn chỉnh nhãn cho đẹp
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
threshold_outlier_percent = 0.05  # 5%
groupby_features = ["quan_huyen", "loai_hinh"]
methods = {
    "residual_z": {
        "violate_residual_z": 100 # Trọng số cho điểm số tổng hợp
    }
}
methods_features = ["residual"]
methods_flags = ["violate_residual_z"]

df_anomalies, list_anomalies = report_anomalies(df_pr1_2, groupby_features, methods, methods_features, methods_flags, threshold_outlier_percent)

scatter_highlight_outliers(df_anomalies, list_anomalies)
visualize_anomalies_by_selected_features(df_anomalies, methods_flags)
def histplot_residual_z(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[['residual_z']], kde=True, color='skyblue')

    # Vẽ đường ngưỡng (ví dụ z > 3 hoặc z < -3)
    plt.axvline(x=3, color='red', linestyle='--', label='Upper Threshold (3z)')
    plt.axvline(x=-3, color='red', linestyle='--', label='Lower Threshold (-3z)')

    plt.title("Phân phối của Residual-Z (Sai số chuẩn hóa)")
    plt.legend()
    plt.show()
histplot_residual_z(df_anomalies)
def detect_real_estate_anomalies_by_min_max(df, groupby_features, threshold_pr_min, threshold_pr_max):
    # Tạo bản sao để tránh lỗi SettingWithCopyWarning
    df = df.copy()

    # Tính toán P1 và P99 theo phân khúc bằng transform
    # transform giúp giữ nguyên số lượng dòng của df gốc để gán trực tiếp
    df['price_min'] = df.groupby(groupby_features)['gia_ban'].transform(lambda x: x.quantile(threshold_pr_min))
    df['price_max'] = df.groupby(groupby_features)['gia_ban'].transform(lambda x: x.quantile(threshold_pr_max))

    # Tính toán các cột vi phạm (violate)
    # Trong Pandas, phép so sánh trả về True/False, ta dùng .astype(int) để chuyển về 1/0
    df['violate_min'] = (df['gia_ban'] < df['price_min']).astype(int)
    df['violate_max'] = (df['gia_ban'] > df['price_max']).astype(int)

    # Vi phạm chung (kết hợp min hoặc max)
    df['violate_min_max'] = ((df['violate_min'] == 1) | (df['violate_max'] == 1)).astype(int)

    return df
threshold_outlier_percent = 0.05
threshold_pr_min = 0.01
threshold_pr_max = 0.99
groupby_features = ["quan_huyen", "loai_hinh"]
methods = {
    "min_max": {
        "violate_min_max": 100
    }
}
methods_features = ["violate_min", "violate_max", "price_min", "price_max"]
methods_flags = ["violate_min_max"]

df_anomalies, list_anomalies = report_anomalies(df_pr1_2, groupby_features, methods, methods_features, methods_flags, threshold_outlier_percent, \
                 threshold_pr_min=threshold_pr_min, threshold_pr_max=threshold_pr_max)

scatter_highlight_outliers(df_anomalies, list_anomalies)
visualize_anomalies_by_selected_features(df_anomalies, methods_flags)
def detect_real_estate_anomalies_by_outside_conf(df, groupby_features, threshold_po_min, threshold_po_max):
    # Tạo bản sao để tránh SettingWithCopyWarning
    df = df.copy()

    # Xác định tên cột động dựa trên tham số (%)
    col_min = f"P{int(threshold_po_min * 100)}"
    col_max = f"P{int(threshold_po_max * 100)}"

    # Tính toán percentile theo từng nhóm và map ngược lại vào df gốc
    # transform('quantile') sẽ trả về một Series có cùng độ dài với df
    df[col_min] = df.groupby(groupby_features)['gia_ban'].transform(lambda x: x.quantile(threshold_po_min))
    df[col_max] = df.groupby(groupby_features)['gia_ban'].transform(lambda x: x.quantile(threshold_po_max))

    # Gắn nhãn các vi phạm (Sử dụng vectorization của numpy/pandas để đạt tốc độ cao)
    df['outside_conf_min'] = (df['gia_ban'] < df[col_min]).astype(int)
    df['outside_conf_max'] = (df['gia_ban'] > df[col_max]).astype(int)

    # 4. Tạo flag tổng hợp (1 nếu vi phạm ít nhất 1 đầu)
    df['outside_conf_flag'] = ((df['outside_conf_min'] == 1) | (df['outside_conf_max'] == 1)).astype(int)

    return df
threshold_outlier_percent = 0.05    # 5%
threshold_po_min = 0.1              # P10
threshold_po_max = 0.9              # P90
groupby_features = ["quan_huyen", "loai_hinh"]
methods = {
    "outside_conf": {
        "outside_conf_flag": 100
    }
}
methods_features = ["outside_conf_min", "outside_conf_max"]
methods_flags = ["outside_conf_flag"]

# Using explicit keyword arguments to avoid positional mapping errors with optional parameters
df_anomalies, list_anomalies = report_anomalies(
                                                df=df_pr1_2,
                                                groupby_features=groupby_features,
                                                methods=methods,
                                                methods_features=methods_features,
                                                methods_flags=methods_flags,
                                                threshold_outlier_percent=threshold_outlier_percent,
                                                threshold_po_min=threshold_po_min,
                                                threshold_po_max=threshold_po_max
                                            )
scatter_highlight_outliers(df_anomalies, list_anomalies)
visualize_anomalies_by_selected_features(df_anomalies, methods_flags)
from sklearn.model_selection import train_test_split
import joblib

def detect_real_estate_anomalies_by_isolation_forest(df, groupby_features, feature_cols, isf_contamination):
    # Tách train/test (80% train, 20% test)
    train_df, test_df = train_test_split(df.copy(), test_size=0.2, random_state=42)
    
    # Xử lý dữ liệu thiếu cho các feature
    X_train = train_df[feature_cols].fillna(0)
    X_test = test_df[feature_cols].fillna(0)

    # Khởi tạo và huấn luyện mô hình TRÊN TẬP TRAIN
    model = IsolationForest(contamination=isf_contamination, random_state=42)
    model.fit(X_train)

    # Dự đoán và tính score TRÊN TẬP TEST
    test_df['iforest_score'] = model.decision_function(X_test)
    preds = model.predict(X_test)
    test_df['is_anomaly_iforest'] = (preds == -1).astype(int)
    
    # Lưu lại model trained (.pkl)
    joblib.dump(model, 'model_anomaly_detection_IsolationForest.pkl')
    
    # Phân tích insight thống kê từ Dữ Liệu Gốc 
    # (Để Backend API / GUI có nền tảng tri thức phục vụ explaination)
    price_col = 'gia_ban' if 'gia_ban' in df.columns else 'gia'
    area_col = 'dien_tich_su_dung' if 'dien_tich_su_dung' in df.columns else 'dien_tich'
    
    if price_col in df.columns and area_col in df.columns:
        pm2 = df[price_col] / df[area_col]
        stats_dict = {
            'median_gia_ban': df[price_col].median(),
            'median_dien_tich': df[area_col].median(),
            'median_price_per_m2': pm2.median()
        }
        if 'dia_chi_cu' in df.columns:
            stats_dict['local_price_per_m2'] = df.assign(pm2=pm2).groupby('dia_chi_cu')['pm2'].median().to_dict()
        if 'loai_hinh' in df.columns:
            target_area = 'dien_tich_dat' if 'dien_tich_dat' in df.columns else area_col
            stats_dict['type_median_area'] = df.groupby('loai_hinh')[target_area].median().to_dict()
            
        joblib.dump(stats_dict, 'anomaly_stats.pkl')
        
    return test_df.sort_values(by="iforest_score", ascending=True)
threshold_percent = 0.05
isf_contamination = 0.05
feature_cols = ["dien_tich_su_dung", "gia_ban", "final_prediction"]
groupby_features = ["quan_huyen", "loai_hinh"]
methods = {
    "isolation_forest": {
        "is_anomaly_iforest": 100 # Trọng số cho điểm số tổng hợp
    }
}
methods_features = ["iforest_score"]
methods_flags = ["is_anomaly_iforest"]

df_anomalies, list_anomalies = report_anomalies(df_pr1_2, groupby_features, methods, methods_features, methods_flags, threshold_percent, \
                                                feature_cols=feature_cols, isf_contamination=isf_contamination)

scatter_highlight_outliers(df_anomalies, list_anomalies)
visualize_anomalies_by_selected_features(df_anomalies, methods_flags)
threshold_outlier_percent = 0.05    # 5%  - Lấy 5% top bất thường nhất
threshold_pr_min = 0.01             # P1  - Min-Max
threshold_pr_max = 0.99             # P99 - Min-Max
threshold_po_min = 0.1              # P10 - Ngoài khoảng tin cậy
threshold_po_max = 0.9              # P90 - Ngoài khoảng tin cậy
isf_contamination = 0.05
feature_cols = ["dien_tich_su_dung", "gia_ban", "final_prediction"]
groupby_features = ["quan_huyen", "loai_hinh"]
methods = {
    "residual_z": {
        "violate_residual_z": 30
    },
    "min_max": {
        "violate_min_max": 25
    },
    "outside_conf": {
        "outside_conf_flag": 15
    },
    "isolation_forest": {
        "is_anomaly_iforest": 30
    }
}
methods_features = ["residual", "violate_min", "violate_max", "price_min", "price_max", "outside_conf_min", "outside_conf_max", "iforest_score"]
methods_flags = ["violate_residual_z", "violate_min_max", "outside_conf_flag", "is_anomaly_iforest"]

df_anomalies, list_anomalies = report_anomalies(df_pr1_2, groupby_features, methods, methods_features, methods_flags=methods_flags, \
                                                threshold_outlier_percent=threshold_outlier_percent, \
                                                threshold_pr_min=threshold_pr_min, threshold_pr_max=threshold_pr_max, \
                                                threshold_po_min=threshold_po_min, threshold_po_max=threshold_po_max, \
                                                feature_cols=feature_cols, isf_contamination=isf_contamination)
scatter_highlight_outliers(df_anomalies, list_anomalies)
visualize_anomalies_by_selected_features(df_anomalies, methods_flags)
data = data_pr2_2.copy()
data.head(1)
df = data.copy()

# Định nghĩa các hàm phụ trợ để xử lý ép kiểu và làm tròn (tương đương F.round + F.cast)
def fmt_num(series, prefix="", suffix=""):
    # Ép kiểu sang float, làm tròn 2 chữ số và nối chuỗi
    # Xử lý trường hợp NaN để tránh lỗi
    return prefix + series.fillna(0).round(2).astype(str) + suffix

def fmt_int(series, suffix=""):
    return series.fillna(0).astype(int).astype(str) + suffix

# Tạo danh sách các cột đã được định dạng
parts = [
    df["tieu_de"].fillna(""),
    df["mo_ta"].fillna(""),
    fmt_num(df["dien_tich_dat"], "dien_tich_dat_", "_m2"),
    fmt_num(df["dien_tich_su_dung"], "dien_tich_su_dung_", "_m2"),
    fmt_num(df["chieu_ngang"], "chieu_ngang_", "_m"),
    fmt_int(df["tong_so_tang"], "_tang"),
    fmt_int(df["so_phong_ngu"], "_phong_ngu"),
    df["loai_hinh"].fillna(""),
    df["giay_to_phap_ly"].fillna(""),
    df["tinh_trang_noi_that"].fillna(""),
    "huong_cua_chinh_" + df["huong_cua_chinh"].fillna(""),
    df["dac_diem"].fillna(""),
    df["dia_chi_cu"].fillna(""),
    df["dia_chi_moi"].fillna(""),
    fmt_num(df["gia_ban"], "gia_ban_", "_ty")
]

# Nối tất cả lại bằng khoảng trắng (tương đương F.concat_ws)
df["description"] = parts[0]
for part in parts[1:]:
    df["description"] = df["description"] + " " + part

# Xử lý Regex và Trim (tương đương F.regexp_replace và F.trim)
pattern_to_remove = r"chua_xac_dinh|hien_trang_khac"
df["description"] = (
    df["description"]
    .str.replace(pattern_to_remove, "", regex=True) # Replace theo pattern
    .str.replace(r'\s+', ' ', regex=True)          # Thu gọn nhiều khoảng trắng thừa do replace tạo ra
    .str.strip()                                   # Trim đầu cuối
)
# df = df.reset_index(names='id')
df = df.reset_index(drop=True)
df['id'] = df.index.values
STOP_WORD_FILE = 'files/vietnamese-stopwords.txt'
with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
    stop_words = file.read()

stop_words = stop_words.split('\n')
def clean_html_icon(text):
    if pd.isna(text) or text == "":
        return ""

    # Loại bỏ các thẻ HTML
    # Dùng 'lxml' hoặc 'html.parser' tùy thư viện bạn đã cài
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # Loại bỏ Emojis và các ký tự biểu tượng đặc biệt
    # Giữ lại ký tự Latin, số và dải Unicode tiếng Việt (\u00C0-\u1EF9)
    text = re.sub(r'[^\x00-\x7F\u00C0-\u1EF9]+', ' ', text)

    # Làm sạch khoảng trắng và xuống dòng
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()

# Áp dụng vào DataFrame bằng .apply()
df["description_clean"] = df["description"].apply(clean_html_icon)

# Hiển thị kết quả (không cắt chữ)
pd.set_option('display.max_colwidth', None)
def vn_tokenize(text):
    if text is None:
        return ""
    return ViTokenizer.tokenize(text)

# Áp dụng cho pandas DataFrame
df["description_clean"] = df["description_clean"].apply(vn_tokenize)
df[["id", "description_clean"]].head(5)
# Bước 1 + 2: Tokenize + HashingTF (gộp luôn bằng HashingVectorizer)
vectorizer = HashingVectorizer(
    n_features=10000,   # giống numFeatures
    tokenizer=lambda x: x.split(),  # vì đã có text dạng "nhà_phố đẹp"
    lowercase=False     # giữ nguyên tiếng Việt
)

X_tf = vectorizer.transform(df["description_clean"])

# Bước 3: IDF
tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X_tf)

# Gán lại vào DataFrame
df["features"] = list(X_tfidf)

# Hiển thị
df[["id", "features"]].head(5)
from sklearn.preprocessing import normalize

# Lấy ma trận TF-IDF (X_tfidf đã tạo ở bước trước)
X_norm = normalize(X_tfidf, norm='l2')

# Gán lại vào DataFrame
df["normFeatures"] = list(X_norm)

# Lấy dạng giống (id, vector)
features_list = list(zip(df["id"], df["normFeatures"]))

# Xem thử
print(features_list[:5])
df.loc[df["id"] == 10, ["id", "description_clean"]]
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Giả sử bạn đã có:
# X_norm (ma trận normalized TF-IDF)
# df (DataFrame chứa id, description_clean)

target_id = 10  # Changed target_id from 10 to 0, as 10 might not exist after data cleaning

# Lấy index của dòng có id = 10
target_idx = df.index[df["id"] == target_id][0]

# Lấy vector của target
target_vector = X_norm[target_idx]

# Tính cosine similarity với toàn bộ dataset
sk_similarities = cosine_similarity(target_vector, X_norm).flatten()

# Gán vào DataFrame
df["similarity_score"] = sk_similarities

# Lọc bỏ chính nó + lấy top 5
recommendations = (
    df[df["id"] != target_id]
    .sort_values(by="similarity_score", ascending=False)
    [["id", "description_clean", "similarity_score"]]
    .head(5)
)

recommendations
# Lấy 2 cột cần xử lý
pdf = df[["id", "description_clean"]]

# Định nghĩa hàm làm sạch và tách từ cho từng dòng
def clean_and_tokenize(text):
    if text is None:
        return ""

    # Thay thế gạch dưới bằng khoảng trắng để ViTokenizer nhận diện từ ghép tốt hơn
    text_clean = text.replace("_", " ")

    # Tokenize tiếng Việt (sân thượng -> sân_thượng)
    tokenized = ViTokenizer.tokenize(text_clean)

    # Custom nối lại các cụm số + đơn vị (ví dụ: dài_3m, ngang_5m)
    # Regex này tìm từ 'dài' hoặc 'ngang' theo sau là khoảng trắng và con số
    final_text = re.sub(r'(dài|ngang|rộng)\s(\d+)', r'\1_\2', tokenized)

    return final_text

# Áp dụng hàm xử lý cho toàn bộ cột trong Pandas
pdf["description_clean"] = pdf["description_clean"].apply(clean_and_tokenize)

# Chuyển chuỗi thành danh sách từ (List of Lists) để Gensim hiểu
documents = [doc.split() for doc in pdf["description_clean"]]

# Tạo từ điển (Dictionary)
dictionary = corpora.Dictionary(documents)

# Chuyển văn bản thành dạng Bag-of-Words (BoW)
corpus = [dictionary.doc2bow(doc) for doc in documents]

# Huấn luyện mô hình TF-IDF
tfidf_model = gensim_models.TfidfModel(corpus)
corpus_tfidf = tfidf_model[corpus]
df.loc[df["id"] == 10, ["id", "description_clean"]]
def get_recommendations(target_id, top_n=5):
    # Logic lấy Similarity từ Gensim (giữ nguyên)
    target_idx = pdf[pdf['id'] == target_id].index[0]
    query_bow = corpus[target_idx]
    query_tfidf = tfidf_model[query_bow]

    # The previous del globals()['similarities'] was not effective for the line outside the function.
    # Using an alias for gensim.similarities directly addresses the naming conflict.
    sims = index[query_tfidf]
    sims = builtins.sorted(builtins.enumerate(sims), key=lambda item: -item[1]) # Use builtins.sorted

    # Lấy dữ liệu thô
    results = []
    for i, score in sims[1:top_n+1]:
        results.append((
            builtins.int(pdf.iloc[i]["id"]),
            pdf.iloc[i]["description_clean"],
            builtins.round(builtins.float(score), 4) # Explicitly use builtins.round for Python float
        ))

    recommend_df = pd.DataFrame(results)

    return recommend_df

# --- CHẠY VÀ IN KẾT QUẢ ---
# Tạo chỉ mục tương đồng (Similarity Index)
# num_features là số lượng từ trong từ điển
index = gensim_similarities.MatrixSimilarity(corpus_tfidf, num_features=len(dictionary))

target_id_test = 10
final_recommend_df = get_recommendations(target_id_test)

final_recommend_df.head(5)
# Chuẩn hóa GIÁ
scaler = MinMaxScaler()

# reshape(-1,1) vì sklearn cần 2D array
df["price_scaled"] = scaler.fit_transform(df[["gia_ban"]])

# One-hot encoding QUẬN
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

loc_encoded = encoder.fit_transform(df[["dia_chi_cu"]])

# Tạo tên cột cho one-hot
loc_columns = encoder.get_feature_names_out(["dia_chi_cu"])

# Chuyển thành DataFrame
df_loc = pd.DataFrame(loc_encoded, columns=loc_columns, index=df.index)

# Gộp lại với DataFrame gốc
df_transformed = pd.concat([df, df_loc], axis=1)
def get_content_similarity_dict(target_id, df_vectorized, X_norm):
    """
    df_vectorized: pandas DataFrame chứa 'id'
    X_norm: ma trận vector đã normalize (sparse matrix hoặc numpy array)
    """

    # Tìm index của target
    matches = df_vectorized.index[df_vectorized["id"] == target_id]
    if len(matches) == 0:
        return {}

    target_idx = matches[0]

    # Lấy vector target
    target_vector = X_norm[target_idx]

    # Tính cosine similarity (dot product vì đã normalize)
    similarities = cosine_similarity(target_vector, X_norm).flatten()

    # Trả về dict {id: score}
    return {
        int(df_vectorized.iloc[i]["id"]): float(similarities[i])
        for i in range(len(similarities))
    }
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_hybrid_recommendations(target_id, df_transformed, X_norm, loc_matrix, top_n=5):
    """
    df_transformed: DataFrame chứa ['id', 'description_clean', 'gia_ban', 'dia_chi_cu', 'price_scaled']
    X_norm: ma trận TF-IDF đã normalize (content features)
    loc_matrix: ma trận one-hot location (từ OneHotEncoder)
    """

    # Lấy index của target
    matches = df_transformed.index[df_transformed["id"] == target_id]
    if len(matches) == 0:
        return pd.DataFrame()

    target_idx = matches[0]

    # Content similarity
    content_sims = cosine_similarity(X_norm[target_idx], X_norm).flatten()

    # Price similarity
    target_price = df_transformed.loc[target_idx, "price_scaled"]
    price_sims = 1.0 - np.abs(df_transformed["price_scaled"].values - target_price)

    # Location similarity
    target_loc_vec = loc_matrix[target_idx].reshape(1, -1) # Reshape to 2D array
    loc_sims = cosine_similarity(target_loc_vec, loc_matrix).flatten()

    # Hybrid score
    alpha, beta, gamma = 0.6, 0.25, 0.15
    hybrid_scores = (alpha * content_sims) + (beta * price_sims) + (gamma * loc_sims)

    # Gán vào DataFrame
    df_transformed = df_transformed.copy()
    df_transformed["hybrid_score"] = hybrid_scores

    # Lọc + sort
    results = (
        df_transformed[df_transformed["id"] != target_id]
        .sort_values(by="hybrid_score", ascending=False)
        [["id", "description_clean", "gia_ban", "dia_chi_cu", "hybrid_score"]]
        .head(top_n)
    )

    return results
df.loc[df["id"] == 10, ["id", "description_clean"]]
# Execute recommendation
final_table = get_hybrid_recommendations(
    target_id=10,
    df_transformed=df_transformed,
    X_norm=X_norm,
    loc_matrix=loc_encoded
)

final_table
df2 = data_pr2_1.copy()
origin_numerical_cols = df2.select_dtypes(include=[np.number]).columns.tolist()
df2["quan"] = df2["dia_chi_cu"].str.extract(r"(quan_.*)$", expand=False)
df2.head(5)
visualize_skewness_pandas(df2)
visualize_outliers_pandas(df2)
lower_percentile = 0.004
upper_percentile = 0.996
cols_to_fix = ["chieu_ngang", "dien_tich_su_dung"]
df2 = trim_outliers(df2, cols_to_fix, lower_percentile, upper_percentile)
num_cols = origin_numerical_cols

selector = VarianceThreshold(threshold=0.01)
selector.fit(df[num_cols])

selected_cols = df[num_cols].columns[selector.get_support()].tolist()

print(f"Giữ lại {len(selected_cols)}/{len(num_cols)} features: {selected_cols}")
def vectorize_and_scale(df, numerical_cols):
    # Lấy dữ liệu số
    X = df[numerical_cols]

    # Scale bằng RobustScaler
    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0)  # tương đương lower=0.25, upper=0.75
    )

    X_scaled = scaler.fit_transform(X)

    # Gán lại vào DataFrame
    scaled_cols = [f"{col}_scaled" for col in numerical_cols]
    df_scaled = pd.DataFrame(X_scaled, columns=scaled_cols, index=df.index)

    # Gộp với DataFrame gốc
    vectorized_df = pd.concat([df, df_scaled], axis=1)

    return vectorized_df
def pca_dimension_reducing(df, numerical_cols):
    # Scale dữ liệu
    df_scaled = vectorize_and_scale(df, numerical_cols)
    X_scaled = df_scaled[[f"{col}_scaled" for col in numerical_cols]].values

    # PCA
    k = min(3, len(numerical_cols))
    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X_scaled)

    # Variance explained
    explained = pca.explained_variance_ratio_
    cumvar = explained.cumsum()
    k_95 = int((cumvar < 0.95).sum()) + 1

    print(f"Cần {k_95} PC để giữ 95% variance")
    print(f"Top PC variance: {[f'{v:.3f}' for v in explained[:5]]}")

    # Loading matrix
    loading_matrix = pca.components_.T  # shape: (n_features, k)

    # Đặt tên PC
    def make_pc_name(pc_idx, loadings_1d, feature_names, top_n=3):
        abs_load = np.abs(loadings_1d)
        top_idx = np.argsort(abs_load)[::-1][:top_n]
        top_names = [feature_names[i] for i in top_idx]
        return f"PC{pc_idx + 1}({','.join(top_names)})"

    pc_col_names = [
        make_pc_name(i, loading_matrix[:, i], numerical_cols, top_n=3)
        for i in range(k)
    ]

    print("Tên cột PC:", pc_col_names)

    # Tạo DataFrame PCA
    df_pca = df.copy()

    for i, col_name in enumerate(pc_col_names):
        df_pca[col_name] = X_pca[:, i]

    return df_pca
def execute_kmeans(df, k, pca_cols):
    """
    df: DataFrame chứa các cột PCA (ví dụ: PC1, PC2, PC3)
    pca_cols: list tên các cột PCA
    """

    # Lấy dữ liệu =====
    X = df[pca_cols].values

    # KMeans
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        init="k-means++",   # tương đương k-means||
        max_iter=100,
        tol=1e-6,
        n_init=10           # sklearn mặc định (quan trọng)
    )

    model = kmeans.fit(X)

    # Dự đoán cluster
    df_result = df.copy()
    df_result["prediction"] = model.labels_

    return model, df_result
def find_optimal_k_and_export_results(df, max_k, numerical_cols):

    silhouette_scores = []
    wcss_list = []
    result_kmeans = {}

    # PCA
    df_pca = pca_dimension_reducing(df, numerical_cols)

    # Lấy các cột PCA
    pca_cols = [col for col in df_pca.columns if col.startswith("PC")]
    X = df_pca[pca_cols].values

    # Loop K
    for k in range(2, max_k + 1):

        # Train KMeans
        model, predictions = execute_kmeans(df_pca, k, pca_cols)

        labels = predictions["prediction"].values

        # Silhouette Score
        silhouette = silhouette_score(X, labels)
        silhouette_scores.append((k, silhouette))

        # WCSS (inertia)
        wcss = model.inertia_
        wcss_list.append((k, wcss))

        print(f"K={k}: Silhouette={silhouette:.4f}, WCSS={wcss:.2f}")

        # Lưu kết quả
        result_kmeans[k] = {
            "model": model,
            "predictions": predictions,
            "silhouette": silhouette,
            "wcss": wcss
        }

    return silhouette_scores, wcss_list, result_kmeans
numerical_cols = ["dien_tich_dat", "dien_tich_su_dung", "chieu_ngang", "tong_so_tang", "so_phong_ngu", "gia_ban"]
# Áp dụng pipeline và trả về kết quả (chọn max_k = 10: K-Means khởi chạy 10 vòng lặp để tìm K tối ưu)
max_k = 10
silhouette_scores, wcss_list, result_kmeans = find_optimal_k_and_export_results(df2, max_k, numerical_cols)

# Lấy danh sách k và wcsse
k_elbow_values = [x[0] for x in wcss_list]
wcss_values = [x[1] for x in wcss_list]

# Tìm K có Silhouette cao nhất
best_silhouette_tuple = builtins.max(silhouette_scores, key=lambda x: x[1])
best_k_for_silhouette = best_silhouette_tuple[0]
max_silhouette_score_value = best_silhouette_tuple[1]

elbow_k = 4  # Điểm K tự chọn

# Tính tốc độ giảm WCSSE (chênh lệch giữa các K liên tiếp)
wcsse_diff = []

for i in range(1, len(wcss_values)):
    diff = wcss_values[i-1] - wcss_values[i]  # Mức giảm khi tăng K thêm 1
    wcsse_diff.append(diff)

k_diff = k_elbow_values[1:]  # K từ 3 trở đi (vì diff bắt đầu từ K=3)

# Vẽ biểu đồ
plt.figure(figsize=(10, 4))
plt.bar(k_diff, wcsse_diff, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Số cụm K', fontsize=12)
plt.ylabel('Mức giảm WCSSE khi tăng K', fontsize=12)
plt.title('Tốc độ giảm WCSSE - Elbow rõ hơn ở cột cao nhất', fontsize=12)
plt.xticks(range(3, 31, 2))
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Đánh dấu điểm giảm mạnh nhất
max_diff_k = k_diff[wcsse_diff.index(builtins.max(wcsse_diff))]
plt.axvline(x=max_diff_k, color='red', linestyle='--', linewidth=2,
            label=f'Giảm mạnh nhất tại K={max_diff_k}')
plt.legend()
plt.tight_layout()
plt.show()

print(f"K có mức giảm WCSSE lớn nhất: K = {max_diff_k}")
print(f"Gợi ý: Elbow nằm ở K = {max_diff_k} hoặc K = {max_diff_k + 1}")
print('-'*100 + '\n')

# Tạo figure với 2 subplot cạnh nhau
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# ========== Biểu đồ 1: Elbow Method (WSSSE) ==========
axes[0].plot(k_elbow_values, wcss_values, 'bo-', linewidth=2, markersize=6)
axes[0].set_xlabel('Số cụm K', fontsize=12)
axes[0].set_ylabel('WCSSE (Within-cluster Sum of Squared Errors)', fontsize=12)
axes[0].set_title(f'Phương pháp Elbow - Chọn K tối ưu (K = 2 đến {max_k})', fontsize=14)
axes[0].set_xticks(range(2, 31))
axes[0].grid(True, linestyle='--', alpha=0.7)

# Đánh dấu điểm khuỷu tay
elbow_k_optimal = max_diff_k

axes[0].axvline(x=elbow_k_optimal, color='red', linestyle='--', linewidth=2, label=f'Elbow tại K={elbow_k_optimal}')
axes[0].axvline(x=elbow_k, color='green', linestyle='--', linewidth=2, label=f'Chọn K={elbow_k}')

axes[0].scatter([elbow_k], [wcss_values[elbow_k-2]], color='red', s=140, zorder=5, marker='*')
axes[0].legend(fontsize=10)

# ========== Biểu đồ 2: Silhouette Score ==========
k_sil_values = [x[0] for x in silhouette_scores] # K values for silhouette
sil_scores_values = [x[1] for x in silhouette_scores] # Silhouette scores

axes[1].plot(k_sil_values, sil_scores_values, 'go-', linewidth=2, markersize=6)
axes[1].set_xlabel('Số cụm K', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title(f'Silhouette Score theo K (K = 2 đến {max_k})', fontsize=14)
axes[1].set_xticks(range(2, 31))
axes[1].grid(True, linestyle='--', alpha=0.7)

# Đánh dấu điểm có silhouette cao nhất
axes[1].axvline(x=best_k_for_silhouette, color='red', linestyle='--', linewidth=2, label=f'Best K={best_k_for_silhouette}')
axes[1].axvline(x=elbow_k, color='green', linestyle='--', linewidth=2, label=f'Chọn K={elbow_k}')
axes[1].scatter([elbow_k], [sil_scores_values[elbow_k-2]], color='red', s=140, zorder=5, marker='*')
axes[1].legend(fontsize=10)

# Thêm đường tham chiếu Silhouette = 0.5 (ngưỡng "chấp nhận được")
axes[1].axhline(y=0.5, color='orange', linestyle=':', linewidth=1.5, label='Ngưỡng 0.5')

plt.tight_layout()
plt.savefig('elbow_silhouette_k30.png', dpi=150, bbox_inches='tight')  # Lưu hình
plt.show()
predictions = result_kmeans[elbow_k]["predictions"]

print(predictions.head())
df_clustered = predictions.copy()

pca_numerical_cols = [
    'x1_phan_khuc_gia_theo_ti_le_dt_dat_dt_su_dung',
    'x2_phan_khuc_gia_theo_ti_le_chieu_ngang_tong_so_tang',
    'x3_phan_khuc_gia_theo_ti_le_so_phong_ngu_tong_so_tang'
]

# Lấy các cột PCA hiện có
pca_cols = [col for col in df_clustered.columns if col.startswith("PC")]

# Rename PC → tên mới
rename_dict = dict(zip(pca_cols, pca_numerical_cols))
df_clustered = df_clustered.rename(columns=rename_dict)

df_clustered.head(1)
centers = result_kmeans[elbow_k]['model'].cluster_centers_
centers_df = pd.DataFrame(centers)
centers_df = centers_df.rename(columns={0: "x1_center", 1: "x2_center", 2: "x3_center"})
print("Clustering Centers: ")
centers_df.head()
def visualize_2D_clustering_scatter(df, prediction_col, selected_dimensions, centers_df_pairing, labels):
    #
    pdf_clustered = df.copy()

    # Khởi tạo biểu đồ
    fig, ax = plt.subplots(figsize=(10, 7))

    # Thiết lập bảng màu tự động dựa trên số lượng cụm
    clusters = sorted(pdf_clustered[prediction_col].unique())
    colors = plt.cm.get_cmap('viridis', len(clusters))
    for i, cluster in enumerate(clusters):
        # Lọc dữ liệu cho từng cụm
        cluster_data = pdf_clustered[pdf_clustered[prediction_col] == cluster]

        # Kiểm tra nếu cụm có dữ liệu mới vẽ
        if not cluster_data.empty:
            ax.scatter(
                cluster_data[selected_dimensions[0]],
                cluster_data[selected_dimensions[1]],        # Trục Y dùng Giá bán để thấy insight rõ hơn
                label=f'Cụm {cluster}',
                alpha=0.6,
                edgecolors='w',
                s=80 # Kích thước điểm
            )

        # Vẽ center lên plot
        sns.scatterplot(data=centers_df,
                        x=centers_df_pairing[0],
                        y=centers_df_pairing[1],
                        color="black",
                        marker="x",
                        linewidths=4,
                        s=300)

    # Trang trí biểu đồ
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(labels[2])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.show()
# def combine_feature_pairs(numerical_cols):
#     # Đầu vào nguyên bản
#     # numerical_cols = ["dien_tich_dat", "dien_tich_su_dung", "chieu_ngang", "tong_so_tang", "so_phong_ngu", "gia_ban"]

#     # Tạo danh sách tên mới với tiền tố x1, x2, x3...
#     # enumerate(..., 1) giúp bắt đầu đếm từ 1
#     numerical_cols_renamed = [f"x{i}_{col}" for i, col in enumerate(numerical_cols, 1)]

#     # Tạo danh sách các cặp từ danh sách đã đổi tên
#     numerical_cols_vectorized_pairing = [list(pair) for pair in combinations(numerical_cols_renamed, 2)]

#     # Hiển thị kết quả
#     print("Danh sách các cột đã đánh số:")
#     print(numerical_cols_renamed)
#     print("\nDanh sách các cặp feature (pairing):")
#     for pair in numerical_cols_vectorized_pairing:
#         print(pair)

#     return numerical_cols_vectorized_pairing
prediction_col = 'prediction'
mapping = dict(zip(pca_numerical_cols, centers_df.columns))
numerical_pairs = list(combinations(pca_numerical_cols, 2))
centers_pairs = [
    [mapping[col1], mapping[col2]]
    for col1, col2 in numerical_pairs
]

for (dim1, dim2), (c1, c2) in zip(numerical_pairs, centers_pairs):
    #
    labels = [
        dim1,
        dim2,
        f'Phân nhóm BĐS theo {dim1} và {dim2}'
    ]
    #
    visualize_2D_clustering_scatter(
        df_clustered,
        prediction_col,
        selected_dimensions=[dim1, dim2],
        centers_df_pairing=[c1, c2],
        labels=labels
    )
def visualize_clustering_boxplots(df, prediction_col, features_to_plot):
    pdf_clustered = df.copy()

    clusters = sorted(pdf_clustered[prediction_col].unique())
    num_clusters = len(clusters)
    num_features = len(features_to_plot)

    # Dynamically create subplots
    num_cols_per_row = min(num_features, 3) # Max 3 columns per row
    num_rows = math.ceil(num_features / num_cols_per_row)
    fig, axes = plt.subplots(num_rows, num_cols_per_row, figsize=(num_cols_per_row * 5, num_rows * 4))
    axes = axes.flatten() if num_rows > 1 or num_cols_per_row > 1 else [axes] # Ensure axes is iterable

    # Set up a colormap for clusters
    # Ensure num_clusters is at least 1 for cmap indexing
    cluster_colors_list = [plt.cm.get_cmap('viridis', max(1, num_clusters))(c_idx / max(1, num_clusters - 1)) for c_idx in range(num_clusters)]

    for idx, feature in enumerate(features_to_plot):
        if idx < len(axes):
            sns.boxplot(
                data=pdf_clustered,
                x=prediction_col,
                y=feature,
                ax=axes[idx],
                palette=cluster_colors_list
            )
            axes[idx].set_title(f'Distribution of x{idx+1} by Cluster', fontsize=12)
            axes[idx].set_xlabel('Cluster')
            axes[idx].set_ylabel(feature)
            axes[idx].grid(True, linestyle='--', alpha=0.6)
        else:
            print(f"Warning: Not enough subplots for feature {feature}. Skipping.")

    # Remove any unused subplots
    for i in range(num_features, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()
visualize_clustering_boxplots(df_clustered, prediction_col, pca_numerical_cols)
def visualize_2D_clustering_heatmap(df, prediction_col, features_to_plot):
    # Calculate the mean of features_to_plot for each cluster
    cluster_means = df.groupby(prediction_col)[features_to_plot].mean()

    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_means.T, annot=True, cmap='YlGnBu', fmt='.3f',
                yticklabels=features_to_plot,
                xticklabels=[f'Cụm {i}' for i in cluster_means.index])
    plt.title('Trung bình các thành phần PCA theo cụm', fontsize=14)
    plt.ylabel('Thành phần PCA')
    plt.xlabel('Cụm')
    plt.show()
visualize_2D_clustering_heatmap(df_clustered, prediction_col, pca_numerical_cols)
def visualize_3D_clustering(df, prediction_col, selected_dimensions, model):
    #
    pdf_clustered = df.copy()

    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'blue', 'green', 'orange', 'purple']

    # Vẽ các điểm
    for cluster in pdf_clustered[prediction_col].unique():
        cluster_data = pdf_clustered[pdf_clustered[prediction_col] == cluster]
        ax.scatter(
            cluster_data[selected_dimensions[0]],
            cluster_data[selected_dimensions[1]],
            cluster_data[selected_dimensions[2]],
            c=colors[cluster % len(colors)],
            label=f'Cụm {cluster}',
            alpha=0.6,
            s=6
        )

    # Trang trí
    ax.set_xlabel(selected_dimensions[0], fontsize=12)
    ax.set_ylabel(selected_dimensions[1], fontsize=12)
    ax.set_zlabel(selected_dimensions[2], fontsize=12)

    # Chỉnh góc nhìn để thấy rõ nhãn Z
    # ax.view_init(elev=20, azim=45)

    ax.set_title(f'Phân cụm K-Means (K={elbow_k})', fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(0.8, 0.9), fontsize=12)
    plt.tight_layout()
    plt.show()


prediction_col = 'prediction'
model = result_kmeans[elbow_k]["model"]
selected_dimensions = ['dien_tich_su_dung', 'chieu_ngang', 'gia_ban']
visualize_3D_clustering(df_clustered, prediction_col, selected_dimensions, model)
def separated_clustered_df(df, prediction_col):
    # Lấy các cluster unique
    clusters = sorted(df[prediction_col].dropna().unique())

    sc_df_list = []

    for cluster_id in clusters:
        # Lọc dữ liệu theo từng cluster
        cluster_data = df[df[prediction_col] == cluster_id]

        sc_df_list.append(cluster_data)

    return sc_df_list
df_clustering = separated_clustered_df(df_clustered, prediction_col)
pc_df_1 = df_clustering[0]
pc_df_1.head()

# for i, cluster in enumerate(list(separated_clustered_df(pdf_clustered, prediction_col)), start=0):
#     print(cluster[i])
tmp_pc_df_1 = pc_df_1.drop(columns=['prediction'] + [c for c in pc_df_1.columns if c.startswith('PC_')])
visualize_correlation(tmp_pc_df_1)
cluster_price_stats = predictions.groupby("prediction").agg(
    Avg_Price=('gia_ban', 'mean'),
    Min_Price=('gia_ban', 'min'),
    Max_Price=('gia_ban', 'max'),
    Count=('gia_ban', 'count') # Hoặc dùng cột 'id' nếu có: ('id', 'count')
).reset_index()

# Sắp xếp theo cluster ID (tương đương .orderBy trong Spark)
cluster_price_stats = cluster_price_stats.sort_values("prediction")

# Melt DataFrame sang dạng long format để vẽ biểu đồ với Seaborn
cluster_price_melted = cluster_price_stats.melt(
    id_vars=['prediction', 'Count'],
    value_vars=['Avg_Price', 'Min_Price', 'Max_Price'],
    var_name='Price_Metric',
    value_name='Price_Value'
)

# Vẽ biểu đồ
plt.figure(figsize=(12, 7))
sns.barplot(
    x='prediction', 
    y='Price_Value', 
    hue='Price_Metric', 
    data=cluster_price_melted, 
    palette='viridis'
)

plt.xlabel('Cluster ID')
plt.ylabel('Price (Tỷ VNĐ)')
plt.title('Giá Trung bình, Nhỏ nhất và Lớn nhất theo từng Cụm (Cluster)')
plt.legend(title='Chỉ số giá')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
# Nhóm dữ liệu và tính toán thống kê diện tích bằng Pandas
cluster_dien_tich_dat_stats = predictions.groupby("prediction").agg(
    Avg_dien_tich_dat=('dien_tich_dat', 'mean'),
    Min_dien_tich_dat=('dien_tich_dat', 'min'),
    Max_dien_tich_dat=('dien_tich_dat', 'max'),
    Count=('gia_ban', 'count')
).reset_index()

# Sắp xếp theo Cluster ID
cluster_dien_tich_dat_stats = cluster_dien_tich_dat_stats.sort_values("prediction")

# Melt DataFrame sang dạng long format để Seaborn có thể vẽ được
cluster_dien_tich_dat_melted = cluster_dien_tich_dat_stats.melt(
    id_vars=['prediction', 'Count'],
    value_vars=['Avg_dien_tich_dat', 'Min_dien_tich_dat', 'Max_dien_tich_dat'],
    var_name='dien_tich_dat_Metric',
    value_name='dien_tich_dat_Value'
)

# Cấu hình và vẽ biểu đồ
plt.figure(figsize=(12, 7))
sns.barplot(
    x='prediction', 
    y='dien_tich_dat_Value', 
    hue='dien_tich_dat_Metric', 
    data=cluster_dien_tich_dat_melted, 
    palette='viridis'
)

# Việt hóa các nhãn biểu đồ cho chuyên nghiệp
plt.xlabel('Cụm (Cluster ID)')
plt.ylabel('Diện tích đất ($m^2$)')
plt.title('Thống kê Diện tích đất: Trung bình, Nhỏ nhất và Lớn nhất theo từng Cụm')
plt.legend(title='Chỉ số diện tích')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
cluster_dien_tich_su_dung_stats = predictions.groupby("prediction").agg(
    Avg_dien_tich_su_dung=('dien_tich_su_dung', 'mean'),
    Min_dien_tich_su_dung=('dien_tich_su_dung', 'min'),
    Max_dien_tich_su_dung=('dien_tich_su_dung', 'max'),
    Count=('gia_ban', 'count')
).reset_index()

# Sắp xếp theo ID của cụm
cluster_dien_tich_su_dung_stats = cluster_dien_tich_su_dung_stats.sort_values("prediction")

# 2. Chuyển đổi DataFrame sang dạng long format (Melt) để vẽ biểu đồ
cluster_dien_tich_su_dung_melted = cluster_dien_tich_su_dung_stats.melt(
    id_vars=['prediction', 'Count'],
    value_vars=['Avg_dien_tich_su_dung', 'Min_dien_tich_su_dung', 'Max_dien_tich_su_dung'],
    var_name='dien_tich_su_dung_Metric',
    value_name='dien_tich_su_dung_Value'
)

# 3. Trực quan hóa dữ liệu
plt.figure(figsize=(12, 7))
sns.barplot(
    x='prediction', 
    y='dien_tich_su_dung_Value', 
    hue='dien_tich_su_dung_Metric', 
    data=cluster_dien_tich_su_dung_melted, 
    palette='viridis'
)

# Tinh chỉnh các nhãn hiển thị
plt.xlabel('Mã cụm (Cluster ID)')
plt.ylabel('Diện tích sử dụng (m2)')
plt.title('Thống kê Diện tích sử dụng: Trung bình, Nhỏ nhất và Lớn nhất theo từng Cụm')
plt.legend(title='Chỉ số diện tích sử dụng')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
cluster_chieu_ngang_stats = predictions.groupby("prediction").agg(
    Avg_chieu_ngang=('chieu_ngang', 'mean'),
    Min_chieu_ngang=('chieu_ngang', 'min'),
    Max_chieu_ngang=('chieu_ngang', 'max'),
    Count=('gia_ban', 'count')
).reset_index()

# Sắp xếp theo ID của cụm (tương đương .orderBy trong Spark)
cluster_chieu_ngang_stats = cluster_chieu_ngang_stats.sort_values("prediction")

# 2. Chuyển đổi DataFrame sang dạng long format (Melt) để Seaborn vẽ barplot
cluster_chieu_ngang_melted = cluster_chieu_ngang_stats.melt(
    id_vars=['prediction', 'Count'],
    value_vars=['Avg_chieu_ngang', 'Min_chieu_ngang', 'Max_chieu_ngang'],
    var_name='chieu_ngang_Metric',
    value_name='chieu_ngang_Value'
)

# 3. Trực quan hóa dữ liệu
plt.figure(figsize=(12, 7))
sns.barplot(
    x='prediction', 
    y='chieu_ngang_Value', 
    hue='chieu_ngang_Metric', 
    data=cluster_chieu_ngang_melted, 
    palette='viridis'
)

# Tinh chỉnh các nhãn hiển thị cho dự án Quận 7
plt.xlabel('Mã cụm (Cluster ID)')
plt.ylabel('Chiều ngang (m)')
plt.title('Thống kê Chiều ngang: Trung bình, Nhỏ nhất và Lớn nhất theo từng Cụm')
plt.legend(title='Chỉ số chiều ngang')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
cluster_tong_so_tang_stats = predictions.groupby("prediction").agg(
    Avg_tong_so_tang=('tong_so_tang', 'mean'),
    Min_tong_so_tang=('tong_so_tang', 'min'),
    Max_tong_so_tang=('tong_so_tang', 'max'),
    Count=('gia_ban', 'count')
).reset_index()

# Sắp xếp theo ID của cụm
cluster_tong_so_tang_stats = cluster_tong_so_tang_stats.sort_values("prediction")

# 2. Chuyển đổi sang dạng long format (Melt) để vẽ biểu đồ
cluster_tong_so_tang_melted = cluster_tong_so_tang_stats.melt(
    id_vars=['prediction', 'Count'],
    value_vars=['Avg_tong_so_tang', 'Min_tong_so_tang', 'Max_tong_so_tang'],
    var_name='tong_so_tang_Metric',
    value_name='tong_so_tang_Value'
)

# 3. Trực quan hóa
plt.figure(figsize=(12, 7))
sns.barplot(
    x='prediction', 
    y='tong_so_tang_Value', 
    hue='tong_so_tang_Metric', 
    data=cluster_tong_so_tang_melted, 
    palette='viridis'
)

# Tinh chỉnh hiển thị
plt.xlabel('Mã cụm (Cluster ID)')
plt.ylabel('Tổng số tầng')
plt.title('Thống kê Tổng số tầng: Trung bình, Nhỏ nhất và Lớn nhất theo từng Cụm')
plt.legend(title='Chỉ số số tầng')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()