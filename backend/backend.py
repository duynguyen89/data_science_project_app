import os
import re
import math
import unicodedata
import joblib
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import streamlit as st

# Required for xgboost pipeline unpickling
def convert_to_string(x):
    return x.astype(str)
import sys
sys.modules['__main__'].convert_to_string = convert_to_string

# Global variables to store trained artifacts
lambdas = {}
xgboost_model = None

@st.cache_resource
def load_xgboost_model():
    global xgboost_model
    if xgboost_model is None:
        xgboost_model = joblib.load('models/model_gia_nha_xgboost.pkl')
    return xgboost_model

def standardize_price(price_str):
    if pd.isna(price_str): return None
    price_str = str(price_str).lower().replace(',', '.')
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", price_str)
    if not nums: return None
    val = float(nums[0])
    if 'tỷ' in price_str: return val
    if 'triệu' in price_str: return val / 1000
    return val

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
    if tang_matches: return int(tang_matches[0])
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
        for f in floor_numbers: total_floors += int(f)
        for m in mezz_numbers: total_floors += int(m)
        if has_lau and not floor_numbers: total_floors += 1
        if has_lung and not mezz_numbers: total_floors += 1
        return total_floors
    return None

def sanitize_string(s):
    if s is None or pd.isna(s): return None
    s = str(s).lower()
    s = unicodedata.normalize('NFKD', s)
    s = "".join([c for c in s if not unicodedata.combining(c)])
    s = s.replace('đ', 'd').replace('Đ', 'D')
    s = re.sub(r'[^\w\s]', '_', s)
    s = re.sub(r'[\s,:\\[\]{}]+', '_', s)
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    return s

@st.cache_data
def load_and_clean_data():
    global lambdas
    data = pd.read_csv("data/cleaned_data/clean_data_without_outliers_with_skewness.csv")
    
    # Fill any null values left over or required for display
    data['mo_ta'] = data['mo_ta'].fillna('')
    data['tieu_de'] = data['tieu_de'].fillna('')
    data['giay_to_phap_ly'] = data['giay_to_phap_ly'].fillna('Đang chờ sổ')
    
    return data

@st.cache_data
def _unused_legacy_loader():
    
    # Preprocessing
    data['gia_ban'] = data['gia_ban'].apply(standardize_price)
    
    # Numeric conversion
    cols_to_convert = ['dien_tich', 'dien_tich_dat', 'dien_tich_su_dung', 'so_phong_ngu', 'chieu_ngang', 'chieu_dai', 'tong_so_tang']
    for col in cols_to_convert:
        if col in data.columns:
            data[col] = data[col].astype(str).str.replace(r'[^0-9,.]', '', regex=True).str.replace(',', '.')
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Floor extraction
    data['mo_ta_so_tang'] = data['mo_ta'].apply(extract_floor_count)
    data['tieu_de_so_tang'] = data['tieu_de'].apply(extract_floor_count)
    mask_all_null = data['tong_so_tang'].isna() & data['tieu_de_so_tang'].isna() & data['mo_ta_so_tang'].isna()
    data.loc[mask_all_null, 'tong_so_tang'] = 1
    data['tong_so_tang'] = data['tong_so_tang'].fillna(data['mo_ta_so_tang'])
    data['tong_so_tang'] = data['tong_so_tang'].fillna(data['tieu_de_so_tang'])
    
    # Drop rows without required basics just to stabilize distributions
    data = data.dropna(subset=['dien_tich_dat'])
    
    # Add an ID column
    data = data.reset_index(drop=True)
    data['id'] = data.index.values
    
    if 'dia_chi' in data.columns:
        data['dia_chi_cu'] = data['dia_chi'].str.extract(r'(Phường\s+[^,]+,\s+Quận\s+[^,]+)', expand=False)
        data['dia_chi_moi'] = data['dia_chi'].str.extract(r'\((Phường\s+[^,]+)', expand=False)
        data['dia_chi_cu'] = data['dia_chi_cu'].replace(['', np.nan], 'Chưa xác định')
        data['dia_chi_moi'] = data['dia_chi_moi'].replace(['', np.nan], 'Chưa xác định')
    
    # Calculate boxcox lambdas
    for col in ['dien_tich_su_dung', 'chieu_ngang', 'tong_so_tang']:
        valid_data = data[col].dropna()
        valid_data = valid_data[valid_data > 0]
        if len(valid_data) > 0:
            _, lmbda = stats.boxcox(valid_data)
            lambdas[col] = lmbda
            
    return data

@st.cache_resource
def get_recommendation_system(df):
    df_rec = df.dropna(subset=['tieu_de', 'mo_ta']).copy()
    df_rec = df_rec.reset_index(drop=True)
    df_rec['id'] = df_rec.index.values
    def clean_text(text):
        if pd.isna(text): return ""
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.strip()
    
    df_rec['text_combined'] = (df_rec['tieu_de'].fillna("") + " " + df_rec['mo_ta'].fillna("")).apply(clean_text)
    vectorizer = HashingVectorizer(n_features=5000, lowercase=False)
    tfidf_matrix = vectorizer.transform(df_rec['text_combined'])
    return df_rec, tfidf_matrix

def recommend_houses(df_rec, tfidf_matrix, target_id, top_n=5):
    if target_id not in df_rec['id'].values:
        return pd.DataFrame()
    idx = df_rec[df_rec['id'] == target_id].index[0]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[-(top_n+1):-1][::-1]
    return df_rec.iloc[sim_indices]

@st.cache_resource
def get_kmeans_pipeline_model():
    model_path = 'models/model_kmeans_pipeline.pkl'
    results_path = 'data/cleaned_data/clustering_results_kmeans.csv'
    
    import joblib
    import os
    if os.path.exists(model_path) and os.path.exists(results_path):
        pipeline = joblib.load(model_path)
        df_cluster = pd.read_csv(results_path)
        return pipeline, df_cluster
    else:
        raise Exception(f"Không tìm thấy Model tại {model_path} hoặc Data tại {results_path}")

@st.cache_resource
def get_anomaly_model(df):
    model_path = 'models/model_anomaly_detection_IsolationForest.pkl'
    stats_path = 'models/anomaly_stats.pkl'
    
    num_cols = ["dien_tich_su_dung", "gia_ban", "so_phong_ngu", "chieu_ngang", "dien_tich_dat", "tong_so_tang"]
    req_cols = num_cols + ["dia_chi_cu", "loai_hinh"]
    
    # Drop where essential numericals are mostly missing
    df_anomaly = df.dropna(subset=["dien_tich_su_dung", "gia_ban"]).copy()
    
    # Fill remaining numerical NaNs with medians to allow training
    for col in num_cols:
        if col in df_anomaly.columns:
            df_anomaly[col] = df_anomaly[col].fillna(df_anomaly[col].median())
    
    X = df_anomaly[num_cols].fillna(0)
    
    # Calculate global stats
    df_anomaly['price_per_m2'] = df_anomaly['gia_ban'] / df_anomaly['dien_tich_su_dung']
    stats_dict = {
        'median_gia_ban': df_anomaly['gia_ban'].median(),
        'median_dien_tich': df_anomaly['dien_tich_su_dung'].median(),
        'median_price_per_m2': df_anomaly['price_per_m2'].median()
    }
    
    # Calculate group stats for Local comparisons
    if 'dia_chi_cu' in df_anomaly.columns:
        local_stats = df_anomaly.groupby('dia_chi_cu')['price_per_m2'].median().to_dict()
        stats_dict['local_price_per_m2'] = local_stats
        
    if 'loai_hinh' in df_anomaly.columns:
        type_stats = df_anomaly.groupby('loai_hinh')['dien_tich_dat'].median().to_dict()
        stats_dict['type_median_area'] = type_stats
    
    if os.path.exists(model_path) and os.path.exists(stats_path):
        isf = joblib.load(model_path)
        stats_dict = joblib.load(stats_path)
    else:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df_anomaly.copy(), test_size=0.2, random_state=42)
        
        X_train = train_df[num_cols].fillna(0)
        
        isf = IsolationForest(contamination=0.05, random_state=42)
        isf.fit(X_train)
        
        joblib.dump(isf, model_path)
        joblib.dump(stats_dict, stats_path)
        
    return isf, stats_dict

def predict_price(input_features, lambdas_dict):
    model = load_xgboost_model()
    # Create single row df
    df_input = pd.DataFrame([input_features])
    
    # Apply transformations expected by the pipeline
    # Expected: ['dien_tich_dat' 'dien_tich_su_dung_boxcox' 'chieu_ngang_boxcox' 'tong_so_tang_boxcox' 'so_phong_ngu_sqrt' 'loai_hinh' 'giay_to_phap_ly' 'tinh_trang_noi_that' 'huong_cua_chinh' 'dac_diem' 'dia_chi_cu' 'dia_chi_moi']
    df_model_input = pd.DataFrame()
    
    # Numerics
    df_model_input['dien_tich_dat'] = df_input['dien_tich_dat']
    df_model_input['so_phong_ngu_sqrt'] = np.sqrt(df_input['so_phong_ngu'].astype(float))
    
    for col in ['dien_tich_su_dung', 'chieu_ngang', 'tong_so_tang']:
        lmbda = lambdas_dict.get(col, 1.0) # Fallback to 1.0 if not found
        val = df_input[col].astype(float).values[0]
        if val > 0:
            df_model_input[f'{col}_boxcox'] = stats.boxcox([val], lmbda=lmbda)[0]
        else:
            df_model_input[f'{col}_boxcox'] = 0.0
            
    # Categoricals
    cat_cols = ['loai_hinh', 'giay_to_phap_ly', 'tinh_trang_noi_that', 'huong_cua_chinh', 'dac_diem', 'dia_chi_cu', 'dia_chi_moi']
    for col in cat_cols:
        df_model_input[col] = df_input[col].apply(sanitize_string)
        
    pred = model.predict(df_model_input)
    return pred[0]
