import pandas as pd
import numpy as np
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Mocking stats_dict to simulate the data
stats_dict = {
    'median_price_per_m2': 0.1,
    'median_dien_tich': 50.0,
    'median_gia_ban': 5.0,
    'local_price_per_m2': {'Phường 7, Quận Bình Thạnh': 0.1},
    'type_median_area': {'Biệt thự, nhà liền kề': 100.0, 'Nhà ngõ, hẻm': 40.0}
}

def explain_anomaly_extended(dtsd, price, ngang, phong, dtd, tang, loai_hinh, phap_ly, noi_that, dac_diem, huong, dia_chi_cu, dia_chi_moi, anomaly_pred):
    print("="*60)
    print(f"TEST CASE INPUTS:")
    print(f"Loại hình: {loai_hinh} | Diện tích: {dtd}m2 đất, {dtsd}m2 SD | Số tầng: {tang} | Số phòng: {phong} | Ngang: {ngang}m")
    print(f"Giá: {price} Tỷ | Pháp lý: {phap_ly} | Nội thất: {noi_that} | Địa chỉ: {dia_chi_cu}")
    print("-" * 60)
    
    if anomaly_pred == 1:
        print("✅ Phân tích: Tin đăng bình thường.")
        print("="*60 + "\n")
        return

    print("⚠️ Phân tích: ĐÂY LÀ TIN ĐĂNG BẤT THƯỜNG (Outlier)")
    
    reasons = []
    price_per_m2 = price / dtsd if dtsd > 0 else 0
    local_price = stats_dict.get('local_price_per_m2', {}).get(dia_chi_cu, stats_dict['median_price_per_m2'])
    type_area = stats_dict.get('type_median_area', {}).get(loai_hinh, stats_dict['median_dien_tich'])

    # 1. Price checks
    if price_per_m2 < (local_price * 0.3):
        reasons.append(f"- [Chênh lệch giá khu vực] Giá/m2 = {price_per_m2:.3f} quá thấp so với kv ({local_price:.3f}).")
    
    # 2. Structural checks (Area vs Floors)
    expected_usability = dtd * tang * 1.2
    if dtsd > expected_usability and tang > 0:
        reasons.append(f"- [Diện tích phi lý] Diện tích SD ({dtsd}) vô lý so với {dtd}m2 đất x {tang} tầng.")
        
    # 3. Categorical checks
    if loai_hinh == 'Biệt thự, nhà liền kề' and dtd < 40:
        reasons.append(f"- [Phân loại sai lệch] Khai là '{loai_hinh}' nhưng diện tích đất siêu nhỏ ({dtd} m2).")
    
    if loai_hinh == 'Nhà ngõ, hẻm' and ngang > 10 and dtd < 50:
        reasons.append("- [Sai lệch ngõ hẻm] Nhà trong hẻm nhưng chiều ngang cực kỳ to so với diện tích đất.")
        
    if phap_ly == 'Đã có sổ' and price < (local_price * dtsd * 0.3):
        reasons.append("- [Pháp lý ảo] Có sổ nhưng giá cực rẻ -> Thường là sổ giả hoặc đánh tráo vị trí.")
        
    if pd.notna(noi_that) and 'Cao cấp' in str(noi_that) and price < (local_price * dtsd * 0.5):
        reasons.append("- [Tin câu view Nội thất] Nội thất 'Cao cấp' nhưng giá rẽ mạt.")

    if phong > (tang * 4) and tang > 0:
        reasons.append(f"- [Số phòng ảo] {phong} phòng nhưng chỉ có {tang} tầng -> Không thể thiết kế.")

    if not reasons:
        reasons.append("- [Tổ hợp] Bất thường do máy học phát hiện (không thuộc rule cứng).")
        
    for r in reasons:
        print(r)
    print("="*60 + "\n")

# -- RUNNING THE 7 TEST CASES --

# 1. Chênh lệch giá (Price is dirt cheap for local)
explain_anomaly_extended(
    dtsd=100.0, price=0.5, ngang=4.0, phong=2, dtd=50.0, tang=2,
    loai_hinh='Nhà mặt phố', phap_ly='Khác', noi_that='Cơ bản', dac_diem='', huong='', dia_chi_cu='Phường 7, Quận Bình Thạnh', dia_chi_moi='', anomaly_pred=-1
)

# 2. Diện tích phi lý (DTSD = 500 but Land=30, Floor=1)
explain_anomaly_extended(
    dtsd=500.0, price=3.0, ngang=4.0, phong=2, dtd=40.0, tang=1,
    loai_hinh='Nhà mặt phố', phap_ly='Khác', noi_that='Cơ bản', dac_diem='', huong='', dia_chi_cu='Phường 7, Quận Bình Thạnh', dia_chi_moi='', anomaly_pred=-1
)

# 3. Phân loại sai lệch (Biệt thự but DTD=20m2)
explain_anomaly_extended(
    dtsd=60.0, price=5.0, ngang=4.0, phong=2, dtd=20.0, tang=3,
    loai_hinh='Biệt thự, nhà liền kề', phap_ly='Khác', noi_that='Cơ bản', dac_diem='', huong='', dia_chi_cu='Phường 7, Quận Bình Thạnh', dia_chi_moi='', anomaly_pred=-1
)

# 4. Sai lệch ngõ hẻm (Hẻm but Ngang=15m, DTD=30m)
explain_anomaly_extended(
    dtsd=60.0, price=2.0, ngang=15.0, phong=2, dtd=30.0, tang=2,
    loai_hinh='Nhà ngõ, hẻm', phap_ly='Khác', noi_that='Cơ bản', dac_diem='', huong='', dia_chi_cu='Phường 7, Quận Bình Thạnh', dia_chi_moi='', anomaly_pred=-1
)

# 5. Pháp lý ảo & 6. Nội thất ảo (Đã có sổ + Nội thất Cao cấp, cheap price)
explain_anomaly_extended(
    dtsd=100.0, price=0.5, ngang=5.0, phong=2, dtd=50.0, tang=2,
    loai_hinh='Nhà mặt phố', phap_ly='Đã có sổ', noi_that='Nội thất Cao cấp', dac_diem='', huong='', dia_chi_cu='Phường 7, Quận Bình Thạnh', dia_chi_moi='', anomaly_pred=-1
)

# 7. Số phòng ảo (20 rooms on 1 floor)
explain_anomaly_extended(
    dtsd=100.0, price=5.0, ngang=5.0, phong=20, dtd=100.0, tang=1,
    loai_hinh='Nhà mặt phố', phap_ly='Khác', noi_that='Cơ bản', dac_diem='', huong='', dia_chi_cu='Phường 7, Quận Bình Thạnh', dia_chi_moi='', anomaly_pred=-1
)

# Combined SUPER ANOMALY (breaks multiple rules)
explain_anomaly_extended(
    dtsd=1000.0, price=0.1, ngang=12.0, phong=18, dtd=25.0, tang=2,
    loai_hinh='Biệt thự, nhà liền kề', phap_ly='Đã có sổ', noi_that='Nội thất Cao cấp', dac_diem='', huong='', dia_chi_cu='Phường 7, Quận Bình Thạnh', dia_chi_moi='', anomaly_pred=-1
)
