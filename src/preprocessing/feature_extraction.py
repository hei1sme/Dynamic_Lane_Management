import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# Định nghĩa thư mục động
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
os.makedirs(FEATURES_DIR, exist_ok=True)  # Đảm bảo thư mục tồn tại

# Load dữ liệu đã xử lý
def load_data(file_name):
    file_path = os.path.join(PROCESSED_DATA_DIR, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File {file_path} không tồn tại!")
    return pd.read_csv(file_path)

# Tính toán độ dày phương tiện trên từng làn
def calculate_vehicle_density(df, num_lanes=3):
    # Kiểm tra nếu dữ liệu có cột 'lane' (số làn)
    if "lane" not in df.columns:
        print("⚠️ Không có cột 'lane'. Dùng giả định 3 làn đường.")
        df["lane"] = np.random.choice(range(1, num_lanes + 1), size=len(df))
        
    # Đếm số phương tiện trên mỗi làn
    vehicle_density = df.groupby("lane")["num_vehicles"].sum() / num_lanes
    return vehicle_density

# Tạo timestamp nếu thiếu
def add_timestamp(df):
    if "timestamp" not in df.columns:
        print("⚠️ Không có cột 'timestamp'. Tạo timestamp mặc định.")
        df["timestamp"] = pd.date_range(start="2025-01-01", periods=len(df), freq="H")
    
    # Đảm bảo rằng cột 'timestamp' có kiểu datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    return df


# Tính toán các đặc trưng
def extract_features(df, num_lanes=3):
    # Tạo timestamp nếu chưa có
    df = add_timestamp(df)
    
    # Tính toán mật độ phương tiện trên từng làn
    vehicle_density = calculate_vehicle_density(df, num_lanes)
    
    # Thêm các đặc trưng vào dataframe
    df["vehicle_density_per_lane"] = df["lane"].map(vehicle_density)
    
    # Tính toán thêm các đặc trưng khác (ví dụ: rolling mean)
    df["rolling_avg_num_vehicles"] = df["num_vehicles"].rolling(window=5).mean()
    
    # Thêm các đặc trưng từ timestamp (giờ trong ngày, ngày trong tuần)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    
    # Xử lý các giá trị NaN sau khi tính toán
    df.fillna(method='bfill', axis=0, inplace=True)  # Điền giá trị thiếu (NaN) từ phía sau
    
    return df

# Hàm chính
def feature_extraction(file_name, num_lanes=3):
    df = load_data(file_name)
    df = extract_features(df, num_lanes)
    
    # Lưu kết quả vào file
    output_file = os.path.join(FEATURES_DIR, "extracted_features.csv")
    df.to_csv(output_file, index=False)
    print(f"✅ Extracted features saved to: {output_file}")
    
    return df

# Chạy script
if __name__ == "__main__":
    file_name = "cleaned_traffic_data.csv"  # File đầu vào
    extracted_df = feature_extraction(file_name)
    print(extracted_df.head())
