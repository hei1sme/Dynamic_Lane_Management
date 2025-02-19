import pandas as pd
import numpy as np

# Đường dẫn file dữ liệu thô và file lưu dữ liệu đã làm sạch
input_path = r"C:\Users\Le Nguyen Gia Hung\Dropbox\Codes\.PROJECTS\major-project\Dynamic_Lane_Management\data\raw\traffic_raw_data.csv"
output_path = r"C:\Users\Le Nguyen Gia Hung\Dropbox\Codes\.PROJECTS\major-project\Dynamic_Lane_Management\data\processed\traffic_cleaned_data.csv"

# Đọc dữ liệu thô từ file CSV
df = pd.read_csv(input_path)

# --- Làm sạch cột vehicle_count ---
# Chuyển đổi sang kiểu số, nếu không chuyển đổi được sẽ trở thành NaN
df['vehicle_count'] = pd.to_numeric(df['vehicle_count'], errors='coerce')
# Nếu có giá trị âm thì coi là lỗi, thay bằng NaN
df['vehicle_count'] = df['vehicle_count'].apply(lambda x: x if pd.isna(x) or x >= 0 else np.nan)
# Điền giá trị thiếu (NaN) bằng giá trị trung bình của cột
vehicle_count_mean = df['vehicle_count'].mean()
df['vehicle_count'].fillna(vehicle_count_mean, inplace=True)

# --- Làm sạch cột traffic_flow ---
df['traffic_flow'] = pd.to_numeric(df['traffic_flow'], errors='coerce')
df['traffic_flow'] = df['traffic_flow'].apply(lambda x: x if pd.isna(x) or x >= 0 else np.nan)
traffic_flow_mean = df['traffic_flow'].mean()
df['traffic_flow'].fillna(traffic_flow_mean, inplace=True)

# --- Làm sạch cột avg_speed ---
df['avg_speed'] = pd.to_numeric(df['avg_speed'], errors='coerce')
df['avg_speed'] = df['avg_speed'].apply(lambda x: x if pd.isna(x) or x >= 0 else np.nan)
avg_speed_mean = df['avg_speed'].mean()
df['avg_speed'].fillna(avg_speed_mean, inplace=True)

# Lưu dữ liệu đã làm sạch vào file CSV
df.to_csv(output_path, index=False)

print(f"Dữ liệu đã được làm sạch và lưu tại: {output_path}")