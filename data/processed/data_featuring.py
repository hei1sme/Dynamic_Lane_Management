import pandas as pd

# Đường dẫn tới dữ liệu đã làm sạch
cleaned_data_path = r"C:\Users\Le Nguyen Gia Hung\Dropbox\Codes\.PROJECTS\major-project\Dynamic_Lane_Management\data\processed\traffic_cleaned_data.csv"
output_path = r"C:\Users\Le Nguyen Gia Hung\Dropbox\Codes\.PROJECTS\major-project\Dynamic_Lane_Management\data\processed\traffic_featured_data.csv"

# 1. Đọc dữ liệu đã làm sạch
df = pd.read_csv(cleaned_data_path)

# 2. Chuyển đổi cột timestamp sang kiểu datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 3. Trích xuất đặc trưng từ timestamp
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['day_of_week'] = df['timestamp'].dt.dayofweek  # 0: Thứ Hai, 6: Chủ Nhật

# 4. Tạo cột is_peak: Đánh dấu khung giờ cao điểm (ví dụ: 7-9 và 16-19)
df['is_peak'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)

# 5. Tạo thêm đặc trưng ví dụ: vehicle_density (số xe trên mỗi xe/phút)
# Lưu ý: Nếu traffic_flow <= 0, đặt density là 0 để tránh chia cho 0
df['vehicle_density'] = df.apply(lambda row: row['vehicle_count'] / row['traffic_flow'] 
                                 if row['traffic_flow'] > 0 else 0, axis=1)

# Lưu dữ liệu sau feature engineering
df.to_csv(output_path, index=False)

print("Feature engineering hoàn thành. Dữ liệu có đặc trưng mới đã được lưu tại:")
print(output_path)
