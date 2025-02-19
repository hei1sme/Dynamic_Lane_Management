import pandas as pd
import numpy as np
import random

# Nhập số lượng dòng dữ liệu từ người dùng (khuyến nghị nhập >200 cho khung giờ cao điểm)
num_rows = int(input("Nhập số lượng dòng dữ liệu cần tạo: "))

# Tạo danh sách timestamp, mỗi 5 phút một mẫu (bạn có thể điều chỉnh thời gian bắt đầu nếu cần)
timestamps = pd.date_range(start="2025-02-01 07:00:00", periods=num_rows, freq="5min")

vehicle_count = []
traffic_flow = []
avg_speed = []

for _ in range(num_rows):
    # vehicle_count: 80% trường hợp dữ liệu hợp lệ (0-50), 10% NaN, 10% giá trị âm (lỗi)
    r = random.random()
    if r < 0.8:
        vc = random.randint(0, 500)
    elif r < 0.9:
        vc = np.nan
    else:
        vc = -random.randint(1, 10)
    vehicle_count.append(vc)
    
    # traffic_flow: 80% trường hợp hợp lệ (1-10 xe/phút), 10% NaN, 10% giá trị âm (lỗi)
    r = random.random()
    if r < 0.8:
        tf = round(random.uniform(1, 10), 2)
    elif r < 0.9:
        tf = np.nan
    else:
        tf = -round(random.uniform(1, 5), 2)
    traffic_flow.append(tf)
    
    # avg_speed: 80% trường hợp hợp lệ (20-70 km/h), 10% NaN, 10% giá trị âm (lỗi)
    r = random.random()
    if r < 0.8:
        sp = random.randint(20, 70)
    elif r < 0.9:
        sp = np.nan
    else:
        sp = -random.randint(1, 10)
    avg_speed.append(sp)

# Tạo DataFrame từ các danh sách trên
df = pd.DataFrame({
    "timestamp": timestamps,
    "vehicle_count": vehicle_count,
    "traffic_flow": traffic_flow,
    "avg_speed": avg_speed
})

# Đường dẫn lưu file CSV (sử dụng raw string để tránh lỗi escape)
output_path = r"C:\Users\Le Nguyen Gia Hung\Dropbox\Codes\.PROJECTS\major-project\Dynamic_Lane_Management\data\raw\traffic_raw_data.csv"

# Lưu dữ liệu vào file CSV
df.to_csv(output_path, index=False)

print(f"Dữ liệu thô ({num_rows} dòng) đã được tạo và lưu tại: {output_path}")
