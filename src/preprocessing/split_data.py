import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Định nghĩa thư mục động
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "features")
SPLIT_DATA_DIR = os.path.join(BASE_DIR, "data", "splits")
os.makedirs(SPLIT_DATA_DIR, exist_ok=True)

def split_dataset(file_name, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    file_path = os.path.join(PROCESSED_DATA_DIR, file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File {file_path} không tồn tại!")

    print(f"📥 Loading file: {file_path}")
    df = pd.read_csv(file_path)

    # Kiểm tra tỷ lệ hợp lệ
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("❌ Tổng các tỷ lệ phải bằng 1!")

    # Chia tập train và temp (val + test)
    train_data, temp_data = train_test_split(df, test_size=(1 - train_ratio), random_state=random_state, shuffle=True)
    
    # Chia tiếp temp thành val và test
    val_data, test_data = train_test_split(temp_data, test_size=(test_ratio / (test_ratio + val_ratio)), 
                                           random_state=random_state, shuffle=True)

    # Định nghĩa đường dẫn output
    train_file = os.path.join(SPLIT_DATA_DIR, "train_data.csv")
    val_file = os.path.join(SPLIT_DATA_DIR, "val_data.csv")
    test_file = os.path.join(SPLIT_DATA_DIR, "test_data.csv")

    # Lưu các file
    train_data.to_csv(train_file, index=False)
    val_data.to_csv(val_file, index=False)
    test_data.to_csv(test_file, index=False)

    print("✅ Data successfully split and saved:")
    print(f"   🟢 Train set: {len(train_data)} samples → {train_file}")
    print(f"   🔵 Validation set: {len(val_data)} samples → {val_file}")
    print(f"   🔴 Test set: {len(test_data)} samples → {test_file}")

    return train_data, val_data, test_data

# Chạy script
if __name__ == "__main__":
    file_name = "extracted_features.csv"  # Thay bằng tên file thực tế
    split_dataset(file_name)
