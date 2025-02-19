import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib  # Để lưu scaler

# Định nghĩa thư mục động
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)  # Đảm bảo thư mục tồn tại

# Load dataset
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File {file_path} không tồn tại!")
    return pd.read_csv(file_path)

# Xử lý missing values
def handle_missing_values(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        df[col].fillna("Unknown", inplace=True)
    
    return df

# Loại bỏ duplicates
def remove_duplicates(df):
    return df.drop_duplicates()

# Xử lý outliers (clip giá trị)
def handle_outliers(df, columns):
    for col in columns:
        if col in df.columns:  # Kiểm tra nếu cột tồn tại
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Kiểm tra nếu lower_bound < upper_bound trước khi thực hiện clipping
            if lower_bound < upper_bound:
                df[col] = np.clip(df[col], lower_bound, upper_bound)
            else:
                print(f"⚠️ Cột {col} có giá trị bất thường, không thực hiện clipping.")
    return df

# Chuẩn hóa dữ liệu (sử dụng scaler đã lưu)
def normalize_data(df, columns, scaler_path):
    scaler = MinMaxScaler()
    # Kiểm tra nếu scaler đã tồn tại, nếu có thì load scaler đã lưu
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        df[columns] = scaler.transform(df[columns])  # Sử dụng transform cho dữ liệu mới
    else:
        df[columns] = scaler.fit_transform(df[columns])
        joblib.dump(scaler, scaler_path)  # Lưu scaler để dùng cho val/test
    return df

# One-Hot Encoding với kiểm tra giá trị không xác định
def encode_categorical(df, columns, encoder_dict):
    for col in columns:
        if col in df.columns and df[col].nunique() > 1:  # Kiểm tra tồn tại và có giá trị khác nhau
            # Thay thế giá trị 'Unknown' hoặc giá trị không xác định với giá trị mặc định
            df[col].fillna('Unknown', inplace=True)
            encoder = OneHotEncoder(sparse_output=False, drop="first")
            encoded = encoder.fit_transform(df[[col]])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
            df = df.drop(col, axis=1).reset_index(drop=True)
            df = pd.concat([df, encoded_df], axis=1)
            
            # Lưu encoder để áp dụng sau này
            encoder_dict[col] = encoder
    return df, encoder_dict

# Hàm chính
def preprocess_data(file_name):
    file_path = os.path.join(RAW_DATA_DIR, file_name)
    print(f"📥 Loading file: {file_path}")

    df = load_data(file_path)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = handle_outliers(df, ["num_vehicles", "density_vehicles", "num_stopped_vehicles"])
    
    # Chuẩn hóa dữ liệu
    scaler_path = os.path.join(PROCESSED_DATA_DIR, "scaler.pkl")
    df = normalize_data(df, ["num_vehicles", "density_vehicles", "num_stopped_vehicles"], scaler_path)

    # One-Hot Encoding
    encoder_dict = {}
    df, encoder_dict = encode_categorical(df, ["traffic_signal_status"], encoder_dict)
    
    # ✅ Lưu output vào đúng `data/processed`
    output_file = os.path.join(PROCESSED_DATA_DIR, "cleaned_traffic_data.csv")
    df.to_csv(output_file, index=False)
    print(f"✅ Cleaned data saved to: {output_file}")
    
    # Lưu các encoder để dùng cho dữ liệu test và validation
    encoder_file = os.path.join(PROCESSED_DATA_DIR, "encoders.pkl")
    joblib.dump(encoder_dict, encoder_file)

    return df, scaler_path, encoder_file

# Chạy script
if __name__ == "__main__":
    file_name = "synthetic_traffic_data.csv"  # File đầu vào
    cleaned_df, scaler_file, encoder_file = preprocess_data(file_name)
    print(cleaned_df.head())
