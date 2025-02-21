import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib  # Import để load scaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Định nghĩa thư mục động
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLIT_DATA_DIR = os.path.join(BASE_DIR, "data", "splits")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Đường dẫn đến scaler.pkl
SCALER_PATH = os.path.join(BASE_DIR, "data", "processed", "scaler.pkl")

# Hàm kiểm tra file tồn tại
def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File {file_path} không tồn tại!")

# Load dataset
def load_data(file_path):
    check_file_exists(file_path)
    df = pd.read_csv(file_path)

    # Chuyển đổi timestamp và sắp xếp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp")
    df = df.drop(columns=["timestamp"])  # Loại bỏ timestamp trước khi train
    
    # Xử lý giá trị NaN
    df = df.fillna(df.median(numeric_only=True))
    
    return df

# Tạo chuỗi dữ liệu (sequence) cho mô hình LSTM
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps][-1])  # Cột cuối cùng là nhãn dự đoán
    return np.array(X), np.array(y)

# Thiết lập tham số
N_STEPS = 5  # Số bước để dự đoán tiếp theo
BATCH_SIZE = 16
EPOCHS = 20

# Load scaler đã được fit từ preprocessing
scaler = joblib.load(SCALER_PATH)

# Load và tiền xử lý dữ liệu
train_file = os.path.join(SPLIT_DATA_DIR, "train_data.csv")
val_file = os.path.join(SPLIT_DATA_DIR, "val_data.csv")
test_file = os.path.join(SPLIT_DATA_DIR, "test_data.csv")

train_df = load_data(train_file)
val_df = load_data(val_file)
test_df = load_data(test_file)

# Đảm bảo train_df, val_df và test_df có các cột giống như lúc huấn luyện scaler
expected_columns = scaler.feature_names_in_  # Cột mà scaler đã học

# Lọc chỉ các cột mong đợi
train_df = train_df[expected_columns]
val_df = val_df[expected_columns]
test_df = test_df[expected_columns]

# Chuẩn hóa dữ liệu bằng scaler đã lưu
scaled_train = scaler.transform(train_df)
scaled_val = scaler.transform(val_df)
scaled_test = scaler.transform(test_df)

# Tạo dữ liệu đầu vào cho LSTM
X_train, y_train = create_sequences(scaled_train, N_STEPS)
X_val, y_val = create_sequences(scaled_val, N_STEPS)
X_test, y_test = create_sequences(scaled_test, N_STEPS)

# Xây dựng mô hình LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(N_STEPS, X_train.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Phân loại nhị phân (giờ cao điểm hay không)
])

# Compile mô hình
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Huấn luyện mô hình
print("🚀 Training model...")
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

# Đánh giá trên tập test
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

# Lưu mô hình
model_path = os.path.join(MODEL_DIR, "lstm_traffic_model.keras")
model.save(model_path)
print(f"💾 Model saved to {model_path}")

