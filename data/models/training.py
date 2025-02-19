import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu đã qua xử lý
data_path = r"C:\Users\Le Nguyen Gia Hung\Dropbox\Codes\.PROJECTS\major-project\Dynamic_Lane_Management\data\processed\traffic_featured_data.csv"
df = pd.read_csv(data_path)

# Chọn đặc trưng và biến mục tiêu
features = ['hour', 'minute', 'day_of_week', 'is_peak', 'vehicle_count', 'traffic_flow', 'vehicle_density']
target = 'avg_speed'

X = df[features]
y = df[target]

# Chuẩn hóa dữ liệu đầu vào
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình MLP
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),  # Tránh overfitting
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)  # Output là giá trị avg_speed
])

# Compile mô hình
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Huấn luyện mô hình với nhiều epoch
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))

# Đánh giá mô hình
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"\n🔹 Đánh giá trên tập kiểm thử:\n  - Loss (MSE): {test_loss:.4f}\n  - MAE: {test_mae:.4f}")

# Vẽ đồ thị loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss Over Epochs')
plt.legend()
plt.show()

# Lưu mô hình đã huấn luyện
model.save(r"C:\Users\Le Nguyen Gia Hung\Dropbox\Codes\.PROJECTS\major-project\Dynamic_Lane_Management\data\models\traffic_speed_model.h5")
print("\n✅ Mô hình đã được lưu thành công!")