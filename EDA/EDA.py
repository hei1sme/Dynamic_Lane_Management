import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đường dẫn tới dữ liệu đã làm sạch
cleaned_data_path = r"C:\Users\Le Nguyen Gia Hung\Dropbox\Codes\.PROJECTS\major-project\Dynamic_Lane_Management\data\processed\traffic_cleaned_data.csv"

# Đọc dữ liệu
df = pd.read_csv(cleaned_data_path)

# 1. Hiển thị 5 dòng đầu tiên của dữ liệu
print("5 dòng đầu tiên của dữ liệu:")
print(df.head())

# 2. Thống kê mô tả
print("\nThống kê mô tả của dữ liệu:")
print(df.describe())

# 3. Vẽ biểu đồ phân phối cho các cột số
numerical_columns = ['vehicle_count', 'traffic_flow', 'avg_speed']
for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True, bins=20, color='skyblue')
    plt.title(f'Phân phối của {col}')
    plt.xlabel(col)
    plt.ylabel('Tần suất')
    plt.tight_layout()
    plt.show()

# 4. Vẽ Boxplot để kiểm tra giá trị ngoại lai
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numerical_columns], palette='pastel')
plt.title('Boxplot của các biến số')
plt.xlabel('Biến số')
plt.ylabel('Giá trị')
plt.tight_layout()
plt.show()

# 5. Vẽ ma trận tương quan giữa các biến số
plt.figure(figsize=(8, 6))
corr_matrix = df[numerical_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Ma trận tương quan giữa các biến số")
plt.tight_layout()
plt.show()