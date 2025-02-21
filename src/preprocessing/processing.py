import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib  # Để lưu scaler và encoder

class TrafficDataProcessor:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.raw_data_dir = os.path.join(base_dir, "data", "raw")
        self.processed_data_dir = os.path.join(base_dir, "data", "processed")
        self.features_dir = os.path.join(base_dir, "data", "features")
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        
        self.scaler_path = os.path.join(self.processed_data_dir, "scaler.pkl")
        self.encoder_path = os.path.join(self.processed_data_dir, "encoders.pkl")
        self.scaler = MinMaxScaler()
        self.encoder_dict = {}

        # Load scaler và encoder nếu có sẵn
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
        if os.path.exists(self.encoder_path):
            self.encoder_dict = joblib.load(self.encoder_path)

    def load_data(self, file_name):
        file_path = os.path.join(self.base_dir, "data", "raw", file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File {file_path} không tồn tại!")
        return pd.read_csv(file_path)

    def handle_missing_values(self, df):
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df.fillna("Unknown", inplace=True)
        return df

    def remove_duplicates(self, df):
        return df.drop_duplicates()

    def handle_outliers(self, df, columns):
        for col in columns:
            if col in df.columns:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                if lower_bound < upper_bound:
                    df[col] = np.clip(df[col], lower_bound, upper_bound)
        return df

    def normalize_data(self, df, columns):
        if os.path.exists(self.scaler_path):
            df[columns] = self.scaler.transform(df[columns])
        else:
            df[columns] = self.scaler.fit_transform(df[columns])
            joblib.dump(self.scaler, self.scaler_path)
        return df

    def encode_categorical(self, df, columns):
        for col in columns:
            if col in df.columns and df[col].nunique() > 1:
                df[col].fillna('Unknown', inplace=True)
                encoder = OneHotEncoder(sparse_output=False, drop="first")
                encoded = encoder.fit_transform(df[[col]])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
                df = df.drop(col, axis=1).reset_index(drop=True)
                df = pd.concat([df, encoded_df], axis=1)
                self.encoder_dict[col] = encoder
        joblib.dump(self.encoder_dict, self.encoder_path)
        return df

    def preprocess_data(self, file_name):
        df = self.load_data(file_name)
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        df = self.handle_outliers(df, ["num_vehicles", "density_vehicles", "num_stopped_vehicles"])
        df = self.normalize_data(df, ["num_vehicles", "density_vehicles", "num_stopped_vehicles"])
        df = self.encode_categorical(df, ["traffic_signal_status"])
        output_file = os.path.join(self.processed_data_dir, "cleaned_traffic_data.csv")
        df.to_csv(output_file, index=False)
        return df

    def add_timestamp(self, df):
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.date_range(start="2025-01-01", periods=len(df), freq="H")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def calculate_vehicle_density(self, df, num_lanes=3):
        if "lane" not in df.columns:
            df["lane"] = np.random.choice(range(1, num_lanes + 1), size=len(df))
        vehicle_density = df.groupby("lane")["num_vehicles"].sum() / num_lanes
        return vehicle_density

    def extract_features(self, df, num_lanes=3):
        df = self.add_timestamp(df)
        vehicle_density = self.calculate_vehicle_density(df, num_lanes)
        df["vehicle_density_per_lane"] = df["lane"].map(vehicle_density)
        df["rolling_avg_num_vehicles"] = df["num_vehicles"].rolling(window=5).mean()
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df.fillna(method='bfill', axis=0, inplace=True)
        return df

    def process_and_extract_features(self, file_name, num_lanes=3):
        df = self.preprocess_data(file_name)
        df = self.extract_features(df, num_lanes)
        output_file = os.path.join(self.features_dir, "extracted_features.csv")
        df.to_csv(output_file, index=False)
        return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processor = TrafficDataProcessor(base_dir)
    file_name = "synthetic_traffic_data.csv"
    processed_df = processor.process_and_extract_features(file_name)
    print(processed_df.head())
