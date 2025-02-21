import tensorflow as tf
import numpy as np
import pandas as pd
import os
import logging
import sys

# Ensure the preprocessing module is accessible
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(base_dir, "src", "preprocessing"))

from processing import TrafficDataProcessor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TrafficPredictor:
    def __init__(self, model_path, base_dir):
        logging.info("Initializing TrafficPredictor...")
        self.model_path = model_path
        self.processor = TrafficDataProcessor(base_dir)
        self.model = self.load_model()

    def load_model(self):
        logging.info(f"Loading model from {self.model_path}...")
        if not os.path.exists(self.model_path):
            logging.error("❌ Model file not found!")
            raise FileNotFoundError("Model file not found!")
        model = tf.keras.models.load_model(self.model_path)
        logging.info("✅ Model loaded successfully.")
        return model

    def predict(self, file_name):
        logging.info(f"Processing file: {file_name}")
        df = self.processor.process_and_extract_features(file_name)
        
        # Select only numerical features for prediction
        feature_columns = ['num_vehicles', 'density_vehicles', 'num_stopped_vehicles',
                           'vehicle_density_per_lane', 'rolling_avg_num_vehicles', 'hour', 'day_of_week']
        if not all(col in df.columns for col in feature_columns):
            logging.error("❌ Missing required feature columns!")
            raise ValueError("Missing required feature columns in preprocessed data!")
        
        X = df[feature_columns].values.reshape(1, df.shape[0], len(feature_columns))
        logging.info("✅ Data reshaped for LSTM input.")
        
        # Predict traffic flow
        logging.info("Running prediction...")
        predictions = self.model.predict(X)
        logging.info("✅ Prediction completed.")
        
        df['predicted_traffic_flow'] = predictions.flatten()
        output_file = os.path.join(self.processor.features_dir, "predicted_traffic_flow.csv")
        df.to_csv(output_file, index=False)
        logging.info(f"✅ Predictions saved to {output_file}")
        
        return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "models", "lstm_traffic_model.keras")
    file_name = "synthetic_traffic_data.csv"
    
    predictor = TrafficPredictor(model_path, base_dir)
    predictions_df = predictor.predict(file_name)
    print(predictions_df.head())
