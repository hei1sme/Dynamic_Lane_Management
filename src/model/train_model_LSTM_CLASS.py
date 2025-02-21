import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import argparse
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

class LSTMTrafficPredictor:
    def __init__(self, base_dir, n_steps=5, batch_size=16, epochs=20):
        """Initialize the LSTM Traffic Predictor with configuration parameters."""
        self.base_dir = base_dir
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Set up directory paths
        self.split_data_dir = os.path.join(base_dir, "data", "splits")
        self.model_dir = os.path.join(base_dir, "models")
        self.scaler_path = os.path.join(base_dir, "data", "processed", "scaler.pkl")
        
        # Create necessary directories
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize model and scaler as None
        self.model = None
        self.scaler = None
        self.history = None

    def _check_file_exists(self, file_path):
        """Verify if a file exists at the given path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ File {file_path} không tồn tại!")

    def _load_data(self, file_path):
        """Load and preprocess the dataset from a CSV file."""
        self._check_file_exists(file_path)
        df = pd.read_csv(file_path)
        
        # Convert timestamp and sort
        if 'timestamp' in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values(by="timestamp")
            df = df.drop(columns=["timestamp"])
        
        # Handle missing values
        df = df.fillna(df.median(numeric_only=True))
        
        return df

    def _create_sequences(self, data):
        """Create sequences for LSTM input from the data."""
        X, y = [], []
        for i in range(len(data) - self.n_steps):
            X.append(data[i:i + self.n_steps])
            y.append(data[i + self.n_steps][-1])
        return np.array(X), np.array(y)

    def _build_model(self, input_shape):
        """Build and compile the LSTM model."""
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        return self.model

    def load_and_prepare_data(self):
        """Load and prepare all datasets for training."""
        print("Loading and preparing data...")
        
        # Load scaler
        self._check_file_exists(self.scaler_path)
        self.scaler = joblib.load(self.scaler_path)
        
        # Load datasets
        train_df = self._load_data(os.path.join(self.split_data_dir, "train_data.csv"))
        val_df = self._load_data(os.path.join(self.split_data_dir, "val_data.csv"))
        test_df = self._load_data(os.path.join(self.split_data_dir, "test_data.csv"))
        
        print("Data loaded successfully!")
        print(f"Training set shape: {train_df.shape}")
        print(f"Validation set shape: {val_df.shape}")
        print(f"Test set shape: {test_df.shape}")
        
        # Filter columns based on scaler features
        expected_columns = self.scaler.feature_names_in_
        train_df = train_df[expected_columns]
        val_df = val_df[expected_columns]
        test_df = test_df[expected_columns]
        
        # Scale data
        scaled_train = self.scaler.transform(train_df)
        scaled_val = self.scaler.transform(val_df)
        scaled_test = self.scaler.transform(test_df)
        
        # Create sequences
        X_train, y_train = self._create_sequences(scaled_train)
        X_val, y_val = self._create_sequences(scaled_val)
        X_test, y_test = self._create_sequences(scaled_test)
        
        print("\nSequences created:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_val shape: {X_val.shape}")
        print(f"X_test shape: {X_test.shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def train(self):
        """Train the LSTM model on the prepared data."""
        print("🚀 Starting training process...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.load_and_prepare_data()
        
        print("\n🏗️ Building model...")
        self._build_model((self.n_steps, X_train.shape[2]))
        self.model.summary()
        
        print("\n🚀 Training model...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            verbose=1  # Show progress bar
        )
        
        # Evaluate on test set
        print("\n📊 Evaluating model on test set...")
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=1)
        print(f"✅ Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
        
        return self.history

    def save_model(self, model_name="lstm_traffic_model.keras"):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("❌ No trained model to save! Please train the model first.")
        
        model_path = os.path.join(self.model_dir, model_name)
        self.model.save(model_path)
        print(f"💾 Model saved to {model_path}")

    def load_model(self, model_name="lstm_traffic_model.keras"):
        """Load a trained model from disk."""
        model_path = os.path.join(self.model_dir, model_name)
        self._check_file_exists(model_path)
        self.model = load_model(model_path)
        print(f"📂 Model loaded from {model_path}")

    def predict(self, input_sequence):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("❌ No model available! Please train or load a model first.")
        
        if self.scaler is None:
            self.scaler = joblib.load(self.scaler_path)
        
        # Ensure input sequence is scaled
        scaled_sequence = self.scaler.transform(input_sequence)
        X, _ = self._create_sequences(scaled_sequence)
        
        return self.model.predict(X)

def parse_args():
    parser = argparse.ArgumentParser(description='LSTM Traffic Predictor Training')
    
    # Add arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16)')
    parser.add_argument('--n_steps', type=int, default=5,
                        help='Number of time steps for LSTM (default: 5)')
    parser.add_argument('--model_name', type=str, default='lstm_traffic_model.keras',
                        help='Name of the model file to save (default: lstm_traffic_model.keras)')
    
    return parser.parse_args()

# Main execution block
if __name__ == "__main__":
    print("Starting LSTM Traffic Predictor...")
    
    # Parse command line arguments
    args = parse_args()
    
    # Get the project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Project root directory: {BASE_DIR}")
    
    try:
        # Create predictor instance with command line arguments
        predictor = LSTMTrafficPredictor(
            base_dir=BASE_DIR,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            epochs=args.epochs
        )
        
        # Train the model
        history = predictor.train()
        
        # Save the model with specified name
        predictor.save_model(args.model_name)
        
        print("\n✨ Training completed successfully!")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        raise e