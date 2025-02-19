import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Train XGBoost model
def train_xgboost(train_path, val_path, test_path):
    # Load datasets
    df_train = load_data(train_path)
    df_val = load_data(val_path)
    df_test = load_data(test_path)
    
    # Features and target
    X_train = df_train.drop(columns=['is_peak_hour', 'timestamp'])
    y_train = df_train['is_peak_hour']
    X_val = df_val.drop(columns=['is_peak_hour', 'timestamp'])
    y_val = df_val['is_peak_hour']
    X_test = df_test.drop(columns=['is_peak_hour', 'timestamp'])
    y_test = df_test['is_peak_hour']
    
    # Initialize and train model
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    
    # Predictions and evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    
    # Save the trained model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xgboost_traffic_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model

if __name__ == "__main__":
    base_path = "data/splits"
    train_path = os.path.join(base_path, "train_data.csv")
    val_path = os.path.join(base_path, "val_data.csv")
    test_path = os.path.join(base_path, "test_data.csv")
    
    trained_model = train_xgboost(train_path, val_path, test_path)
