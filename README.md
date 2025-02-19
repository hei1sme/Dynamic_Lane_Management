# Dynamic Traffic Management Using Mobile Barriers

## Project Overview
This project aims to develop an AI-powered traffic management system that dynamically adjusts lane distribution using mobile barriers. The system will analyze real-time traffic patterns to predict peak hours and suggest optimal barrier movements to improve traffic flow.

## Objectives
- Collect and analyze traffic data using open-source cameras.
- Train a model to predict peak traffic hours based on real-world data.
- Develop a decision-making system that suggests when to adjust lane distribution.
- Implement a simulation to test model performance before real-world application.

## Folder Structure
```
Dynamic_Traffic_Management/
│-- data/                # Data storage
│   ├── raw/            # Raw images and datasets
│   ├── processed/      # Cleaned and preprocessed data
│   ├── fake_data/      # Synthetic data for testing
│
│-- notebooks/          # Jupyter notebooks for data analysis and modeling
│   ├── data_analysis.ipynb
│   ├── model_training.ipynb
│
│-- src/                # Source code
│   ├── data_collection/
│   │   ├── scrape_images.py       # Web scraping for images
│   │   ├── remove_duplicates.py   # Remove redundant images
│   ├── preprocessing/
│   │   ├── clean_data.py          # Data cleaning and preprocessing
│   │   ├── feature_extraction.py  # Extract features from images
│   ├── model/
│   │   ├── train_model.py         # Train the traffic prediction model
│   │   ├── predict.py             # Run inference on new data
│   ├── utils/
│   │   ├── image_utils.py         # Image processing utilities
│   │   ├── data_utils.py          # Data handling utilities
│
│-- models/              # Trained models storage
│-- logs/                # Logs for model training and experiments
│-- reports/             # Research papers, presentations, or reports
│-- requirements.txt     # Dependencies list
│-- README.md            # Project documentation
```

## Data Collection
- **Source**: Open-source traffic cameras (11s per frame update interval)
- **Features Extracted**:
  - Total number of vehicles
  - Vehicle density (vehicles/m²)
  - Vehicle type ratios (motorbikes, cars, trucks, buses)
  - Number of stopped vehicles
  - Active lanes
  - Traffic signal status
  - Traffic congestion level
  - Peak hour label (Yes/No)

## Technology Stack
- **Programming Language**: Python
- **Data Processing**: Pandas, NumPy
- **Computer Vision**: OpenCV, YOLOv8
- **Machine Learning**: TensorFlow, PyTorch, Scikit-learn
- **Traffic Simulation**: SUMO, MATLAB
- **Database**: PostgreSQL, MongoDB
- **APIs**: OpenTraffic API, OpenWeather API

## Getting Started
### 1. Install Dependencies
```sh
pip install -r requirements.txt
```

### 2. Run Data Collection
```sh
python src/data_collection/scrape_images.py
```

### 3. Preprocess Data
```sh
python src/preprocessing/clean_data.py
```

### 4. Train the Model
```sh
python src/model/train_model.py
```

### 5. Run Predictions
```sh
python src/model/predict.py
```

## Contributors
- **Lê Nguyễn Gia Hưng (Group Leader)**
- Võ Tấn Phát
- Huỳnh Quốc Việt
- Phạm Mạnh Quân

## License
This project is licensed under the MIT License.

---
For further inquiries, please contact **Lê Nguyễn Gia Hưng**.

