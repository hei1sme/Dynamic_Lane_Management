import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Generate synthetic traffic data
num_samples = 300000  # Number of rows
timestamps = [datetime(2025, 2, 19, 6, 0, 0) + timedelta(seconds=i*12) for i in range(num_samples)]

def generate_traffic_data(timestamp):
    hour = timestamp.hour
    is_peak_hour = 1 if (7 <= hour <= 9 or 16 <= hour <= 19) else 0
    num_vehicles = random.randint(5, 50) if is_peak_hour else random.randint(1, 20)
    density_vehicles = round(num_vehicles / random.uniform(100, 300), 3)
    
    motorbike_ratio = round(random.uniform(0.5, 0.8), 2)
    car_ratio = round(random.uniform(0.1, 0.3), 2)
    truck_ratio = round(random.uniform(0.05, 0.15), 2)
    bus_ratio = round(1 - (motorbike_ratio + car_ratio + truck_ratio), 2)
    
    num_stopped_vehicles = random.randint(0, int(num_vehicles * 0.3))
    active_lanes = 3
    traffic_signal_status = random.choice([0, 1, 2])  # 0 = Green, 1 = Yellow, 2 = Red
    traffic_congestion_level = min(3, max(0, int(num_vehicles / 15)))
    
    return [timestamp.strftime('%Y-%m-%d %H:%M:%S'), num_vehicles, density_vehicles,
            motorbike_ratio, car_ratio, truck_ratio, bus_ratio,
            num_stopped_vehicles, active_lanes, traffic_signal_status,
            traffic_congestion_level, is_peak_hour]

data = [generate_traffic_data(ts) for ts in timestamps]

columns = ['timestamp', 'num_vehicles', 'density_vehicles', 'motorbike_ratio',
           'car_ratio', 'truck_ratio', 'bus_ratio', 'num_stopped_vehicles',
           'active_lanes', 'traffic_signal_status', 'traffic_congestion_level', 'is_peak_hour']

df = pd.DataFrame(data, columns=columns)

# Get script directory and save CSV in the same folder
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, 'synthetic_traffic_data.csv')
df.to_csv(output_path, index=False)

print(f"Synthetic traffic data generated and saved to {output_path}")