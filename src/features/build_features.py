# src/features/build_features.py

import pandas as pd
import os
from statsmodels.tsa.seasonal import STL

def load_cycle_lengths(file_path):
    cycle_lengths = []
    with open(file_path, 'r') as f:
        for line in f:
            _, length = line.strip().split(': ')
            cycle_lengths.append(int(length))
    return cycle_lengths

def calculate_average_cycle_length(cycle_lengths):
    return int(sum(cycle_lengths) / len(cycle_lengths))

def decompose_time_series(df, period, seasonal=13):
    stl_speed = STL(df['Speed'], seasonal=seasonal, period=period)
    res_speed = stl_speed.fit()
    df['trend_speed'] = res_speed.trend
    df['seasonal_speed'] = res_speed.seasonal
    df['residual_speed'] = res_speed.resid

    stl_acceleration = STL(df['Acceleration'], seasonal=seasonal, period=period)
    res_acceleration = stl_acceleration.fit()
    df['trend_acceleration'] = res_acceleration.trend
    df['seasonal_acceleration'] = res_acceleration.seasonal
    df['residual_acceleration'] = res_acceleration.resid
    
    return df

def create_lag_features(df, lag=10):
    for i in range(1, lag + 1):
        df[f'speed_lag_{i}'] = df['Speed'].shift(i)
        df[f'acceleration_lag_{i}'] = df['Acceleration'].shift(i)
        df[f'trend_speed_lag_{i}'] = df['trend_speed'].shift(i)
        df[f'seasonal_speed_lag_{i}'] = df['seasonal_speed'].shift(i)
        df[f'residual_speed_lag_{i}'] = df['residual_speed'].shift(i)
        df[f'trend_acceleration_lag_{i}'] = df['trend_acceleration'].shift(i)
        df[f'seasonal_acceleration_lag_{i}'] = df['seasonal_acceleration'].shift(i)
        df[f'residual_acceleration_lag_{i}'] = df['residual_acceleration'].shift(i)
    
    df.dropna(inplace=True)
    return df

def save_features(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'features_data.csv')
    df.to_csv(output_file, index=False)
    print(f"Feature-engineered data saved to {output_file}")

def main():
    processed_data_dir = "data/processed"
    processed_file = os.path.join(processed_data_dir, 'combined_data.csv')
    cycle_lengths_file = os.path.join(processed_data_dir, 'cycle_lengths.txt')
    
    # Load cycle lengths and calculate the average
    cycle_lengths = load_cycle_lengths(cycle_lengths_file)
    average_cycle_length = calculate_average_cycle_length(cycle_lengths)
    
    # Load the combined dataset
    df = pd.read_csv(processed_file)
    
    # Decompose the time series using the average cycle length
    df = decompose_time_series(df, period=average_cycle_length, seasonal=13)
    
    # Create lag features
    df = create_lag_features(df, lag=10)
    
    # Save the feature-engineered data
    save_features(df, processed_data_dir)

if __name__ == "__main__":
    main()