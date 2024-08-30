# src/models/manual_split_test.py

import os
import numpy as np
import pandas as pd
import joblib
import sys
import logging
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the absolute path of the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..', '..')

# Add the parent directory to the sys.path
sys.path.append(parent_dir)

# Now you should be able to import the src module
import src.visualization.visualize as viz

def load_model(model_dir):
    model_file = os.path.join(model_dir, 'xgboost_model.pkl')
    model = joblib.load(model_file)
    return model

def make_forecast(X, model):
    predictions = model.predict(X)
    return predictions

def manual_split_data(df, new_cycle_lengths, train_perc=70, val_perc=15, test_perc=15):
    data_size = len(df)
    num_cycles = len(new_cycle_lengths)

    # Calculate test set size
    num_test_cycles = np.argmin(abs(np.cumsum(new_cycle_lengths[::-1])-data_size*test_perc/100))+1
    test_size = np.sum(new_cycle_lengths[-num_test_cycles:])
    logging.info(f'{test_perc}% of the data was selected as test data i.e. Last {num_test_cycles} cycles')
    logging.info(f'Test size = {test_size}')

    # Calculate validation set size
    num_val_cycles = np.argmin(abs(np.cumsum(new_cycle_lengths[::-1])-(data_size - test_size)*val_perc/100))+1
    val_size = np.sum(new_cycle_lengths[-(num_val_cycles+num_test_cycles):-num_test_cycles])
    logging.info(f'{val_perc}% of the data was selected as validation data i.e. Last {num_val_cycles} cycles')
    logging.info(f'Validation size = {val_size}')

    # Calculate train set size
    num_train_cycles = num_cycles - (num_test_cycles + num_val_cycles)
    train_size = np.sum(new_cycle_lengths[:num_train_cycles])
    logging.info(f'{train_perc}% of the data was selected as train data i.e. Last {num_train_cycles} cycles')
    logging.info(f'Train size = {train_size}')

    logging.info(f"""Train size      = {train_size}{' '*2}-> %{train_perc}
    Validation size = {val_size}{' '*(2+len(str(train_size))-len(str(val_size)))}-> %{val_perc}
    Test size       = {test_size}{' '*(2+len(str(train_size))-len(str(test_size)))}-> %{test_perc}""")

    # Split the dataset
    X_train = df.iloc[:train_size]
    X_val = df.iloc[train_size:train_size + val_size]
    X_test = df.iloc[train_size + val_size:]

    return X_train, X_val, X_test, train_size, val_size, test_size

def main():
    processed_data_dir = "data/processed"
    processed_file = os.path.join(processed_data_dir, 'features_data.csv')
    cycle_lengths_file = os.path.join(processed_data_dir, 'cycle_lengths.txt')

    df = pd.read_csv(processed_file)
    
    # Load cycle lengths
    cycle_lengths = []
    with open(cycle_lengths_file, 'r') as f:
        for line in f:
            _, length = line.strip().split(': ')
            cycle_lengths.append(int(length))
    
    # Split the data manually based on cycles
    X_train, X_val, X_test, train_size, val_size, test_size = manual_split_data(df, cycle_lengths, train_perc=70, val_perc=15, test_perc=15)

    # Load the trained model
    model_dir = "models"
    model = load_model(model_dir)

    # Make predictions on the test set
    y_test = X_test['Speed']
    X_test = X_test.drop(columns=['Time', 'Speed', 'Acceleration', 'Cycle'])
    y_test_pred = make_forecast(X_test, model)

    # Plot the dataset split
    train_size = len(X_train)
    val_size = len(X_val)
    viz.plot_train_val_test_split(X_train['Speed'], X_val['Speed'], y_test, train_size, val_size, save=True)

    # Plot the results
    # viz.plot_actual_vs_predicted(y_test, y_test_pred, title="Manual Test Set: Actual vs Predicted Speed", save=True)
    # viz.plot_residuals(y_test, y_test_pred, title="Manual Test Set: Residuals", save=True)
    
    # Plot last cycle
    y_test_last_cylce = y_test.loc[18588:].reset_index(drop=True)
    y_test_pred_last_cylce = y_test_pred[1938:]
    viz.plot_actual_vs_predicted(y_test_last_cylce, y_test_pred_last_cylce, title="Test Set: Actual vs Predicted Speed", save=True)
    viz.plot_residuals(y_test_last_cylce, y_test_pred_last_cylce, title="Test Set: Residuals", save=True)

if __name__ == "__main__":
    main()