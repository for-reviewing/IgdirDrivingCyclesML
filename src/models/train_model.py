# src/models/train_model.py

import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import sys
import logging

# Set up logging to file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("reports/training_output.txt"),
                              logging.StreamHandler(sys.stdout)])

# Get the absolute path of the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..', '..')

# Add the parent directory to the sys.path
sys.path.append(parent_dir)

import src.visualization.visualize as viz

def train_xgboost_model(X_train, y_train, X_val, y_val, n_estimators=100, learning_rate=0.1, early_stopping_rounds=10):
    model_speed = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        objective='reg:squarederror',
        eval_metric='rmse',
        early_stopping_rounds=early_stopping_rounds
    )

    logging.info("Training the XGBoost model with early stopping...")

    # Train the model with early stopping
    model_speed.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # Retrieve the evaluation results
    evals_result = model_speed.evals_result()

    return model_speed, evals_result

def save_validation_rmse(evals_result, output_dir):
    rmse_values = evals_result['validation_0']['rmse']
    os.makedirs(output_dir, exist_ok=True)
    rmse_file = os.path.join(output_dir, "validation_rmse.txt")
    with open(rmse_file, "w") as f:
        for i, rmse in enumerate(rmse_values):
            f.write(f"Iteration {i + 1}: RMSE = {rmse}\n")
    logging.info(f"Validation RMSE values saved to {rmse_file}")

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    return y_pred, rmse, mae

def save_metrics(rmse, mae, output_dir, dataset_type="Validation"):
    os.makedirs(output_dir, exist_ok=True)
    metrics_file = os.path.join(output_dir, f"{dataset_type.lower()}_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"RMSE: {rmse}\n")
        f.write(f"MAE: {mae}\n")
    logging.info(f"{dataset_type} metrics saved to {metrics_file}")

def save_model(model, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model_file = os.path.join(output_dir, 'xgboost_model.pkl')
    joblib.dump(model, model_file)
    logging.info(f"Model saved to {model_file}")

def main():
    processed_data_dir = "data/processed"
    processed_file = os.path.join(processed_data_dir, 'features_data.csv')
    
    df = pd.read_csv(processed_file)
    
    # Prepare the dataset
    X = df.drop(columns=['Time', 'Speed', 'Acceleration', 'Cycle'])
    y = df['Speed']

    # Split the data into train, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Train the model
    model_speed, evals_result = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    # Save validation RMSE values
    save_validation_rmse(evals_result, "reports")
    
    # Plot RMSE values from file
    viz.plot_rmse_from_file("reports/validation_rmse.txt", save=True)
    
    # Evaluate on validation set
    y_val_pred, val_rmse, val_mae = evaluate_model(model_speed, X_val, y_val)
    save_metrics(val_rmse, val_mae, "reports", "Validation")
    
    # Plot validation results
    viz.plot_actual_vs_predicted(y_val, y_val_pred, title="Validation: Actual vs Predicted Speed", save=True)
    viz.plot_residuals(y_val, y_val_pred, title="Validation: Residuals", save=True)

    # Plot feature importance
    feature_names = X.columns.tolist()  # Get feature names from the dataset
    # Plot the top 5 most important features
    viz.plot_feature_importance(model_speed, feature_names, top_n=5, save=True)
    
    # Evaluate on test set
    y_test_pred, test_rmse, test_mae = evaluate_model(model_speed, X_test, y_test)
    save_metrics(test_rmse, test_mae, "reports", "Test")
    
    # Plot test results
    viz.plot_actual_vs_predicted(y_test, y_test_pred, title="Test: Actual vs Predicted Speed", save=True)
    viz.plot_residuals(y_test, y_test_pred, title="Test: Residuals", save=True)
    
    # Save the model
    models_dir = "models"
    save_model(model_speed, models_dir)

if __name__ == "__main__":
    main()
