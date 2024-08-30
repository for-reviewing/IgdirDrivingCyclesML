# src/models/predict_model.py

import os
import pandas as pd
import joblib
import sys
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

def make_forecast(df, model):
    X = df.drop(columns=['Time', 'Speed', 'Acceleration', 'Cycle'])
    predictions = model.predict(X)
    df['Predicted_Speed'] = predictions
    return df

def save_predictions(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'predictions.csv')
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def main():
    processed_data_dir = "data/processed"
    processed_file = os.path.join(processed_data_dir, 'features_data.csv')
    model_dir = "models"
    
    df = pd.read_csv(processed_file)
    model = load_model(model_dir)
    
    predictions_df = make_forecast(df, model)
    
    reports_dir = "reports"
    save_predictions(predictions_df, reports_dir)
    
    # to get following values first run manual_split_test.py script
    train_size=13563
    val_size=3087
    # Visualize predictions
    # viz.plot_actual_vs_predicted(predictions_df['Speed'], predictions_df['Predicted_Speed'],title='All prediction: Actual vs Predicted', save=False,)
    viz.plot_actual_vs_predicted_with_split(predictions_df['Speed'], predictions_df['Predicted_Speed'], train_size, val_size, title='All predictions: Actual vs Predicted', save=True)

if __name__ == "__main__":
    main()
