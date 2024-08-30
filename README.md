# Advanced Driving Cycle Generation Using XGBoost: A Case Study in Iğdır, Türkiye

## Overview

This project focuses on generating advanced driving cycles using On-Board Diagnostics (OBD) data and the Extreme Gradient Boosting (XGBoost) algorithm. The case study is centered on the city of [Igdir, Turkey](https://www.google.com/maps/dir/39.9186162,44.0435286/39.9228033,44.0456321/39.930725,44.047373/39.9374025,44.0801313/39.9044083,44.0610306/39.9185878,44.0435193/@39.917033,44.0667609,14z/data=!4m2!4m1!3e0!5m1!1e1?entry=ttu&g_ep=EgoyMDI0MDgyNy4wIKXMDSoASAFQAw%3D%3D), where complex and variable traffic conditions make accurate driving cycle modeling essential for vehicle performance and emissions evaluation.

## Project Structure

The project is organized into several directories and files, each serving a specific purpose:

```sh
IgdirDrivingCyclesML/
├── data/
│   ├── processed/                 # Processed datasets
│   │   ├── combined_data.csv       # Combined driving data from all cycles
│   │   ├── cycle_lengths.txt       # Length of each driving cycle
│   │   └── features_data.csv       # Feature-engineered dataset
│   └── raw/                        # Raw datasets
│       ├── D01.txt                 # Raw driving data for cycle D01
│       ├── D02.txt                 # Raw driving data for cycle D02
│       ├── ...                     # (Additional raw data files)
│       └── segment_sizes.mat       # MATLAB file containing segment sizes
├── docs/                           # Documentation (additional materials or instructions)
├── models/                         # Trained models
│   └── xgboost_model.pkl           # Serialized XGBoost model
├── notebooks/                      # Jupyter notebooks for analysis and experimentation
│   └── driving_cycle.ipynb         # Notebook for driving cycle analysis
├── references/                     # Reference materials and scripts
│   ├── create_project_structure.py # Script to create project structure
│   └── project_structure.txt       # Text file outlining project structure
├── reports/                        # Reports, figures, and metrics
│   ├── figures/                    # Generated figures from the analysis
│   ├── predictions.csv             # Model predictions
│   ├── test_metrics.txt            # Test set performance metrics
│   ├── training_output.txt         # Output log from model training
│   ├── validation_metrics.txt      # Validation set performance metrics
│   └── validation_rmse.txt         # RMSE values during validation
├── src/                            # Source code for data processing, feature engineering, modeling, and visualization
│   ├── data/                       # Data processing scripts
│   │   ├── make_dataset.py         # Script to create processed datasets
│   ├── features/                   # Feature engineering scripts
│   │   ├── build_features.py       # Script to create and save features
│   ├── models/                     # Modeling scripts
│   │   ├── train_model.py          # Script to train XGBoost model
│   │   ├── predict_model.py        # Script to generate predictions using trained model
│   │   ├── manual_split_test.py    # Script for manual dataset split and testing
│   ├── visualization/              # Visualization scripts
│   │   ├── plot_settings.py        # Script to set global plot settings
│   │   ├── visualize.py            # Script for various visualization functions
├── README.md                       # This README file
├── environment.yml                 # Conda environment configuration
├── requirements.txt                # Python dependencies
└── setup.py                        # Setup script for the project
```

## Installation

### Prerequisites

- Python 3.7 or higher
- Conda (recommended) or Python's `virtualenv`

### Setting Up the Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/IgdirDrivingCyclesML.git
   cd IgdirDrivingCyclesML
   ```

2. **Create the Conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate igdir-driving-cycles
   ```

   Alternatively, if you're using `virtualenv`, you can install dependencies with:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup the project**:
   ```bash
   python setup.py install
   ```

## Usage

### Step 1: Data Processing

The first step is to process the raw data to create a combined dataset.

```bash
python src/data/make_dataset.py
```

This script will:
- Load the raw data from the `data/raw/` directory.
- Combine the individual cycle datasets.
- Save the combined dataset to `data/processed/combined_data.csv`.
- Generate visualizations of the time vs. speed and speed vs. acceleration for the dataset.

### Step 2: Feature Engineering

Once the data is processed, you can generate features for the model.

```bash
python src/features/build_features.py
```

This script will:
- Decompose the time series into trend, seasonal, and residual components.
- Create lagged features to capture temporal dependencies.
- Save the feature-engineered dataset to `data/processed/features_data.csv`.

### Step 3: Model Training

Train the XGBoost model using the prepared features.

```bash
python src/models/train_model.py
```

This script will:
- Split the dataset into training, validation, and test sets.
- Train the XGBoost model with early stopping based on validation performance.
- Save the trained model to `models/xgboost_model.pkl`.
- Generate and save various visualizations, such as feature importance and validation results.

### Step 4: Model Prediction

Generate predictions using the trained model.

```bash
python src/models/predict_model.py
```

This script will:
- Load the trained XGBoost model.
- Apply the model to the test dataset to make predictions.
- Save the predictions to `reports/predictions.csv`.
- Generate and save visualizations comparing actual vs. predicted speeds.

### Step 5: Manual Data Split and Testing

You can manually split the data into training, validation, and test sets for further analysis.

```bash
python src/models/manual_split_test.py
```

This script will:
- Split the dataset manually based on predefined cycle lengths.
- Evaluate the model on the manually split test set.
- Generate and save visualizations for the manual test split results.

## Project Documentation

### Figures and Visualizations

All figures generated during the data processing, feature engineering, and modeling steps are saved in the `reports/figures/` directory. These include:

- **Time vs Speed** for individual driving cycles.
- **Speed vs Acceleration** distribution.
- **Feature Importance** as determined by the XGBoost model.
- **Actual vs Predicted Speeds** for both validation and test datasets.
- **Residuals** for the model predictions.

### Reports and Logs

Performance metrics and training logs are saved in the `reports/` directory. These include:

- `validation_metrics.txt`: RMSE and MAE for the validation set.
- `test_metrics.txt`: RMSE and MAE for the test set.
- `training_output.txt`: Detailed log of the training process, including hyperparameter settings and training progress.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or raise an issue.

## Usage
You are free to use the code and data provided in this repository for research, analysis and development. 
If you use the code or data in this repository in your work, please cite our article:
```diff
@Article{
}
```
```diff

```

## License
The code and data provided in this repository are licensed under the MIT License. See the [![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

# Contact
If you have any questions or feedback, please contact at: [email@gmail.com](mailto:email@gmail.com)
