# src/visualization/visualize.py
import matplotlib.pyplot as plt
from . import plot_settings
import os
import re

# Apply global plot style settings
plot_settings.set_plot_style()
# plt.plot(y_pred, label='Predicted', color='blue', marker='o', linestyle='--', linewidth=2, markersize=6, alpha=0.7)

def sanitize_filename(title):
    """Sanitize the title to create a valid filename."""
    return re.sub(r'[<>:"/\\|?*]', '_', title).replace(' ', '_').lower()

def plot_actual_vs_predicted(y_true, y_pred, title='Actual vs Predicted', save=False, output_dir="reports/figures"):
    plt.figure()
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted', linestyle=':', alpha=0.9, linewidth=2)
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Speed')
    plt.legend()
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        filename = sanitize_filename(title)
        plt.savefig(os.path.join(output_dir, f"{filename}.png"))
        plt.savefig(os.path.join(output_dir, f"{filename}.eps"))
    else:
        plt.show()

def plot_residuals(y_true, y_pred, title='Residuals (Actual - Predicted)', save=False, output_dir="reports/figures"):
    residuals = y_true - y_pred
    plt.figure()
    plt.plot(residuals.values, label='Residuals', color='purple')
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Residual')
    plt.axhline(0, color='red', linestyle='--')
    plt.legend()
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        filename = sanitize_filename(title)
        plt.savefig(os.path.join(output_dir, f"{filename}.png"))
        plt.savefig(os.path.join(output_dir, f"{filename}.eps"))
    else:
        plt.show()

def plot_feature_importance(model, feature_names, top_n=5, save=False, output_dir="reports/figures"):
    importance = model.get_booster().get_score(importance_type='weight')
    
    try:
        importance = {feature_names[int(k[1:])]: v for k, v in importance.items()}
    except (ValueError, IndexError):
        importance = {k: v for k, v in importance.items()}
    
    # Sort the features by importance and select the top N
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Unzip the sorted importance into keys and values for plotting
    keys, values = zip(*sorted_importance)

    plt.figure()
    plt.barh(keys, values)
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"top_{top_n}_feature_importance.png"))
        plt.savefig(os.path.join(output_dir, f"top_{top_n}_feature_importance.eps"))
    else:
        plt.show()

def plot_rmse_from_file(file_path, save=False, output_dir="reports/figures"):
    iterations = []
    rmse_values = []

    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(": RMSE = ")
            iteration = int(parts[0].split(" ")[1])
            rmse = float(parts[1])
            iterations.append(iteration)
            rmse_values.append(rmse)
    
    # plt.figure(figsize=(15, 5))
    plt.plot(iterations, rmse_values, marker='o', linestyle='-', color='b')
    plt.title('Validation RMSE per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.grid(True)
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "validation_rmse.png"))
        plt.savefig(os.path.join(output_dir, "validation_rmse.eps"))
    else:
        plt.show()

def plot_time_vs_speed(df, cycle_name, save=False, output_dir="reports/figures"):
    """
    Plots Time vs Speed for a specific cycle.

    Parameters:
    df (pd.DataFrame): DataFrame containing the cycle data.
    cycle_name (str): Name of the cycle (e.g., 'D01').
    save (bool): Whether to save the plot or show it.
    output_dir (str): Directory to save the plot if save=True.
    """
    plt.figure()
    plt.plot(df['Time'], df['Speed'], label='Speed')
    # plt.title(f'Time vs Speed for {cycle_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        filename = sanitize_filename(f'Time_vs_Speed_{cycle_name}')
        plt.savefig(os.path.join(output_dir, f"{filename}.png"))
        plt.savefig(os.path.join(output_dir, f"{filename}.eps"))
    else:
        plt.show()

def plot_speed_vs_acceleration(df, save=False, output_dir="reports/figures"):
    """
    Plots Speed vs Acceleration scatter plot.

    Parameters:
    df (pd.DataFrame): DataFrame containing the combined data.
    save (bool): Whether to save the plot or show it.
    output_dir (str): Directory to save the plot if save=True.
    """
    plt.figure()
    plt.scatter(df['Speed'], df['Acceleration'], alpha=0.3)
    plt.title('Speed vs Acceleration')
    plt.xlabel('Speed (m/s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        filename = sanitize_filename('Speed_vs_Acceleration')
        plt.savefig(os.path.join(output_dir, f"{filename}.png"))
        plt.savefig(os.path.join(output_dir, f"{filename}.eps"))
    else:
        plt.show()
        

def plot_train_val_test_split(y_train, y_val, y_test, train_size, val_size, save=False, output_dir="reports/figures"):
    """
    Plots the dataset split showing Training, Validation, and Test sets.

    Parameters:
    y_train (pd.Series): Training set target values.
    y_val (pd.Series): Validation set target values.
    y_test (pd.Series): Test set target values.
    train_size (int): Size of the training set.
    val_size (int): Size of the validation set.
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    y_train.plot(ax=ax)
    y_val.plot(ax=ax)
    y_test.plot(ax=ax, alpha=0.7)
    
    ax.axvline(train_size, color='black', linestyle='--')
    ax.axvline(train_size + val_size, color='red', linestyle='--')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.legend(['Training Set', 'Validation Set', 'Test Set'], loc='upper right')
    plt.title('Dataset Train, Validation/Test split')
    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, "train_val_test_split.png"))
        plt.savefig(os.path.join(output_dir, "train_val_test_split.eps"))
    else:
        plt.show()
        
def plot_actual_vs_predicted_with_split(y_true, y_pred, train_size, val_size, title='Actual vs Predicted', save=False, output_dir="reports/figures"):
    plt.subplots(figsize=(15, 5))

    # Plot the actual and predicted values
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted', linestyle=':', alpha=0.9, linewidth=2)
    
    # Add vertical lines to indicate the training, validation, and test splits
    plt.axvline(train_size, color='black', linestyle='--')
    plt.axvline(train_size + val_size, color='red', linestyle='--')
    
    # Update the legend and title
    plt.legend(['Actual', 'Predicted'], loc='upper left')
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Speed')
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.tight_layout()

    if save:
        os.makedirs(output_dir, exist_ok=True)
        filename = sanitize_filename(title)
        plt.savefig(os.path.join(output_dir, f"{filename}_with_split.png"))
        plt.savefig(os.path.join(output_dir, f"{filename}_with_split.eps"))
    else:
        plt.show()
