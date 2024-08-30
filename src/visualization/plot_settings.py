# src/visualization/plot_settings.py

import matplotlib.pyplot as plt
# print(plt.style.available)

import matplotlib.pyplot as plt

def set_plot_style():
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (8, 6)  # Adjusted figure size
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.spines.top'] = False  # Hide top spine
    plt.rcParams['axes.spines.right'] = False  # Hide right spine
    plt.rcParams['axes.grid'] = True  # Enable grid by default

