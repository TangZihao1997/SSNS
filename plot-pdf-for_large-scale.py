import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'legend.fontsize': 15,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
})

def load_results(identifier):
    with open(f'large-scale-result/{identifier}.pkl', 'rb') as file:
        results = pickle.load(file)
    return results

def plot(identifier):
    results = load_results(identifier)
    if not os.path.exists('pic'):
        os.makedirs('pic')

    metrics = ['obj_vals', 'obj_vals', 'log10_mar_errs', 'log10_mar_errs']
    x_labels = ['iterations', 'run_times', 'iterations', 'run_times']
    x_labels_formatted = ['Iteration Number', 'Run time (seconds)', 'Iteration Number', 'Run time (seconds)']
    y_labels = ['Objective Values', 'Objective Values', 'Log10 Marginal Errors', 'Log10 Marginal Errors']
    file_formats = ['pdf', 'pdf', 'pdf', 'pdf']

    for metric, x_label, x_label_formatted, y_label, file_format in tqdm(
            zip(metrics, x_labels, x_labels_formatted, y_labels, file_formats), total=len(metrics),
            desc="Plotting and Saving"):
        df = pd.DataFrame()

        for result in tqdm(results, desc=f"Processing results"):
            temp_df = pd.DataFrame({
                'X': np.array(result[x_label]) / (1000.0 if x_label == 'run_times' else 1),
                'Y': np.array(result[metric]),
                'Method': result['method_name']
            })
            df = pd.concat([df, temp_df])

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='X', y='Y', hue='Method', style='Method', linewidth=2.5,
                     style_order=['SSNS','BCD'],hue_order=['SSNS','BCD'])
        plt.xlabel(x_label_formatted)
        plt.ylabel(y_label)
        plt.title(f"Synthetic data (m={m}, n={n})")
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(f'pic/{x_label_formatted}.{identifier}.{file_format}', bbox_inches='tight')
        plt.close()

# Example usage
values = [1000,5000,10000]
for value in values:
    n = m = value
    identifier = f"Synthetic_data_m={m}_n={n}"
    plot(identifier)



