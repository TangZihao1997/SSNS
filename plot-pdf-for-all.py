import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tqdm import tqdm


sns.set(style="whitegrid", context="talk", font_scale=1.2)
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'legend.fontsize': 15,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
})
def load_results(identifier, norm, reg):
    base_path = f'./save/{norm}/reg={reg}'
    with open(os.path.join(base_path, f'{identifier}.pkl'), 'rb') as file:
        results = pickle.load(file)
    return results


def plot(identifier, cutoff_times, cutoff_iters, norm, reg):
    results = load_results(identifier, norm, reg)
    base_pic_dir = f'./picsave-test2/{norm}/reg={reg}'
    if not os.path.exists(base_pic_dir):
        os.makedirs(base_pic_dir)

    metrics = ['obj_vals', 'obj_vals', 'log10_mar_errs', 'log10_mar_errs']
    x_labels = ['iterations', 'run_times', 'iterations', 'run_times']
    x_labels_formatted = ['Iteration Number', 'Run time (seconds)', 'Iteration Number', 'Run time (seconds)']
    y_labels = ['Objective Values', 'Objective Values', 'Log10 Marginal Errors', 'Log10 Marginal Errors']

    file_formats = ['pdf', 'pdf', 'pdf', 'pdf']
    line_styles = [(0, (3, 5, 1, 5)), '--', '-.', ':', '-']
    line_width = 2.5

    for key in cutoff_iters:
        if key in identifier:
            cutoff_iter = cutoff_iters[key]
            break

    for key in cutoff_times:
        if key in identifier:
            cutoff_time = cutoff_times[key]
            break

    for metric, x_label, x_label_formatted, y_label, file_format in tqdm(
            zip(metrics, x_labels, x_labels_formatted, y_labels, file_formats), total=len(metrics),
            desc="Plotting and Saving"):

        plt.figure(figsize=(10, 6))
        sns.set_palette("muted")

        for result, line_style in tqdm(zip(results, line_styles), total=len(results),
                                       desc=f"Processing results"):
            x_data = np.array(result[x_label])
            y_data = np.array(result[metric])

            if cutoff_time and x_label == 'run_times':
                mask1 = x_data <= cutoff_time
                x_data = x_data[mask1] / 1000.0
                y_data = y_data[mask1]

            if cutoff_iter and x_label == 'iterations':
                mask2 = x_data <= cutoff_iter
                x_data = x_data[mask2]
                y_data = y_data[mask2]

            sns.lineplot(x=x_data, y=y_data, label=result['method_name'], linestyle=line_style, linewidth=line_width)

        plt.xlabel(x_label_formatted)
        plt.ylabel(y_label)


        if 'MNIST' in identifier and 'Fashion' in identifier:
            parts = identifier.split('_')
            title = f"FashionMNIST (ID1={parts[1]}, ID2={parts[2]})"

        if 'MNIST' in identifier and 'Fashion' not in identifier:
            parts = identifier.split('_')
            title = f"MNIST (ID1={parts[1]}, ID2={parts[2]})"

        elif 'feature' in identifier:
            parts = identifier.split('_')
            feature_mapping = {
                'feature1': 'Tench',
                'feature2': 'English springer',
                'feature3': 'Cassette player',
                'feature4': 'Chain saw',
                'feature5': 'Church',
                'feature6': 'French horn'
            }
            feature_name_1 = feature_mapping.get(parts[1], parts[1])
            feature_name_2 = feature_mapping.get(parts[3], parts[3])
            title = f"ImageNet ({feature_name_1} vs {feature_name_2})"

        plt.title(title)
        plt.legend(loc='lower right')
        plt.grid(True)
        save_path = os.path.join(base_pic_dir, f'{x_label_formatted}.{identifier}.{file_format}')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


np.random.seed(123)
cutoff_times = {
    'MNIST': 3000,
    'feature': 3000
}

cutoff_iters = {
    'MNIST': 300,
    'feature': 300
}

p_list = [34860, 2, 239, 37372, 17390]
q_list = [45815, 54698, 43981, 54698, 49947]

method_names = ["BCD", "APDAGD", "Dual L-BFGS", "Newton", "SSNS"]

feature_list_one = ['train_feature1', 'train_feature1', 'train_feature1', 'train_feature1', 'train_feature1']
feature_list_two = ['train_feature2', 'train_feature3', 'train_feature4', 'train_feature5', 'train_feature6']

for norm in ['2-norm-square', '1-norm']:
    for reg in ['0.01', '0.001']:
        # For MNIST and FashionMNIST
        for p_id, q_id in tqdm(zip(p_list, q_list), total=len(p_list),
                               desc=f"Processing MNIST/FashionMNIST for {norm}, reg={reg}"):
            identifier = f"MNIST_{p_id}_{q_id}"
            plot(identifier, cutoff_times, cutoff_iters, norm, reg)
            identifier = f"FashionMNIST_{p_id}_{q_id}"
            plot(identifier, cutoff_times, cutoff_iters, norm, reg)

        # For ImageNet feature data
        for feature_one, feature_two in tqdm(zip(feature_list_one, feature_list_two), total=len(feature_list_one),
                                             desc=f"Processing ImageNet features for {norm}, reg={reg}"):
            identifier = f"{feature_one}_{feature_two}"
            plot(identifier, cutoff_times, cutoff_iters, norm, reg)
