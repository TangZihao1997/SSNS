import numpy as np
import regot
import os
import pickle
from scipy.spatial.distance import cdist
from tqdm import tqdm

method_functions = {
    "BCD": regot.sinkhorn_bcd,
    "APDAGD": regot.sinkhorn_apdagd,
    "Dual L-BFGS": regot.sinkhorn_lbfgs_dual,
    "Newton": regot.sinkhorn_newton,
    "SSNS": regot.sinkhorn_ssns
}

def load_feature_vectors_from_folder(folder_path):
    feature_vectors = []
    for file in os.listdir(folder_path):
        if file.endswith('.npy'):
            file_path = os.path.join(folder_path, file)
            feature_vector = np.load(file_path)
            feature_vectors.append(feature_vector)
    return np.array(feature_vectors)

def example_real_data(base_path, feature_one, feature_two, metric='euclidean'):
    full_path_one = os.path.join(base_path, feature_one)
    full_path_two = os.path.join(base_path, feature_two)
    M_feature_one = load_feature_vectors_from_folder(full_path_one)
    M_feature_two = load_feature_vectors_from_folder(full_path_two)
    M = cdist(M_feature_one, M_feature_two, metric=metric)
    M = M * M if metric == 'euclidean' else M
    M /= np.max(M)
    a = 1 / M_feature_one.shape[0] * np.ones(M_feature_one.shape[0])
    b = 1 / M_feature_two.shape[0] * np.ones(M_feature_two.shape[0])
    return M, a, b


def process_and_save(M, a, b, reg, identifier, method_names, metric, save_path):
    results = []
    for method_name in tqdm(method_names, desc=f"Processing methods for {identifier}"):
        method = method_functions[method_name]
        res = method(M, a, b, reg, tol=1e-8, max_iter=500)
        results.append({
            'method_name': method_name,
            'obj_vals': res.obj_vals,
            'run_times': res.run_times,
            'log10_mar_errs': np.log10(res.mar_errs),
            'iterations': np.arange(len(res.obj_vals))
        })

    norm_dir = '2-norm-square' if metric == 'euclidean' else '1-norm'
    base_save_path = os.path.join(save_path, f'reg={reg}', norm_dir)
    os.makedirs(base_save_path, exist_ok=True)
    file_path = os.path.join(base_save_path, f'{feature_one}_{feature_two}.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(results, file)

base_paths = ['./train_feature_60', './train_feature_90']
save_paths = ['./feature-pkl/pkl-for-imagenet-60', './feature-pkl/pkl-for-imagenet-90']
# feature_list_one = ['train_feature1', 'train_feature1', 'train_feature1', 'train_feature1', 'train_feature1']
# feature_list_two = ['train_feature2', 'train_feature3', 'train_feature4', 'train_feature5', 'train_feature6']
# reg_list = [0.01, 0.001]
# metrics = ['euclidean', 'cityblock']
feature_list_one = ['train_feature1']
feature_list_two = ['train_feature3']
reg_list = [0.001]
metrics = ['cityblock']
method_names = ["BCD", "APDAGD", "Dual L-BFGS", "Newton", "SSNS"]


for base_path, save_path in zip(base_paths, save_paths):
    for reg in reg_list:
        for metric in metrics:
            for feature_one, feature_two in tqdm(zip(feature_list_one, feature_list_two), total=len(feature_list_one),
                                                 desc=f"Processing real data pairs for reg={reg}, metric={metric}"):
                M, a, b = example_real_data(base_path, feature_one, feature_two, metric=metric)
                identifier = f"{feature_one}_{feature_two}_reg_{reg}_metric_{metric}"
                process_and_save(M, a, b, reg, identifier, method_names, metric, save_path)
