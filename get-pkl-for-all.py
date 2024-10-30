import numpy as np
import regot
import os
from mnist.loader import MNIST
from scipy.spatial.distance import cdist
from tqdm import tqdm
import pickle

def example_MNIST(mndata, p_id, q_id, metric='euclidean'):
    images, labels = mndata.load_training()

    def mnist(eps, p_id, q_id, n):
        p, q = np.float64(images[p_id]), np.float64(images[q_id])
        p, q = p / sum(p), q / sum(q)
        p = (1 - eps / 8) * p + eps / (8 * n)
        q = (1 - eps / 8) * q + eps / (8 * n)
        return p, q

    def cartesian_product(*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    n = len(images[0])
    m = int(np.sqrt(n))
    a, b = mnist(eps=0.01, p_id=p_id, q_id=q_id, n=n)
    M = np.arange(m)
    M = cartesian_product(M, M)
    M = cdist(M, M, metric=metric)
    M = M * M if metric == 'euclidean' else M
    M /= np.max(M)
    return M, a, b


def example_FashionMNIST(fashionmndata, p_id, q_id, metric='euclidean'):
    images, labels = fashionmndata.load_training()

    def mnist(eps, p_id, q_id, n):
        p, q = np.float64(images[p_id]), np.float64(images[q_id])
        p, q = p / sum(p), q / sum(q)
        p = (1 - eps / 8) * p + eps / (8 * n)
        q = (1 - eps / 8) * q + eps / (8 * n)
        return p, q

    def cartesian_product(*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    n = len(images[0])
    m = int(np.sqrt(n))
    a, b = mnist(eps=0.01, p_id=p_id, q_id=q_id, n=n)
    M = np.arange(m)
    M = cartesian_product(M, M)
    M = cdist(M, M, metric=metric)
    M = M * M if metric == 'euclidean' else M
    M /= np.max(M)
    return M, a, b


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


method_functions = {
    "BCD": regot.sinkhorn_bcd,
    "APDAGD": regot.sinkhorn_apdagd,
    "Dual L-BFGS": regot.sinkhorn_lbfgs_dual,
    "Newton": regot.sinkhorn_newton,
    "SSNS": regot.sinkhorn_ssns
}


def process_and_save(M, a, b, reg, identifier, method_names, metric):
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
    base_save_path = f'./save/{norm_dir}/reg={reg}'
    os.makedirs(base_save_path, exist_ok=True)
    file_path = os.path.join(base_save_path, f'{identifier}.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(results, file)


# Main script
np.random.seed(123)
mndata = MNIST('./data_mnist/')
fashionmndata = MNIST('./fashion/')
base_path = './train_feature'
feature_list_one = ['train_feature1', 'train_feature1', 'train_feature1', 'train_feature1', 'train_feature1']
feature_list_two = ['train_feature2', 'train_feature3', 'train_feature4', 'train_feature5', 'train_feature6']
p_list = [34860, 2, 239, 37372, 17390]
q_list = [45815, 54698, 43981, 54698, 49947]
reg_list = [0.01, 0.001]

method_names = ["BCD", "APDAGD", "Dual L-BFGS", "Newton", "SSNS"]
metrics = ['euclidean', 'cityblock']

for reg in reg_list:
    for metric in metrics:
        # MNIST
        for p_id, q_id in tqdm(zip(p_list, q_list), total=len(p_list),
                               desc=f"Processing MNIST data pairs for reg={reg}, metric={metric}"):
            M, a, b = example_MNIST(mndata, p_id, q_id, metric=metric)
            identifier = f"MNIST_{p_id}_{q_id}"
            process_and_save(M, a, b, reg, identifier, method_names, metric)

        # FashionMNIST
        for p_id, q_id in tqdm(zip(p_list, q_list), total=len(p_list),
                               desc=f"Processing FashionMNIST data pairs for reg={reg}, metric={metric}"):
            M, a, b = example_FashionMNIST(fashionmndata, p_id, q_id, metric=metric)
            identifier = f"FashionMNIST_{p_id}_{q_id}"
            process_and_save(M, a, b, reg, identifier, method_names, metric)

        # Real ImagineNet data
        for feature_one, feature_two in tqdm(zip(feature_list_one, feature_list_two), total=len(feature_list_one),
                                             desc=f"Processing real data pairs for reg={reg}, metric={metric}"):
            M, a, b = example_real_data(base_path, feature_one, feature_two, metric=metric)
            identifier = f"{feature_one}_{feature_two}"
            process_and_save(M, a, b, reg, identifier, method_names, metric)
