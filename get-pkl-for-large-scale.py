import numpy as np
from scipy.stats import expon, norm
import regot
from tqdm import tqdm
import pickle
import time

def example1(n, m):
    x1 = np.linspace(0.0, 5.0, num=n)
    x2 = np.linspace(0.0, 5.0, num=m)
    distr1 = expon(scale=1.0)
    distr2 = norm(loc=1.0, scale=0.2)
    distr3 = norm(loc=3.0, scale=0.5)
    a = distr1.pdf(x1)
    a = a / np.sum(a)
    b = 0.2 * distr2.pdf(x2) + 0.8 * distr3.pdf(x2)
    b = b / np.sum(b)
    M = np.square(x1.reshape(n, 1) - x2.reshape(1, m))
    M /= np.max(M)
    return M, a, b

method_functions = {
    "BCD": regot.sinkhorn_bcd,
    "SSNS": regot.sinkhorn_ssns
}

def process_and_save(M, a, b, reg, identifier, method_names):
    results = []
    for method_name in tqdm(method_names, desc=f"Processing methods for {identifier}"):
        method = method_functions[method_name]
        res = method(M, a, b, reg, tol=1e-8, max_iter=500)
        results.append({
            'method_name': method_name,
            'obj_vals': res.obj_vals,
            'run_times': res.run_times,
            'log10_mar_errs': np.log10(res.mar_errs),
            'iterations': np.arange(len(res.obj_vals))  #
        })

    with open(f'large-scale-result/{identifier}.pkl', 'wb') as file:
        pickle.dump(results, file)



np.random.seed(123)
reg = 0.001
method_names = ["BCD", "SSNS"]
values = [1000, 5000, 10000]
for value in values:
    n = m = value
    M, a, b = example1(n, m)
    identifier = f"Synthetic_data_m={m}_n={n}"
    t1 = time.time()
    process_and_save(M, a, b, reg, identifier, method_names)
    t2 = time.time()
    print(f"Time for n=m={value}: {t2 - t1} seconds")


