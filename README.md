# SSNS

## Environment
Use the following commands to create a virtual environment and install necessary packages:

```bash
conda create -n ssns
conda activate ssns
conda install python=3.10 numpy scipy matplotlib notebook ipywidgets
pip install pot cvxpy clarabel scs qpsolvers python-mnist
import regot

## Experiments
We evaluate the SSNS algorithm's performance through various numerical experiments, focusing on entropic-regularized optimal transport (OT). We compare SSNS with other optimization methods, including the Sinkhorn algorithm, APDAGD, L-BFGS, and the globalized Newton method.

### Datasets and Experiment Setup
We use three benchmark datasets to define the OT problem, with experiments conducted on both (Fashion-)MNIST and ImageNet datasets:
- **(Fashion-)MNIST**: Flat and normalized pixel values are used, employing either ℓ1-distances or squared Euclidean distances between pixels.
- **ImageNet**: Images are processed through a ResNet18 network followed by dimension reduction to derive 30-dimensional feature vectors. Costs are calculated based on these features.
  
### Code on Experiments
#### Retrieving Data for Different Algorithms:
- **Comparative Experiments**: ```bash get-pkl-for-all.py```
- **Impact of Feature Dimension**: ```bash get-pkl-for-feature.py```
- **Scalability**: ```bash get-pkl-for-large-scale.py```

#### Plotting from PKL Data：
- **Comparative Experiments**: ```bash plot-pdf-for-all.py```
- **Impact of Feature Dimension**: ```bash plot-pdf-for-feature.py```
- **Scalability**: ```bash plot-pdf-for_large-scale.py```

### Results
Detailed results and comparisons are presented in Section A.3 of the main document, highlighting the efficiency and effectiveness of the SSNS algorithm against the backdrop of established methods.
