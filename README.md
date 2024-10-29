# SSNS

## Environment
Use the following commands to create a virtual environment and install necessary packages:

```bash
conda create -n ssns
conda activate ssns
conda install python=3.10 numpy scipy matplotlib notebook ipywidgets
pip install pot cvxpy clarabel scs qpsolvers python-mnist
import regot
```
## Experiments
We evaluate the SSNS algorithm's performance through various numerical experiments, focusing on entropic-regularized optimal transport (OT). We compare SSNS with other optimization methods, including the Sinkhorn algorithm, APDAGD, L-BFGS, and the globalized Newton method.

### Datasets and Experiment Setup
We use three benchmark datasets to define the OT problem, with experiments conducted on both (Fashion-)MNIST and ImageNet datasets:
- **(Fashion-)MNIST**: ```fashion``` and ```data_mnist```
- **ImageNet**: ```train_feature```, ```train_feature_60``` and ```train_feature_90```
  
### Code on Experiments
#### Retrieving Data for Different Algorithms:
- **Comparative Experiments**: ``` get-pkl-for-all.py```
- **Impact of Feature Dimension**: ``` get-pkl-for-feature.py```
- **Scalability**: ``` get-pkl-for-large-scale.py```

#### Plotting from PKL Data：
- **Comparative Experiments**: ``` plot-pdf-for-all.py```
- **Impact of Feature Dimension**: ``` plot-pdf-for-feature.py```
- **Scalability**: ``` plot-pdf-for_large-scale.py```

