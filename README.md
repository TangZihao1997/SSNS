# SSNS
 code for Safe and Sparse Newton Method for Entropic-Regularized Optimal Transport

## Section: Numerical Experiments and Algorithm Performance

### Overview
We evaluate the SSNS algorithm's performance through various numerical experiments, focusing on entropic-regularized optimal transport (OT). We compare SSNS with other optimization methods, including the Sinkhorn algorithm, APDAGD, L-BFGS, and the globalized Newton method.

### Datasets and Experiment Setup
We use three benchmark datasets to define the OT problem, with experiments conducted on both (Fashion-)MNIST and ImageNet datasets. The MNIST dataset involves flat and normalized pixel values, using either ℓ1-distances or squared Euclidean distances between pixels. For ImageNet, images are processed through a ResNet18 network followed by dimension reduction to derive 30-dimensional feature vectors. Costs are calculated based on these features.

### Algorithm Settings
- **Entropic Regularization**: We normalize cost matrices to unit infinity norms and set the regularization parameter η at 0.01 and 0.001.
- **Evaluation Metric**: Performance is assessed by the marginal error of the transport plan, evaluated by the norm of the gradient.

### Results
Detailed results and comparisons are presented in Section A.3 of the main document, highlighting the efficiency and effectiveness of the SSNS algorithm against the backdrop of established methods. 

 
