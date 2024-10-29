## Section: Numerical Experiments and Algorithm Performance

### Overview
We evaluate the SSNS algorithm's performance through various numerical experiments, focusing on entropic-regularized optimal transport (OT). We compare SSNS with other optimization methods, including the Sinkhorn algorithm, APDAGD, L-BFGS, and the globalized Newton method.

### Datasets and Experiment Setup
We use three benchmark datasets to define the OT problem, with experiments conducted on both (Fashion-)MNIST and ImageNet datasets:
- **(Fashion-)MNIST**: Flat and normalized pixel values are used, employing either ℓ1-distances or squared Euclidean distances between pixels.
- **ImageNet**: Images are processed through a ResNet18 network followed by dimension reduction to derive 30-dimensional feature vectors. Costs are calculated based on these features.
  
#### Improvements on Experiments
Per reviewer suggestions, we have made the following adjustments to enhance the experiments:
1. **Reproducibility**: We improve reproducibility by using randomly selected image IDs from prior literature and including additional test cases in Figures 4 and 5.
2. **Impact of Feature Dimension**: Our study demonstrates that the convergence property of SSNS is robust to the feature dimension of input images.
3. **Regularization Parameters**: We analyze the impact of different regularization parameters on the performance of optimization algorithms, with results presented in Section 5. The convergence tolerance and cost matrix normalization are precisely set to ensure comparability.
4. **Scalability**: We evaluate SSNS on large synthetic OT problems with various continuous distributions to test scalability, confirming SSNS's efficiency and fast convergence speed even for very large problem sizes.

### Algorithm Settings
- **Entropic Regularization**: Cost matrices are normalized to unit infinity norms with regularization parameters η set at 0.01 and 0.001.
- **Evaluation Metric**: Performance is measured by the marginal error of the transport plan, evaluated by the norm of the gradient.

### Results
Detailed results and comparisons are presented in Section A.3 of the main document, highlighting the efficiency and effectiveness of the SSNS algorithm against the backdrop of established methods.
