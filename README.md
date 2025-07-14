# Thesis Formulation Report (TFR)
# Plug-and-play regularisation techniques (PnP) for inverse problems
This folder consists of Python code used for the TFR report and project involving PnP techniques for inverse problems. Primarily the application of computed tomopgraphy (CT) scanning was used for applications involving image reconstruction of MNIST digits, Shepp-Logan Phantom and natural images. 

PnP algorithms:
- Proximal gradient descent (PGD)

PnP denoisers:
- Total variation (TV) - taken from adapted skimage library for torch tensors (in the utils folder as torch_denoise_tv_chambolle)
- Block-matching 3D (BM3D) - taken from bm3d library
- Denoising convolutional neural network (DnCNN) - taken from deepinv library
- Dilated-Residual U-network (DRUNet) - taken from deepinv library
- Gradient-step DRUNet (GS-DRUNet) - - taken from deepinv library


Other subfolders:
- Natural_Images - contains example images including MNIST and butterfly.png (butterfly in my childhood home garden) for application use for CT and general image denoising
- 

## Functionality of the files
There was mainly a use of Jupyter notebooks for fast and easy use of running the respective algorithms and denoisers. Each notebook had an array of functions to fit their purpose and use similar elements across the board (e.g algorithms, forward operators, noise etc)
For the deep denoisers' training weights, deepinv Pytorch library was used with help from their documentation (Github: https://github.com/deepinv/deepinv
Documentation:https://deepinv.github.io/deepinv/index.html). Note that all files are implemented to work with torch Python library (inputs as torch tensors, can be adapted to numpy as well).

Notebook files (.ipynb):
- PhantomCTClassicalGradientDescent - Implementation of gradient descent without regularisation then with Tikhonov (L2) and soft-thresholding (L1) 
- PhantomCTPnP - PnP-PGD and PnP-ADMM implementation across the five denoisers for Shepp-Logan Phantom 

Python files (.py):
- radon.py - Discretised forward CT operator (Radon), add noise functions (Poisson and Gaussian)
- algorithms.py - PnP-PGD, PnP-FISTA, PnP-ADMM algorithms with the denoiser choice function used in the above .ipynb files 

## Libraries
In this project, a number of Python libraries were used for model construction, mathematical calculations, plotting and data manipulation (including the standard numpy, matplotlib etc) 
The non-trivial extensively used Python libraries include:
- deepinv -> downloading neural networks (NNs) training weights for the deep denoisers including DnCNN, DRUNet and GSDRUNet (Training weights obtained via https://huggingface.co/deepinv)
- torch and torchvision -> algorithm construction use and use of MNIST digit dataset for application of the algorithms
- skimage --> Shepp-Logan Phantom images and use of Radon transform and filtered back-projection (FBP)
- statsmodels --> alongside numpy and matplotlib, helpful for probability density function plots of noise distributions and histograms plots as well.

Full list of libraries used are stated in the dependencies.yml file.






























