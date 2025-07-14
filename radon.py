import numpy as np
import matplotlib.pyplot as plt

#CT Radon transform forward model and denoising functions
from skimage.transform import radon, resize, iradon
from skimage.data import shepp_logan_phantom
from skimage.restoration import denoise_tv_chambolle

#MNIST dataset
import torch
import torch.nn.functional as F

# Deep inverse imports
import deepinv as dinv
from deepinv.models import DnCNN, DRUNet



def forward_operator_radon(n, angles):
    """
    Forward operator - Radon transform discretised version with matrix A^(m x n^2)
    n - number of pixels of input image
    m - number of angles for the Radon transform to be performed muliplied by number of detector elements
    """
    circle = False
    theta = np.linspace(0., 180., angles, endpoint=False)
    size = radon(np.zeros((n, n)), theta=theta, circle=circle).shape
    M = size[0] * size[1]
    A = torch.zeros((M, n**2), dtype=torch.float32)
    
    for i in range(n**2):
        e = torch.zeros((n**2,), dtype=torch.float32)
        e[i] = 1
        e = e.view(n, n)
        
        # Convert e to numpy for radon
        sinogram = radon(e.numpy(), theta=theta, circle=circle)
        
        # Convert sinogram back to torch and assign to A
        A[:, i] = torch.from_numpy(sinogram.flatten())
    
    return A



# Add noise to the sinogram (Gaussian and Poisson)
def add_noise(sinogram, noise_type='gaussian', sigma=0.01):
    """
    Add different types of noise to a sinogram
    sinogram - original sinogram input
    noise_type - the type of noise to add ['gaussian', 'poisson']
    sigma - standard deviation for Gaussian noise input
    """

    if noise_type == 'gaussian':
        n = sinogram.shape[0]
        m = sinogram.shape[1]
        noise = sigma * torch.randn(n,m)
        noisy_sinogram = sinogram + noise
    
    elif noise_type == 'poisson':
        noisy_sinogram = torch.poisson(sinogram)


    return noisy_sinogram


# FBP reconstruction in torch
def fbp(sinogram, angles):
    sinogram_np = sinogram.numpy()
    theta = np.linspace(0., 180., angles, endpoint=False)
    reconstructed_image = iradon(sinogram_np, theta=theta, circle=False)
    return torch.from_numpy(reconstructed_image)