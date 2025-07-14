import numpy as np
import matplotlib.pyplot as plt

#CT Radon transform forward model and denoising functions
from skimage.transform import radon, resize, iradon
from skimage.data import shepp_logan_phantom
from skimage.restoration import denoise_tv_chambolle

#TV and BM3D denoisers with tracking
from tqdm import tqdm
from bm3d import bm3d
from utils.torch_denoise_tv_chambolle import *

#MNIST dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

# Deep inverse imports
import deepinv as dinv
from deepinv.models import DnCNN, DRUNet


from radon import fbp, forward_operator_radon, add_noise



def PSNR(original, img):
    """
    Calculate peak-signal-to-noise-ratio for a estimate against the ground truth image
    """
    max_value =  1
    mse = F.mse_loss(original.flatten(), img.flatten())
    if mse == 0:
        return 100
    psnr = 20 * torch.log10(max_value / (torch.sqrt(mse)))
    return psnr.detach()



def power_method(A, num_iterations=1000, tol=1e-6):
    """
    Estimate the largest eigenvalue of a square matrix - used to upper bound the Lipschitz constant of A
    """
    # Initialize a random vector b
    n = A.size(0)
    b = torch.randn(n, 1)

    # Perform the power method iterations
    for _ in range(num_iterations):
        # Multiply A with b
        Ab = torch.mm(A, b)

        # Estimate the eigenvalue
        eigenvalue_estimate = torch.dot(b.squeeze(), Ab.squeeze()) / torch.dot(b.squeeze(), b.squeeze())

        # Normalize b
        b = Ab / torch.norm(Ab)

        # Check for convergence
        if torch.norm(Ab - b * eigenvalue_estimate) < tol:
            break

    return eigenvalue_estimate.item()



def soft_thresh(x, l):
    return torch.sign(x) * F.relu(torch.abs(x) - l)




#Deep denoisers
channels = 1
# DnCNN Zhang 2017 denoiser
dncnn = DnCNN(
    in_channels=channels,
    out_channels=channels,
    pretrained="download",  
    device="cpu",
)

# Zhang 2021 denoiser
drunet = DRUNet(
    in_channels= channels,
    out_channels= channels,
    pretrained="download",
    device="cpu",
)



def apply_denoiser(x, l, imsize, sigma = None,  method = 'tv', ):
        
        """
        Denoiser options to replace the proximal step
        3 handcrafted denoisers: 'TV', 'BM3D', 'proximal'
        2 deep denoisers: 'DnCNN', 'DRUNet'
        """
        if method == 'tv':
            return denoise_tv_chambolle_torch(x, weight = l)
        elif method == 'bm3d':
            print()
            #x = torch.to_numpy(x)
            #print(x.shape)
            x = bm3d(x.reshape(imsize), sigma_psd = sigma*1e+1).flatten()
            return torch.from_numpy(x).float()
        elif method == 'proximal':
            return soft_thresh(x, l)
        elif method == 'DnCNN':
            #print('DnCNN')
            x = x.unsqueeze(0).unsqueeze(0)
            x = dncnn(x, sigma = sigma)
            return x.squeeze(0).squeeze(0)
        elif method == 'DRUNet':
            x = x.reshape(imsize)
            x = x.unsqueeze(0).unsqueeze(0)
            #print("x shape before:", x.shape)
            x = drunet(x, sigma = sigma)
            return x.squeeze(0).squeeze(0).flatten().detach()
        
        


#PnP-PGD
def pnp_pgd(A, b, x_truth, method = None, reg_l = 1e-5, iters = 50, tol = 1e-3, sigma = 0.05, L = 0):
    """
    PnP iterative shrinkage thrseholding algorithm (PnP-ISTA)
    """
    n = int(np.sqrt(A.shape[1]))
    imsize = (n,n)
    x = torch.zeros_like(x_truth, requires_grad=False)
    #x = fbp(b, n_angles).flatten()
    psnrs = []
    iterates_pairs = []
    differences = []
    increments =[]
    gsteps = []
    #L = torch.norm(A) ** 2  # Lipschitz constant
    if L == 0:
        L = power_method(A.T @ A)
    t = 1 / L # Initial stepsize
    b = b.flatten()

    #f = lambda x: 0.5 * torch.norm(A @ x - b) ** 2
    #grad_f =lambda x: A.T @ (A @ x - b)
    #psnr.append(PSNR(x_truth, x))
    for i in tqdm(range(iters), desc = str(method) + '-PnP PGD iterations'):
        
        #gradient descent step
        current_grad = A.T @ (A @ x - b)
        #print("current gradient:", current_grad)
        x_new = x - (t * current_grad)
           
        """
        #backtracking line search (Armijo condition)
        while True:
            x_new = x - (t * current_grad)
            if (torch.norm(A @ (x_new - x)) ** 2) <=  t*(current_grad.T @ current_grad):
                break
            t *= 0.5
        """

        #denoising step (proximal step)
        denoised_x = apply_denoiser(x_new, reg_l, imsize,  method = method, sigma = sigma)
        #x = soft_thresh(x_descent, l / L)

        #inverted and denoised iterates stored
        iterates_pairs.append((x_new, denoised_x))
        #difference between "noisy and denoised iterates"
        diff = x_new - denoised_x
        differences.append(diff)

        #store current gradient desecent step for analysis later
        gsteps.append(x_new)
        #print("MSE for new iterate:", torch.norm(x_truth - denoised_x)**2)
        
        increments.append(torch.norm(x - denoised_x))
        #new estimate
        x = denoised_x
        psnr = PSNR(x_truth, x).detach()
        psnrs.append(psnr)
        if torch.norm(current_grad) <= tol:  # Termination criterion
            print('Iteration {}: gradient norm {:.4e} is less than tolerance {}\n'.format(i, torch.norm(current_grad), tol))
            break

    print(f"PnP-{method} Final PSNR: {psnr:.2f} dB")

    return x, psnrs, gsteps, iterates_pairs, increments



#PnP-FISTA
def pnp_fista(A, b, x_truth, method = None, reg_l = 1e-5, iters = 50, tol = 1e-3, sigma = 0.05, L = 0):
    """
    PnP FISTA (accelerated PGD)
    """

    n = int(np.sqrt(A.shape[1]))
    imsize = (n,n)
    #ground = 0.5 * np.linalg.norm(A.dot(x_g) - b) ** 2 + l * np.linalg.norm(x_g, 1)
    x = torch.zeros_like(x_truth, requires_grad=False)
    #x = fbp(b, n_angles).flatten()
    psnrs = []
    iterates_pairs = []
    differences = []
    increments = []
    if L == 0:
        L = power_method(A.T @ A)  # Lipschitz constant

    b = b.flatten()
    

    #Initialisation of parameters (step size and initial guesses)
    t = 1
    z = x.clone()
    
    for i in tqdm(range(iters), desc = 'PnP FISTA iterations'):
        grad_g = A.T @ (A @ x - b)
        xold = x.clone()
        z = z + A.T @(b - A @ z) / L
        zold = z.clone()
        x = apply_denoiser(z, reg_l, imsize, method= method, sigma = sigma)
        t0 = t
        t = (1 + torch.sqrt(torch.tensor(1 + 4 * t ** 2))) / 2.
        z = x + ((t0 - 1) / t) * (x - xold)
        
        incr = torch.norm(x - xold)
        increments.append(incr)

        #inverted and denoised iterates stored
        iterates_pairs.append((zold, x))
        #difference between "noisy and denoised iterates"
        diff = zold - x 
        differences.append(diff)

        #print("MSE for new iterate:", torch.norm(x_truth - denoised_x)**2)

        psnr = PSNR(x_truth, x).detach()
        psnrs.append(psnr)
        if torch.norm(grad_g) <= tol:  # Termination criterion
            print('Iteration {}: gradient norm {:.4e} is less than tolerance {}\n'.format(i, torch.norm(grad_g), tol))
            break

    print(f"PnP-{method} Final PSNR: {psnr:.2f} dB")
    return x, psnrs, differences, iterates_pairs, increments


def pnp_admm(A, b, x_ground_truth, denoiser, angles = 60, niter = 50, beta = 1e+1, lamb = 1, sigma = 0.05, tol = 1e-6):
    """
    Plug-and-play alternat directions method of multipliers
    """
    n = int(np.sqrt(A.shape[1]))
    imsize = (n,n)
    m = A.shape[0]
    detectors = m//angles
    
    # Define the 3 variables for use
    x = torch.zeros_like(x_ground_truth)
    u = torch.zeros_like(x_ground_truth)
    v = torch.zeros_like(x_ground_truth)
    psnrs = []
    iterates_pairs = []
    diffs = []
    increments =[]
    inv_beta = 1/beta

    # PnP ADMM iteration performed on the 3 variables
    for i in tqdm(range(niter), desc= str(denoiser) + '-PnP ADMM iterations'):
        x_k = x
        #print("b shape:", b.shape)
        #measurement = b.reshape((int(torch.ceil(n * torch.sqrt(torch.tensor(2)))), n))
        correction = (A @ (inv_beta * (u - v)))
        #print("correction shape:", correction.shape)
        #print("b shape:", b.shape)
        sinogram = b + correction.reshape(detectors, angles)
        #print("b shape:", b.shape)
        x = fbp(sinogram, angles).flatten().float()
        #print("x fbp shape:", x.shape)

        
        u = apply_denoiser(x + v, lamb, imsize, method = denoiser, sigma = sigma).float()
        #print("u shape:", u.shape)
    

        v = (v + x - u).float()

        # Flattening u and v for the next iteration (2D to 1D)
        #u = u.flatten()
        #v = v.flatten()


        increment = torch.norm(x - x_k)
        diff = x - u

        #inverted and denoised iterates stored
        iterates_pairs.append((x, u))
        
        #print("MSE for new iterate:", torch.norm(x_truth - denoised_x)**2)
        
        increments.append(increment)
        #difference between "noisy and denoised iterates"
        diffs.append(diff)

        current_psnr = PSNR(x_ground_truth, x).detach()
        psnrs.append(current_psnr)
        if increment <= tol:
            print('Iteration {}: error between sucessive iterates {:.4e} is less than tolerance {}\n'.format(i, increment, tol))
            break

    print(f"Final PSNR: {current_psnr:.2f} dB")

    return x, psnrs, diffs, iterates_pairs, increments
    