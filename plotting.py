import numpy as np
import matplotlib.pyplot as plt

#CT Radon transform forward model and denoising functions
from skimage.transform import radon, resize, iradon
from skimage.data import shepp_logan_phantom
from skimage.restoration import denoise_tv_chambolle

#MNIST dataset
import torch
import torch.nn.functional as F

from algorithms import *



def comparison_plot(norm_res, method, ytitle='PSNR', color='k', title = 'PSNR'):
    K = len(norm_res)

    plt.plot(range(1, len(norm_res) + 1),norm_res, color=color, label=method)
    k = np.arange(K)
    plt.title(title)
    plt.ylabel(ytitle)
    plt.xlabel('number of iterations: k')
    plt.legend()
    plt.grid(True)

def display_images(image_list, image_titles, ground_truth):
    titles = []
    for img, title in zip(image_list, image_titles):
        
        #cur_min = np.round(np.amin(img), 1)
        #cur_max = np.round(np.amax(img), 1)
        #bounds = '{} to {}'.format(str(cur_min), str(cur_max))
        psnr = PSNR(ground_truth, img)
        titles.append(title + f' PSNR: {psnr:.2f} dB \n')
        

        #nrmse = pnpm.nrmse(img, ground_truth)
        #titles.append(title + ' [NRMSE: ' + str(nrmse) + ']')

    for img, title in zip(image_list, titles):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 5))
        

        ax.imshow(img, cmap='gray')
        #pnpm.display_image(img, fig=fig, ax=ax, cmap='gray')

        plt.suptitle(title)
        plt.tight_layout()
        fig.show()


def plotting_estimates1(fbp, x1, x2, x3, x4, x5, psnrs_list, title):
    """
    Plot of FBP and the five denoisers' reconstructions
    """
    size = 19
    fig, axs = plt.subplots(2, 3, figsize=(10, 6.5))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle(title, size = 20, y = 1.02)

    vmin = min(fbp.min(), x1.min(), x2.min(), x3.min(), x4.min(), x5.min())
    vmax = max(fbp.max(), x1.max(), x2.max(), x3.max(), x4.max(), x5.max())

    # Plotting the images
    im0 = axs[0, 0].imshow(fbp, cmap='gray', vmin=vmin, vmax=vmax)
    axs[0, 0].set_title(f"FBP \n PSNR: {psnrs_list[0]:.2f} dB", fontsize=size)
    axs[0, 1].imshow(x1, cmap='gray', vmin=vmin, vmax=vmax)
    axs[0, 1].set_title(f'TV \n PSNR: {psnrs_list[1]:.2f} dB', fontsize=size)
    axs[0, 2].imshow(x2, cmap='gray', vmin=vmin, vmax=vmax)
    axs[0, 2].set_title(f'BM3D \n PSNR: {psnrs_list[2]:.2f} dB', fontsize=size)
    axs[1, 0].imshow(x3, cmap='gray', vmin=vmin, vmax=vmax)
    axs[1, 0].set_title(f'DnCNN \n PSNR: {psnrs_list[3]:.2f} dB', fontsize=size)
    axs[1, 1].imshow(x4, cmap='gray', vmin=vmin, vmax=vmax)
    axs[1, 1].set_title(f'DRUNet \n PSNR: {psnrs_list[4]:.2f} dB', fontsize=size)
    axs[1, 2].imshow(x5, cmap='gray', vmin=vmin, vmax=vmax)
    axs[1, 2].set_title(f'GS-DRUNet \n PSNR: {psnrs_list[5]:.2f} dB', fontsize=size)

    # Adding a color bar for the whole figure
    cbar = fig.colorbar(im0, ax=axs, orientation='horizontal', fraction=0.08, pad=0.1)
    cbar.ax.tick_params(labelsize=size)

    for i in range(2):
        for j in range(3):
            axs[i, j].axis('off')