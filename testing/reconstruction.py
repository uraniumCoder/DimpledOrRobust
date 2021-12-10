"""
Calculates the average L2 distance between images and reconstructed images by an autoencoder.
"""
from DimpledOrRobust.data_preprocessing.cifar10 import DatasetMaker
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from data_preprocessing.cifar10 import Cifar10DatasetFiltered
denormalize = Cifar10DatasetFiltered.denormalize

def show_reconstructed_images(model, dataloader, save_path):
    """
    Show 5 sample images and corresponding reconstructed images
    """

    fig = plt.figure(figsize=(25, 10))
    ax_arr = fig.subplots(2, 5)

    plt.axis('off')
    for X, y in dataloader:
        for i in range(5):
            image_orig = torch.permute(denormalize(X.cuda()), (0,2,3,1))
            model_reconstructed = model(X.cuda())
            if type(model_reconstructed) is tuple:
                model_reconstructed = model_reconstructed[0]
            image_regen = torch.permute(torch.clamp(denormalize(model_reconstructed), 0, 1), (0,2,3,1))
            ax_arr[0][i].imshow(image_orig[i].cpu().numpy())
            ax_arr[1][i].imshow(image_regen[i].cpu().detach().numpy())
        break
    ax_arr[0][2].set_title('Original CIFAR-10 Images')
    ax_arr[1][2].set_title('Reconstructed Images')
    fig.tight_layout()
    plt.savefig(save_path)

def average_l2_dist(model, dataloader):
    total_l2 = 0
    count = 0
    model.eval()
    for X, y in tqdm(dataloader):
        image_orig = denormalize(X.cuda())
        model_reconstructed = model(X.cuda())
        if type(model_reconstructed) is tuple:
            model_reconstructed = model_reconstructed[0]
        image_regen = torch.clamp(denormalize(model_reconstructed), 0, 1)
        err = image_regen - image_orig

        l2_dist = torch.sqrt((err**2).sum(dim=1).sum(dim=1).sum(dim=1))
        total_l2 += l2_dist.sum().item()
        count += X.shape[0]
        
    print('Average L2 Distance: ', total_l2 / count)
    return total_l2 / count