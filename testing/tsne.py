"""
Visualize the latent space of a VAE using TSNE.
"""
import pathlib
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def draw_digits_latent_space(z_mean_test, test_label, ax):
    scatter = ax.scatter(z_mean_test[:,0],z_mean_test[:,1],c=test_label)#,cmap=plt.cm.get_cmap('jet', 10))
    plt.colorbar(scatter, ax=ax)

PERPLEXITIES = [2, 5, 10, 50, 100, 500]

def tsne_plots_model(vae_model, dataloader, save_path, perplexities=PERPLEXITIES, pca_components=150):
    """
    TSNE plots for the VAE model.
    This function projects onto 150 dimensions first using PCA, then uses TSNE to project onto 2 dimensions.
    """
    fig = plt.figure(figsize=(30, 15))
    fig.tight_layout()
    ax_arr = fig.subplots(2, 3)
    axs = [x for y in ax_arr for x in y]

    Z_all = []
    labels = []
    with torch.no_grad():
        for X, y in dataloader:
            Z, _ = vae_model.encode(X.cuda())
            Z_all.append(Z.cpu())
            labels.append(y)

    Z_stacked = torch.cat(Z_all, dim=0)
    labels_stack = torch.cat(labels, dim=0)


    for i, purp in enumerate(perplexities):
        pca = PCA(n_components=pca_components)
        tsne = TSNE(n_components=2, random_state=42, perplexity=purp)

        z_pca = pca.fit_transform(Z_stacked[:])
        z_tsne = tsne.fit_transform(z_pca)

        draw_digits_latent_space(z_tsne, labels_stack[:], axs[i])
        axs[i].set_title(f'T-SNE Perplexity {purp}')

    plt.savefig(save_path)