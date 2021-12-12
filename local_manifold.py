import torch
import math

def get_local_approximation(encoder, decoder, image, eps=None, batch_size=None):
    """
    Gets set of vectors spanning the tangent space at image of the image manifold
    image is a tensor of shape (3, H, W)
    """
    z = encoder(image[None, :, :, :])
    if type(z) is tuple:
        z = z[0]
    z_flat = z.reshape((1, -1))
    image_on_manifold = decoder(z)
    if type(image_on_manifold) is tuple:
        image_on_manifold = image_on_manifold[0]
    if eps is None:
        eps = 0.01
    z_perturbed_flat = torch.eye(z_flat.shape[1], device='cuda')*eps + z_flat
    
    if batch_size is None:
        batch_size = z_perturbed_flat.shape[0]

    image_perturbed = torch.zeros(z_perturbed_flat.shape[0], 3, image.shape[1], image.shape[2], device='cuda')
    for i in range(math.ceil(z_perturbed_flat.shape[0]/batch_size)):
        end = min(i*batch_size+batch_size, z_perturbed_flat.shape[0])
        z_perturbed = z_perturbed_flat[i*batch_size:end, :].reshape((-1, *z.shape[1:]))
        image_on_manifold_perturbed = decoder(z_perturbed)
        if type(image_on_manifold_perturbed) is tuple:
            image_on_manifold_perturbed = image_on_manifold_perturbed[0]
        image_perturbed[i*batch_size:end, :, :, :] = image_on_manifold_perturbed.detach()

    return (image_perturbed - image_on_manifold).detach() / eps

def get_orthonormal_basis(tangent_vectors):
    """
    Gets orthonormal basis of tangent space of image manifold
    tangent_vectors is a tensor of shape (imagespace_dim, latent_dim)
    """
    u, s, v = torch.linalg.svd(tangent_vectors, full_matrices=False)
    return u

def sample_from_orthonormal_basis(orthonormal_basis, num_samples):
    """
    Samples num_samples from orthonormal basis
    orthonormal_basis is a tensor of shape (imagespace_dim, latent_dim)
    returns a tensor of shape (num_samples, imagespace_dim)
    """
    return orthonormal_basis @ torch.randn(*orthonormal_basis.shape[1:], num_samples, device='cuda')
