import torch

def get_local_approximation(encoder, decoder, image):
    """
    Gets set of vectors spanning the tangent space at image of the image manifold
    image is a tensor of shape (3, H, W)
    """
    z = encoder(image[None, :, :, :])
    z_flat = z.reshape((1, -1))
    image_on_manifold = decoder(z)
    eps = 0.01
    z_perturbed_flat = torch.eye(z_flat.shape[1], device='cuda')*eps + z_flat
    z_perturbed = z_perturbed_flat.reshape((-1, *z.shape[1:]))
    image_perturbed = decoder(z_perturbed)
    return (image_perturbed - image_on_manifold).detach()
