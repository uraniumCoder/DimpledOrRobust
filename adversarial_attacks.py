import advertorch.attacks.iterative_projected_gradient
from advertorch.attacks.iterative_projected_gradient import *
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from projection  import projection

def patch_perturb(project_onto_k=False, project_onto_not_k=False, get_local_manifold_approx=None, LATENT_DIM=None, IMAGESPACE_DIM=None,):
    def greedy_perturb_iterative(xvar, yvar, predict, nb_iter, eps, eps_iter, loss_fn,
                                                delta_init=None, minimize=False, ord=np.inf,
                                                clip_min=0.0, clip_max=1.0,
                                                l1_sparsity=None):
        """
        Iteratively maximize the loss over the input. It is a shared method for
        iterative attacks including IterativeGradientSign, LinfPGD, etc.
        :param xvar: input data.
        :param yvar: input labels.
        :param predict: forward pass function.
        :param nb_iter: number of iterations.
        :param eps: maximum distortion.
        :param eps_iter: attack step size.
        :param loss_fn: loss function.
        :param delta_init: (optional) tensor contains the random initialization.
        :param minimize: (optional bool) whether to minimize or maximize the loss.
        :param ord: (optional) the order of maximum distortion (inf or 2).
        :param clip_min: mininum value per input dimension.
        :param clip_max: maximum value per input dimension.
        :param l1_sparsity: sparsity value for L1 projection.
                                    - if None, then perform regular L1 projection.
                                    - if float value, then perform sparse L1 descent from
                                        Algorithm 1 in https://arxiv.org/pdf/1904.13000v1.pdf
        :return: tensor containing the perturbed input.
        """
        if delta_init is not None:
            delta = delta_init
        else:
            delta = torch.zeros_like(xvar)

        delta.requires_grad_()
        for ii in range(nb_iter):
            outputs = predict(xvar + delta)
            if outputs.argmax(-1).item() != yvar.item():
                break
            loss = loss_fn(outputs, yvar)
            if minimize:
                    loss = -loss

            loss.backward()
            if ord == np.inf:
                grad_sign = delta.grad.data.sign()
                delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
                delta.data = batch_clamp(eps, delta.data)
                delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                                                    ) - xvar.data

            elif ord == 2:
                grad = delta.grad.data

                grad_shape = grad.shape

                # projection onto image manifold
                if project_onto_k or project_onto_not_k:
                    local_manifold = get_local_manifold_approx(xvar)
                    local_manifold_flat = local_manifold.reshape((LATENT_DIM, IMAGESPACE_DIM)).transpose(0, 1).cuda()
                    grad_flat = grad.reshape((IMAGESPACE_DIM, 1)).cuda

                    on_manifold = projection(grad_flat,local_manifold_flat)

                    if project_onto_k:
                        grad = on_manifold.reshape(grad_shape)

                    if project_onto_not_k:
                        off_manifold = grad_flat - on_manifold
                        grad = off_manifold.reshape(grad_shape)

                grad = normalize_by_pnorm(grad)
                delta.data = delta.data + batch_multiply(eps_iter, grad)
                delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                                                    ) - xvar.data
                if eps is not None:
                    delta.data = clamp_by_pnorm(delta.data, ord, eps)

            elif ord == 1:
                grad = delta.grad.data
                abs_grad = torch.abs(grad)

                batch_size = grad.size(0)
                view = abs_grad.view(batch_size, -1)
                view_size = view.size(1)
                if l1_sparsity is None:
                    vals, idx = view.topk(1)
                else:
                    vals, idx = view.topk(
                        int(np.round((1 - l1_sparsity) * view_size)))

                out = torch.zeros_like(view).scatter_(1, idx, vals)
                out = out.view_as(grad)
                grad = grad.sign() * (out > 0).float()
                grad = normalize_by_pnorm(grad, p=1)
                delta.data = delta.data + batch_multiply(eps_iter, grad)

                delta.data = batch_l1_proj(delta.data.cpu(), eps)
                if xvar.is_cuda:
                    delta.data = delta.data.cuda()
                delta.data = clamp(xvar.data + delta.data, clip_min, clip_max
                                                    ) - xvar.data
            else:
                error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
                raise NotImplementedError(error)
            delta.grad.data.zero_()

        x_adv = clamp(xvar + delta, clip_min, clip_max)
        return x_adv

    advertorch.attacks.iterative_projected_gradient.perturb_iterative = greedy_perturb_iterative
    return greedy_perturb_iterative