from torch.utils.data import DataLoader
from tqdm import tqdm
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
import advertorch
import torch
import numpy as np
import torch.nn.functional as F
import math
from matplotlib import pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from local_manifold import get_local_approximation, get_orthonormal_basis, sample_from_orthonormal_basis
from projection import projection
from adversarial_attacks import patch_perturb

class AdversarialProjectionExperiment():
    def __init__(self,
                 dataset,
                 autoencoder_model,
                 classifier_model,
                 latent_dim,
                 imagespace_dim,
                 device,
                 denormalize
                 ):
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.autoencoder_model = autoencoder_model
        self.classifier_model = classifier_model
        self.LATENT_DIM = latent_dim
        self.IMAGESPACE_DIM = imagespace_dim
        self.device = device
        self.denormalize = denormalize

    def first_experiment(self, order=2, plot_perturbations=True, save_path=None, **kwargs):
        """
        Replicates first experiment in Adi Shamir paper
        """
        advertorch.attacks.iterative_projected_gradient.perturb_iterative = patch_perturb()
        if order == 2:
            params = {'eps': 20.0, 'eps_iter': 1.0, 'nb_iter': 50,    
                'rand_init': False, 'targeted': False, 'clip_min': -10.0, 'clip_max': 10.0}
            params.update(kwargs)
            adversary_cifar10 = L2PGDAttack(
                self.classifier_model, **params)
        else:
            params = {'eps': 3.0, 'eps_iter': 0.1, 'nb_iter': 50,    
                'rand_init': False, 'targeted': False, 'clip_min': -10.0, 'clip_max': 10.0}
            params.update(kwargs)
            adversary_cifar10 = LinfPGDAttack(
                self.classifier_model, **params)

        norm_ratios, rand_ratios = [], []
        try:
            for cln_data, true_label in tqdm(self.dataloader):
                cln_data, true_label = cln_data.to(self.device), true_label.to(self.device)
                adv_untargeted = adversary_cifar10.perturb(cln_data, true_label)

                perturbation_vector = adv_untargeted - cln_data
                perturbation_vector_flat = perturbation_vector.reshape((self.IMAGESPACE_DIM,))[:, None]

                local_manifold = get_local_approximation(self.autoencoder_model.encode, self.autoencoder_model.decode, cln_data[0])
                local_manifold_flat = local_manifold.reshape((self.LATENT_DIM, self.IMAGESPACE_DIM)).transpose(0, 1).cuda()
                on_manifold_vector = projection(perturbation_vector_flat, local_manifold_flat) 

                random_manifold = torch.normal(torch.zeros(self.IMAGESPACE_DIM, self.LATENT_DIM), torch.ones(self.IMAGESPACE_DIM, self.LATENT_DIM)).cuda()
                on_random_vector = projection(perturbation_vector_flat, random_manifold) 

                off_manifold_vector = perturbation_vector_flat - on_manifold_vector
                
                norm_ratio = (on_manifold_vector.norm()/perturbation_vector.norm()).item()
                random_ratio = (on_random_vector.norm()/perturbation_vector.norm()).item()
                
                if not np.isnan(norm_ratio):
                    norm_ratios.append(norm_ratio)

                if not np.isnan(random_ratio):
                    rand_ratios.append(random_ratio)
        except KeyboardInterrupt:
            pass

        if plot_perturbations:
            self.visualize_perturbations(save_path, perturbation_vector, on_manifold_vector, off_manifold_vector, local_manifold_flat, cln_data)

        return norm_ratios, rand_ratios

    def norm_ratio_histographs(self, norm_ratios, rand_ratios, save_path):
        """
        Plots histograms of norm ratios and random ratios in the same figure
        """
        bins = np.linspace(0, 1, 200)
        plt.hist(norm_ratios, bins=bins, alpha=0.5)
        plt.hist(rand_ratios, bins=bins, alpha=0.5)
        plt.legend(['on manifold', 'random'])
        plt.xlabel('norm ratio')
        plt.ylabel('count')
        plt.title('Norm ratio histograms, on manifold vs random')

        plt.savefig(save_path)

    @staticmethod
    def classifier_predictions(classifier_model, inputs):
        logits = classifier_model(inputs)
        return F.softmax(logits)

    def second_experiment(self, **kwargs):
        def get_local_manifold_approx(x_image):
            return get_local_approximation(self.autoencoder_model.encode, self.autoencoder_model.decode, x_image[0])
        def do_restricted_attacks():
            """
            Performs restricted attacks on the images in the dataset. 
            
            In order to do attacks involving projections, use the patch_perturb function.
            We do iterative gradient attacks. We attack for 200 iterations, or until the model misclassifies the image.

            Returns a list of perturbation vector norms, and whether or not each attack succeeded in making the model
            misclassify the image.
            """

            params = {'eps': 200.0, 'eps_iter': 0.5, 'nb_iter': 50,    
                'rand_init': False, 'targeted': False, 'clip_min': -10.0, 'clip_max': 10.0}
            params.update(kwargs)
            adversary_cifar10 = L2PGDAttack(
                self.classifier_model, **params)
            
            norms, successes = [], []

            for cln_data, true_label in tqdm(self.dataloader):
                try:
                    cln_data, true_label = cln_data.to(self.device), true_label.to(self.device)
                    adv_untargeted = adversary_cifar10.perturb(cln_data, true_label)

                    perturbation_vector = adv_untargeted - cln_data
                    norm = perturbation_vector.norm().item()
                    norms.append(norm)

                    attacked_performance = self.classifier_predictions(self.classifier_model, adv_untargeted)[0, true_label].item()
                    successes.append(attacked_performance < 0.5)
                except KeyboardInterrupt:
                    break
            return norms, successes

        advertorch.attacks.iterative_projected_gradient.perturb_iterative = patch_perturb()
        norms_unrestricted, success_unrestricted = do_restricted_attacks()

        print('Unrestricted attacks done')

        advertorch.attacks.iterative_projected_gradient.perturb_iterative = patch_perturb(
            project_onto_k=True, 
            get_local_manifold_approx=get_local_manifold_approx, 
            LATENT_DIM=self.LATENT_DIM, 
            IMAGESPACE_DIM=self.IMAGESPACE_DIM)
        norms_on_manifold, success_on_manifold = do_restricted_attacks()

        print('On manifold attacks done')

        advertorch.attacks.iterative_projected_gradient.perturb_iterative = patch_perturb(
            project_onto_not_k=True, 
            get_local_manifold_approx=get_local_manifold_approx, 
            LATENT_DIM=self.LATENT_DIM, 
            IMAGESPACE_DIM=self.MAGESPACE_DIM)
        norms_off_manifold, success_off_manifold = do_restricted_attacks()
        
        print('Off manifold attacks done')

        return norms_unrestricted, norms_on_manifold, norms_off_manifold, success_unrestricted, success_on_manifold, success_off_manifold

    def visualize_perturbations(self, save_path, perturbation_vector, on_manifold_vector, off_manifold_vector, local_manifold_flat, img):
        """
        Visualizes the perturbations.

        We save the perturbations in the following order:
        1. Perturbation vector
        2. Original image
        3. On manifold vector
        4. Off manifold vector
        5-9: Samples on the manifold
        """
        orthobasis_flat = get_orthonormal_basis(local_manifold_flat)
        samples = sample_from_orthonormal_basis(orthobasis_flat, 6)

        fig = plt.figure(figsize=(25, 25))
        axs = [x for y in fig.subplots(3,3) for x in y]
        axs[0].imshow((perturbation_vector[0].permute((1, 2, 0))/(20*perturbation_vector.std()) + 0.5).cpu())
        axs[0].set_title('Perturbation vector')

        axs[1].imshow((self.denormalize(img[0]).permute((1, 2, 0))).cpu())
        axs[1].set_title('Original image')

        axs[2].imshow((on_manifold_vector[:, 0].reshape(*img[0].shape).permute((1, 2, 0))/(20*on_manifold_vector.std()) + 0.5).cpu())
        axs[2].set_title('On manifold vector')

        axs[3].imshow((off_manifold_vector[:, 0].reshape(*img[0].shape).permute((1, 2, 0))/(20*off_manifold_vector.std()) + 0.5).cpu())
        axs[3].set_title('Off manifold vector')

        for i in range(5):
            axs[i+4].imshow((samples[:, i].reshape(*img[0].shape).permute((1, 2, 0)) + 2).cpu()/4)
            axs[i+4].set_title('Sample from manifold')

        plt.savefig(save_path)
    
    def plot_perturbation_lengths(self, norms_unrestricted, norms_onmanifold, norms_offmanifold, successes_unrestricted, successes_onmanifold, successes_offmanifold, save_path):
        """
        Plots the perturbation lengths
        
        First, we plot histograms of the lengths of perturbations required to change the model's label, 
        we plot for all 3 kinds (unrestricted, on manifold, off manifold) of attacks in one plot.

        Then, we return a table of values, including the success rates for all 3 kinds of attacks, 
        as well as the ratio of perturbation lengths between on/off manifold attacks vs unrestricted attack
        in the attacks that were successful.
        """
        norms_unrestricted = np.array(norms_unrestricted)
        norms_onmanifold = np.array(norms_onmanifold)
        norms_offmanifold = np.array(norms_offmanifold)
        successes_unrestricted = np.array(successes_unrestricted)
        successes_onmanifold = np.array(successes_onmanifold)
        successes_offmanifold = np.array(successes_offmanifold)
        
        plt.hist([math.log(x) for x in norms_unrestricted[successes_unrestricted] if x > 0.1], bins=np.linspace(0, 4, 200), alpha=0.3)
        plt.hist([math.log(x) for x in norms_onmanifold[successes_onmanifold] if x > 0.1], bins=np.linspace(0, 4, 200), alpha=0.3)
        plt.hist([math.log(x) for x in norms_offmanifold[successes_offmanifold] if x > 0.1], bins=np.linspace(0, 4, 200), alpha=0.3)
        plt.legend(['Unrestricted', 'On manifold', 'Off manifold'])
        plt.title('Lengths of perturbations required to change the model\'s label')
        plt.xlabel('Log of perturbation length')
        plt.ylabel('Number of samples')

        plt.savefig(save_path)

        n_successes = (sum(successes_unrestricted), sum(successes_onmanifold), sum(successes_offmanifold))
        tot_onmanifold = sum(norms_onmanifold[successes_onmanifold])
        tot_offmanifold = sum(norms_offmanifold[successes_offmanifold])
        tot_unrestricted_success_onmanifold = sum(norms_unrestricted[successes_onmanifold])
        tot_unrestricted_success_offmanifold = sum(norms_unrestricted[successes_offmanifold])

        onmanifold_norm_ratio = tot_onmanifold / tot_unrestricted_success_onmanifold
        offmanifold_norm_ratio = tot_offmanifold / tot_unrestricted_success_offmanifold

        print('Successes on manifold:', n_successes[1])
        print('Successes off manifold:', n_successes[2])
        print('Successes unrestricted:', n_successes[0])

        print('On manifold norm ratio:', onmanifold_norm_ratio)
        print('Off manifold norm ratio:', offmanifold_norm_ratio)

        return n_successes, onmanifold_norm_ratio, offmanifold_norm_ratio


