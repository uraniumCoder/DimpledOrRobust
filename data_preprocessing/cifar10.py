from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

import numpy as np
import torch

class Cifar10DatasetFiltered(Dataset):
    """
    Dataset for Cifar 10 images, pre-processed (class filtered) and augmented
    """

    DEFAULT_CLASSES = ['plane', 'car', 'ship', 'truck']
    CLASS_DICT = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
                    'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

    def __init__(self, data_root, split, kwargs = {}):
        # Downloading/Louding CIFAR10 data
        super().__init__()
        selected_classes = self.DEFAULT_CLASSES

        if split == 'train':
            self.data_transform = DatasetMaker.TRANSFORM_WITH_AUG
            raw_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True)
        
        else:
            self.data_transform = DatasetMaker.TRANSFORM_NO_AUG
            raw_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True)

        X, y = raw_set.data, raw_set.targets
        self.processed_set = DatasetMaker([
            self.get_class_i(X, y, self.CLASS_DICT[class_]) for class_ in selected_classes
        ], self.data_transform)

    @classmethod
    def denormalize(img):
        """
        Denormalizes a tensor image
        """
        return DatasetMaker.INV_NRM(img)

    # Define a function to separate CIFAR classes by class index
    @staticmethod
    def get_class_i(x, y, i):
        """
        x: trainset.train_data or testset.test_data
        y: trainset.train_labels or testset.test_labels
        i: class label, a number between 0 to 9
        return: x_i
        """
        # Convert to a numpy array
        y = np.array(y)
        # Locate position of labels that equal to i
        pos_i = np.argwhere(y == i)
        # Convert the result into a 1-D list
        pos_i = list(pos_i[:, 0])
        # Collect all data that match the desired label
        x_i = [x[j] for j in pos_i]

        return x_i

    def __getitem__(self, idx):
        return self.processed_set.__getitem__(idx)
    
    def __len__(self):
        return self.processed_set.__len__()

class DatasetMaker(Dataset):

    # Transformations
    RHF = transforms.RandomHorizontalFlip()
    IMG_MEANS = torch.tensor((0.4914, 0.4822, 0.4465))
    IMG_STDS = torch.tensor((0.2470, 0.2435, 0.2616))
    NRM = transforms.Normalize(IMG_MEANS, IMG_STDS)
    INV_NRM = transforms.Normalize(-IMG_MEANS/IMG_STDS, 1/IMG_STDS)
    TT = transforms.ToTensor()
    RC = transforms.RandomCrop(32)
    P = transforms.Pad(4)
    TPIL = transforms.ToPILImage()

    # Transforms object for trainset with augmentations
    TRANSFORM_WITH_AUG = transforms.Compose([TPIL, P, RC, RHF, TT, NRM])
    # Transforms object for testset with NO augmentation
    TRANSFORM_NO_AUG = transforms.Compose([TT, NRM])

    def __init__(self, datasets, transformFunc=TRANSFORM_NO_AUG):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class