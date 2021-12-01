from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

class Cifar10Dataset(Dataset):

    def __init__(self, split, kwargs = {}):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if split == 'train':
            self.orig_set = torchvision.datasets.CIFAR10(root='../data/cifar10/', train=True,
                                        download=True, transform=transform)

        print(self.orig_set[0])
    

if __name__ == "__main__":
    Cifar10Dataset('train')