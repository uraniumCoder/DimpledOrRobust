import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from IPython import embed
from collections import OrderedDict
import torch
model_urls = {
    'cifar10': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar10-d875770b.pth',
    'cifar100': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar100-3a55a987.pth',
}

"""
Defining the model architechture we trained on cifar10
"""

class CIFAR(nn.Module):
    def __init__(self, features, n_channel, num_classes, dropout):
        super(CIFAR, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_channel, num_classes)
        )
        #print(self.features)
        #print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False, dropout=0.0):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                if i == len(cfg)-1:
                    layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
                else:
                    layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU(), nn.Dropout(dropout)]
            else:
                layers += [conv2d, nn.ReLU(), nn.Dropout(dropout)]
            in_channels = out_channels
    return nn.Sequential(*layers)

def cifar10model(n_channel, pretrained=None, dropout=0.0):
    cfg = [n_channel, n_channel, 'M', 2*n_channel, 2*n_channel, 'M', 4*n_channel, 4*n_channel, 'M', (8*n_channel, 0), 'M']
    layers = make_layers(cfg, batch_norm=True, dropout=dropout/20)
    model = CIFAR(layers, n_channel=8*n_channel, num_classes=10, dropout=dropout)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['cifar10'])
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model

#if __name__ == '__main__':
#    model = cifar10(128, pretrained='log/cifar10/best-135.pth')
#    embed()