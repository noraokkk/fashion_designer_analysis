import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


def accuracy(pos_samples, neg_samples):
    """ pos_samples: Distance between positive pair
        neg_samples: Distance between negative pair
    """
    is_cuda = pos_samples.is_cuda
    margin = 0
    pred = (pos_samples - neg_samples - margin).cpu().data
    acc = (pred > 0).sum() * 1.0 / pos_samples.size()[0]
    acc = torch.from_numpy(np.array([acc], np.float32))
    if is_cuda:
        acc = acc.cuda()
    return Variable(acc)


class resnet_model(nn.Module):
    def __init__(self, num_labels, backbone, remove_last_layer=True):
        super(resnet_model, self).__init__()
        # ResNet backbone
        self.fc = nn.Linear(1000, num_labels)
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
            if remove_last_layer:
                self.backbone.fc = torch.nn.Identity()
                self.fc = nn.Linear(512, num_labels)
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
            if remove_last_layer:
                self.backbone.fc = torch.nn.Identity()
                self.fc = nn.Linear(2048, num_labels)

    def forward(self, images):  # uidx is the user idx
        features = self.backbone(images)
        output = self.fc(features)
        # output = nn.functional.softmax(features, dim=1)
        return output


class resnet(nn.Module):
    def __init__(self, num_labels):
        super(resnet, self).__init__()
        hidden = 512  # this should match the backbone output feature size #resnet18
        # ResNet backbone
        self.backbone = models.resnet18(pretrained=True)
        self.fc = nn.Linear(1000, num_labels)

    def forward(self, images):  # uidx is the user idx
        features = self.backbone(images)
        # features = self.fc(features)
        # output = nn.functional.softmax(features, dim=1)
        return features
