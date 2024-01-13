import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from transformers import ViTModel


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
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            if remove_last_layer:
                self.backbone.fc = torch.nn.Identity()
                self.fc = nn.Linear(512, num_labels)
        elif backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            if remove_last_layer:
                self.backbone.fc = torch.nn.Identity()
                self.fc = nn.Linear(2048, num_labels)
        elif backbone == "resnet101":
            self.backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            if remove_last_layer:
                self.backbone.fc = torch.nn.Identity()
                self.fc = nn.Linear(2048, num_labels)
        else:
            print('invalid backbone of resnet!')
            exit(0)

    def forward(self, images):  # uidx is the user idx
        features = self.backbone(images)
        output = self.fc(features)
        # output = nn.functional.softmax(features, dim=1)
        return output

class Vit(nn.Module):
    def __init__(self, num_labels=10, backbone=None):
        super(Vit, self).__init__()
        # self.fc = nn.Linear()
        if backbone is not None:
            self.model = ViTModel.from_pretrained(backbone)
        else:
            self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.num_labels = num_labels
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

    def forward(self,images):
        output = self.model(images)
        return self.classifier(output.last_hidden_state[:, 0])



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
