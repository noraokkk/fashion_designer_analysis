import torch
from scipy.sparse import csc_matrix
import pickle
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms
import json
import h5py
import os
from PIL import Image
from dataloaders.data_utils import get_unk_mask_indices

def load_dataset(args):
    normTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            normTransform
        ]
    )

    train_dataset = Designer(args=args, image_transform=transform, is_training=True, data_augument=False,mode='train',root=args.data_dir)
    val_dataset = Designer(args=args, image_transform=transform, mode='val',root=args.data_dir)
    test_dataset = Designer(args=args, image_transform=transform,mode='test',root=args.data_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    return train_loader, val_loader, test_loader #resnet101


class Designer(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 is_training=False,
                 data_augument=False,
                 image_transform=None,
                 mode='',
                 root='.'
                 ):
        if root is None or root=='':
            root='.'
        self.root = root
        self.args = args
        self.num_class = args.num_classes
        self.is_training = is_training
        self.data_augument = data_augument
        self.image_transform = image_transform
        self.data = []
        json_file = root+"/"+mode+".json"
        with open(json_file, 'r') as json_file:
            self.data = json.load(json_file)


    def __getitem__(self, index):
        data = self.data[index]
        img_pth = os.path.join(self.root,data["file_path"])
        image = self.image_transform(Image.open(img_pth).convert('RGB'))
        # label = data["season_label"]
        label = data["label"]
        label_vec = np.zeros(self.num_class)
        label_vec[label]=1
        # return image,label_vec
        return image, label

    def __len__(self):
        return len(self.data)