import argparse,math,time,warnings,copy, numpy as np, os.path as path 
import torch, torch.nn as nn, torch.nn.functional as F
from pdb import set_trace as stop
from tqdm import tqdm
from utils import custom_replace
import random
from PIL import Image
import cv2
from torchvision import transforms
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt

unnormalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

# unnormalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_pil = transforms.ToPILImage()

def run_epoch(args,model,data,optimizer=None,likelihood=None,mll=None,train=False,warmup_scheduler=None):
    if train:
        model.train()
        likelihood.train()
        optimizer.zero_grad()
    else:
        model.eval()
        likelihood.eval()
    correct = 0
    batch_idx = 0
    loss_total = 0
    model = model.to(args.device)

    for i, (image, target) in enumerate(data):
        str = '{}/{} '.format(i,int(len(data.dataset)/args.batch_size))
        image = image.to(args.device)
        target = target.to(args.device)

        if train:
           output = model(image)
        else:
            with torch.no_grad():
                output = model(image)
        predition = likelihood(output)  # This gives us 16 samples from the predictive distribution
        pred = predition.probs.mean(0).argmax(-1)
        loss = -mll(output, target)
        str += ' loss: {:.2f}:'.format(loss.data)
        print(str)

        if train:
            loss.backward()
            # Grad Accumulation
            if ((batch_idx+1)%args.grad_ac_steps == 0):
                optimizer.step()
                optimizer.zero_grad()
                if warmup_scheduler is not None:
                    warmup_scheduler.step()

        correct += pred.eq(target.view_as(pred)).cpu().sum()
        batch_idx += 1
        loss_total += loss

    acc = 100. * correct / float(len(data.dataset))
    # print('Test set: Accuracy: {}/{} ({}%)'.format(correct, len(data.dataset), acc))
    return acc, loss_total



def run_epoch_test(args, model, likelihood, mll, data):
    model.eval()
    likelihood.eval()
    correct = 0
    loss_total = 0
    model = model.to(args.device)
    for i, (image, target) in enumerate(data):
        str = '{}/{} '.format(i, int(len(data.dataset) / args.batch_size))
        image = image.to(args.device)
        target = target.to(args.device)
        with torch.no_grad():
            output = model(image)  # This gives us 16 samples from the predictive distribution
            features = gp_layer_before(image, model)
            test_x, test_y, test_x_mat, test_y_mat = gp_layer_feeding(features,model)

            predition = likelihood(output)  # This gives us 10 samples from the predictive distribution
            pred = predition.probs.mean(0).argmax(-1)  # Taking the mean over all of the sample we've drawn
            loss = -mll(output, target)
            str += ' loss: {:.2f}:'.format(loss.data)
            print(str)
        correct += pred.eq(target.view_as(pred)).cpu().sum()
        loss_total += loss


    acc = 100. * correct / float(len(data.dataset))
    print('Test set: Accuracy: {}/{} ({}%)'.format(correct, len(data.dataset), acc))
    plot_class(test_x, test_y, test_x_mat, target,2)
    return acc, loss_total

def gp_layer_before(x,model):
    features = model.feature_extractor(x)
    features = model.fc(features)
    features = model.scale_to_bounds(features)
    # This next line makes it so that we learn a GP for each feature
    return features

def gp_layer_feeding(x,model):
    test_x = torch.as_tensor(np.stack(np.meshgrid(np.linspace(-1,1),np.linspace(-1,1),), axis=-1))
    test_y = model.gp_layer(test_x.to(x).view(-1,2).transpose(-1, -2).unsqueeze(-1))
    return test_x, test_y, x, None

def plot_class(grid_x, grid_y, test_x, test_y, num_class):
    N = grid_x.shape[0]
    fig, ax = plt.subplots(1, num_class, figsize = (15, 5))
    pred_samples = grid_y.sample(torch.Size((256,))).exp()
    probabilities = (pred_samples / pred_samples.sum(-1, keepdim=True)).mean(0)

    # levels = np.linspace(0, 1.05, 20)
    for i in range(num_class):
        im = ax[i].contourf(
            grid_x.numpy()[...,0], grid_x.numpy()[...,1],
            probabilities[:,i].detach().cpu().numpy().reshape((N,N))
        )
        ax[i].scatter(
            test_x[:, 0].numpy(force=True), test_x[:, 1].numpy(force=True),
            c=test_y.numpy(force=True), cmap='Blues',
            marker='X', edgecolor='k'
        )
        fig.colorbar(im, ax=ax[i])
        ax[i].set_title("Probabilities: Class " + str(i), fontsize = 20)


def plot_bound(test_x, test_y, test_x_mat, test_y_mat, num_class, target):
    test_x_mat, test_y_mat = np.meshgrid(x.detach()[:,0].cpu().numpy(), x.detach()[:,1].cpu().numpy())
    test_x_mat, test_y_mat = torch.Tensor(test_x_mat), torch.Tensor(test_y_mat)
    pred_means = test_y.loc
    N = test_x_mat.shape[0]
    fig, ax = plt.subplots(1, num_class, figsize = (15, 5))
    pred_samples = test_y.sample(torch.Size((256,))).exp()
    probabilities = (pred_samples / pred_samples.sum(-2, keepdim=True)).mean(0)

    # levels = np.linspace(0, 1.05, 20)
    for i in range(num_class):
        im = ax[i].contourf(
            test_x_mat.detach().cpu().numpy(), test_y_mat.detach().cpu().numpy(),probabilities.detach()[:,i].cpu().numpy().reshape((N,N))
        )
        fig.colorbar(im, ax=ax[i])
        ax[i].set_title("Probabilities: Class " + str(i), fontsize = 20)

    fig, ax = plt.subplots(1,2, figsize=(10, 5))

    ax[0].contourf(test_x_mat.numpy(), test_y_mat.numpy(), test_labels.numpy())
    ax[0].set_title('True Response', fontsize=20)

    ax[1].contourf(test_x_mat.numpy(), test_y_mat.numpy(), pred_means.max(0)[1].reshape((20,20)))
    ax[1].set_title('Estimated Response', fontsize=20)