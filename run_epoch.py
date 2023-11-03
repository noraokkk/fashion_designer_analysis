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

def run_epoch(args,model,data,optimizer=None,train=False,warmup_scheduler=None):
    if train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

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
        pred = output.argmax(-1)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output,target)
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
    return acc, loss_total


def run_epoch_test(args, model, data):
    model.eval()
    correct = 0
    loss_total = 0
    model = model.to(args.device)
    for i, (image, target) in enumerate(data):
        str = '{}/{} '.format(i, int(len(data.dataset) / args.batch_size))
        image = image.to(args.device)
        target = target.to(args.device)
        with torch.no_grad():
            output = model(image)  # This gives us 16 samples from the predictive distribution
            pred = output.argmax(-1)
            output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, target)
            str += ' loss: {:.2f}:'.format(loss.data)
            print(str)
        correct += pred.eq(target.view_as(pred)).cpu().sum()
        loss_total += loss


    acc = 100. * correct / float(len(data.dataset))
    print('Test set: Accuracy: {}/{} ({}%)'.format(correct, len(data.dataset), acc))
    return acc, loss_total



# def run_epoch(args,model,data,optimizer,train=False,warmup_scheduler=None):
#     if train:
#         model.train()
#         optimizer.zero_grad()
#     else:
#         model.eval()
#
#     # pre-allocate full prediction and target tensors
#     all_predictions = torch.zeros(len(data.dataset),args.num_classes).cpu()
#     all_targets = torch.zeros(len(data.dataset),args.num_classes).cpu()
#     batch_idx = 0
#     loss_total = 0
#     for i, (image, target) in enumerate(data):
#         # if i>10:
#         #     break
#         # if i%100==0:
#         #     print('{}/{}'.format(i,int(len(data.dataset)/args.batch_size)))
#         str = '{}/{} '.format(i,int(len(data.dataset)/args.batch_size))
#         model.to(args.device)
#         image = image.to(args.device)
#         target = target.to(args.device)
#
#         if train:
#            pred = model(image)
#         else:
#             with torch.no_grad():
#                 pred = model(image)
#
#         # loss = F.binary_cross_entropy_with_logits(pred,target,reduction='none')
#         loss = (loss.sum() / args.num_classes)
#         loss = F.nll_loss(pred, target)
#         str += ' loss: {:.2f}:'.format(loss.data)
#         print(str)
#
#         if train:
#             loss.backward()
#             # Grad Accumulation
#             if ((batch_idx+1)%args.grad_ac_steps == 0):
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 if warmup_scheduler is not None:
#                     warmup_scheduler.step()
#
#         ## Updates ##
#         start_idx,end_idx=(batch_idx*data.batch_size),((batch_idx+1)*data.batch_size)
#         if pred.size(0) != all_predictions[start_idx:end_idx].size(0):
#             pred = pred.view(target.size(0),-1)
#
#         all_predictions[start_idx:end_idx] = pred.data.cpu()
#         all_targets[start_idx:end_idx] = target.data.cpu()
#         batch_idx +=1
#
#     loss_total = loss_total/float(all_predictions.size(0))
#
#     return all_predictions,all_targets,loss_total
#
#
# def run_epoch_test(args, model, data):
#     model.eval()
#     # pre-allocate full prediction and target tensors
#     all_predictions = torch.zeros(len(data.dataset), args.num_classes).cpu()
#     all_targets = torch.zeros(len(data.dataset), args.num_classes).cpu()
#     batch_idx = 0
#     loss_total = 0
#     for i, (image, target) in enumerate(data):
#         if i > 10:
#             break
#         # if i%100==0:
#         #     print('{}/{}'.format(i,int(len(data.dataset)/args.batch_size)))
#         str = '{}/{} '.format(i, int(len(data.dataset) / args.batch_size))
#         model.to(args.device)
#         image = image.to(args.device)
#         target = target.to(args.device)
#         with torch.no_grad():
#             pred = model(image)
#
#         loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
#         loss = (loss.sum() / args.num_classes)
#         # loss = F.nll_loss(pred, target)
#         str += ' loss: {:.2f}:'.format(loss.data)
#         print(str)
#
#         ## Updates ##
#         start_idx, end_idx = (batch_idx * data.batch_size), ((batch_idx + 1) * data.batch_size)
#         if pred.size(0) != all_predictions[start_idx:end_idx].size(0):
#             pred = pred.view(target.size(0), -1)
#
#         all_predictions[start_idx:end_idx] = pred.data.cpu()
#         all_targets[start_idx:end_idx] = target.data.cpu()
#         batch_idx += 1
#
#     loss_total = loss_total / float(all_predictions.size(0))
#
#     return all_predictions, all_targets, loss_total


