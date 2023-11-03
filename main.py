import torch
import argparse
from resnet_model import resnet_model
from config_args import get_args
import util_tool.evaluate as evaluate
import util_tool.logger as logger
from optim_schedule import WarmupLinearSchedule
from run_epoch import run_epoch, run_epoch_test
import yaml
#personality
from dataset import load_dataset
import matplotlib.pyplot as plt

args = get_args(argparse.ArgumentParser())
torch.manual_seed(0)
train_loader, val_loader, test_loader = load_dataset(args)
print('Labels: {}'.format(args.num_classes))

def load_saved_model(saved_model_name,model,device):
    # checkpoint = torch.load(saved_model_name,map_location='cuda:0')
    checkpoint = torch.load(saved_model_name,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model

print(args.model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device
model = resnet_model(args.num_classes,backbone="resnet18")
model = model.to(args.device)


####test
if args.inference:
    model = load_saved_model(args.saved_model_name, model,device)
    if test_loader is not None:
        data_loader = test_loader
    else:
        data_loader = val_loader

    acc_test, loss_test = run_epoch_test(args, model, data_loader)
    print('======================== testing acc: {:.2f}%========================'.format(acc_test))
    exit(0)

if args.freeze_backbone:
    for name, p in model.backbone.named_parameters():
        print(name)
        p.requires_grad = False
if args.optim == 'adam':
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr)#, weight_decay=0.0004)
else:
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)

if args.warmup_scheduler:
    step_scheduler = None
    scheduler_warmup = WarmupLinearSchedule(optimizer, 1, 300000)
else:
    scheduler_warmup = None
    if args.scheduler_type == 'plateau':
        step_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=5)
    elif args.scheduler_type == 'step':
        step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    else:
        step_scheduler = None

metrics_logger = logger.Logger(args)
loss_logger = logger.LossLogger(args.model_name)

loss_train_all = []
loss_val_all = []
loss_test_all = []

for epoch in range(1,args.epochs+1):
    print('======================== {} ========================'.format(epoch))
    for param_group in optimizer.param_groups:
        print('LR: {}'.format(param_group['lr']))

    train_loader.dataset.epoch = epoch
    ################### Train #################
    acc_train, loss_train = run_epoch(args,model,train_loader,optimizer,train=True,warmup_scheduler=scheduler_warmup)
    loss_train_all.append(loss_train)

    ################### Valid #################
    acc_val, loss_val = run_epoch(args,model,val_loader,optimizer,train=False,warmup_scheduler=scheduler_warmup)
    loss_val_all.append(loss_val)

    ################### Test #################
    if test_loader is not None:
        acc_test, loss_test = run_epoch(args,model,test_loader,optimizer,train=False,warmup_scheduler=scheduler_warmup)
        loss_test_all.append(loss_test)
    else:
        acc_test = acc_val
    if step_scheduler is not None:
        if args.scheduler_type == 'step':
            step_scheduler.step(epoch)
        elif args.scheduler_type == 'plateau':
            step_scheduler.step(loss_val)

    ############## Log and Save ##############
    # best_valid,best_test = metrics_logger.evaluate(train_metrics,valid_metrics,test_metrics,epoch,0,model,valid_loss,test_loss,all_preds,all_targs,None,args)
    best_valid,best_test = metrics_logger.evaluate_AUC_genearl(acc_train,acc_val,acc_test,epoch,0,model,loss_val,loss_test,args)
    print(args.model_name)

f,ax = plt.subplots(1,1,tight_layout=True,figsize=(10,3))
ax.plot([i for i in range(0,args.epochs)], [loss.detach().cpu() for loss in loss_train_all])
ax.set_ylabel('training loss');ax.set_xlabel('epoch');

f,ax = plt.subplots(1,1,tight_layout=True,figsize=(10,3))
ax.plot([i for i in range(0,args.epochs)], [loss.detach().cpu() for loss in loss_val_all])
ax.set_ylabel('val loss');ax.set_xlabel('epoch');

f,ax = plt.subplots(1,1,tight_layout=True,figsize=(10,3))
ax.plot([i for i in range(0,args.epochs)], [loss.detach().cpu() for loss in loss_test_all])
ax.set_ylabel('test loss');ax.set_xlabel('epoch');


# for epoch in range(1,args.epochs+1):
#     print('======================== {} ========================'.format(epoch))
#     for param_group in optimizer.param_groups:
#         print('LR: {}'.format(param_group['lr']))
#
#     train_loader.dataset.epoch = epoch
#     ################### Train #################
#     all_predictions,all_targets,train_loss = run_epoch(args,model,train_loader,optimizer,train=True,warmup_scheduler=scheduler_warmup)
#     train_metrics = evaluate.compute_metrics_normal(all_predictions,all_targets)
#     loss_logger.log_losses('train.log',epoch,train_loss,train_metrics)
#
#     ################### Valid #################
#     all_predictions,all_targets,valid_loss = run_epoch(args,model,val_loader,None)
#     valid_metrics = evaluate.compute_metrics_normal(all_predictions,all_targets)
#     loss_logger.log_losses('valid.log',epoch,valid_loss,valid_metrics)
#
#     ################### Test #################
#     if test_loader is not None:
#         all_predictions,all_targets,test_loss = run_epoch(args,model,test_loader,None)
#         test_metrics = evaluate.compute_metrics_normal(all_predictions,all_targets)
#     else:
#         test_loss,test_metrics = valid_loss,valid_metrics
#     loss_logger.log_losses('test.log',epoch,test_loss,test_metrics)
#
#     if step_scheduler is not None:
#         if args.scheduler_type == 'step':
#             step_scheduler.step(epoch)
#         elif args.scheduler_type == 'plateau':
#             step_scheduler.step(valid_loss)
#
#     ############## Log and Save ##############
#     # best_valid,best_test = metrics_logger.evaluate(train_metrics,valid_metrics,test_metrics,epoch,0,model,valid_loss,test_loss,all_preds,all_targs,None,args)
#     best_valid,best_test = metrics_logger.evaluate_AUC(train_metrics,valid_metrics,test_metrics,epoch,0,model,valid_loss,test_loss,all_predictions,all_targets,None,args)
#
#     print(args.model_name)
