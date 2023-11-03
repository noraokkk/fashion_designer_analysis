import torch
import argparse
from resnet_model import resnet
from config_args_gp import get_args
import util_tool.evaluate as evaluate
import util_tool.logger as logger
from optim_schedule import WarmupLinearSchedule
from run_epoch_gp import run_epoch, run_epoch_test
import yaml
#personality
from dataset import load_dataset
from gp_layer import DirichletClassificationLikelihood
from gp_layer import DKLModel, DKLModel_PCA
import gpytorch

args = get_args(argparse.ArgumentParser())
torch.manual_seed(0)
with open(args.config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if args.dataset_path != 'from_yaml':
    config['dataset']['dataset_path'] = args.dataset_path
if args.bb_path != 'from_yaml':
    config['models']['encoder_params']['pretrained_path'] = args.bb_path
if args.bb_path != 'from_yaml':
    config['models']['gnn_params']['pretrained_path'] = args.gnn_path
if args.net_type != 'from_yaml':
    config['models']['encoder_params']['net_type'] = args.net_type
if args.is_apex != 'from_yaml':
    config['train_params']['is_apex'] = args.is_apex

args.config = config
criterion = torch.nn.MarginRankingLoss(margin = args.margin)
train_loader, val_loader, test_loader = load_dataset(args)
print('Labels: {}'.format(args.num_classes))


def load_saved_model(saved_model_name,model,likelihood):
    checkpoint = torch.load(saved_model_name,map_location='cuda:0')
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    model.load_state_dict(checkpoint['state_dict'])
    return model, likelihood

print(args.model_name)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.device = device
backbone = resnet(args.num_classes,criterion,device=device)

gp_latent_dim = 2
model = DKLModel(backbone, num_dim=gp_latent_dim)

# gp_latent_dim = 10
# model = DKLModel_PCA(backbone, num_dim=gp_latent_dim)
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=gp_latent_dim, num_classes=args.num_classes)
model = model.to(args.device)
likelihood = likelihood.to(args.device)
mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))

for name, p in model.named_parameters():
    print(name)

####test
if args.inference:
    model,likelihood = load_saved_model(args.saved_model_name,model,likelihood)
    if test_loader is not None:
        data_loader = test_loader
    else:
        data_loader = val_loader
    
    acc_test, loss_test = run_epoch_test(args,model,likelihood,mll,data_loader)
    print('======================== testing acc: {:.2f}%========================'.format(acc_test))
    exit(0)

if args.freeze_backbone:
    for name, p in model.named_parameters():
        print(name)
    for name, p in model.feature_extractor.named_parameters():
        # print(name)
        p.requires_grad = False
    # for p in model.backbone.parameters():
    #     print(p.name)
    #     p.requires_grad=False
    # for p in model.backbone.base_network.layer4.parameters():
    #     p.requires_grad=True

if args.optim == 'adam':
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr)#, weight_decay=0.0004)
else:
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.SGD([
        {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4},
        {'params': model.gp_layer.hyperparameters(), 'lr': args.lr * 0.01},
        # {'params': model.gp_layer.hyperparameters()},
        {'params': model.gp_layer.variational_parameters()},
        {'params': likelihood.parameters()},
    ], lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0)


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

for epoch in range(1,args.epochs+1):
    print('======================== {} ========================'.format(epoch))
    for param_group in optimizer.param_groups:
        print('LR: {}'.format(param_group['lr']))

    train_loader.dataset.epoch = epoch
    ################### Train #################
    acc_train, loss_train = run_epoch(args,model,train_loader,optimizer,likelihood,mll,train=True,warmup_scheduler=scheduler_warmup)

    ################### Valid #################
    acc_val, loss_val = run_epoch(args,model,val_loader,optimizer,likelihood,mll,train=False,warmup_scheduler=scheduler_warmup)

    ################### Test #################
    if test_loader is not None:
        acc_test, loss_test = run_epoch(args,model,test_loader,optimizer,likelihood,mll,train=False,warmup_scheduler=scheduler_warmup)
    else:
        acc_test = acc_val
    if step_scheduler is not None:
        if args.scheduler_type == 'step':
            step_scheduler.step(epoch)
        elif args.scheduler_type == 'plateau':
            step_scheduler.step(loss_val)

    ############## Log and Save ##############
    # best_valid,best_test = metrics_logger.evaluate(train_metrics,valid_metrics,test_metrics,epoch,0,model,valid_loss,test_loss,all_preds,all_targs,None,args)
    best_valid,best_test = metrics_logger.evaluate_AUC_gp(acc_train,acc_val,acc_test,epoch,0,model,likelihood,loss_val,loss_test,args)
    print(args.model_name)
