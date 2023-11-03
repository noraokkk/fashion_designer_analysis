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
from gp_layer import ResnetExtractor, DirichletGPModel
import gpytorch
import botorch
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from resnet_model import resnet_model

torch.manual_seed(0)
args = get_args(argparse.ArgumentParser())
train_loader, val_loader, test_loader = load_dataset(args)
print('Labels: {}'.format(args.num_classes))

def load_saved_model(saved_model_name,model,likelihood):
    checkpoint = torch.load(saved_model_name,map_location='cuda:0')
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    model.load_state_dict(checkpoint['state_dict'])
    return model, likelihood

print(args.model_name)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
args.device = device
class_labels = ["alexander_mcqueen", "donatella_versace", "john_galliano", "karl_lagerfeld", "yves_saint_laurent"]

resnet_extractor = resnet_model(args.num_classes,backbone="resnet18")
resnet_extractor = resnet_extractor.to(args.device)
checkpoint = torch.load(args.saved_backbone_name, map_location=device)
resnet_extractor.load_state_dict(checkpoint['state_dict'])

if args.freeze_backbone:
    for name, p in resnet_extractor.named_parameters():
        print(name)
        p.requires_grad = False

train_embeddings = []
train_classes = []
for i, (image, target) in enumerate(train_loader):
    # if i>20:
    #     break
    image = image.to(args.device)
    target = target.to(args.device)
    train_embeddings.append(resnet_extractor(image))
    train_classes.append(target)

train_embeddings = torch.cat(train_embeddings,dim=0)
train_classes = torch.cat(train_classes,dim=0)

likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(train_classes.to(torch.int64), learn_additional_noise=True)
model = DirichletGPModel(train_embeddings, likelihood.transformed_targets, likelihood, num_classes=likelihood.num_classes)
model = model.to(device)
likelihood = likelihood.to(device)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
mll.train()

optimization_trace = []
botorch.optim.fit.fit_gpytorch_mll_torch(mll, step_limit=1000, optimizer=lambda p: torch.optim.Adam(p, lr=0.01), callback=lambda _,i: optimization_trace.append(i))
# botorch.optim.fit.fit_gpytorch_mll_scipy(mll, callback=lambda _,i: optimization_trace.append(i))

f,ax = plt.subplots(1,1,tight_layout=True,figsize=(10,3))
ax.plot([r.runtime for r in optimization_trace], [r.fval for r in optimization_trace])
ax.set_ylabel('Marginal log-lik');ax.set_xlabel('Seconds');
# plt.show(f);plt.close(f)
print(optimization_trace[-1].status, optimization_trace[-1].message)


####testing
test_embeddings = []
test_classes = []
for i, (image, target) in enumerate(test_loader):
    image = image.to(args.device)
    target = target.to(args.device)
    test_embeddings.append(resnet_extractor(image))
    test_classes.append(target)
test_embeddings = torch.cat(test_embeddings,dim=0)
test_classes = torch.cat(test_classes,dim=0)

model.eval()
likelihood.eval()
with torch.no_grad():
    test_pred_dist = model(test_embeddings)

test_pred_samples = test_pred_dist.sample(torch.Size((256,))).exp()
test_probabilities = (test_pred_samples / test_pred_samples.sum(1, keepdim=True)).mean(0)

test_classes = test_classes.detach().cpu()
test_probabilities = test_probabilities.detach().cpu()

sklearn.metrics.ConfusionMatrixDisplay(
    sklearn.metrics.confusion_matrix(test_classes, (torch.argmax(test_probabilities,dim=0,keepdim=False))),
    display_labels=class_labels
).plot();

print(test_probabilities[:,1])

torch.softmax(test_probabilities,dim=0).t()

print('ROC-AUC',np.round(sklearn.metrics.roc_auc_score(
    test_classes,
    torch.softmax(test_probabilities,dim=0).t(),multi_class='ovr'
),2))

print('Acc',np.round(sklearn.metrics.accuracy_score(
    test_classes,
    (torch.argmax(test_probabilities,dim=0,keepdim=False))
)*100,2),'%')
