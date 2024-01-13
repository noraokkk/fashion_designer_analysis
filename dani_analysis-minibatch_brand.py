# CUDA_LAUNCH_BLOCKING=1
# TORCH_USE_CUDA_DSA=1
from resnet_model import resnet_model
from torch.utils.data import DataLoader, TensorDataset
import os
import torchvision.transforms as transforms
import numpy as np
import sklearn.metrics
import pandas as pd
import sklearn.model_selection
import torch
import torchvision
import gpytorch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import PIL
import base64
from functools import cache
import wandb
import socket
import warnings

tensor_to_image = torchvision.transforms.ToPILImage()
pil_to_tensor = torchvision.transforms.PILToTensor()


@cache
def tensor_to_url(tensor, size=128):
    return fr"data:image/png;base64,{base64.b64encode(PIL.ImageOps.contain(tensor_to_image(tensor), (size, size))._repr_png_()).decode('ascii')}"


# Make all photos square
def pad_image(img):
    h, w = img.shape[1:]
    if h != w:
        new_w = max(h, w)
        pad_h, rem_h = divmod(new_w - h, 2)
        pad_w, rem_w = divmod(new_w - w, 2)
        padding = [pad_w, pad_h, pad_w + rem_w, pad_h + rem_h]
        return torchvision.transforms.functional.pad(img, padding, padding_mode='edge')
    return img


class DirichletGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_inducing, num_classes, input_dim, latent_dim):
        self.batch_shape = torch.Size([num_classes])
        # self.inducing_inputs = torch.randn(1, num_inducing, latent_dim)
        self.inducing_inputs = torch.randn(num_classes, num_inducing, latent_dim)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing,
                                                                                        batch_shape=self.batch_shape)
        variational_strategy = gpytorch.variational.VariationalStrategy(self, self.inducing_inputs,
                                                                        variational_distribution,
                                                                        learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=self.batch_shape)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=self.batch_shape),
            batch_shape=self.batch_shape,
        )
        self.scaler = gpytorch.utils.grid.ScaleToBounds(-1, 1)
        self.fc = torch.nn.Linear(input_dim, latent_dim)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def transform(self, x):
        x = self.fc(x)
        x = self.scaler(x)
        return x

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, inputs, prior: bool = False, **kwargs):
        if inputs is not None and inputs.dim() == 1:
            inputs = inputs.unsqueeze(-1)
        if inputs is not None:
            inputs = self.transform(inputs)
        return self.variational_strategy(inputs, prior=prior, **kwargs)

    def embedding_posterior(self, z):
        '''Compute the posterior over z = self.transform(x)'''
        return self.variational_strategy(z, prior=False)


class Designer(torch.utils.data.Dataset):
    def __init__(self,
                 num_classes,
                 is_training=False,
                 data_augument=False,
                 image_transform=None,
                 data=None,
                 root=""
                 ):
        self.num_class = num_classes
        self.is_training = is_training
        self.data_augument = data_augument
        self.image_transform = image_transform
        self.data = data
        self.root = root

    def __getitem__(self, index):
        img_pth = self.data[0][index]
        label = self.data[1][index]
        img_pth = os.path.join(self.root, img_pth)
        # print(img_pth)
        image = self.pad_image(torchvision.io.read_image(img_pth, mode=torchvision.io.ImageReadMode.RGB))
        image = self.image_transform(image)
        # label = data["season_label"]
        # label_vec = np.zeros(self.num_class)
        # label_vec[label]=1
        # return image,label_vec
        return image, label, index

    def __len__(self):
        return len(self.data[0])

    # Make all photos square
    def pad_image(self, img):
        h, w = img.shape[1:]
        if h != w:
            new_w = max(h, w)
            pad_h, rem_h = divmod(new_w - h, 2)
            pad_w, rem_w = divmod(new_w - w, 2)
            padding = [pad_w, pad_h, pad_w + rem_w, pad_h + rem_h]
            return torchvision.transforms.functional.pad(img, padding, padding_mode='edge')
        return img

class MinibatchedDirichletClassificationLikelihood(gpytorch.likelihoods.DirichletClassificationLikelihood):
    def _shaped_noise_covar(self, base_shape, *params, selected_noise=None, **kwargs):
        assert selected_noise is not None
        if len(params) > 0:
            # we can infer the shape from the params
            shape = None
        else:
            # here shape[:-1] is the batch shape requested, and shape[-1] is `n`, the number of points
            shape = base_shape

        res = self.noise_covar(*params, shape=shape, noise=selected_noise, **kwargs)

        if self.second_noise_covar is not None:
            res = res + self.second_noise_covar(*params, shape=shape, **kwargs)
        elif isinstance(res, gpytorch.linear_operator.ZeroLinearOperator):
            warnings.warn(
                "You have passed data through a FixedNoiseGaussianLikelihood that did not match the size "
                "of the fixed noise, *and* you did not specify noise. This is treated as a no-op.",
                gpytorch.utils.warnings.GPInputWarning,
            )

        return res

if __name__ == '__main__':
    #
    # wandb.init(config={
    #     "dataset": "fashion_brand",
    # },
    #     project="fashion_gp_project",
    #     entity="fashion_gp",
    #     notes=socket.gethostname(),
    #     dir="E://coding//wandb",
    #     job_type="training",
    #     reinit=True)

    # Load dataset
    torch.manual_seed(0)
    img_size = 512
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    img_root = Path('F://datasets//fashion_brands//')

    train_metadata = json.loads((img_root / 'train.json').read_text())
    test_metadata = json.loads((img_root / 'test.json').read_text())
    val_metadata = json.loads((img_root / 'val.json').read_text())

    # class_labels = ["alexander_mcqueen","donatella_versace","karl_lagerfeld","yves_saint_laurent"]
    # class_labels = ["alexander_mcqueen","donatella_versace","john_galliano","karl_lagerfeld","yves_saint_laurent"]

    n_train = len(train_metadata)
    n_test = len(test_metadata)
    n_val = len(val_metadata)
    n_all = n_train + n_test + n_val

    all_classes = torch.empty(n_all, dtype=torch.int)
    all_images_path = [None] * n_all
    # all_images = [None]*n_all
    for i, meta in enumerate((*train_metadata, *test_metadata, *val_metadata)):
        # print(i)
        all_classes[i] = meta['label']
        all_images_path[i] = str(img_root / meta['file_path'])

        # images
        # all_images[i] = torchvision.io.read_image(str(img_root / meta['file_path']),mode=torchvision.io.ImageReadMode.RGB)
        # all_images[i] = pad_image(torchvision.transforms.functional.resize(all_images[i], img_size, antialias=True))

    n_classes = all_classes.max() + 1
    print(n_classes)
    # latent_dim_res = n_classes
    latent_dim_res = 512  # res18

    resnet_extractor = resnet_model(n_classes, backbone="resnet18").to(device)
    checkpoint = torch.load('results/fashion_brands_101_res18.3layer.bsz_128sz_224.sgd0.002//best_model.pt',
                            map_location=device)
    resnet_extractor.load_state_dict(checkpoint['state_dict'])

    for p in resnet_extractor.parameters():
        p.requires_grad = False

    # Define resnet feature extractor
    resnet_input_transform = torchvision.models.ResNet18_Weights.DEFAULT.transforms()
    crop_size = resnet_input_transform.crop_size[0]

    ##################################k-fold#############################################
    # # kfold = sklearn.model_selection.ShuffleSplit(n_splits=1, random_state=19960111)
    # kfold = sklearn.model_selection.ShuffleSplit(n_splits=5, test_size=0.2, train_size=0.7,
    #                                              random_state=19960111)  # modm
    # train_idx, test_idx = next(kfold.split(np.empty(len(all_classes)), all_classes))
    #
    # train_classes = all_classes[train_idx]
    # train_images_pth = [all_images_path[i] for i in train_idx]
    # train_data = [train_images_pth, train_classes]
    #
    # test_classes = all_classes[test_idx]
    # test_images_pth = [all_images_path[i] for i in test_idx]
    # test_data = [test_images_pth, test_classes]
    ##################################k-fold#############################################

    ##################################fixed train val test#############################################
    train_classes = []
    train_images_pth = []
    for i, meta in enumerate(train_metadata):
        # print(i)
        train_classes.append(meta['label'])
        train_images_pth.append(str(img_root / meta['file_path']))
    train_classes = torch.tensor(train_classes)
    train_data = [train_images_pth, train_classes]

    test_classes = []
    test_images_pth = []
    for i, meta in enumerate(test_metadata):
        # print(i)
        test_classes.append(meta['label'])
        test_images_pth.append(str(img_root / meta['file_path']))
    test_classes = torch.tensor(test_classes)
    test_data = [test_images_pth, test_classes]

    ##################################fixed train val test#############################################


    print("training num:{}".format(len(train_images_pth)))
    print("testing num:{}".format(len(test_images_pth)))

    normTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),  # for the pad image function: tensor -> Image
            transforms.Resize((crop_size, crop_size)),
            transforms.ToTensor(),
            normTransform
        ]
    )

    # load dataset
    batch_size = 16
    train_dataset = Designer(num_classes=len(all_classes), image_transform=transform, is_training=True,
                             data_augument=False, data=train_data, root=img_root)
    test_dataset = Designer(num_classes=len(all_classes), image_transform=transform, is_training=False,
                            data_augument=False, data=test_data, root=img_root)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)



    # # resnet backbone testing
    # test_pred_classes = torch.empty(len(test_classes), dtype=torch.int64)
    # test_pred_probabilities = torch.empty(n_classes, len(test_classes), dtype=torch.float32)
    # with torch.no_grad():
    #     for i, (x_batch, y_batch, batch_index) in enumerate(test_loader):
    #         j = (i * test_loader.batch_size)
    #         x_batch = x_batch.to(device)
    #         y_batch = y_batch.to(device)
    #         test_probabilities = resnet_extractor(x_batch)
    #         test_probabilities = torch.nn.functional.softmax(test_probabilities, dim=1)
    #         test_pred_classes[j:j + len(x_batch)] = torch.argmax(test_probabilities, dim=1, keepdim=False)
    #         test_pred_probabilities[:, j:j + len(x_batch)] = test_probabilities.T
    #
    # acc = sklearn.metrics.accuracy_score(test_classes.detach().cpu(), test_pred_classes.detach().cpu())
    # auc = sklearn.metrics.roc_auc_score(test_classes.detach().cpu(), test_pred_probabilities.detach().cpu().T,
    #                                     multi_class="ovo")
    # metrics = pd.Series(index=['acc', 'auc'])
    # metrics.at['acc'] = acc
    # metrics.at['auc'] = auc
    # print("testing on resnet backbone:")
    # print(metrics)

    #key parameters
    # num_inducing = 600
    # latent_dim = 15
    num_epochs = 50  # 100

    param_idx = 1
    for num_inducing in [300,900,1500,1800,2100]: #[300, 900, 1200]:
        # for latent_dim in [20, 15, 10, 5]:
        for latent_dim in [15]:
            para_str = "inducing_"+str(num_inducing)+"dim_"+str(latent_dim)
            wandb.init(
                project="fashion_gp_project",
            )

            # #define our custom x axis metric
            wandb.define_metric("test/step")
            #set all other test/ metrics to use this step
            wandb.define_metric("test/*", step_metric="test/step")

            #define likelihood
            # likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(train_classes.long(), learn_additional_noise=True) #original
            likelihood = MinibatchedDirichletClassificationLikelihood(train_classes.long(), learn_additional_noise=True)
            # likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(train_classes.long(), learn_additional_noise=True)
            train_transformed_targets = likelihood.transformed_targets
            # train_transformed_targets = likelihood.transformed_targets.t()

            #define model
            model = DirichletGPModel(
                num_inducing=num_inducing,
                input_dim=latent_dim_res,
                num_classes=likelihood.num_classes,
                latent_dim=latent_dim
            )

            likelihood = likelihood.to(device)
            model = model.to(device)

            # Our loss object. We're using the VariationalELBO
            mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_classes.size(0))
            mll.to(device)
            mll.train()

            # model training
            model.train()
            initial_lengthscales = model.covar_module.base_kernel.lengthscale.numpy(force=True)
            optimization_trace = []
            optimizer = torch.optim.Adam(mll.parameters(), lr=0.01)
            step = 1
            for epoch in range(num_epochs):
                print("epoch: {}/{}".format(epoch, num_epochs))
                for j, (x_batch, y_batch, batch_index) in enumerate(train_loader):
                    # for x_batch, y_batch in train_loader:
                    # if j>100:
                    #     break
                    if j % 50 == 0:
                        print("{}/{}".format(j, str(int(len(train_loader.dataset) / batch_size) + 1)))
                    optimizer.zero_grad()
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    y_batch_tf = train_transformed_targets[:, batch_index]
                    y_batch_tf = y_batch_tf.to(device)
                    noise_batch = likelihood.noise_covar.noise[:, batch_index]
                    noise_batch = noise_batch.to(device)
                    # output = model(resnet_extractor(x_batch))
                    output = model(resnet_extractor.backbone(x_batch))
                    # loss = -mll(output, y_batch.t()).sum(dim=0)
                    # loss = -mll(output, y_batch.t()).mean(dim=0)
                    full_loss = -mll(output, y_batch_tf, selected_noise=noise_batch)
                    with torch.no_grad():
                        wandb.log({
                            'loss'+'_'+para_str: full_loss.mean(dim=0).item(),
                            'full_loss'+'_'+para_str: wandb.Histogram(full_loss.numpy(force=True)),
                            'hyper/lengthscale'+'_'+para_str: wandb.Histogram(
                                model.covar_module.base_kernel.lengthscale.view(-1).numpy(force=True)),
                            'hyper/outputscale'+'_'+para_str: wandb.Histogram(model.covar_module.outputscale.numpy(force=True)),
                            'hyper/second_noise'+'_'+para_str: wandb.Histogram(likelihood.second_noise.view(-1).numpy(force=True)),
                            'hyper/fc_eighs'+'_'+para_str: torch.linalg.eigvalsh(model.fc.weight.T @ model.fc.weight).sum().item(),
                        }, step=step)
                    loss = full_loss.mean(dim=0)
                    optimization_trace.append(full_loss.mean(dim=0).item())
                    #optimization_trace.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    # wandb.log({
                    #     'training_curve': loss.item(),
                    #     'lengthscale': model.covar_module.base_kernel.lengthscale.numpy(force=True).reshape(-1),
                    #     'variance': likelihood.noise.numpy(force=True).reshape(-1),
                    # }, step=step)
                    step += 1

            # wandb.finish()
            # f, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(10, 2 * 1), sharex=True)
            # ax.plot(optimization_trace)
            # ax.set_ylabel('Marginal log-lik')
            # ax.set_xlabel('Steps')
            # plt.show(f)
            # plt.close(f)

            ###
            # f, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(10, 2 * 1), sharex=True)
            # ax.plot(optimization_trace)
            # ax.set_ylabel('Marginal log-lik')
            # ax.set_xlabel('Steps')
            # plt.show(f)
            # plt.close(f)

                # test on each epoch
                metrics = pd.Series(index=['acc', 'auc'])
                n_plots = 1
                n_cols = 5
                n_rows = int(np.ceil(n_plots / n_cols))

                model.eval()
                test_pred_classes = torch.empty(len(test_classes), dtype=torch.int64)
                test_pred_probabilities = torch.empty(n_classes, len(test_classes), dtype=torch.float32)
                with torch.no_grad():
                    for i, (x_batch, y_batch, batch_index) in enumerate(test_loader):
                        j = (i * test_loader.batch_size)
                        x_batch = x_batch.to(device)
                        y_batch = y_batch.to(device)
                        test_pred_dist = model(resnet_extractor.backbone(x_batch))

                        test_pred_samples = test_pred_dist.sample(torch.Size((256,))).exp()
                        test_probabilities = (test_pred_samples / test_pred_samples.sum(1, keepdim=True)).mean(0)
                        test_pred_classes[j:j + len(x_batch)] = torch.argmax(test_probabilities, dim=0, keepdim=False)
                        test_pred_probabilities[:, j:j + len(x_batch)] = test_probabilities

                acc = sklearn.metrics.accuracy_score(test_classes.detach().cpu(), test_pred_classes.detach().cpu())
                auc = sklearn.metrics.roc_auc_score(test_classes.detach().cpu(), test_pred_probabilities.detach().cpu().T,
                                                    multi_class="ovo")
                metrics = pd.Series(index=['acc', 'auc'])
                metrics.at['acc'] = acc
                metrics.at['auc'] = auc
                print(metrics)
                with torch.no_grad():
                    wandb.log({
                        'test/acc'+'_' + para_str: acc,
                        'test/auc'+'_' + para_str: auc
                    })
            wandb.finish()





