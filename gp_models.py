import gpytorch
import torch



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



class MeanFieldDecoupledModel(gpytorch.models.ApproximateGP):
    '''A batch of 3 independent MeanFieldDecoupled PPGPR models.'''

    def __init__(self, num_inducing, num_classes, input_dim, latent_dim):
        self.batch_shape = torch.Size([num_classes])
        # self.inducing_inputs = torch.randn(1, num_inducing, latent_dim)
        self.inducing_inputs = torch.randn(num_classes, num_inducing, latent_dim)
        '''The variational parameters have a batch_shape of [3]'''
        variational_distribution = gpytorch.variational.MeanFieldVariationalDistribution(
             num_inducing, batch_shape=self.batch_shape,
         )
        variational_strategy = gpytorch.variational.BatchDecoupledVariationalStrategy(
             self, self.inducing_inputs, variational_distribution, learn_inducing_locations=True,
             mean_var_batch_dim=-1
         )

        '''The mean/covar modules have a batch_shape of [3, 2]'''
        '''where the last batch dim corresponds to the mean & variance hyperparameters'''
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_classes,1]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
             gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_classes,1])),
             batch_shape=torch.Size([num_classes,1]),
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


class MAPApproximateGP(gpytorch.models.ApproximateGP):
    '''A batch of 3 independent MeanFieldDecoupled PPGPR models.'''

    def __init__(self, num_inducing, num_classes, input_dim, latent_dim):
        self.batch_shape = torch.Size([num_classes])
        # self.inducing_inputs = torch.randn(1, num_inducing, latent_dim)
        self.inducing_inputs = torch.randn(num_classes, num_inducing, latent_dim)
        '''The variational parameters have a batch_shape of [3]'''
        variational_distribution = gpytorch.variational.DeltaVariationalDistribution(
             num_inducing, batch_shape=self.batch_shape,
         )
        variational_strategy = gpytorch.variational.VariationalStrategy(
             self, self.inducing_inputs, variational_distribution, learn_inducing_locations=True)

        '''The mean/covar modules have a batch_shape of [3, 2]'''
        '''where the last batch dim corresponds to the mean & variance hyperparameters'''
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
class OrthDecoupledApproximateGP(gpytorch.models.ApproximateGP):
    '''A batch of 3 independent MeanFieldDecoupled PPGPR models.'''

    def __init__(self, num_inducing, num_classes, input_dim, latent_dim):
        self.batch_shape = torch.Size([num_classes])
        # self.inducing_inputs = torch.randn(1, num_inducing, latent_dim)
        self.inducing_inputs = torch.randn(num_classes, num_inducing, latent_dim)

        '''The variational parameters have a batch_shape of [3]'''
        mean_inducing_points = torch.randn(num_classes,num_inducing, latent_dim, dtype=torch.float32)
        covar_inducing_points = torch.randn(100, latent_dim, dtype=torch.float32)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(covar_inducing_points.size(-2)
                                                                                    )
        covar_variational_strategy = gpytorch.variational.VariationalStrategy(self, covar_inducing_points,
                                                                        variational_distribution,
                                                                        learn_inducing_locations=True)
        variational_strategy = gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
            covar_variational_strategy, mean_inducing_points,
            gpytorch.variational.DeltaVariationalDistribution(mean_inducing_points.size(-2),batch_shape=self.batch_shape)
        )

        '''The mean/covar modules have a batch_shape of [3, 2]'''
        '''where the last batch dim corresponds to the mean & variance hyperparameters'''
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