from typing import Optional, Union

import gpytorch
import torch
from gpytorch.constraints import Interval, Positive
from linear_operator.operators import LinearOperator, MatmulLinearOperator, RootLinearOperator

__all__ = ['CosineKernel', 'LinearDirichletGPModel']


class CosineKernel(gpytorch.kernels.Kernel):
    def __init__(
            self,
            variance_constraint: Optional[Interval] = None,
            **kwargs
    ):
        super(CosineKernel, self).__init__(**kwargs)
        if variance_constraint is None:
            variance_constraint = Positive()

        self.register_parameter(name="raw_variance", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))
        self.register_constraint("raw_variance", variance_constraint)

    @property
    def variance(self) -> torch.Tensor:
        return self.raw_variance_constraint.transform(self.raw_variance)

    @variance.setter
    def variance(self, value: Union[float, torch.Tensor]):
        self._set_variance(value)

    def _set_variance(self, value: Union[float, torch.Tensor]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_variance)
        self.initialize(raw_variance=self.raw_variance_constraint.inverse_transform(value))

    def forward(
            self, x1: torch.Tensor, x2: torch.Tensor, diag: Optional[bool] = False,
            last_dim_is_batch: Optional[bool] = False,
            **params
    ) -> LinearOperator:
        assert not last_dim_is_batch
        x1_ = x1
        x1_norm = (torch.linalg.norm(x1_, dim=-1, keepdim=True) + 1).sqrt().reciprocal()

        if x1.size() == x2.size() and torch.equal(x1, x2):
            # Use RootLinearOperator when x1 == x2 for efficiency when composing
            # with other kernels
            prod = RootLinearOperator(x1_).add(torch.as_tensor([[1.]]).to(x1_))
            prod *= RootLinearOperator(x1_norm * self.variance.sqrt())
        else:
            x2_ = x2
            x2_norm = (torch.linalg.norm(x2_, dim=-1, keepdim=True) + 1).sqrt().reciprocal()
            prod = MatmulLinearOperator(
                x1_ * x1_norm * self.variance.sqrt(),
                (x2_ * x2_norm * self.variance.sqrt()).transpose(-2, -1)
            )
            prod += MatmulLinearOperator(
                x1_norm * self.variance.sqrt(),
                x2_norm.transpose(-2, -1) * self.variance.sqrt()
            )

        if diag:
            return prod.diagonal(dim1=-1, dim2=-2)
        else:
            return prod


class LinearDirichletGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_inducing, num_classes, input_dim, latent_dim=None, kernel_type='cosine'):
        self.batch_shape = torch.Size([num_classes])
        self.inducing_inputs = torch.randn(num_classes, num_inducing, input_dim)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing, batch_shape=self.batch_shape
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, self.inducing_inputs, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=self.batch_shape)
        if kernel_type == 'cosine':
            self.covar_module = CosineKernel(batch_shape=self.batch_shape)
            # No need to scale data
            self.scaler = lambda x: x
        else:
            self.covar_module = gpytorch.kernels.LinearKernel(batch_shape=self.batch_shape)
            self.scaler = gpytorch.utils.grid.ScaleToBounds(-1, 1)

    def transform(self, x):
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
        """Compute the posterior over z = self.transform(x)"""
        return self.variational_strategy(z, prior=False)
