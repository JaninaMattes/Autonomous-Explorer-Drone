import torch
from torch import nn
from torch.distributions import MultivariateNormal


class DiagGaussianPolicy(nn.Module):
    def __init__(self, action_dim, init_std: float = 0.5, learned: bool = False) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.init_std = init_std
        self.learned = learned

        if learned:
            # PyTorch Parameters: are associated with the PyTorch module
            # and are subject to optimization and gradient updates.
            # They can be transferred between devices (i.e. CPU to GPU).
            self.log_std = nn.Parameter(
                torch.ones(action_dim) * torch.log(torch.tensor(init_std))
            )
        else:
            # PyTorch Buffers: are tensor attributes associated with parameters, 
            # but are other than parameters not part of the gradient update.
            # However, if our model is moved these tensor attributes 
            # will be themselves located on the GPU as well
            self.register_buffer(
                "std",
                torch.ones(action_dim) * init_std
            )

    def forward(self, mean: torch.Tensor) -> MultivariateNormal:
        if self.learned:
            std = torch.exp(self.log_std)
        else:
            std = self.std

        # Handle batching
        if mean.dim() == 2:
            std = std.expand(mean.size(0), -1)

        cov = torch.diag_embed(std ** 2)
        return MultivariateNormal(mean, covariance_matrix=cov)


if __name__ == "__main__":
    # Simple example
    action_dim = 4
    dist_fn = DiagGaussianPolicy(action_dim=3, learned=False)
    params = dict(dist_fn.named_parameters())
    mean = torch.zeros(action_dim)
    dist = dist_fn(mean)
    
    print("Without learned std parameters.")
    print(f"Params: {params}")
    print(f"Action dim: {action_dim}")
    print(f"Mean shape: {dist.mean.shape}")
    
    action_dim = 4
    dist_fn = DiagGaussianPolicy(action_dim=3, learned=True)
    params = dict(dist_fn.named_parameters())
    mean = torch.zeros(action_dim)
    dist = dist_fn(mean)

    print("\nWith learned std parameters.")
    print(f"Params: {params}")
    print(f"Action dim: {action_dim}")
    print(f"Mean shape: {dist.mean.shape}")
    