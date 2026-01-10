# drone_explorer/utils/utility.py
 
import torch
from torch import nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """ Initialize the hidden layers with orthogonal initialization
        Engstrom, Ilyas, et al., (2020)
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def normalize(x: torch.Tensor, eps: float = 1e-8):
    """Standard normalization"""
    return (x - x.mean()) / (x.std() + eps)



if __name__ == "__main__":
    # Simple examples as test layer_init
    print("Testing layer_init...")
    test_layer = nn.Linear(10, 5)
    initialized_layer = layer_init(test_layer)
    print(f"Weight shape: {initialized_layer.weight.shape}")
    print(f"Bias values (should be close to zero): {initialized_layer.bias.data}")
    print(f"Weight mean: {initialized_layer.weight.data.mean():.3f}")
    
    # Test normalize
    print("\nTesting normalize...")
    test_tensor = torch.randn(100) * 5 + 10  # Random data with mean ~10, std ~5
    print(f"Mean: {test_tensor.mean():.3f}, Std: {test_tensor.std():.3f}")
    normalized = normalize(test_tensor)
    print(f"Mean: {normalized.mean():.6f}, Std: {normalized.std():.3f}")
    print("Should be close to mean=0, std=1")