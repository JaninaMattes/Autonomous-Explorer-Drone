import torch
from torch import nn

""" Base model """
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()