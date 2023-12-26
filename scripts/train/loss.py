import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_fn(inputs, targets):
    inputs = F.softmax(inputs, dim=1)
    loss = F.cross_entropy(inputs, targets)
    return loss