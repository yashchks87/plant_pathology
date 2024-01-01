import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_fn(inputs, targets):
    cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)
    # inputs = F.softmax(inputs, dim=1)
    targets = targets.view(-1)
    # loss = F.cross_entropy(inputs, targets)
    loss = cross_entropy(inputs, targets)
    return loss