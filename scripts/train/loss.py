import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_fn(inputs, targets, class_weights=None):
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).cuda()
        cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
    else:
        cross_entropy = nn.CrossEntropyLoss()
    targets = targets.view(-1)
    loss = cross_entropy(inputs, targets)
    return loss