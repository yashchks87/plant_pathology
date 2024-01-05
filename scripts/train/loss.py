import torch
import torch.nn as nn
import torch.nn.functional as F

def loss_fn(inputs, inputs_aux_1, inputs_aux_2, targets, class_weights=None):
    # if class_weights is not None:
    #     class_weights = torch.FloatTensor(class_weights).cuda()
    #     cross_entropy = nn.CrossEntropyLoss(weight=class_weights)
    # else:
    #     cross_entropy = nn.CrossEntropyLoss()
    cross_entropy = nn.CrossEntropyLoss()
    if inputs_aux_1 != None:
        cross_entropy_aux_1 = nn.CrossEntropyLoss()
        cross_entropy_aux_2 = nn.CrossEntropyLoss()
    targets = targets.view(-1)
    loss = cross_entropy(inputs, targets)
    if inputs_aux_1 != None:
        loss_aux_1 = cross_entropy_aux_1(inputs_aux_1, targets)
        loss_aux_2 = cross_entropy_aux_2(inputs_aux_2, targets)
    return loss + 0.3 * loss_aux_1 + 0.3 * loss_aux_2