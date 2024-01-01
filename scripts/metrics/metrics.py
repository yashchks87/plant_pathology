import torch
import torch.nn as nn
import torch.nn.functional as F

def confusion_matrix(final_preds, one_hot_truth, num_classes):
  c_m = torch.zeros(num_classes, num_classes).int()
  for x in range(num_classes):
    # actual_ones = one_hot_truth[torch.where(one_hot_truth[:, x] == 1)]
    c_m[x] = final_preds[torch.where(one_hot_truth[:, x] == 1)].sum(dim=0)
  return c_m

def generate_precision(c_m):
  return (c_m.diagonal() / c_m.sum(dim=0))  # add epsilon to avoid division by zero (nan

def generate_recall(c_m):
  return (c_m.diagonal() / c_m.sum(dim=1))  # add epsilon to avoid division by zero (nan

def generate_metrics(preds, targets, num_classes):
  targets = targets.view(-1)
  preds = F.softmax(preds, dim = 1)
  one_hot_truth = F.one_hot(targets, num_classes=num_classes)
  final_preds = F.one_hot(torch.argmax(preds, dim = 1), num_classes=num_classes)
  # return one_hot_truth, final_preds
  c_m = confusion_matrix(final_preds, one_hot_truth, num_classes)
  # return c_m
  precision = generate_precision(c_m)
  recall = generate_recall(c_m)
  precision = torch.nan_to_num(precision, nan=0.0)
  recall = torch.nan_to_num(recall, nan=0.0)
  return c_m, precision, recall