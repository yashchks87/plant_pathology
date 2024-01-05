import torch
import torch.nn as nn
import torch.nn.functional as F

class IncepBlock(nn.Module):
  def __init__(self, 
               in_channels, 
               channels_1x1_out,
               channels_3x3_reduce,
               channels_3x3,
               channels_5x5_reduce,
               channels_5x5,
               out_1x1):
    super(IncepBlock, self).__init__()
    self.branch1 = nn.Sequential(
        nn.Conv2d(in_channels, channels_1x1_out, kernel_size=1, stride=1),
        nn.ReLU(),
    )
    self.branch2 = nn.Sequential(
        nn.Conv2d(in_channels, channels_3x3_reduce, kernel_size=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(channels_3x3_reduce, channels_3x3, kernel_size=3, stride=1, padding=1),
        nn.ReLU()
    )
    self.branch3 = nn.Sequential(
        nn.Conv2d(in_channels, channels_5x5_reduce, kernel_size=1, stride=1),
        nn.ReLU(),
        nn.Conv2d(channels_5x5_reduce, channels_5x5, kernel_size=5, stride=1, padding=2),
        nn.ReLU()
    )
    self.branch4 = nn.Sequential(
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        nn.Conv2d(in_channels, out_1x1, kernel_size=1, stride=1),
        nn.ReLU()
    )
  
  def forward(self, x):
    b1 = self.branch1(x)
    b2 = self.branch2(x)
    b3 = self.branch3(x)
    b4 = self.branch4(x)
    return torch.cat((b1, b2, b3, b4), dim=1)
  
class AuxLoss(nn.Module):
  def __init__(self, 
               conv_in_channels,
               dense_inputs,
               num_classes=12):
    super(AuxLoss, self).__init__()
    self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
    self.conv1 = nn.Conv2d(conv_in_channels, 128, kernel_size=1, stride=1)
    self.relu = nn.ReLU()
    self.dense1 = nn.Linear(dense_inputs, 1024)
    self.dropout = nn.Dropout(p=0.7)
    self.dense2 = nn.Linear(1024, num_classes)
  
  def forward(self, x):
    ap1 = self.avgpool(x)
    c1 = self.conv1(ap1)
    r1 = self.relu(c1)
    r1 = r1.view(r1.size(0), -1)
    d1 = self.dense1(r1)
    d1 = self.dropout(d1)
    d2 = self.dense2(d1)
    return d2
  
class IncepV1(nn.Module):
  def __init__(self, num_classes=12):
    super(IncepV1, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
    self.relu = nn.ReLU()
    # Look for here if error occurs
    self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(64, 192, kernel_size=1, stride=1)
    self.conv3 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
    self.block3a = IncepBlock(192, 64, 96, 128, 16, 32, 32)
    self.block3b = IncepBlock(256, 128, 128, 192, 32, 96, 64)
    self.block4a = IncepBlock(480, 192, 96, 208, 16, 48, 64)
    self.aux1 = AuxLoss(512, 2048)
    self.block4b = IncepBlock(512, 160, 112, 224, 24, 64, 64)
    self.block4c = IncepBlock(512, 128, 128, 256, 24, 64, 64)
    self.block4d = IncepBlock(512, 112, 144, 288, 32, 64, 64)
    self.aux2 = AuxLoss(528, 2048)
    self.block4e = IncepBlock(528, 256, 160, 320, 32, 128, 128)
    self.block5a = IncepBlock(832, 256, 160, 320, 32, 128, 128)
    self.block5b = IncepBlock(832, 384, 192, 384, 48, 128, 128)
    self.avgpool = nn.AvgPool2d(7, 1)
    self.dropout = nn.Dropout(p=0.4)
    self.linear = nn.Linear(1024, num_classes)

  def forward(self, x):
    c1 = self.conv1(x)
    r1 = self.relu(c1)
    m1 = self.max1(r1)
    c2 = self.conv2(m1)
    r2 = self.relu(c2)
    c3 = self.conv3(r2)
    r3 = self.relu(c3)
    m2 = self.max1(r3)
    b3a = self.block3a(m2)
    b3b = self.block3b(b3a)
    m3 = self.max1(b3b)
    b4a = self.block4a(m3)
    o1 = self.aux1(b4a)
    b4b = self.block4b(b4a)
    b4c = self.block4c(b4b)
    b4d = self.block4d(b4c)
    o2 = self.aux2(b4d)
    b4e = self.block4e(b4d)
    m4 = self.max1(b4e)
    b5a = self.block5a(m4)
    b5b = self.block5b(b5a)
    ap1 = self.avgpool(b5b)
    ap1 = self.dropout(ap1)
    ap1 = ap1.view(ap1.size(0), -1)
    l1 = self.linear(ap1)
    return l1, o1, o2
  
def get_incep_v1(num_classes=12):
  return IncepV1(num_classes=num_classes)