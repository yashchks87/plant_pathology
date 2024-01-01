import torch
import torch.nn as nn

class VGGBlock(nn.Module):
  def __init__(self, in_channels, out_channels, num_repeats):
    super(VGGBlock, self).__init__()
    self.modules = []
    for _ in range(num_repeats):
      self.modules.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
      self.modules.append(nn.ReLU())
      in_channels = out_channels
    self.modules.append(nn.MaxPool2d(kernel_size=2))
    self.conv = nn.Sequential(*self.modules)
  
  def forward(self, x):
    return self.conv(x)

class VGG16(nn.Module):
  def __init__(self, num_classes):
    super(VGG16, self).__init__()
    self.relu = nn.ReLU()
    self.block1 = VGGBlock(3, 64, 2)
    self.block2 = VGGBlock(64, 128, 2)
    self.block3 = VGGBlock(128, 256, 3)
    self.block4 = VGGBlock(256, 512, 3)
    self.block5 = VGGBlock(512, 512, 3)
    self.fc1 = nn.Linear(25088, 4096)
    self.fc2 = nn.Linear(4096, 4096)
    self.fc3 = nn.Linear(4096, num_classes)
  
  def forward(self, x):
    b1 = self.block1(x)
    b2 = self.block2(b1)
    b3 = self.block3(b2)
    b4 = self.block4(b3)
    b5 = self.block5(b4)
    # return b5
    b5 = b5.view(b5.size(0), -1)
    f1 = self.relu(self.fc1(b5))
    f2 = self.relu(self.fc2(f1))
    f3 = self.fc3(f2)
    return f3
  

def get_vgg16(num_classes):
  return VGG16(num_classes)