import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
  # Downsample meaning residual connections are applied
  def __init__(self, in_channels, out_channels, stride = 1, downsample=None):
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(out_channels)
    )
    self.downsample = downsample
    self.relu = nn.ReLU()
    self.out_channels = out_channels

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    # return x
    out = self.conv2(out)
    if self.downsample:
      # So if we have to downsample this will becomes new residual values
      # to match it with summing values because number of channels now changed and
      # height and width of image is changed as well.
      residual = self.downsample(x)
    out += residual
    out = self.relu(out)
    return out

class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes = 10):
  # def __init__(self):
    """
      Args:
        block: Residual block class is being used
        layers: Number of resblocks
        num_classes: Number of classes in the dataset
    """
    """
      Some notes on dotted lines called as downsampling
      https://towardsdatascience.com/residual-blocks-building-blocks-of-resnet-fd90ca15d6ec
    """
    super(ResNet, self).__init__()
    self.inplanes = 64
    self.conv1 = nn.Sequential(
                  nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                  nn.BatchNorm2d(64),
                  nn.ReLU()
                )
    self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
    self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
    self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
    self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
    self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
    self.fc = nn.Linear(512, num_classes)
    self.avgpool = nn.AvgPool2d(7, stride = 1)

  def _make_layer(self, block, planes, blocks, stride = 1):
    downsample = None
    if stride != 1 or self.inplanes != planes:
      downsample = nn.Sequential(
          nn.Conv2d(self.inplanes, planes, kernel_size = 1, stride = stride),
          nn.BatchNorm2d(planes)
      )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))
    return nn.Sequential(*layers)

  # def _make_layer()
  def forward(self, x):
    x = self.conv1(x)
    x = self.maxpool(x)
    x = self.layer0(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x
  
  def get_predictions(self, x):
    self.eval()
    with torch.no_grad():
      x = self.forward(x)
      return x

def get_model(num_classes = 12):
    model = ResNet(ResidualBlock, [3, 4, 6, 3], num_classes = num_classes)
    return model