import torch
import torch.nn as nn
import torch.nn.functional as F


#Defining the layers that constitute the basic structure of the residual block
class BaseBlock(nn.Module):
  def __init__(self, input, channels, stride=1, dilate=1):
    self.stride = stride
    super().__init__()

    self.conv1 = nn.Conv1d(input, channels, kernel_size=3, stride=stride, padding=dilate, dilation=dilate)
    self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, stride=1, padding=dilate, dilation=dilate)
    if stride > 1:
      self.convd = nn.Conv1d(input, channels, kernel_size=1, stride=stride, padding=0)
    self.bnorm1 = nn.BatchNorm1d(channels)
    self.bnorm2 = nn.BatchNorm1d(channels)

  def forward(self, x):
    h1 = F.relu(self.bnorm1(self.conv1(x)))
    if self.stride > 1:
      res = self.convd(x)
    else:
      res = x
    h2 = res + self.bnorm2(self.conv2(h1))
    return F.relu(h2)

#Defining the number of BaseBlocks that make up a single ResBlock
class ResBlock(nn.Module):
  def __init__(self, input, channels, n_block, dilate=1):
    self.n_block = n_block
    super().__init__()

    #Creating a ResBlock form BaseBlocks in a loop
    self.block0 = BaseBlock(input, channels, 2, dilate)
    self.res = nn.ModuleList()
    for i in range(n_block):
      self.res.append(BaseBlock(channels, channels, 1, dilate))

  def forward(self, x):
    h = self.block0(x)
    for res in self.res:
      h = res(h)
    return h

#Defining the structure of the ResNet model
class ResNet34(nn.Module):
  def __init__(self):
    super().__init__()

    self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3)
    self.bnorm1 = nn.BatchNorm1d(64)
    self.resblock0 = ResBlock(64, 64, 3)
    self.resblock1 = ResBlock(64, 128, 4)
    self.resblock2 = ResBlock(128, 256, 6)
    self.resblock3 = ResBlock(256, 512, 3)
    self.avgpool = nn.AdaptiveAvgPool1d(1)
    self.l1 = nn.Linear(512, 2)

  def forward(self, x):
    h1 = F.relu(self.bnorm1(self.conv1(x)))
    h2 = F.max_pool1d(h1, 3, 2)
    h3 = self.resblock0(h2)
    h4 = self.resblock1(h3)
    h5 = self.resblock2(h4)
    h6 = self.resblock3(h5)
    h7 = self.avgpool(h6)
    h7 = torch.flatten(h7, 1)
    h8 = self.l1(h7)
    return h8
