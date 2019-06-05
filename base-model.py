import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def conv3x3(ni, no, ks=3, s=2, p=1):
    '''
    Conv2D with stride 2: change image sz to (image sz / 2)
    '''
    return nn.Conv2d(ni, no, kernel_size=3, stride=s, padding=p)


def conv1x1(ni, no, ks=1, s=1, p=0):
    '''
    Conv2D: increase number of channels fromm ni to no
    '''
    return nn.Conv2d(ni, no, kernel_size=ks, stride=s, padding=p)


class ConvBlock(nn.Module):
    '''
    ConvBlock: BatchNorm -> ReLU -> Conv
    '''
    def __init__(self, ni, no):
        super().__init__()
        self.conv = conv3x3(ni, no)
        self.bn = nn.BatchNorm2d(no)

    def forward(self, x):
        return self.bn(F.relu_(self.conv(x)))


class Lambda(nn.Module):
    '''
    Applies a function to the input
    '''
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)


class Net(nn.Module):
    '''
    Create a neural network.
    layers: # of Layers with number of channels
    c: # of classes
    '''
    def __init__(self, layers, c):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.layers = nn.ModuleList(
            [ConvBlock(layers[i], layers[i+1]) for i in range(len(layers) - 1)])
        self.flatten = Lambda(lambda x: x.view(x.size(0), -1))
        self.linear = nn.Linear(layers[-1], c)

    def forward(self, x):
        x = self.conv1(x)                           # x: bs x 3 x sz x sz
        for layer in self.layers: x = layer(x)      # x: bs x channels x sz/2 x sz/2
        x = F.adaptive_avg_pool2d(x, 1)             # x: bs x channels x 1 x 1
        x = self.flatten(x)                         # x: bs x channels
        x = self.linear(x)                          # x: bs x c

        return torch.sigmoid(x)


def get_model(args):
    model = Net([16, 32, 64], 17)
    model.to(args.device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    return model, optimizer