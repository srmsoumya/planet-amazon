import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class PlanetResNet(nn.Module):
    def __init__(self, M, c):
        super().__init__()
        self.features = nn.Sequential(*(list(M.children())[:-2]))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(in_features=512, out_features=c)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.sigmoid(x)


class PlanetWrapper(object):
    def __init__(self, M):
        self.model = M

    @staticmethod
    def freeze(layers):
        for param in layers.parameters(): param.requires_grad_(False)
            
    @staticmethod
    def unfreeze(layers):
        for param in layers.parameters(): param.requires_grad_(True)
        
    def freeze_features(self, arg=True):
        if arg: self.freeze(self.model.features)
        else:   self.unfreeze(self.model.features)
            
    def partial_freeze_features(self, pct=0.2):
        size = len(list(self.model.features.children()))
        freeze_point = int(size * (1 - pct))
        
        for idx, child in enumerate(self.model.features.children()):
            if idx < freeze_point: self.freeze(child)
            else: self.unfreeze(child)
        
    def freeze_classifier(self, arg=True):
        if arg: self.freeze(self.model.classifier)
        else:   self.unfreeze(self.model.classifier)

    def summary(self):
        print('\n\n')
        for idx, (name, child) in enumerate(self.model.features.named_children()):
            print(f'{idx}: {name}-{child}')
            for param in child.parameters():
                print(f'{param.requires_grad}')

        for idx, (name, child) in enumerate(self.model.classifier.named_children()):
            print(f'{idx}: {name}-{child}')
            for param in child.parameters():
                print(f'{param.requires_grad}')
        print('\n\n')


def get_model(args):
    base = PlanetResNet(resnet18(), 17)
    resnet = PlanetWrapper(base)

    # freeze or unfreeze features
    if args.freeze_features: resnet.freeze_features(True)
    else: resnet.freeze_features(False)

    # freeze or unfreeze classifier
    if args.freeze_classifier: resnet.freeze_classifier(True)
    else: resnet.freeze_classifier(False)

    # partial freeze features
    if args.freeze_pct:
        resnet.partial_freeze_features(args.freeze_pct)

    resnet.model.to(args.device)
    # optimizer = optim.SGD(resnet.model.parameters(), lr=args.lr)
    optimizer = optim.SGD(resnet.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    resnet.summary()    
    return resnet.model, optimizer


def get_model_test(args):
    model = PlanetResNet(resnet18(), 17)
    model.to(args.device)

    return model