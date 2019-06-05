import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models 

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


class PlanetDenseNet(nn.Module):
    def __init__(self, M, c):
        super().__init__()
        self.features = M.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(in_features=1024, out_features=c)
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
    # resnet18 = models.resnet18(pretrained=True)
    densenet121 = models.densenet121(pretrained=True, drop_rate=args.drop_rate)
    base = PlanetDenseNet(densenet121, 17)
    wrapper = PlanetWrapper(base)

    # freeze or unfreeze features
    if args.freeze_features: wrapper.freeze_features(True)
    else: wrapper.freeze_features(False)

    # freeze or unfreeze classifier
    if args.freeze_classifier: wrapper.freeze_classifier(True)
    else: wrapper.freeze_classifier(False)

    # partial freeze features
    if args.freeze_pct:
        wrapper.partial_freeze_features(args.freeze_pct)

    wrapper.model.to(args.device)
    # optimizer = optim.SGD(wrapper.model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(wrapper.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    optimizer = optim.Adam(wrapper.model.parameters(), lr=args.lr, weight_decay=1e-4, amsgrad=True)

    wrapper.summary()    
    return wrapper.model, optimizer


def get_model_test(args):
    model = PlanetDenseNet(models.densenet121(pretrained=False, drop_rate=args.drop_rate), 17)
    model.to(args.device)

    return model
