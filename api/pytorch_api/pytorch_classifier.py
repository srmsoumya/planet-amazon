import cv2
import numpy as np
import torch
from torchvision import models
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensor
from argparse import ArgumentParser
from pathlib import Path

from model import PlanetDenseNet
from dataset import mlb


use_gpu = True
device = torch.device(
    'cuda') if use_gpu and torch.cuda.is_available() else torch.device('cpu')
print('Using device: ', device)

def load_model(model_path, device=device):
    print('pytorch_classifier.py: Loading model...')
    checkpoint = torch.load(model_path, map_location=device)

    model = PlanetDenseNet(models.densenet121(pretrained=False), 17)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device=device)
    model.eval()

    return model

def classify(model, bin_img):
    SZ = 256
    MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array(
        [0.229, 0.224, 0.225])

    transform = {
        'test': Compose([
            Resize(height=SZ, width=SZ),
            Normalize(mean=MEAN, std=STD),
            ToTensor()
        ])
    }
    
    img = np.frombuffer(bin_img, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform['test'](image=img)['image']
    img = img.unsqueeze_(dim=0)

    with torch.no_grad():
        output = model(img.to(device))
        output = output.detach().cpu().numpy() > 0.2
        output = mlb.inverse_transform(output)

        return output[0]
