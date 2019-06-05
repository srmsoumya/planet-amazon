import cv2
import numpy as np
import torch
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensor
from argparse import ArgumentParser
from pathlib import Path

from model import get_model_test
from utils import load_checkpoint_test
from dataset import mlb


def get_args():
    parser = ArgumentParser(description='Planet Amazon from Space Challenge: Predict')

    parser.add_argument('--cpu', action='store_true', default=True)
    parser.add_argument('--cp_file', type=str, default='cp_best.pt.tar')
    parser.add_argument('--img_path', type=str, default='sample.jpg')
    parser.add_argument('--drop_rate', type=float, default=0.0)

    args = parser.parse_args()
    return args


def load_model(args):
    cwd = Path.cwd()
    path = Path(cwd/'checkpoint'/args.cp_file)
    
    model = get_model_test(args)
    load_checkpoint_test(model, path, args)
    model.eval()

    return model


def load_data(input, url=False):
    SZ = 256
    MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    
    transform = {
        'test': Compose([
            Resize(height=SZ, width=SZ),
            Normalize(mean=MEAN, std=STD),
            ToTensor()
        ])
    }

    if url:
        img = np.frombuffer(input, np.uint8) # read from bytes
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(str(input)) # reads JPG image
    
    img = transform['test'](image=img)['image'] # converts to pytorch tensor

    return img.unsqueeze_(dim=0)


def predict(model, args):
    data = load_data(args.img_path)
    
    with torch.no_grad():
        output = model(data.to(args.device))
        output = output.detach().cpu().numpy() > 0.2
        output = mlb.inverse_transform(output)

        return output
        

def main():
    args = get_args()
    args.device = torch.device('cpu') if args.cpu else torch.device('cuda')
    model = load_model(args)
    return args, model


if __name__ == "__main__":
    args, model = main()
    results = predict(model, args)
    print(results)
