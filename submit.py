import torch
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path

from model import get_model_test
from utils import load_checkpoint_test
from dataset import get_data_test

def get_args():
    parser = ArgumentParser(description='Planet Amazon from Space Challenge')

    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--cp_file', type=str, default='cp_best.pt.tar')
    parser.add_argument('--sub_file', type=str, default='submission.csv')

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    store = list()

    if not args.cpu: args.device = torch.device('cuda')
    cwd = Path.cwd()
    path = Path(cwd/'checkpoint'/args.cp_file)

    model = get_model_test(args)
    load_checkpoint_test(model, path, args)
    (mlb, test_dl) = get_data_test(args.img_size, args.batch_size) 
    
    model.eval()
    with torch.no_grad():
        for data, filenames in test_dl:
            output = model(data.to(args.device))
            output = output.detach().cpu().numpy() > 0.2
            output = mlb.inverse_transform(output)
            for name, tags in zip(filenames, output):
                store.append((name, ' '.join(tags)))
    
    store = pd.DataFrame(store, columns=['image_name', 'tags'])
    store.to_csv(args.sub_file, index=False) 
        

if __name__ == "__main__":
    main()




