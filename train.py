import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time
from pathlib import Path

from config import get_args
from dataset import get_data
from model import get_model
from utils import Logger, save_checkpoint, load_checkpoint, f2_score


def loop(model, criterion, optimizer, data, target, device, loss_log, acc_log, end, train=True):
    
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = criterion(output, target)
    acc = f2_score(output, target)

    # record the loss and accuracy 
    loss_log.update(loss.item(), data.size(0))
    acc_log.update(acc, data.size(0))

    if train:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def train(epoch, model, criterion, optimizer, dataloader, args):
    print(f'TRAIN {epoch}')
    batch_log = Logger()
    data_log = Logger()
    loss_log = Logger()
    acc_log = Logger()

    model.train()
    end = time.time()
    for idx, (data, target) in enumerate(dataloader, 1):
        # measure data loading time
        data_log.update(time.time() - end)

        loop(model, criterion, optimizer, data, target, args.device, loss_log, acc_log, end, train=True)
        
        # measure time taken by a batch
        batch_log.update(time.time() - end)
        end = time.time()

        if idx % args.log_freq == 0:
            print(f'''T.Epoch: [{epoch}][{idx}/{len(dataloader)}] \t \
                  Time {batch_log.val:.3f}s ({batch_log.avg:.3f}s)\t \
                  Data {data_log.val:.3f}s ({data_log.avg:.3f}s)  \t \
                  Loss {loss_log.val:.4f}  ({loss_log.avg:.4f})   \t \
                  Acc  {acc_log.val:.3f}   ({acc_log.avg:.3f})''')
            # print(f'Epoch: {epoch}: [{idx*args.BS}/{args.BS*args.BATCHES}] \
            #         {int((idx*args.BS)/(args.BS*args.BATCHES)*100)}% {loss.item()}')

def validate(epoch, model, criterion, optimizer, dataloader, args):
    print(f'VALIDATE {epoch}')
    batch_log = Logger()
    data_log = Logger()
    loss_log = Logger()
    acc_log = Logger()
    
    model.eval()
    with torch.no_grad():
        end = time.time()
        for idx, (data, target) in enumerate(dataloader, 1):
            # measure data loading time
            data_log.update(time.time() - end)

            loop(model, criterion, optimizer, data, target, args.device, loss_log, acc_log, end, train=False)

            # measure time taken by a batch
            batch_log.update(time.time() - end)
            end = time.time()
            if idx % (args.log_freq) == 0:
                print(f'''V.Epoch: [{epoch}][{idx}/{len(dataloader)}]\t \
                    Time {batch_log.val:.3f}s ({batch_log.avg:.3f}s) \t \
                    Data {data_log.val:.3f}s  ({data_log.avg:.3f}s)  \t \
                    Loss {loss_log.val:.4f}  ({loss_log.avg:.4f})    \t \
                    Acc  {acc_log.val:.3f}   ({acc_log.avg:.3f})''')

        print(f'** Validation Accuracy: {acc_log.avg:.3f} **')
        return acc_log.avg, loss_log.avg

        # print(f'Val Loss: {val_loss.item()/args.BS*len(val_dl)}')

def fit(model, criterion, optimizer, train_dl, val_dl, args):

    for epoch in range(args.epoch_start, args.epochs + 1):
        train(epoch, model, criterion, optimizer, train_dl, args)
        acc, args.loss = validate(epoch, model, criterion, optimizer, val_dl, args)

        is_best = acc > args.best_acc   
        args.best_acc = max(acc, args.best_acc)
        save_checkpoint({
            'epoch': epoch,
            'best_acc': args.best_acc,
            'loss': args.loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, is_best)

    return model

def main():
    args = get_args()
    cwd = Path.cwd()

    # Create the device
    args.device = torch.device('cuda') if (
        not args.cpu and torch.cuda.is_available()) else torch.device('cpu')
    print(f'Oven: {args.device}')

    # Load the train-dataloader and validation-dataloader
    train_dl, val_dl = get_data(args.img_size, args.batch_size)

    # Load the model, define the loss & optim
    model, optimizer = get_model(args)
    criterion = nn.BCELoss()

    # Create checkpoint directory to store the state
    cp_dir = cwd/'checkpoint'
    cp_dir.mkdir(exist_ok=True)

    if args.scratch:
        print('Fresh Bake! Training the network from scratch.')
    else:
        path = cp_dir/args.cp_file
        args.epoch_start, args.best_acc, args.loss = load_checkpoint(model, optimizer, path, args)
        print(f'Warming Up! Loading the network from: {path}')
        print(f'Start Epoch: {args.epoch_start}, Accuracy: {args.best_acc}')

    # Call model fit
    fit(model, criterion, optimizer, train_dl, val_dl, args)

if __name__ == "__main__":
    main()
