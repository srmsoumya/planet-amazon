import torch
import shutil
from pathlib import Path
from sklearn.metrics import fbeta_score
from datetime import datetime

class Logger:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.count, self.sum, self.avg = (0, 0, 0, 0)

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.sum += val * n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best):
    print('Storing the cake in the freezer!')
    cp_dir = Path('checkpoint')
    filename = f"cp_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{state['best_acc']:.6f}.pt.tar"
    path = cp_dir/filename
    torch.save(state, path)
    if is_best:
        print('One of the best cakes ever made! Saving it to cp_best.pt.tar')
        shutil.copyfile(path, cp_dir/'cp_best.pt.tar')

def load_checkpoint(model, optimizer, path, args):
    cp = torch.load(path)
    epoch_start = cp['epoch']
    best_acc = cp['best_acc']
    loss = cp['loss']
    model.load_state_dict(cp['model_state_dict'])
    optimizer.load_state_dict(cp['optimizer_state_dict'])

    # update the LR based on args
    for g in optimizer.param_groups:
        g['lr'] = args.lr
    optimizer.lr = args.lr

    return epoch_start, best_acc, loss

def load_checkpoint_test(model, path):
    cp = torch.load(path)
    best_acc = cp['best_acc']
    model.load_state_dict(cp['model_state_dict'])
    print(f'Accuracy of the model: {best_acc}')

def f2_score(output, target, threshold=0.2):
    target, output = target.cpu().numpy(), output.detach().cpu().numpy()
    output = output > threshold
    return fbeta_score(target, output, beta=2, average='samples')
