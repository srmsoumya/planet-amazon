from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser(description='Planet Amazon from Space Challenge')

    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--scratch', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--epoch_start', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--best_acc', type=float, default=.0)
    parser.add_argument('--loss', type=float, default=.0)
    parser.add_argument('--cp_file', type=str, default='cp_best.pt.tar')
    parser.add_argument('--freeze_features', action='store_true', default=False)
    parser.add_argument('--freeze_classifier', action='store_true', default=False)
    parser.add_argument('--freeze_pct', type=float, default=0.0)
    parser.add_argument('--drop_rate', type=float, default=0.2)

    args = parser.parse_args()
    print(f'Configuration: {args}')
    return args
