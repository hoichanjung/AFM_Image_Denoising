import argparse


def getConfig():
    parser = argparse.ArgumentParser()

    parser.add_argument('action', choices=('train', 'test'))
    parser.add_argument('--exp_num', default='1', type=str,
                        help='experiment number')
    parser.add_argument('--neptune', default='sync', type=str,
                        help='update on neptune (sync/offline)')
                     
    # dataset
    parser.add_argument('--noise', '-n', default='Random', type=str,
                        help='noise type (Random/Line/Scar/Hum)')
    parser.add_argument('--resize', '-i', default=256, type=int,metavar='N',
                        help='image size (default: 512)')
    parser.add_argument('--workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 6)')

    # optimizer config
    parser.add_argument('--op', default='adam', type=str,
                        help='the name of optimizer(adam,adadelta)')
    parser.add_argument('--scheduler', '--s',  default='ReduceLROnPlateau', type=str,
                        help='the name of scheduler(ReduceLROnPlateau,CosineAnnealingWarmRestarts)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, metavar='W',
                        help='weight decay (default: 1e-5)')
    parser.add_argument('--patience', '--patience', default=3, type=int,
                        help='patience')

    # model config
    parser.add_argument('--model', default='UNET', type=str,
                        help='model name(UNET/HINET/MPRNET/RESTORMER/UFORMER)')

    # training config
    parser.add_argument('--gid', default='0,1',
                        help='gpu id list(eg: 0,1,2...)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--b', '--batch-size', default=16, type=int,metavar='N',
                        help='mini-batch size (default: 16)')
    parser.add_argument('--ckpt', default=None, type=str, metavar='checkpoint_path',
                        help='path to save checkpoint (default: checkpoint)')

    args = parser.parse_args()

    return args



if __name__ == '__main__':
    config = getConfig()
    config = vars(config)
    print(config)