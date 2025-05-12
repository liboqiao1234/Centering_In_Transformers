import argparse
from torchvision import models

def get_args():
    parser = argparse.ArgumentParser(description="vit")

    parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_small',
                        help='model architecture')
    
    parser.add_argument('--m', '--method', type=str, default="ori", help="method on model")

    parser.add_argument("--data_path", type=str, default="/path/to/imagenet",
                        help="path to dataset repository")
    
    parser.add_argument("--dump_path", type=str, default="/path/to/result",
                    help="path to dataset repository")

    parser.add_argument("--epochs", default=100, type=int,
                        help="number of total epochs to run")

    parser.add_argument('--img_size', default=224, type=int,
                        help='resolution of input image')

    parser.add_argument("--batch_size", default=256, type=int,
                        help="batch size per gpu, i.e. how many unique instances per gpu")

    parser.add_argument("--patch_size", default=16, type=int,
                        help="patch size")

    parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial (base) learning rate for train', dest='lr')
    
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float,
                    metavar='W', help='weight decay', dest='weight_decay')

    parser.add_argument("--warmup_epochs", default=5, type=int, help="number of warmup epochs (5% of total epochs)")

    parser.add_argument("--workers", default=4, type=int,
                        help="number of data loading workers")

    parser.add_argument('--num_classes', default=1000, type=int,   
                    help='number of classes')
    
    parser.add_argument('--seed', default=-1, type=int, help='manual seed')

    parser.add_argument('--wandb', default=False, type=bool, help='wandb log')
    
    parser.add_argument('--norm_type', type=str, default="ln", choices=['ln', 'rms'],
                        help='normalization type: ln for LayerNorm, rms for RMSNorm')
    
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout rate')

    return parser.parse_args()
