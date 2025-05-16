import math
import os
import builtins
import shutil
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import CosineAnnealingLR
from args import get_args
import models.vision_transformer as vits
import models.vision_transformer_cbwc as vits_cbwc
import models.vision_transformer_rms as vits_rms
import models.vision_transformer_cbwc_wc as vits_cbwc_wc
import wandb
import time
from data import CustomDataset, make_dataset
from torchvision.datasets import CIFAR10
from Taiyi.taiyi.monitor import Monitor
from Taiyi.visualize import Visualization


# 线性学习率调度器，包含预热期和线性衰减
class LinearWarmupLinearDecayLR:
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.last_epoch = last_epoch
        self._step_count = 0
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        
    def get_lr(self):
        if self._step_count <= self.warmup_epochs:
            # 线性预热
            return [base_lr * (self._step_count / self.warmup_epochs) for base_lr in self._last_lr]
        else:
            # 线性衰减到0
            remaining = max(0, (self.total_epochs - self._step_count) / (self.total_epochs - self.warmup_epochs))
            return [base_lr * remaining for base_lr in self._last_lr]
    
    def step(self):
        self._step_count += 1
        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
        return values


def main():

    args = get_args()

    datapath = args.data_path
    # dataset = datapath.split('/')[-1]
    dataset = 'cifar10'

    model_name = str(args.arch) + '_' + args.m + '_' + dataset + '_e' + str(args.epochs) + '_bs' + str(args.batch_size) + '_lr' + str(args.lr) + '_wd' + str(args.weight_decay)
    model_name = model_name + '_wre' + str(args.warmup_epochs) + '_wk' + str(args.workers) + '_nc' + str(args.num_classes) + '_s' + str(args.seed) + '_ps' + str(args.patch_size) + '_' + args.norm_type

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Data loading code
    datadir = args.data_path
    print("Loading data from '{}'".format(datadir))

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = CIFAR10(
        root=datadir,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = CIFAR10(
        root=datadir,
        train=False,
        download=True,
        transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False
    )



    # train_dataset, test_dataset = make_dataset(root_dir=datadir, splite_rate=0.2)
    #
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers,
    #     pin_memory=True, shuffle=True)
    #
    # val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, num_workers=args.workers,
    #     pin_memory=True, shuffle=False)

    print("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    print("creating model '{}'".format(args.arch))

    # 根据指定的norm_type参数选择模型
    if args.norm_type == 'ln':
        if args.m == 'ori':
            model = vits.__dict__[args.arch](img_size=[args.img_size], num_classes=args.num_classes, patch_size=args.patch_size, drop_rate=args.dropout)
        elif args.m == 'cbwc':
            model = vits_cbwc.__dict__[args.arch](num_classes=args.num_classes, patch_size=args.patch_size, drop_rate=args.dropout)
        elif args.m == 'cbwc-wc':
            model = vits_cbwc_wc.__dict__[args.arch](num_classes=args.num_classes, patch_size=args.patch_size, drop_rate=args.dropout)
    else:  # 'rms'
        model = vits_rms.__dict__[args.arch](img_size=[args.img_size], num_classes=args.num_classes, patch_size=args.patch_size, drop_rate=args.dropout)

    args.lr = args.lr * args.batch_size / 256
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # 使用自定义调度器，实现预热和线性衰减
    scheduler = CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    #scheduler = LinearWarmupLinearDecayLR(optimizer, 
                                        #  warmup_epochs=args.warmup_epochs, 
                                        #  total_epochs=args.epochs)

    print("Building optimizer done.")

    # copy model to GPU
    model.cuda()
    print(model)
    print("Building model done.")
    global step
    step = 0
    if args.wandb:
        wandb.init(
            project="vit",
            name=model_name,
            notes=str(args),
            config={
                "architecture": args.arch,
                "method": args.m,
                "dataset": "cifar10",
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "patch_size": args.patch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_epochs": args.warmup_epochs,
                "workers": args.workers,
                "method": str('origin'),
                "seed": args.seed,
                "norm_type": args.norm_type,
                "dropout": args.dropout,
            }
        )
        taiyi_config = {
            "LayerNorm": [['InputAngleStd','linear(5,0)'], ['InputAngleMean', 'linear(5,0)'],
                          ['OutputAngleStd','linear(5,0)'], ['OutputAngleMean', 'linear(5,0)']],
            "RMSNorm": [['InputAngleStd','linear(5,0)'], ['InputAngleMean', 'linear(5,0)'],
                        ['OutputAngleStd','linear(5,0)'], ['OutputAngleMean', 'linear(5,0)']],
            "Attention": [['InputAngleStd','linear(5,0)'], ['InputAngleMean', 'linear(5,0)'],
                          ['OutputAngleStd','linear(5,0)'], ['OutputAngleMean', 'linear(5,0)']],
            "Mlp": [['InputAngleStd','linear(5,0)'], ['InputAngleMean', 'linear(5,0)'],
                            ['OutputAngleStd','linear(5,0)'], ['OutputAngleMean', 'linear(5,0)']],
        }
        monitor = Monitor(model, taiyi_config)
        vis_wandb = Visualization(monitor, wandb)
    else:
        monitor = None
        vis_wandb = None

    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    dump_path = os.path.join(args.dump_path, model_name)

    if not os.path.exists(dump_path):
        # 如果路径不存在，则创建它
        os.makedirs(dump_path)
        print(f"目录 {dump_path} 已创建。")
    else:
        print(f"目录 {dump_path} 已存在。")

    print("==> Begin Training.")

    for epoch in range(args.epochs):

        # train the network for one epoch
        print("============ Starting epoch %i ... ============" % epoch)

        # train the network
        train(train_loader, model, criterion, optimizer, epoch, args, monitor, vis_wandb)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        if args.wandb:
            wandb.log({"learning_rate": current_lr})

        # save checkpoints
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler._step_count,
        }

        torch.save(
            save_dict,
            os.path.join(dump_path, "checkpoint.pth.tar"),
        )

        acc1, acc5 = validate(val_loader, model, criterion, args)

    if args.wandb:
        wandb.finish()
        vis_wandb.close()
        monitor.get_output()


def train(train_loader, model, criterion, optimizer, epoch, args, monitor=None, vis_wandb=None):
    global step
    batch_time = AverageMeter('Time', ':6.3f')
    fp_time = AverageMeter('FPTime', ':6.3f')
    bp_time = AverageMeter('BPTime', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    torch.cuda.synchronize()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        torch.cuda.synchronize()
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        torch.cuda.synchronize()
        fp_begin = time.time()
        output = model(images)
        torch.cuda.synchronize()
        fp_end = time.time()
        fp_time_batch = (fp_end - fp_begin) * 1e6
        fp_time.update(fp_time_batch)

        loss = criterion(output, target)

        if monitor is not None:
            monitor.track(step)
        if vis_wandb is not None:
            vis_wandb.show(step)
        step += 1
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        torch.cuda.synchronize()
        bp_begin = time.time()
        loss.backward()
        
        bp_end = time.time()
        bp_time_batch = (bp_end - bp_begin) * 1e6
        bp_time.update(bp_time_batch)
        if args.wandb:
            wandb.log({"fp_time":fp_time_batch, "bp_time":bp_time_batch})

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % 100 == 0:
            progress.display(i)

        torch.cuda.synchronize()
        end = time.time()

    if args.wandb:
        wandb.log({"train_loss":losses.avg, "train_acc_top1": top1.avg, "train_acc_top5": top5.avg, "train_epoch":epoch, "train_fp_avg_time": fp_time.avg, "train_bp_avg_time": bp_time.avg})

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    val_time = AverageMeter('val_time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output

            torch.cuda.synchronize()
            val_begin = time.time()
            output = model(images)
            torch.cuda.synchronize()
            val_end = time.time()
            val_time_batch = (val_end - val_begin)*1e6
            val_time.update(val_time_batch)
            if args.wandb:
                wandb.log({"val_time":val_time_batch})

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    if args.wandb:
        wandb.log({"test_loss":losses.avg, "test_acc_top1": top1.avg, "test_acc_top5": top5.avg, "val_avg_time": val_time.avg})
    return top1.avg, top5.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == "__main__":
    main()
