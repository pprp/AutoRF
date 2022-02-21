import argparse
import glob
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import space.genotypes as genotypes
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import utils.utils as utils
from utils.warmup import WarmUpLR
from tensorboardX import SummaryWriter
from thop import profile
from torch.autograd import Variable
from utils.labelsmooth import LSR
from retrain.studentnet import Network
from torch.cuda.amp import autocast


parser = argparse.ArgumentParser("cifar")

############## data

parser.add_argument(
    "--data", type=str, default="/data/public/cifar", help="location of the data corpus"
)
parser.add_argument(
    "--dataset", type=str, default="cifar10", help="cifar10 or cifar100"
)

parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")

parser.add_argument("--cutout_length", type=int, default=8, help="cutout length")

parser.add_argument(
    "--primitives",
    type=str,
    default="fullpool",
    help="choose in autola, fullpool, fullconv, hybrid",
)

############# params
parser.add_argument("--model_name", type=str, default="resnet20", help="name")

parser.add_argument(
    "--model_base", type=str, default="resnet20", help="name of base models"
)
parser.add_argument("--learning_rate", type=float, default=0.1, help="learning rate")
parser.add_argument(
    "--learning_rate_min", type=float, default=0.001, help="min learning rate"
)
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument(
    "--weight_decay", type=float, default=5e-4, help="weight decay 1e-3"
)
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--epochs", type=int, default=500, help="num of training epochs")
parser.add_argument("--report_freq", type=float, default=400, help="report frequency")
parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
parser.add_argument("--save", type=str, default="test", help="experiment name")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")

parser.add_argument(
    "--model_path",
    type=str,
    default="exps/search-exp-20191220-160515/weights.pt",
    help="path of pretrained model",
)
parser.add_argument(
    "--auxiliary", action="store_true", default=False, help="use auxiliary tower"
)

################## settings 

parser.add_argument(
    "--label_smooth", action="store_true", default=False, help="wether use label smooth"
)

parser.add_argument(
    "--scheduler", type=str, default="cosine", help="scheduler choose from cosine, steplr, warmup"
)

parser.add_argument(
    "--warm", type=int, default=5, help="warmup epoch numbers"
)

parser.add_argument(
    "--drop_path_prob", type=float, default=0.2, help="drop path probability"
)

parser.add_argument("--dropout", type=float, default=0.0, help="drop path probability")
parser.add_argument("--seed", type=int, default=0, help="random seed")

parser.add_argument(
    "--arch", type=str, default="Attention", help="which architecture to use"
)

############## end

args = parser.parse_args()
args.save = "{}-{}-{}".format(
    args.model_name, args.save, time.strftime("%Y%m%d-%H%M%S")
)
utils.create_exp_dir(
    os.path.join("exps/retrain", args.save), scripts_to_save=glob.glob("*/*.py")
)

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(
    os.path.join(os.path.join("exps/retrain", args.save), "log.txt")
)
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset == "cifar10":
    CIFAR_CLASSES = 10
elif args.dataset == "cifar100":
    CIFAR_CLASSES = 100


def set_learning_rate(optimizer, epoch):
    if epoch <= 150:
        optimizer.param_groups[0]["lr"] = 0.1
    elif epoch < 300:
        optimizer.param_groups[0]["lr"] = 0.01
    else:
        optimizer.param_groups[0]["lr"] = 0.001


def main():
    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("gpu device = %d" % args.gpu)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    logging.info("genotype = %s", genotype)

    model = Network(args.model_base, CIFAR_CLASSES, genotype, dropout=args.dropout)

    # flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),))
    model = model.cuda()

    # info = f"flops:{flops/1000**3}G params: {params/1000**2}M"
    # logging.info(info)

    test_epoch = 1
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    if args.label_smooth:
        criterion = LSR(e=0.2)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    train_transform, test_transform = utils._data_transforms_cifar(args)
    train_queue, test_queue = utils._data_loader_cifar(
        args, train_transform, test_transform
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs)
    )
    elif args.scheduler == "steplr":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2) #learning rate decay
    elif args.scheduler == "warmup":
        warmup_scheduler = WarmUpLR(optimizer, args.warm * len(train_queue))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2) #learning rate decay

    best_acc = 0.0
    writer = SummaryWriter(os.path.join("exps/retrain", args.save))

    for epoch in range(args.epochs):
        if epoch <= args.warm and args.scheduler == "warmup":
            warmup_scheduler.step() 

        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        # train_acc, train_obj = train_ricap(train_queue, model, criterion, optimizer)

        # logging.info("train_acc %f", train_acc)

        valid_acc, valid_obj = infer(test_queue, model, criterion)
        writer.add_scalar("train_loss", train_obj)
        writer.add_scalar("train_acc", train_acc)
        writer.add_scalar("val_loss", valid_obj)
        writer.add_scalar("val_acc", valid_acc)

        if valid_acc > best_acc:
            best_acc = valid_acc
            logging.info(
                "valid epoch %d, valid_acc %.2f, best_acc %.2f",
                epoch,
                valid_acc,
                best_acc,
            )
            utils.save(
                model,
                os.path.join(
                    os.path.join("exps/retrain", args.save), "weights_retrain.pt"
                ),
            )
        if args.scheduler == "warmup" and epoch > args.warm:
            scheduler.step()
        elif not args.sheduler == "warmup":
            scheduler.step() 


def train_ricap(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()

        I_x, I_y = input.size()[2:]

        w = int(np.round(I_x * np.random.beta(0.3, 0.3)))
        h = int(np.round(I_y * np.random.beta(0.3, 0.3)))
        w_ = [w, I_x - w, w, I_x - w]
        h_ = [h, h, I_y - h, I_y - h]

        cropped_images = {}
        c_ = {}
        W_ = {}
        for k in range(4):
            idx = torch.randperm(input.size(0))
            x_k = np.random.randint(0, I_x - w_[k] + 1)
            y_k = np.random.randint(0, I_y - h_[k] + 1)
            cropped_images[k] = input[idx][:, :, x_k : x_k + w_[k], y_k : y_k + h_[k]]
            c_[k] = target[idx].cuda()
            W_[k] = w_[k] * h_[k] / (I_x * I_y)

        patched_images = torch.cat(
            (
                torch.cat((cropped_images[0], cropped_images[1]), 2),
                torch.cat((cropped_images[2], cropped_images[3]), 2),
            ),
            3,
        )
        patched_images = patched_images.cuda()

        if True:  # amp=True
            with autocast():
                output = model(patched_images)
                loss = sum([W_[k] * criterion(output, c_[k]) for k in range(4)])
        else:
            output = model(patched_images)
            loss = sum([W_[k] * criterion(output, c_[k]) for k in range(4)])

        acc = sum([W_[k] * utils.accuracy(output, c_[k])[0] for k in range(4)])

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info(
                "train %03d %.3f %.2f %.2f", step, objs.avg, top1.avg, top5.avg
            )

    return top1.avg, objs.avg


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # if step % args.report_freq == 0:
        #     logging.info("train %03d %.3f %.2f %.2f", step,
        #                  objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(test_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(test_queue):
            input = input.cuda()
            target = target.cuda()

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    return top1.avg, objs.avg


if __name__ == "__main__":
    main()
