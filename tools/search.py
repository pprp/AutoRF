import argparse
import glob
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# from space.genotypes import PRIMITIVES


import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms

import utils.utils as utils
from torch.autograd import Variable

from search.architect import Architect
from search.supernet import Network



parser = argparse.ArgumentParser("cifar")
parser.add_argument(
    "--data", type=str, default="/data/public/cifar", help="location of the data corpus"
)
parser.add_argument(
    "--dataset", type=str, default="cifar10", help="cifar10 or cifar100 or imagenet"
)

parser.add_argument("--primitives", type=str, default="fullpool", help="choose in autola, fullpool, fullconv, hybrid")
parser.add_argument("--model_name", type=str, default="searching", help="backbone")
parser.add_argument("--comments", type=str, default="cifar100", help="backbone")


parser.add_argument("--batch_size", type=int, default=128, help="batch size")
parser.add_argument(
    "--learning_rate", type=float, default=0.025, help="init learning rate"
)
parser.add_argument(
    "--learning_rate_min", type=float, default=0.0001, help="min learning rate"
)
parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
parser.add_argument("--report_freq", type=float, default=200, help="report frequency")
parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
parser.add_argument("--epochs", type=int, default=50, help="num of training epochs")
parser.add_argument(
    "--model_path", type=str, default="saved_models", help="path to save the model"
)
parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
parser.add_argument("--cutout_length", type=int, default=8, help="cutout length")
parser.add_argument(
    "--drop_path_prob", type=float, default=0.3, help="drop path probability"
)
parser.add_argument("--save", type=str, default="EXP", help="experiment name")
parser.add_argument("--seed", type=int, default=2, help="random seed")
parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
parser.add_argument(
    "--train_portion", type=float, default=0.5, help="portion of training data"
)
parser.add_argument(
    "--unrolled",
    action="store_true",
    default=False,
    help="use one-step unrolled validation loss",
)
parser.add_argument(
    "--arch_learning_rate",
    type=float,
    default=1e-4,
    help="learning rate for arch encoding",
)
parser.add_argument(
    "--arch_weight_decay",
    type=float,
    default=1e-3,
    help="weight decay for arch encoding",
)
args = parser.parse_args()

args.save = "{}-{}-{}-{}".format(
    args.comments, args.model_name, args.save, time.strftime("%Y%m%d-%H%M%S")
)
args.save = os.path.join("exps/search", args.save)

utils.create_exp_dir(args.save, scripts_to_save=glob.glob("*/*.py"))

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(args.save, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

if args.dataset == "cifar10":
    NUM_CLASSES = 10
elif args.dataset == "cifar100":
    NUM_CLASSES = 100 
elif args.dataset == "imagenet":
    NUM_CLASSES = 1000

def main():
    if not torch.cuda.is_available():
        logging.info("no gpu device available")
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = False
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("gpu device = %d" % args.gpu)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    model = Network(num_classes=NUM_CLASSES, model_name=args.model_name, primitives=args.primitives)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    train_transform, valid_transform = utils._data_transforms_cifar(args)

    if args.dataset == "cifar10":
        train_data = dset.CIFAR10(
            root=args.data, train=True, download=True, transform=train_transform
        )
    elif args.dataset == "cifar100":
        train_data = dset.CIFAR100(
            root=args.data, train=True, download=True, transform=train_transform
        )
    elif args.dataset == "imagenet":
        train_data = dset.ImageFolder(
            args.data,
            transform=transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing(), 
    ]))

        valid_data = dset.ImageFolder(args.data, transform=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]))

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True,
        num_workers=8,
    )

    valid_queue = torch.utils.data.DataLoader(
        train_data if args.dataset is not "imagenet" else valid_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True,
        num_workers=8,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min
    )

    architect = Architect(model, args)

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info("epoch = %d lr = %e", epoch, lr)

        genotype = model.genotype()
        logging.info("genotype = %s", genotype)

        print(F.softmax(model.alphas_normal, dim=-1))

        train_acc, train_obj = train(
            train_queue, valid_queue, model, architect, criterion, optimizer, lr
        )
        logging.info("train_acc = %f", train_acc)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info("valid_acc = %f", valid_acc)

        utils.save(model, os.path.join(args.save, "weights.pt"))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        model.train()
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()

        input_search, target_search = next(iter(valid_queue))
        input_search = Variable(input_search, requires_grad=False).cuda()
        target_search = Variable(target_search, requires_grad=False).cuda()

        architect.step(
            input,
            target,
            input_search,
            target_search,
            lr,
            optimizer,
            unrolled=args.unrolled,
        )

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
            logging.info("train step:%03d loss:%.4f top1:%.3f top5:%.3f", step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    # model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda()

        logits = model(input)
        loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info("valid step:%03d loss:%.4f top1:%.3f top5:%.3f", step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == "__main__":
    main()
