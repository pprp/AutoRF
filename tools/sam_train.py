import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import space.genotypes as genotypes
import torch
import torchvision
from retrain.studentnet import Network
from timm.loss import LabelSmoothingCrossEntropy
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from utils.utils import accuracy, AvgrageMeter
from thop import profile,clever_format

# from homura.vision.models.cifar_resnet import wrn28_2, wrn28_10, resnet20, resnet56, resnext29_32x4d
from utils.asam import ASAM, SAM
from utils.utils import Cutout

def load_cifar(data_loader, batch_size=256, num_workers=4):
    if data_loader == CIFAR10:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    # Transforms
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_transform.transforms.append(Cutout(8))

    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    # DataLoader
    train_set = data_loader(
        root="/data/public/cifar", train=True, download=True, transform=train_transform
    )
    test_set = data_loader(
        root="/data/public/cifar", train=False, download=True, transform=test_transform
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


def train(args):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    # Data Loader
    train_loader, test_loader = load_cifar(eval(args.dataset), args.batch_size)
    num_classes = 10 if args.dataset == "CIFAR10" else 100

    print(f"running cifar-{num_classes}")

    # Model
    # model = eval(args.model)(num_classes=num_classes).cuda()
    model = Network(
        args.model, num_classes, eval("genotypes.%s" % args.arch), dropout=0.0
    ).cuda()

    input_for_profile = torch.randn(1, 3, 32, 32).cuda()
    macs, params = profile(model, inputs=(input_for_profile,))
    info = f"flops:{macs/1000**3}G params: {params/1000**2}M"
    print(info)
    macs, params = clever_format([macs, params], "%.3f")
    print(f"flops:{macs}  params:{params}")

    # Minimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    minimizer = eval(args.minimizer)(optimizer, model, rho=args.rho, eta=args.eta)

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        minimizer.optimizer, args.epochs
    )

    # Loss Functions
    if args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0.0
    for epoch in range(args.epochs):    
        # reset 
        objs.reset() 
        top1.reset() 
        top5.reset() 

        # Train
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            n = inputs.size(0)

            # Ascent Step
            predictions = model(inputs)
            batch_loss = criterion(predictions, targets)
            batch_loss.mean().backward()
            minimizer.ascent_step()

            # Descent Step
            criterion(model(inputs), targets).mean().backward()
            minimizer.descent_step()

            with torch.no_grad():
                prec1, prec5 = accuracy(predictions, targets, topk=(1, 5))
                objs.update(batch_loss.sum().item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

        print(
            f"Epoch: {epoch}, Train acc1: {top1.avg:6.2f} %, Train acc5: {top5.avg:6.2f} %, Train loss: {objs.avg:8.5f}"
        )
        scheduler.step()

        # Test
        model.eval()
        # Reset
        objs.reset() 
        top1.reset() 
        top5.reset() 

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.cuda()
                targets = targets.cuda()
                n = inputs.size(0)

                predictions = model(inputs)

                loss = criterion(predictions, targets)
                prec1, prec5 = accuracy(predictions, targets, topk=(1, 5))

                objs.update(loss.sum().item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

        if best_accuracy < prec1.item():
            best_accuracy = prec1.item()
        print(
            f"Epoch: {epoch}, Test acc1:  {top1.avg:6.2f} %, Test acc5: {top5.avg:6.2f} %, Test loss:  {objs.avg:8.5f}"
        )
    print(f"Best test accuracy: {best_accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="CIFAR100", type=str, help="CIFAR10 or CIFAR100."
    )
    parser.add_argument(
        "--model", default="rf_resnet56", type=str, help="Name of model architecure"
    )
    parser.add_argument("--arch", default="P1", type=str, help="Name of model arch")
    parser.add_argument("--minimizer", default="ASAM", type=str, help="ASAM or SAM.")
    parser.add_argument("--lr", default=0.1, type=float, help="Initial learning rate.")
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum.")
    parser.add_argument(
        "--weight_decay", default=5e-4, type=float, help="Weight decay factor."
    )
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    parser.add_argument("--epochs", default=200, type=int, help="Number of epochs.")
    parser.add_argument("--smoothing", default=0.1, type=float, help="Label smoothing.")
    parser.add_argument("--rho", default=0.5, type=float, help="Rho for ASAM.")
    parser.add_argument("--eta", default=0.0, type=float, help="Eta for ASAM.")
    args = parser.parse_args()
    assert args.dataset in [
        "CIFAR10",
        "CIFAR100",
    ], f"Invalid data type. Please select CIFAR10 or CIFAR100"
    assert args.minimizer in [
        "ASAM",
        "SAM",
    ], f"Invalid minimizer type. Please select ASAM or SAM"
    train(args)
