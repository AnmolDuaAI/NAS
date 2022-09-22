import enum
import os
import sched
import sys
import time
import glob
from importlib_metadata import requires
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search import Network
from architect import Architect

import config


def MakeDir(path):
    if (not os.path.exists(path)):
        os.mkdir(path)

def set_seeds(seedv):
    np.random.seed(seedv)
    torch.manual_seed(seedv)
    torch.cuda.manual_seed(seedv)

def initialize_loss():
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    return criterion

def initialize_model(criterion):
    model = Network(
        C = config.init_channels,
        num_classes=10,
        layers = config.layers,
        criterion = criterion
    )

    model = model.cuda()
    return model

def initialize_optimizer(model):
    optimizer = torch.optim.SGD(
        model.parameters(),
        config.lr,
        momentum = config.momentum,
        weight_decay = config.weight_decay
    )

    return optimizer

def getLoaders():
    train_transform, valid_transform = utils._data_transforms_cifar10()
    train_data = dset.CIFAR10(root = config.data, train = True, download = True, transform = train_transform)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(config.train_prop * num_train))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size = config.batch_size,
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory = True,
        num_workers = 2
    )

    valid_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size = config.batch_size,
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory = True,
        num_workers = 2
    )

    return train_loader, valid_loader


def train(train_loader, valid_loader, model, architect, criterion, optimizer, lr):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    for step, (input, target) in enumerate(train_loader):
        model.train()
        n = input.size(0)
        input = Variable(input, requires_grad = False).cuda()
        target = Variable(target, requires_grad = False).cuda()
        input_search, target_search = next(iter(valid_loader))
        input_search = Variable(input_search, requires_grad = False).cuda()
        target_search = Variable(target_search, requires_grad = False).cuda()

        architect.step(
            input,
            target,
            input_search,
            target_search,
            lr,
            optimizer,
            unrolled = config.unrolled # Unrolling
        )

        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()

        # Ignoring gradient clipping for now

        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk = (1,5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if (step % config.report_freq == 0):
            print ("Training :::  Step : " + str(step) + " Loss : " + str(objs.avg) + " Top1 : " + str(top1.avg) + " Top5 : " + str(top5.avg))

    return top1.avg, objs.avg

def infer(valid_loader, model, criterion):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.eval()

    for step, (input,target) in enumerate(valid_loader):
        input = Variable(input, volatile = True).cuda()
        target = Variable(target, volatile = True).cuda() # ASync parameter not working

        logits = model(input)
        loss = criterion(logits, target)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1,5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if (step % config.report_freq):
            print ("Training :::  Step : " + str(step) + " Loss : " + str(objs.avg) + " Top1 : " + str(top1.avg) + " Top5 : " + str(top5.avg))

    return top1.avg, objs.avg
        


def main():
    MakeDir(config.save)

    print ("Setting Seeds")
    set_seeds(config.seed)
    torch.cuda.set_device(config.gpu_device_id)
    cudnn.benchmark = True

    print ("Initializing Loss")
    criterion = initialize_loss()

    print ("Initializing Model!")
    model = initialize_model(criterion)

    print ("Initializing Optimizer!")
    optimizer = initialize_optimizer(model)

    print ("Getting Loaders!")
    train_loader, valid_loader = getLoaders()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        float(config.epochs),
        eta_min = config.lr_min
    )

    print ("Initializing Architect!")
    architect = Architect(model)

    print ("Start Training-----")

    for epoch in range(config.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        print ("Epoch : " + str(epoch) + " :: " + str(lr))
        genotype = model.genotype()
        print ("GenoType : " + str(genotype))

        print (F.softmax(model.alphas_normal, dim = -1))
        print (F.softmax(model.alphas_reduce, dim = -1))

        # Training part
        train_acc, train_obj = train(train_loader, valid_loader, model, architect, criterion, optimizer, lr)
        print ("Train Accuracy : " + str(train_acc))

        valid_acc, valid_obj = infer(valid_loader, model, criterion)
        print ("Valid Accuracy : " + str(valid_acc))

        utils.save(model, os.path.join(config.save, 'weights.pt'))

if (__name__ == "__main__"):
    main()