import gc
import os
import torch
import time

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import utils as utility

from torch.autograd import Variable as V
from torch.utils.data import DataLoader


def graynet_save(file_path, file_name, graynet, optimizer=None):
    """ Saving Model

        Parameters
        ----------
        file_path : str
            saving file path
        file_name : str
            saving file name
        graynet : str
            pytorch model
        optimizer : torch.optim(.SGD or .Adam)
            optimizer 
    """
    state_dict = {
        'net_arch': 'graynet',
        'model': graynet.state_dict()
    }
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()
    torch.save(state_dict, '{}/{}.pth'.format(file_path, file_name))


def graynet_load(file_path, file_name, model, optimizer):
    """ Loading Model

        Parameters
        ----------
        file_path : str
            saving file path
        file_name : str
            saving file name
        graynet : str
            pytorch model
        optimizer : torch.optim(.SGD or .Adam)
            optimizer 

        Returns
        ----------
        tuple
            tuple of model and optimizer
    """
    check_points = torch.load('{}/{}.pth'.format(file_path, file_name))
    keys = check_points.keys()

    assert ('net_arch' in keys) and ('model' in keys) and (
        'optimizer' in keys), 'Cannot read this file in address : {}/{}.pth'.format(file_path, file_name)
    assert check_points['net_arch'] == 'graynet', 'This file model architecture is not \'graynet\''
    model.load_state_dict(check_points['model'])
    optimizer.load_state_dict(check_points['optimizer'])
    return model, optimizer


def graynet_train(graynet,
                  train_data_loader,
                  optimizer,
                  criterion,
                  device,
                  report_path,
                  num_epoch=1,
                  start_epoch=0,
                  batch_size=2,
                  num_workers=1,
                  gpu=False):
    """ Train Model and saving results in reports path

        Parameters
        ----------
        graynet : graynet(nn.Module)
            pytorch model
        train_data_loader : CamVidLoader(Dataset) or any thing else
            data loader
        optimizer : torch.optim(.SGD or .Adam)
            optimizer 
        criterion : torch.nn(.CrossEntropyLoss() or .MSELoss()) or any thing else
            loss function
        device : torch.device
            model and tensors device
        report_path : str
            saving path of training losses, accuracies, and batches size
        num_epoch : int 
            num of epochs for training
        start_epoch : int
            first epoch number
        batch_size : int 
            batch size of model
        num_workers : int
            num of workers for loading data
        gpu : Boolean
            Using gpu or not
    """

    utility.mkdir(os.path.split(report_path)[0], os.path.split(
        report_path)[1], forced_remove=False)
    utility.mkdir(report_path, 'models', forced_remove=False)
    utility.mkdir(report_path, 'train_losses', forced_remove=False)
    utility.mkdir(report_path, 'train_batches_size', forced_remove=False)

    for epoch in range(start_epoch, start_epoch+num_epoch):
        data_loader = DataLoader(train_data_loader,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 pin_memory=gpu and torch.cuda.is_available(),
                                 num_workers=num_workers)

        loss_item = 0
        losses = list()
        batches_size = list()

        for ix, (X, Y) in enumerate(data_loader):
            X = X.permute(0, 3, 1, 2)
            Y = torch.unsqueeze(Y, dim=1)

            X, Y = V(X), V(Y)

            if device != 'cpu' and gpu and torch.cuda.is_available():
                if device.split(':')[0] == 'cuda':
                    X, Y = X.cuda(device=device), Y.cuda(device=device)
                elif device == 'multi':
                    X, Y = nn.DataParallel(X), nn.DataParallel(Y)

            output = graynet(X)

            prev_loss = loss_item
            loss = criterion(output, Y)
            loss_item = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss_item)
            batches_size.append(X.size()[0])
            print('epoch: %d, batch: %d(x%d), prev loss:%.4d , curr loss: %.4d, D(loss)=%.4d' % (
                epoch,
                ix,
                batch_size,
                prev_loss,
                loss_item,
                loss_item-prev_loss
            )
            )

            del X, Y, output
            torch.cuda.empty_cache()
            gc.collect()

        utility.save_np_file(
            '{}/train_losses/train_losses_epoch_{}'.format(report_path, epoch), losses)
        utility.save_np_file(
            '{}/train_batches_size/train_batches_size_epoch_{}'.format(report_path, epoch), batches_size)

        del losses, batches_size
        torch.cuda.empty_cache()
        gc.collect()


def graynet_eval(graynet,
                 eval_data_loader,
                 optimizer,
                 criterion,
                 device,
                 report_path,
                 eval_mode,
                 epoch=None,
                 batch_size=2,
                 num_workers=1,
                 gpu=True):
    """ Evaluating Model for an epoch and saving results in reports path

        Parameters
        ----------
        graynet : graynet(nn.Module)
            pytorch model
        eval_data_loader : CamVidLoader(Dataset) or any thing else
            data loader
        optimizer : torch.optim(.SGD or .Adam)
            optimizer 
        criterion : torch.nn(.CrossEntropyLoss() or .MSELoss()) or any thing else
            loss function
        device : torch.device
            model and tensors device
        report_path : str
            saving path of evaluating losses, accuracies, and batches size
        eval_mode : str
            "val" or "test"
        epoch : int or None
            an integer for validation and any value for testing that corresponds to current epoch.
        batch_size : int 
            batch size of model
        num_workers : int
            num of workers for loading data
        gpu : Boolean
            Using gpu or not
    """
    assert eval_mode == 'val' or eval_mode == 'test', 'Invalid evaluation mode.'
    assert eval_mode == 'val' and epoch is not None and type(
        epoch) == int, 'In validation mode, epoch must be an integer value.'

    utility.mkdir(os.path.split(report_path)[0], os.path.split(
        report_path)[1], forced_remove=False)
    utility.mkdir(report_path, '{}_losses'.format(
        eval_mode), forced_remove=False)
    utility.mkdir(report_path, '{}_batches_size'.format(
        eval_mode), forced_remove=False)

    data_loader = DataLoader(eval_data_loader,
                             batch_size=batch_size,
                             shuffle=False,
                             pin_memory=gpu and torch.cuda.is_available(),
                             num_workers=num_workers)

    loss_item = 0
    losses = list()
    batches_size = list()

    for ix, (X, Y) in enumerate(data_loader):
        X = X.permute(0, 3, 1, 2)
        Y = Y.permute(0, 3, 1, 2)

        X, Y = V(X, requires_grad=False), V(Y, requires_grad=False)

        if device != 'cpu' and gpu and torch.cuda.is_available():
            if device.split(':')[0] == 'cuda':
                X, Y = X.cuda(device=device), Y.cuda(device=device)
            elif device == 'multi':
                X, Y = nn.DataParallel(X), nn.DataParallel(Y)

        output = graynet(X)

        prev_loss = loss_item
        loss = criterion(output, Y)
        loss_item = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss_item)
        batches_size.append(X.size()[0])
        print('batch = %d(x%d), prev loss:%.4d , curr loss: %.4d, D(loss)=%.4d' % (
            ix,
            batch_size,
            prev_loss,
            loss_item,
            loss_item-prev_loss
        )
        )

        del X, Y, output
        torch.cuda.empty_cache()
        gc.collect()

    utility.save_np_file('{}/{}_losses/{}_losses{}'.format(report_path, eval_mode,
                                                           eval_mode, '_epoch_{}'.format(epoch) if eval_mode == 'val' else ''), losses)
    utility.save_np_file('{}/{}_batches_size/{}_batches_size{}'.format(report_path, eval_mode,
                                                                       eval_mode, '_epoch_{}'.format(epoch) if eval_mode == 'val' else ''), batches_size)

    del losses, batches_size
    torch.cuda.empty_cache()
    gc.collect()


def graynet_weight_init(graynet):
    """Initializing wieghts
        Parameters
        ----------
        graynet : graynet(nn.Module)
            pytorch model
        Returns
        ----------
        uent : graynet(nn.Module)
            pytorch model
    """
    if isinstance(graynet, nn.Conv2d):
        nn.init.xavier_normal(graynet.weight.data)
        nn.init.constant(graynet.bias, 0)
    elif isinstance(graynet, nn.BatchNorm2d):
        graynet.weight.data.fill_(1)
        graynet.bias.data.zero_()
    return graynet
