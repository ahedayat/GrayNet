import os
import torch
import warnings

import data_loader.pascalvoc12 as voc12
import torch.nn as nn
import torch.optim as optim
import numpy as np
import utils as util

import torchvision.transforms as transforms

import nets.graynet as graynet


def _main(args):
    annalysis_num = args.analysis

    output_channel, mid_channel, input_channel = (1, 128, 3)
    data_transform = transforms.Compose([transforms.Resize((224, 224))])
    target_transform = transforms.Compose([transforms.Resize((224, 224))])

    train_data_path, val_data_path, test_data_path = [
        './datasets/pascalvoc12/{}'.format(data_type) for data_type in ['train', 'val', 'test']]
    annotations_version = 2

    loading_model_path, loading_model_name = ('./models', 'graynet')
    saving_model_path, saving_model_name = ('./models', 'graynet')

    reports_path = './reports'

    torch.manual_seed(0)
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    warnings.filterwarnings("ignore")

    train_data, val_data = [voc12.loader(
        data_path,
        data_transform=data_transform,
        target_transform=target_transform,
        shuffle_data=False)
        for data_path in [train_data_path, val_data_path]]

    # train_data, val_data, test_data = [voc12.loader(
    #     data_path,
    #     data_transform=data_transform,
    #     target_transform=target_transform,
    #     shuffle_data=False)
    #     for data_path in [train_data_path, val_data_path, test_data_path]]

    model = None

    model = graynet.Model(output_channel=output_channel,
                          mid_channel=mid_channel,
                          input_channel=input_channel)

    device = torch.device(
        "cuda:0" if args.gpu and torch.cuda.is_available() else "cpu")

    if args.gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
        model = model.cuda(device=device)

    criterion = nn.MSELoss()

    assert args.optimization in [
        'adam', 'sgd'], 'Uknown optimization algorithm. for optimization can use adam and sgd.'
    if args.optimization == 'sgd'or annalysis_num == 3:
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr, momentum=args.momentum)
    elif args.optimization == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.load and os.path.isdir(loading_model_path) and os.path.isfile('./{}/{}.pth'.format(loading_model_path, loading_model_name)):
        model, optimizer = graynet.load(
            loading_model_path, loading_model_name, model, optimizer)

    if args.start_epoch != 0:
        model_path = './reports/analysis_{}/models'.format(annalysis_num)
        model_name = 'unet_epoch_{}'.format(args.start_epoch-1)
        model, optimizer = graynet.load(
            model_path, model_name, model, optimizer)

    graynet.train(graynet=model,
                  train_data_loader=train_data,
                  optimizer=optimizer,
                  criterion=criterion,
                  device=device,
                  report_path='{}/{}'.format(reports_path, annalysis_num),
                  num_epoch=args.epochs,
                  start_epoch=args.start_epoch,
                  batch_size=args.batchsize,
                  num_workers=args.num_workers,
                  gpu=args.gpu)

    if args.save:
        if not os.path.isdir(saving_model_path):
            os.makedirs(saving_model_path)
        graynet.save(saving_model_path, saving_model_name, model, optimizer)


if __name__ == '__main__':
    args = util.get_args()
    _main(args)
