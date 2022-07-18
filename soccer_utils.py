import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import os
from collections import OrderedDict

from utils.utils import (train, validate, build_dataflow, get_augmentor, save_checkpoint)
from utils.video_dataset import VideoDataSet
from utils.dataset_config import get_dataset_config

class SoccerDataset(datasets.ImageFolder):

    # def __init__(self, root, transform, vids=100):
        
    #     self.root = root
    #     self.vids = vids
    #     self.transform = transform
    
    def __getitem__(self, index):
  
        img, label = super(SoccerDataset, self).__getitem__(index)
        
        vid_id = self.imgs[index][0].split('/')[-2]
        
        return (img, label , vid_id)

    # def __len__(self):
    #     return self.vids

def make_vidtrackers(args, root_dir):

    vid_dict = {}

    for classes in os.listdir(root_dir):
        class_path = os.path.join(root_dir, classes)
        for video in os.listdir(class_path):
            vid_dict[video] = {'correct': 0, 'total': 0}

    return vid_dict

def soccer_loaders(args, batch_size=None):

    mean, std = get_transforms(args, mean=None, std=None)
    preprocess = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)])

    labels = ['dribble', 'kick', 'run', 'walk']

    train_root = args.base_path + '/train'
    test_root = args.base_path + '/val'

    augmentor = get_augmentor(True, args.input_size, scale_range=None, mean=mean,
                                    std=std,
                                    disable_scaleup=False,
                                    threed_data=False,
                                    is_flow=False,
                                    version='v1')
    
    train_data = SoccerDataset(root=train_root, transform=preprocess)
    test_data = SoccerDataset(root=test_root, transform=preprocess)


    train_loader = DataLoader(train_data, batch_size=args.bs if None else batch_size, 
                                shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_data, batch_size=args.bs if None else batch_size, 
                                shuffle=False, num_workers=args.workers)

    loaders = {'train':train_loader, 'test':test_loader}

    return loaders, labels

def get_loaders(args, model):

    num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file = get_dataset_config(
        args.dataset)

    mean = model.mean(args.modality)
    std = model.std(args.modality)

    train_list = os.path.join(args.datadir, train_list_name)

    train_augmentor = get_augmentor(True, args.input_size, scale_range=args.scale_range, mean=mean,
                                    std=std,
                                    disable_scaleup=args.disable_scaleup,
                                    threed_data=args.threed_data,
                                    is_flow=True if args.modality == 'flow' else False,
                                    version=args.augmentor_ver)

    train_dataset = VideoDataSet(args.datadir, train_list, args.groups, args.frames_per_group,
                                 num_clips=args.num_clips,
                                 modality=args.modality, image_tmpl=image_tmpl,
                                 dense_sampling=args.dense_sampling,
                                 transform=train_augmentor, is_train=True, test_mode=False,
                                 seperator=filename_seperator, filter_video=filter_video)

    train_loader = build_dataflow(train_dataset, is_train=True, batch_size=args.batch_size,
                                  workers=args.workers, is_distributed=False)

    val_list = os.path.join(args.datadir, val_list_name)

    val_augmentor = get_augmentor(False, args.input_size, scale_range=args.scale_range, mean=mean,
                                  std=std, disable_scaleup=args.disable_scaleup,
                                  threed_data=args.threed_data,
                                  is_flow=True if args.modality == 'flow' else False,
                                  version=args.augmentor_ver)

    val_dataset = VideoDataSet(args.datadir, val_list, args.groups, args.frames_per_group,
                               num_clips=args.num_clips,
                               modality=args.modality, image_tmpl=image_tmpl,
                               dense_sampling=args.dense_sampling,
                               transform=val_augmentor, is_train=False, test_mode=False,
                               seperator=filename_seperator, filter_video=filter_video)

    val_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                workers=args.workers,
                                is_distributed=False)

    return train_loader, val_loader


def expand_model(backbone_args, model):

    fc = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2048, 1024)),
                          ('relu1', nn.ReLU()),
                          ('bn1', nn.BatchNorm1d(1024)),
                          ('fc2', nn.Linear(1024,1024)),
                          ('relu2', nn.ReLU()),
                          ('bn2', nn.BatchNorm1d(1024)),
                          ('fc3', nn.Linear(1024, backbone_args.num_classes)),
                          ]))
    
    model.fc = fc

    return model

def get_criterions(args, front_net):

    if args.loss == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss().to(args.device)
    elif args.loss == 'nll':
        loss_fn = nn.NLLLoss().to(args.device)
    else:
        loss_fn = nn.KLDivLoss(reduction="batchmean").to(args.device)
    
    if args.optim == 'Adam':
        optimizer = optim.Adam(front_net.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(front_net.parameters(), lr=args.lr,
                                  momentum=0.9, weight_decay=5e-4)
    #args.optimizer = optimizer
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    criterions = {'loss':loss_fn, 'optimizer': optimizer, 'scheduler': scheduler}

    return criterions


def save_n_restore_model(args, model, front_net, acc, criterions, optimizer, scheduler, restore):

    if not os.path.exists('./ckpts'):
        os.mkdir('./ckpts')

    if restore:

        if front_net:
            restore_path = './ckpts/front_net/'+ f'{args.eval_ckpt}.pt'
            checkpoint = torch.load(restore_path)
            front_net.load_state_dict(checkpoint['model_state_dict'])
            criterions['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
            criterions['scheduler'].load_state_dict(checkpoint['scheduler_state_dict'])
            loss = checkpoint['loss']

        if model:
            if args.distill_ckpt:
                restore_path = './experiments/resnet18_distill/'+ f'{args.distill_ckpt}_teacher/best.pth.tar'
                checkpoint = torch.load(restore_path)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optim_dict'])
                
            else:
                restore_path = './ckpts/backbone_model/'+ f'{args.eval_ckpt}.pt'
                checkpoint = torch.load(restore_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                loss = checkpoint['loss']

        return model, front_net

    else:

        if front_net:
            save_path = './ckpts/front_net/'+ f'{acc:.3f}_{args.loss}_{args.lr}_{args.log_file}.pt'
            torch.save({
                    'model_state_dict': front_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': criterions['loss'],
                    }, save_path)

        if model:
            save_path = './ckpts/backbone_model/'+ f'{acc:.3f}_{args.loss}_{args.lr}_{args.log_file}.pt'
            torch.save({
                    'model_state_dict': model.state_dict(),
                    'loss': criterions['loss'],
                    }, save_path)

  
def get_transforms(args, mean=None, std=None):

    mean = [0.485, 0.456, 0.406] if mean is None else mean
    std = [0.229, 0.224, 0.225] if std is None else std

    return mean, std