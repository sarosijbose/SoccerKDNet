import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from tqdm import tqdm

import kd_utils as utils
import student_models.net as net
import student_models.data_loader as data_loader
import student_models.resnet as resnet
from evaluate import evaluate, evaluate_kd

from soccer_utils import (soccer_loaders, get_loaders, expand_model, get_criterions, 
                            save_n_restore_model, make_vidtrackers)
from models.model_builder import build_model
from models.frontnet import frontnetbn
from opts import arg_parser

def train_kd(args, model, backbone_model, front_net, teacher_model, optimizer, loss_fn_kd, dataloader, metrics, params):
    
    model.train()
    if teacher_model:
        teacher_model.eval()
    if backbone_model:
        backbone_model.to(args.device).eval()
    if front_net:
        front_net.to(args.device).eval()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch, _) in enumerate(dataloader):
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(non_blocking=True), \
                                            labels_batch.cuda(non_blocking=True)
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            output_batch = model(train_batch)

            with torch.no_grad():
                if args.dataset == 'Kinetics400':
                    output_teacher_batch = teacher_model(train_batch)
                elif args.dataset == 'Soccer':
                    output_teacher_batch = front_net(backbone_model(train_batch))
            if params.cuda:
                output_teacher_batch = output_teacher_batch.cuda(non_blocking=True)

            loss = loss_fn_kd(output_batch, labels_batch, output_teacher_batch, params)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data.cpu().numpy()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data)

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate_kd(args, model, teacher_model, train_dataloader, val_dataloader, optimizer,
                       loss_fn_kd, metrics, params, model_dir, backbone_model, front_net):
    """Train the model and evaluate every epoch.

    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)"""

    best_val_acc = 0.0
    
    # Tensorboard logger setup
    # board_logger = utils.Board_Logger(os.path.join(model_dir, 'board_logs'))

    if params.model_version == "resnet18_distill":
        scheduler = StepLR(optimizer, step_size=150, gamma=0.1)

    for epoch in range(params.num_epochs):

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        if args.dataset == 'Kinetics400':
            train_kd(args, model, backbone_model, front_net, teacher_model, optimizer, loss_fn_kd, 
                        train_dataloader, metrics, params)
        elif args.dataset == 'Soccer':
            train_kd(args, model, backbone_model, front_net, teacher_model, optimizer, loss_fn_kd, 
                        train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate_kd(model, val_dataloader, metrics, params)

        val_acc = val_metrics['accuracy']
        is_best = val_acc>=best_val_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

        scheduler.step()


        # #============ TensorBoard logging: uncomment below to turn in on ============#
        # # (1) Log the scalar values
        # info = {
        #     'val accuracy': val_acc
        # }

        # for tag, value in info.items():
        #     board_logger.scalar_summary(tag, value, epoch+1)

        # # (2) Log values and gradients of the parameters (histogram)
        # for tag, value in model.named_parameters():
        #     tag = tag.replace('.', '/')
        #     board_logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
        #     # board_logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)

def main():

    parser = argparse.ArgumentParser(description='Train various student models.')
    
    parser.add_argument('--base_path', type=str, default='/home/SarosijBose/HAR/KDHAR/soccer/images')
    parser.add_argument('--stand_alone', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='Soccer', choices=['Kinetics400', 'Soccer'])
    parser.add_argument('--epochs', type=int, default=100, help='Train Epochs')
    parser.add_argument('--bs', type=int, default=64, help='Batch Size')
    parser.add_argument('--loss', type=str, default='CrossEntropy', choices=['nll', 'CrossEntropy', 'KLD'])
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--workers', type=int, default=12, help='No. of workers')
    parser.add_argument('--gpu',help='Model Choice', default='0')
    parser.add_argument('--input_size', default=224, type=int, metavar='N', help='spatial size')
    parser.add_argument('--eval_ckpt', type=str, default='58.863_CrossEntropy_0.0001_train_n_val6')
    parser.add_argument('--model_dir', default='experiments/resnet18_distill/jointnet_teacher')
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir \
                        containing weights to reload before training")

    args = parser.parse_args()
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    # Load the parameters from json file
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    global backbone_args
    backbone_parser = arg_parser()
    backbone_args = backbone_parser.parse_args()
    if backbone_args.dataset == 'kinetics400':
        backbone_args.num_classes = 400
    if args.dataset == 'Soccer':
        args.num_classes = 4
    else:
        args.num_classes = 400

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    random.seed(230)
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    val_vidtrackers = make_vidtrackers(args, root_dir=args.base_path + '/val')
    
    if params.model_version == 'resnet18_distill':
        model = resnet.ResNet18(num_classes=args.num_classes).cuda() if params.cuda else resnet.ResNet18()
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                                momentum=0.9, weight_decay=5e-4)
        # fetch loss function and metrics definition in model files
        loss_fn_kd = net.loss_fn_kd
        metrics = resnet.metrics

    """ 
        Specify the pre-trained teacher models for knowledge distillation
    """
    if params.teacher == "tam":
        teacher_model, _ = build_model(backbone_args, test_mode=True)
        teacher_model = teacher_model.cuda() if params.cuda else teacher_model
        front_net = backbone_model = None
    elif params.teacher == "jointnet":
        backbone_model, _ = build_model(backbone_args, test_mode=True)
        backbone_model = expand_model(backbone_args, backbone_model)
        front_net = frontnetbn(stand_alone=args.stand_alone, distill=True)

        criterions = get_criterions(args, front_net)

        args.distill_ckpt = False
        backbone_model, front_net = save_n_restore_model(args, backbone_model, front_net, acc=args.eval_ckpt.split('_')[0], 
                                                criterions=criterions, optimizer=None, scheduler=None,
                                                restore=True)
        #teacher_model = front_net(model()).cuda() if params.cuda else teacher_model
        teacher_model = None

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    if args.dataset == 'Kinetics400':
        train_dl, dev_dl = get_loaders(args=backbone_args, model=teacher_model)
    elif args.dataset == 'Soccer':
        loaders, labels = soccer_loaders(args, batch_size=args.bs)
        train_dl, dev_dl = loaders['train'], loaders['test']

    logging.info("- done.")

    # Train the model with KD
    logging.info("Experiment - model version: {}".format(params.model_version))
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    logging.info("First, loading the teacher model and computing its outputs...")

    if args.dataset == 'Kinetics400':
        train_and_evaluate_kd(args, model, teacher_model, train_dl, dev_dl, optimizer, loss_fn_kd,
                            metrics, params, args.model_dir, backbone_model, front_net)
    else:
        train_and_evaluate_kd(args, model, teacher_model, train_dl, dev_dl, optimizer, loss_fn_kd,
                            metrics, params, args.model_dir, backbone_model=backbone_model, front_net=front_net)

if __name__ == '__main__':
    main()