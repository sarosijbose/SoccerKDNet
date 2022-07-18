import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models.frontnet import frontnetbn
import student_models.resnet as resnet
from logger import Logging
import tensorboard_logger
from models.model_builder import build_model
from opts import arg_parser
from soccer_utils import soccer_loaders, expand_model, get_criterions, save_n_restore_model, make_vidtrackers

import numpy as np
import sys
import argparse
import tqdm

def print_args(args, backbone_args):

    print("---BACKBONE CONFIGS---")
    s = "==========================================\n"
    for arg, content in backbone_args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    print(s)

    print("---FINE TUNE CONFIGS---")
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    print(s)

def train_jointnet(args, loaders, criterions, model, front_net, running_loss = 0.0, loss_log=100):

    #cudnn.benchmark = args.cudnn_benchmark
    model.to(args.device).eval()
    model.fc.requires_grad = True

    front_net.to(args.device).train()

    loss_fn, optimizer, scheduler = criterions['loss'], criterions['optimizer'], criterions['scheduler']

    for epoch in tqdm.tqdm(range(args.epochs)):
        for i, (data, labels, _) in enumerate(loaders['train']):
            data, labels = data.to(args.device), labels.to(args.device)
            if args.stand_alone:
                data = data.view(-1, data.shape[-1] * data.shape[-2])
                preds = front_net(model(data))
            else:
                preds = front_net(model(data))
            if args.loss == 'KLD':
                #raise ValueError(preds.shape, preds[-1].shape, labels.shape)
                loss = loss_fn(preds[:0], labels)
            else:
                loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % loss_log == loss_log-1:    # print every 100 mini-batches
                print(f'Epoch:{epoch+1} || loss------> {(running_loss / loss_log):.3f}')
                running_loss = 0.0

        scheduler.step()

    return model, front_net, optimizer, scheduler

def eval_jointnet(args, loaders, model, front_net, tracker):

    model.to(args.device).eval()
    if front_net:
        front_net.to(args.device).eval()

    pbar = tqdm.tqdm(loaders['test'], unit='batches', leave=False, total=len(loaders['test']))

    correct, total, vid_correct, vid_total = 0, 0, 0, 0
    with torch.no_grad():
        for images, labels, vid_id in pbar:
            images, labels = images.to(args.device), labels.to(args.device)
            if front_net:
                outputs = front_net(model(images))
            else:
                outputs = model(images)
            for idx, i in enumerate(outputs):
                if torch.argmax(i) == labels[idx]:
                    tracker[vid_id[idx]]['correct'] += 1
                tracker[vid_id[idx]]['total'] += 1

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    for video, metrics in tracker.items():
        if metrics['correct'] >= (metrics['total']/2):
            vid_correct+=1
        vid_total+=1

    print('~~~ON A FRAME BASIS~~~')
    print(f'No. of correct predictions: {correct} || No. of total samples: {total}')
    print('Accuracy of fine-tuned network on test videos: %.3f %%' % (
        100 * correct / total))

    print('~~~ON A VIDEO BASIS~~~')
    print(f'No. of correct predictions: {vid_correct} || No. of total samples: {vid_total}')
    print('Accuracy of fine-tuned network on test videos: %.3f %%' % (
        100 * vid_correct / vid_total))

    return (100 * correct / total, 100 * vid_correct / vid_total)

def main():

    parser = argparse.ArgumentParser(description='Fine Tune on soccer dataset.')

    parser.add_argument('--base_path', type=str, default='/home/SarosijBose/HAR/KDHAR/soccer/images')
    parser.add_argument('--stand_alone', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=100, help='Train Epochs')
    parser.add_argument('--runs', type=int, default=5, help='Sample results')
    parser.add_argument('--bs', type=int, default=64, help='Batch Size')
    parser.add_argument('--loss', type=str, default='CrossEntropy', choices=['nll', 'CrossEntropy', 'KLD'])
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate')
    parser.add_argument('--workers', type=int, default=12, help='No. of workers')
    parser.add_argument('--input_size', default=224, type=int, metavar='N', help='spatial size')
    parser.add_argument('--gpu',help='Model Choice', default='0')
    parser.add_argument('--save_ckpt', type=bool, default=True)
    parser.add_argument('--eval_only', type=bool, default=True)
    parser.add_argument('--eval_ckpt', type=str, default=None)
    parser.add_argument('--distill_ckpt', type=str, default='jointnet')
    parser.add_argument('--log_file', type=str, default='vid_val_student_ofR101')

    args = parser.parse_args()
    args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    global backbone_args
    backbone_parser = arg_parser()
    backbone_args = backbone_parser.parse_args()
    if backbone_args.dataset == 'kinetics400':
        backbone_args.num_classes = 400

    args.log_dir = './log/' + args.log_file + f'/{args.log_file}.log'
    sys.stdout = Logging(args, args.log_dir)
    print_args(args, backbone_args)

    loaders, labels = soccer_loaders(args, batch_size=args.bs)
    
    val_vidtrackers = make_vidtrackers(args, root_dir=args.base_path + '/val')

    accuracies = []
    best_acc = 0

    if args.eval_only:

        if args.eval_ckpt:

            model, _ = build_model(backbone_args, test_mode=True)
            model = expand_model(backbone_args, model)
            front_net = frontnetbn(stand_alone=args.stand_alone, distill=False)

            criterions = get_criterions(args, front_net)

            model, front_net = save_n_restore_model(args, model, front_net, acc=args.eval_ckpt.split('_')[0], 
                                                    criterions=criterions, optimizer=None, scheduler=None,
                                                    restore=args.eval_only)

            eval_frame_acc, eval_vid_acc = eval_jointnet(args, loaders, model, front_net, tracker=val_vidtrackers)
        
            print(f"Evaluation accuracy obtained on a frame-frame basis: {eval_frame_acc:.3f} %")
            print(f"Evaluation accuracy obtained on video basis: {eval_vid_acc:.3f} %")

        elif args.distill_ckpt:

            student_model = resnet.ResNet18(num_classes=4)
            args.optim == 'SGD'
            criterions = get_criterions(args, student_model)

            student_model, _ = save_n_restore_model(args, model=student_model, front_net=None, acc=None, 
                                                    criterions=criterions, optimizer=criterions['optimizer'], 
                                                    scheduler=None, restore=args.eval_only)


            eval_frame_acc, eval_vid_acc = eval_jointnet(args, loaders, student_model, 
                                                        front_net=None, tracker=val_vidtrackers)
        
            print(f"Evaluation accuracy obtained on a frame-frame basis: {eval_frame_acc:.3f} %")
            print(f"Evaluation accuracy obtained on video basis: {eval_vid_acc:.3f} %")

    else:

        for _ in range(args.runs):

            model, _ = build_model(backbone_args, test_mode=True)
            model = expand_model(backbone_args, model)
            front_net = frontnetbn(stand_alone=args.stand_alone, distill=False)

            criterions = get_criterions(args, front_net)

            model, front_net, optimizer, scheduler = train_jointnet(args, loaders, criterions=criterions, 
                                                model=model, front_net=front_net)
            print('------Training complete------')
            eval_frame_acc, eval_vid_acc = eval_jointnet(args, loaders, model, front_net)
            accuracies.append(eval_frame_acc)
            print('------Evaluation complete. Saving best available checkpoint------')

            if eval_vid_acc > best_acc:
                best_acc = eval_vid_acc
                if args.save_ckpt:
                    save_n_restore_model(args, model, front_net, eval_vid_acc, criterions=criterions, 
                                        optimizer=optimizer, scheduler=scheduler, restore=args.eval_only)

        print(f"Mean accuracy obtained over {args.runs} runs: {np.mean(accuracies):.3f}")
        print(f"Best accuracy obtained over {args.runs} runs: {best_acc:.3f}")
        print([*accuracies])

if __name__ == '__main__':
    main()