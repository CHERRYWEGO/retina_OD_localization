from __future__ import print_function, absolute_import

import os
import argparse
import random
import time
# import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

from evaluation.fashionloss import EyeMeter
from pose import Bar
from pose.datasets.eye import eye_data, eye_testdata
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import accuracy, AverageMeter, final_preds, get_preds
from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate, save_checkpoint2
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap, load_image
from pose.utils.transforms import fliplr, flip_back, color_normalize
import pose.models as models
import pose.datasets as datasets

import numpy as np
import pandas as pd
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def main(args):
    print("==> creating model '{}', stacks={}, blocks={}".format(args.arch, args.stacks, args.blocks))
    model = models.__dict__[args.arch](num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.num_classes)

    model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=args.lr)

    cudnn.benchmark = True

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # train_data = eye_data(args.data_path, args.data_label_path)
    # test_data = eye_data(args.data_path_test, args.data_label_test_path)
    test_data = eye_testdata(args.data_path_test)
    
    # # Data loading code
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.Eye(df=train_data, img_folder=args.data_path, 
    #     mask_folder=args.data_label_path, sigma=args.sigma),
    #     batch_size=args.train_batch, shuffle=True,
    #     num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        # datasets.Eye(df=test_data, img_folder=args.data_path_test,
        datasets.Eyetest(df=test_data, img_folder=args.data_path_test, 
        # mask_folder=args.data_label_test_path, 
        sigma=args.sigma, test_condition=True),
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # optionally resume from a checkpoint
    args.evaluate = True
    if args.evaluate:
        print('\nEvaluation only')
        model.module.load_state_dict(torch.load('OMIA.pth'))
        valid_loss = validate(val_loader, model, criterion, args.num_classes, args.strucloss_alpha,
                              args.debug, args.flip)
        return

    lr = args.lr
    # break_point = 0
    # best_loss = 100000
    # epoch = 0

    best_loss = 0.0000

    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))
        # decay sigma
        if args.sigma_decay > 0:
            train_loader.dataset.sigma *= args.sigma_decay
            val_loader.dataset.sigma *= args.sigma_decay

        # train for one epoch
        train_loss= train(train_loader, model, criterion, optimizer, args, args.debug, args.flip)

        # evaluate on validation set
        valid_loss = validate(val_loader, model, criterion, args.num_classes, args.strucloss_alpha,
                                                      args.debug, args.flip)

        if valid_loss > best_loss:
            best_loss = valid_loss
            real_model = model.module
            torch.save(real_model.state_dict(), str(round(valid_loss, 4)) + "_model.pth")
    #
    real_model = model.module
    torch.save(real_model.state_dict(), "model_last.pth")


def train(train_loader, model, criterion, optimizer, args, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # acces = AverageMeter()
    true_loss = EyeMeter()

    # switch to train mode
    model.train()

    end = time.time()
    # for class condition
    bar = Bar('Processing', max=len(train_loader))
    for i, (inputs, target, meta) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)
        input_var = inputs.cuda()
        target_var = target.cuda()

        # save input_image
        # imgs = input_var.cpu().numpy()

        # compute output
        output = model(input_var)

        # final_output = output[-1].clone()
        # for index in range(1, len(output)):
        #     # if index % 2 != 0:
        #     final_output += output[index].clone() * index

        loss = criterion(output[0], target_var)
        for j in range(1, len(output)):
            loss += criterion(output[j], target_var)
            # loss += criterion(output[j], target_var)
        # measure accuracy and record loss
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inputs.size(0))
        true_loss.update_box(output[-1].clone(), meta, target, True)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
    # print('({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.5f} |Trueloss: {true_loss:.4f} | Trueloss_OD: {true_loss_od:.4f} | Trueloss_F: {true_loss_f:.4f}'.format(
    print('({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.5f} |Trueloss: {true_loss:.4f} '.format(
            batch=i + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            # true_loss_od = true_loss.real_avg_od,
            # true_loss_f = true_loss.real_avg_fovea,
            true_loss = true_loss.real_avg
        ))
    return true_loss.real_avg


def validate(val_loader, model, criterion, num_classes, alpha, debug=False, flip=True):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # acces = AverageMeter()
    true_loss = EyeMeter()
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        gt_win, pred_win = None, None
        end = time.time()
        bar = Bar('Processing', max=len(val_loader))
        # for i, (inputs, target, meta) in enumerate(tqdm(val_loader)):
        for i, (inputs, meta) in enumerate(tqdm(val_loader)):
            # measure data loading time
            data_time.update(time.time() - end)
            input_var = inputs.cuda()
            # target_var = target.cuda()

            # save input_image
            # imgs = input_var.cpu().numpy()
            # np.save('imgs_{}'.format(i), imgs)

            # compute output
            end = time.time()
            output = model(input_var)
            batch_time.update(time.time() - end)
            # loss = criterion(output[0], target_var)

            # for o in output[1:]:
            #     loss += criterion(o, target_var)
            # true_loss.update_box(output[-1], meta, target, True)
            true_loss.update_res(output[-1], meta, bilinear_test=True)
            # true_loss.update_heatmap(output[-1], meta, target)
            # true_loss.update_circle(output[-1], meta, True)
            # output[-1].cpu().numpy()

            # measure accuracy and record loss
            # losses.update(loss.item(), inputs.size(0))
            # acces.update(acc[0], inputs.size(0))

            # measure elapsed time

        loc_res = pd.DataFrame(true_loss.result, columns=['file', 'OD_x', 'OD_y'])
        loc_res.to_csv("OD2019.csv", index=0)

            # plot progress
        print('({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.5f} |Trueloss: {true_loss:.4f}'.format(
        # print('({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.5f} | True_Loss: {true_loss:.4f} | box: {box:.4f} | oval: {oval:.4f} | r8: {r8:.4f} | r4: {r4:.4f} | r2: {r2:.4f} | r1: {r1:.4f} | dis: {dis:.4f}'.format(
                    # print('({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.5f} | True_Loss: {true_loss:.4f} | IOU: {true_real_loss:.4f}'.format(
                batch=i + 1,
                size=len(val_loader),
                data=data_time.val,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                # true_loss_od=true_loss.real_avg_od,
                # true_loss_f=true_loss.real_avg_fovea,
                true_loss = true_loss.real_avg,
                # true_real_loss = true_loss.real_avg_iou
                # box=true_loss.real_avg_box,
                # oval=true_loss.real_avg_oval,
                # r8=true_loss.real_avg_dis8,
                # r4=true_loss.real_avg_dis4,
                # r2=true_loss.real_avg_dis2,
                # r1=true_loss.real_avg_dis1,
                # dis=true_loss.real_avg_dis
        ))
        return true_loss.real_avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OD_localization')
    # Model structure
    parser.add_argument('--validationRatio', type=float, default=0.90, help='test Validation Split.')

    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: hg)')
    parser.add_argument('-s', '--stacks', default=1, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('--scale', default=256, type=int, metavar='N',
                        help='image scale')
    parser.add_argument('--features', default=128, type=int, metavar='N',
                        help='Number of features in the hourglass')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--num-classes', default=1, type=int, metavar='N',
                        help='Number of keypoints')
    # Training strategy
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=8, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=8, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-7, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[20, 50],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    # Data processing
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')

    parser.add_argument('--strucloss_alpha', type=float, default=0,
                        help='alpha of structure loss')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')
    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')


    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')
    parser.add_argument('--kind', type=str, default="skirt", help='Number of Clothes.')

    
    # # FULL DATASETS
    # parser.add_argument('--data_path',
    #                     default='/train/', type=str,
    #                     help='Path to train dataset')
    # parser.add_argument('--data_label_path',
    #                     default='/train.csv',
    #                     type=str,
    #                     help='Path to train groundtruth')
    # parser.add_argument('--data_path_test',
    #                     default='/test/',
    #                     type=str,
    #                     help='Path to test dataset')
    # parser.add_argument('--data_label_test_path',
    #                     default='/test.csv',
    #                     type=str,
    #                     help='Path to test groundtruth')

    # only test images
    parser.add_argument('--data_path_test',
                        default='/testdata/',
                        type=str,
                        help='Path to test dataset')

    main(parser.parse_args())