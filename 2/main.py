import math
import copy
import time
import shutil
import os
import random
os.environ['TORCH_HOME'] = './ckpts'

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2

from dataset import get_pascal_voc2007_data, pascal_voc2007_loader, idx_to_class
from model import FastRCNN
from utils import coord_trans, data_visualizer


def parse_args():
    parser = argparse.ArgumentParser('Faster R-CNN', add_help=False)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_decay', default=1.0, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--overfit_small_data', default=False, action='store_true')
    parser.add_argument('--output_dir', default='./exp/fast_rcnn')

    args = parser.parse_args()

    return args


def main(args):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    if args.overfit_small_data:
        args.output_dir = args.output_dir + "_overfit_small"
    os.makedirs(args.output_dir, exist_ok=True)

    # build dataset & dataloader
    train_dataset = get_pascal_voc2007_data('./data/VOCtrainval_06-Nov-2007/', 'train')
    val_dataset = get_pascal_voc2007_data('./data/VOCtrainval_06-Nov-2007/', 'val')

    train_loader = pascal_voc2007_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers,
                                         proposal_path='data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Proposals')
    val_loader = pascal_voc2007_loader(val_dataset, args.batch_size, num_workers=args.num_workers,
                                       proposal_path='data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Proposals')

    if args.overfit_small_data:
        num_sample = 10
        small_dataset = torch.utils.data.Subset(
            train_dataset,
            torch.linspace(0, len(train_dataset)-1, steps=num_sample).long()
        )
        small_train_loader = pascal_voc2007_loader(small_dataset, 10,
                                                   proposal_path='data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Proposals')
        val_dataset = small_dataset
        train_loader = small_train_loader
        val_loader = small_train_loader

    model = FastRCNN()
    model.cuda()

    # build optimizer
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr
    )
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: args.lr_decay ** epoch
    )

    # load ckpt
    ckpt_path = os.path.join(args.output_dir, 'checkpoint.pth')
    start_epoch = 0
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_sched'])

    if start_epoch < args.epochs:
        train(args, model, train_loader, optimizer, lr_scheduler, start_epoch)
    inference(args, model, val_loader, val_dataset, visualize=args.overfit_small_data)


def train(args, model, train_loader, optimizer, lr_scheduler, start_epoch):
    loss_history = []
    model.train()
    for i in range(start_epoch, args.epochs):
        start_t = time.time()
        for iter_num, data_batch in enumerate(train_loader):
            images, boxes, boxes_batch_ids, proposals, proposal_batch_ids, w_batch, h_batch, _ = data_batch
            resized_boxes = coord_trans(boxes, boxes_batch_ids, w_batch, h_batch, mode='p2a')
            resized_proposals = coord_trans(proposals, proposal_batch_ids, w_batch, h_batch, mode='p2a')
            
            images = images.to(dtype=torch.float, device='cuda')
            resized_boxes = resized_boxes.to(dtype=torch.float, device='cuda')
            boxes_batch_ids = boxes_batch_ids.cuda()
            resized_proposals = resized_proposals.to(dtype=torch.float, device='cuda')
            proposal_batch_ids = proposal_batch_ids.cuda()

            loss = model(images, resized_boxes, boxes_batch_ids, resized_proposals, proposal_batch_ids)
            optimizer.zero_grad()
            loss.backward()
            loss_history.append(loss.item())
            optimizer.step()

            if iter_num % 50 == 0:
                print('(Iter {} / {}) loss: {:.4f}'.format(iter_num, len(train_loader), np.mean(loss_history[-50:])))

        end_t = time.time()
        print('(Epoch {} / {}) loss: {:.4f}, time per epoch: {:.1f}s'.format(
            i, args.epochs, np.mean(loss_history[-len(train_loader):]), end_t-start_t))
        lr_scheduler.step()

        checkpoint = { 
            'epoch': i + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_sched': lr_scheduler.state_dict()}
        torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))

    # plot the training losses
    fig, ax = plt.subplots()
    ax.plot(loss_history)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training loss history')
    fig.savefig(os.path.join(args.output_dir, 'training_loss.png'))
    plt.close()


def inference(args, model, val_loader, dataset, thresh=0.5, nms_thresh=0.5, visualize=False):
    model.eval()
    start_t = time.time()

    if args.output_dir is not None:
        det_dir = os.path.join(args.output_dir, 'mAP_input/detection-results')
        gt_dir = os.path.join(args.output_dir, 'mAP_input/ground-truth')
        vis_dir = os.path.join(args.output_dir, 'visualize')
        os.makedirs(det_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

    for iter_num, data_batch in enumerate(val_loader):
        images, boxes, boxes_batch_ids, proposals, proposal_batch_ids, w_batch, h_batch, img_ids = data_batch
        images = images.to(dtype=torch.float, device='cuda')
        resized_proposals = coord_trans(proposals, proposal_batch_ids, w_batch, h_batch, mode='p2a')
        resized_proposals = resized_proposals.to(dtype=torch.float, device='cuda')
        proposal_batch_ids = proposal_batch_ids.cuda()

        with torch.no_grad():
            final_proposals, final_conf_scores, final_class = \
                model.inference(images, resized_proposals, proposal_batch_ids, thresh=thresh, nms_thresh=nms_thresh)

        # clamp on the proposal coordinates
        batch_size = len(images)
        for idx in range(batch_size):
            torch.clamp_(final_proposals[idx][:, 0::2], min=0, max=w_batch[idx])
            torch.clamp_(final_proposals[idx][:, 1::2], min=0, max=h_batch[idx])

            # visualization
            # get the original image
            # hack to get the original image so we don't have to load from local again...
            i = batch_size*iter_num + idx
            img, _ = dataset.__getitem__(i)

            box_per_img = boxes[boxes_batch_ids==idx]
            final_all = torch.cat((final_proposals[idx], \
                final_class[idx].float(), final_conf_scores[idx]), dim=-1).cpu()
            final_batch_idx = torch.LongTensor([idx] * final_all.shape[0])
            resized_final_proposals = coord_trans(final_all, final_batch_idx, w_batch, h_batch)

            # write results to file for evaluation (use mAP API https://github.com/Cartucho/mAP for now...)
            if args.output_dir is not None:
                file_name = img_ids[idx].replace('.jpg', '.txt')
                with open(os.path.join(det_dir, file_name), 'w') as f_det, \
                open(os.path.join(gt_dir, file_name), 'w') as f_gt:
                    print('{}: {} GT bboxes and {} proposals'.format(img_ids[idx], len(box_per_img), resized_final_proposals.shape[0]))
                    for b in box_per_img:
                        f_gt.write('{} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(idx_to_class[b[4].item()], b[0], b[1], b[2], b[3]))
                    for b in resized_final_proposals:
                        f_det.write('{} {:.6f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(idx_to_class[b[4].item()], b[5], b[0], b[1], b[2], b[3]))

                if visualize:
                    data_visualizer(img, idx_to_class, os.path.join(vis_dir, img_ids[idx]), box_per_img, resized_final_proposals)

    end_t = time.time()
    print('Total inference time: {:.1f}s'.format(end_t-start_t))


if __name__=='__main__':
    args = parse_args()
    main(args)