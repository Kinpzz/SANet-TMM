# -*- coding: utf-8 -*-

"""SANet training routines."""

# Standard lib imports
import os
import time
import argparse
import os.path as osp
from urllib.parse import urlparse

# PyTorch imports
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

# Local imports
from networks.sanet import SANet
from referit_loader import ReferDataset
from utils.pyt_utils import load_model
from engine import Engine

# Other imports
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
from pydensecrf import densecrf
import skimage
parser = argparse.ArgumentParser(
    description='Structured Attention Network for Referring Image Segmentation')

# Dataloading-related settings
parser.add_argument('--data', type=str, default='datasets/refer',
                    help='path to ReferIt splits data folder')
parser.add_argument('--split-root', type=str, default='data',
                    help='path to dataloader splits data folder')
parser.add_argument('--snapshot', default='models/deeplab_resnet.pth.tar',
                    help='path to weight snapshot file')
parser.add_argument('--dataset', default='unc', type=str,
                    help='dataset used to train network')
parser.add_argument('--val', default='val', type=str,
                    help='name of the dataset split used to validate')
parser.add_argument('-j', '--num_workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

# Training procedure settings
parser.add_argument('--no-cuda', default=False, action='store_true',
                    help='Do not use cuda to train model')
parser.add_argument('--batch-size', default=1, type=int,
                    help='Batch size for each gpu')
parser.add_argument('--sync-bn', action='store_true', default=False,
                    help='Use sync batchnorm. Default False')
parser.add_argument('--pin-memory', default=False, action='store_true',
                    help='enable CUDA memory pin on DataLoader')

# Model settings
parser.add_argument('--size', default=320, type=int,
                    help='image size')
parser.add_argument('--os', default=16, type=int,
                    help='output stride. Default 16')
parser.add_argument('--time', default=20, type=int,
                    help='maximum time steps per batch')
parser.add_argument('--emb-size', default=300, type=int,
                    help='word embedding dimensions')
parser.add_argument('--hid-size', default=256, type=int,
                    help='language model hidden size')
parser.add_argument('--vis-size', default=256, type=int,
                    help='visual feature dimensions')
parser.add_argument('--mix-size', default=256, type=int,
                    help='multimodal feature dimensions')
parser.add_argument('--tree-hid-size', default=256, type=int,
                    help='tree-gru hidden state dimensions')
parser.add_argument('--lang-layers', default=1, type=int,
                    help='number of language model (Bi-LSTM) stacked layers')

# Other settings
parser.add_argument('--tensorboard', action='store_true', default=False,
                    help='Using tensorboard for visualization. Default False')
parser.add_argument('--visual-interval', default=100, type=int,
                    help='Using tensorboard for visualization. Default False')
parser.add_argument('--crf', action='store_true', default=False,
                    help='Using crf for refinement')

engine = Engine(custom_parser=parser)

args = parser.parse_args()

verbose = 0
if (not engine.distributed) or (engine.distributed and engine.local_rank ==0):
    verbose = 1

# print argument settings
args_dict = vars(args)
if verbose == 1:
    print('Argument list to program')
    print('\n'.join(['--{0} {1}'.format(arg, args_dict[arg])
                 for arg in args_dict]))
    print('\n\n')
    if args.tensorboard:
        from tensorboardX import SummaryWriter
        # Writer will output to ./runs/ directory by default
        writer = SummaryWriter()

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    torch.backends.cudnn.benchmark = True

image_size = (args.size, args.size)

input_transform = Compose([
    Resize(image_size),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

target_transform = Compose([
    Resize(image_size),
    ToTensor()
])

totensor = ToTensor()

refer_val = ReferDataset(data_root=args.data,
                            dataset=args.dataset,
                            split_root=args.split_root,
                            split=args.val,
                            transform=input_transform,
                            annotation_transform=target_transform,
                            max_query_len=args.time)

val_loader, val_sampler = engine.get_test_loader(refer_val)

net = SANet(dict_size=len(refer_val.corpus),
          emb_size=args.emb_size,
          hid_size=args.hid_size,
          vis_size=args.vis_size,
          mix_size=args.mix_size,
          tree_hid_size = args.tree_hid_size,
          lang_layers=args.lang_layers,
          output_stride=args.os,
          num_classes=1,
	      pretrained_backbone=False,
          pretrained_embedding=None,
          dataset=args.dataset)

if osp.exists(args.snapshot):
    print('Loading state dict from: {0}'.format(args.snapshot))
    net = load_model(model=net, model_file=args.snapshot, is_restore=True)
else:
    ValueError("{} not exists.".format(args.snapshot))

cuda = torch.cuda.is_available() if args.cuda else False
if cuda:
    net.cuda()
if verbose == 1:
    if cuda:
        current_device = torch.cuda.current_device()
        print("Running on", torch.cuda.get_device_name(current_device))
    else:
        print("Running on CPU")

#base_params = list(map(id, net.backbone.parameters()))
#new_params = filter(lambda p: id(p) not in base_params, net.parameters())

net = engine.data_parallel(net)

def compute_mask_IU(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:])
    temp = (masks * target)
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union
# def compute_mask_IU(masks, target):
#     assert(target.shape[-2:] == masks.shape[-2:])
#     I = torch.sum(masks.int() & target.int())
#     U = torch.sum(masks.int() | target.int())
#     return I, U
def compute_mask_IU_np(masks, target):
    assert(target.shape[-2:] == masks.shape[-2:])
    I = np.sum(np.logical_and(masks, target))
    U = np.sum(np.logical_or(masks, target))
    return I, U

def visualize_data(imgs, masks, words, words_len, out, att):
        visual_imgs = imgs.detach().cpu()
        visual_gt = masks.detach().cpu()
        visual_out = out.detach().cpu()
        visual_att = att.detach().cpu()
        img_grid = torchvision.utils.make_grid(visual_imgs, nrow=4, normalize=True, pad_value=1)
        gt_grid = torchvision.utils.make_grid(visual_gt, nrow=4, normalize=True, pad_value=1)
        out_grid = torchvision.utils.make_grid(visual_out, nrow=4, normalize=True, pad_value=1)
        att_grid = torchvision.utils.make_grid(visual_att, nrow=4, normalize=True, pad_value=1)
        phrase = ""
        for i in range(words.size(0)):
            words_idx = words[i].detach().cpu().tolist()
            words_list = refer_val.corpus.dictionary.__getitem__(words_idx)
            phrase += "({})".format(str(i)) + " ".join(words_list[j] for j in range(words_len[i])) + '; '
        return img_grid, gt_grid, out_grid, att_grid, phrase

def evaluate(epoch=0):
    net.eval()
    score_thresh = np.concatenate([np.arange(start=0.00, stop=0.96,
                                             step=0.025)]).tolist()
    cum_I = torch.zeros(len(score_thresh)).cuda()
    cum_U = torch.zeros(len(score_thresh)).cuda()
    eval_seg_iou_list = [.5, .6, .7, .8, .9]

    if args.crf:
        cum_I_dcrf = 0.0
        cum_U_dcrf = 0.0
        seg_correct_dcrf = np.zeros(len(eval_seg_iou_list))
    seg_correct = torch.zeros(len(eval_seg_iou_list), len(score_thresh)).cuda()
    seg_total = 0
    if args.crf:
        seg_correct_dcrf = np.zeros(len(eval_seg_iou_list))
    with tqdm(total=len(val_loader),
              dynamic_ncols=True,
              desc='Validation Epoch #{}'.format(epoch),
              disable=not verbose) as t:
        for batch_idx, (imgs, masks, words, adjs, words_len) in enumerate(val_loader):
            if cuda:
                imgs = imgs.cuda()
                masks = masks.cuda()
                words = words.cuda()
                adjs = adjs.cuda()
                words_len = words_len.cuda()
            with torch.no_grad():
                out, _, out_att = net(imgs, words, adjs, words_len)
                out = torch.sigmoid(out)

            b_cum_I = torch.zeros(len(score_thresh)).cuda()
            b_cum_U = torch.zeros(len(score_thresh)).cuda()
            b_seg_correct = torch.zeros(len(eval_seg_iou_list), len(score_thresh)).cuda()

            for i in range(imgs.size(0)):
                inter = torch.zeros(len(score_thresh)).cuda()
                union = torch.zeros(len(score_thresh)).cuda()

                mask_path = osp.join(refer_val.mask_dir, refer_val.images[seg_total][1])
                mask = Image.open(mask_path)
                mask = totensor(mask).unsqueeze(0).cuda()
                pred = F.interpolate(out[i].unsqueeze(0), size=mask.shape[2:], mode='bilinear', align_corners=True).squeeze(0)

                for idx, thresh in enumerate(score_thresh):
                    thresholded_out = (pred > thresh).float()
                    try:
                        inter[idx], union[idx] = compute_mask_IU(thresholded_out, mask)
                    except AssertionError:
                        inter[idx] = 0
                        union[idx] = mask.sum()

                if args.crf:
                    sigm_val = np.squeeze(out[i].cpu().numpy())
                    d = densecrf.DenseCRF2D(320, 320, 2)
                    U = np.expand_dims(-np.log(sigm_val), axis=0)
                    U_ = np.expand_dims(-np.log(1 - sigm_val), axis=0)
                    unary = np.concatenate((U_, U), axis=0)
                    unary = unary.reshape((2, -1))
                    d.setUnaryEnergy(unary)
                    d.addPairwiseGaussian(sxy=3, compat=3)
                    #import ipdb; ipdb.set_trace()
                    img_path = osp.join(refer_val.im_dir, refer_val.images[seg_total][0])
                    proc_im = Image.open(img_path).convert('RGB').resize((320,320), 1)
                    proc_im = skimage.img_as_ubyte(np.array(proc_im))
                    d.addPairwiseBilateral(sxy=20, srgb=3, rgbim=proc_im, compat=10)
                    Q = d.inference(5)
                    mask_ = mask.squeeze().cpu().numpy()
                    pred_dcrf = np.argmax(Q, axis=0).reshape((320, 320)).astype(np.float32)
                    pred_dcrf = skimage.transform.resize(pred_dcrf, [mask_.shape[0], mask_.shape[1]])
                    inter_dcrf, union_dcrf = compute_mask_IU_np(pred_dcrf, mask_)
                    cum_I_dcrf += inter_dcrf
                    cum_U_dcrf += union_dcrf
                    this_iou_dcrf = inter_dcrf / union_dcrf

                this_iou = inter / union

                for idx, seg_iou in enumerate(eval_seg_iou_list):
                    for jdx in range(len(score_thresh)):
                        b_seg_correct[idx, jdx] += (this_iou[jdx] >= seg_iou)
                    if args.crf:
                        seg_correct_dcrf[idx] += (this_iou_dcrf >= seg_iou)
                seg_total += 1

                b_cum_I += inter
                b_cum_U += union
                if verbose == 1 and args.tensorboard and epoch >= 5 and this_iou.max() < 0.1:
                    failed_idx = [i]
                    img_grid, gt_grid, out_grid, att_grid, phrase = visualize_data(imgs[failed_idx], \
                                            masks[failed_idx], words[failed_idx], words_len[failed_idx], out[failed_idx], out_att[failed_idx])
                    n_iter = epoch*len(val_loader) + batch_idx + i
                    writer.add_image('val_failed/images', img_grid, n_iter)
                    writer.add_image('val_failed/gts', gt_grid, n_iter)
                    writer.add_image('val_failed/output', out_grid, n_iter)
                    writer.add_image('val_failed/att', att_grid, n_iter)
                    writer.add_text('val_failed/phrase', phrase, n_iter)

            if engine.distributed:
                seg_correct += engine.all_reduce_tensor(b_seg_correct.float().detach())
                cum_I += engine.all_reduce_tensor(b_cum_I.float().detach())
                cum_U += engine.all_reduce_tensor(b_cum_U.float().detach())
            else:
                seg_correct += b_seg_correct.float().detach()
                cum_I += b_cum_I.float().detach()
                cum_U += b_cum_U.float().detach()
            # Tensorboard
            if verbose == 1 and args.tensorboard and batch_idx % args.visual_interval == 0:
                img_grid, gt_grid, out_grid, att_grid, phrase = visualize_data(imgs, masks, words, words_len, out, out_att)
                n_iter = epoch*len(val_loader) + batch_idx
                writer.add_image('val/images', img_grid, n_iter)
                writer.add_image('val/gts', gt_grid, n_iter)
                writer.add_image('val/output', out_grid, n_iter)
                writer.add_image('val/att', att_grid, n_iter)
                writer.add_text('val/phrase', phrase, n_iter)
            t.set_postfix({'IoU@{:.2f}'.format(score_thresh[20]): '{:.3f}'.format(float(this_iou[20]))})
            t.update(1)

    # Print final accumulated IoUs
    final_ious = cum_I / cum_U
    max_iou, max_idx = torch.max(final_ious, 0)
    max_iou = float(max_iou.detach().cpu().numpy())
    max_idx = int(max_idx.detach().cpu().numpy())

    # Evaluation finished. Compute total IoUs and threshold that maximizes
    if verbose == 1:
        print('-' * 26)
        for jdx, thresh in enumerate(score_thresh):
            print('prec@X for Threshold {:.3f}'.format(thresh))
            for idx, seg_iou in enumerate(eval_seg_iou_list):
                print('prec@{:s} = {:2.2%}'.format(
                str(seg_iou), seg_correct[idx, jdx] / seg_total))

        print('prec@X for Threshold {:.3f}'.format(score_thresh[max_idx]))
        for idx, seg_iou in enumerate(eval_seg_iou_list):
            print('prec@{:s} = {:2.2%}'.format(
                str(seg_iou), seg_correct[idx, max_idx] / seg_total))
        print('-' * 26 + '\n' + '')
        print('FINAL accumulated IoUs at different thresholds:')
        print('{:4}| {:3} |'.format('Thresholds', 'mIoU'))
        print('-' * 26)
        for idx, thresh in enumerate(score_thresh):
            print('{:.3f}| {:<2.2%} |'.format(thresh, final_ious[idx]))
        print('-' * 26)
        # Print maximum IoU
        print('Maximum Overall IoU: {:2.2%} - Threshold: {:.3f}'.format(
            max_iou, score_thresh[max_idx]))

        if args.crf:
            print('-' * 26)
            print('Results after DCRF')
            for idx, seg_iou in enumerate(eval_seg_iou_list):
                print('prec@{:s} = {:2.2%}'.format(
                str(seg_iou), seg_correct_dcrf[idx] / seg_total))
            print('Overall IoU: {:2.2%}'.format(cum_I_dcrf / cum_U_dcrf))

        if args.tensorboard:
            writer.add_scalar('val/max_iou', max_iou, epoch)
            writer.add_scalar('val/max_iou_threshold', score_thresh[max_idx], epoch)
    return max_iou

# average metrics from distributed training.
class Metric(object):
    def __init__(self, name, engine):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)
        self.engine = engine

    def update(self, value, n=1):
        value = value.detach()
        if self.engine.distributed:
            value = engine.all_reduce_tensor(value)
        self.sum += value
        self.n += n

    def reset(self):
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    @property
    def avg(self):
        return self.sum / self.n

if __name__ == '__main__':
    try:
        evaluate(0)
    except KeyboardInterrupt:
        if args.tensorboard and verbose == 1:
            writer.close()
            torch.cuda.empty_cache()
        print('-' * 89)
        print('Exiting from validation early')
