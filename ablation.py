"""
Evaluation Script
Support Two Modes: Pooling based inference and sliding based inference
"""
# ToDo: Be careful Relu!!!
import os
import logging
import sys
import argparse
import random

from datetime import datetime
from tqdm import tqdm
import cv2
from PIL import Image

from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
import transforms.joint_transforms as joint_transforms
import matplotlib.pyplot as plt

from config import assert_and_infer_cfg
import datasets
from datasets import cityscapes
from datasets import mapillary
from datasets import synthia
from datasets import bdd100k
from datasets import gtav
from datasets import idd
from optimizer import restore_snapshot

from utils.my_data_parallel import MyDataParallel
from utils.misc import fast_hist, save_log, per_class_iu, evaluate_eval_for_inference
from train import parse_for_modelassign
import torchvision.transforms as standard_transforms
import transforms.transforms as extended_transforms
import torch.nn.functional as F
from datasets import cityscapes_labels

import time
import network
from tsnelib import RunTsne

sys.path.append(os.path.join(os.getcwd()))



sys.path.append(os.path.join(os.getcwd(), '../'))

parser = argparse.ArgumentParser(description='evaluation')
parser = parse_for_modelassign(parser)
parser.add_argument('--dataset', nargs='*', type=str, default=['cityscapes'],
                    help='a list of datasets; cityscapes, mapillary, camvid, kitti, gtav, mapillary, synthia')
parser.add_argument('--source_domain', nargs='*', type=str, default=['gtav'],
                    help='a list of datasets; cityscapes, mapillary, camvid, kitti, gtav, mapillary, synthia')
parser.add_argument('--crop_size', nargs='*', type=int, default=[1280,720],
                    help='a list of datasets; tsnemem, clustering')
parser.add_argument('--exp', type=str, default=None)
parser.add_argument('--snapshot', required=True, type=str, default='')
parser.add_argument('--ablation_mode', nargs='*', type=str, default=['tsnemem'],
                    help='a list of datasets; tsnemem, clustering')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='minimum testing (4 items evaluated) to verify nothing failed')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--outdir', default=None, type=str,
                    help='output dir')
parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--mem_actmap', action='store_true', default=False,
                    help='plot memory activation map')
parser.add_argument('--tsne', action='store_true', default=False,
                    help='plot tsne map')
parser.add_argument('--image_in', action='store_true', default=False,
                    help='Image instance normalization')
parser.add_argument('--all_class', action='store_true', default=False,
                    help='visuallize all classes')
parser.add_argument('--tsnecuda', action='store_true', default=False,
                    help='visuallize all classes')
parser.add_argument('--duplication', type=int, default=10)
parser.add_argument('--imagenum_dom', type=int, default=600)



args = parser.parse_args()
assert_and_infer_cfg(args, train_mode=False)
cudnn.benchmark = True
args.world_size = 1
args.target_domain=[]
for x in args.dataset:
    if x not in args.source_domain:
        args.target_domain.append(x)


num_classes = datasets.num_classes
trainId2name = cityscapes_labels.trainId2name
trainId2color = cityscapes_labels.trainId2color
domId2name = {
    0:'gtav',
    1:'synthia',
    2:'cityscapes',
    3:'bdd100k',
    4:'mapillary',
    5:'idd',
}

if 'WORLD_SIZE' in os.environ:
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
try:
    args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)
    print('disturl : ' + args.dist_url)
    torch.distributed.init_process_group(backend='nccl',
                                            init_method=args.dist_url,
                                            world_size=args.world_size, rank=args.local_rank)
except RuntimeError:
    time.sleep(1)
    args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)
    print('disturl : ' + args.dist_url)
    torch.distributed.init_process_group(backend='nccl',
                                            init_method=args.dist_url,
                                            world_size=args.world_size, rank=args.local_rank)


mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
date_str = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))



def showbatchtensor(imgs):
    for ind in range(imgs.shape[0]):
        if imgs[ind].shape[0]==1: # if imgs is gt
            plt.imshow(imgs[ind].squeeze())
            plt.show()
        else:
            plt.imshow(imgs[ind].permute(1, 2, 0))
            plt.show()


def setup_loader():
    """
    Setup Data Loaders
    """
    # Image appearance transformation
    val_input_transform = [
                            standard_transforms.ToTensor()
    ]

    target_transform = [extended_transforms.MaskToTensor()]

    if args.crop_size is not None:
        val_joint_transform_list_global = [
            joint_transforms.CenterCropPad(tuple(args.crop_size))]
    else:
        val_joint_transform_list_global=[]



    val_sets = []
    val_dataset_names = []

    for dataset_name in args.dataset:
        if 'cityscapes' == dataset_name:
            dataset = cityscapes
            val_set = dataset.CityScapes('fine', 'val', 0,
                                         transform=standard_transforms.Compose(val_input_transform),
                                         target_transform=standard_transforms.Compose(target_transform),
                                         joint_transform=joint_transforms.Compose(val_joint_transform_list_global),
                                         eval_mode = True,
                                         cv_split=0,
                                         image_in=args.image_in)
            val_sets.append(val_set)
            val_dataset_names.append('cityscapes')

        if 'idd' == dataset_name:
            dataset = idd
            val_set = dataset.CityScapes('val', 0,
                                         transform=standard_transforms.Compose(val_input_transform),
                                         target_transform=standard_transforms.Compose(target_transform),
                                         joint_transform=joint_transforms.Compose(val_joint_transform_list_global),
                                         eval_mode = True,
                                         cv_split=0,
                                         image_in=args.image_in)
            val_sets.append(val_set)
            val_dataset_names.append('idd')

        if 'bdd100k' == dataset_name:
            dataset = bdd100k
            val_set = dataset.BDD100K('val', 0,
                                      transform=standard_transforms.Compose(val_input_transform),
                                      target_transform=standard_transforms.Compose(target_transform),
                                      joint_transform=joint_transforms.Compose(val_joint_transform_list_global),
                                      eval_mode=True,
                                      cv_split=0,
                                      image_in=args.image_in)
            val_sets.append(val_set)
            val_dataset_names.append('bdd100k')

        if 'gtav' == dataset_name:
            dataset = gtav

            val_set = gtav.GTAV('val', 0,
                                transform=standard_transforms.Compose(val_input_transform),
                                target_transform=standard_transforms.Compose(target_transform),
                                joint_transform=joint_transforms.Compose(val_joint_transform_list_global),
                                eval_mode=True,
                                cv_split=0,
                                image_in=args.image_in)

            val_sets.append(val_set)
            val_dataset_names.append('gtav')

        if 'synthia' == dataset_name:
            dataset = synthia

            val_set = dataset.Synthia('val', 0,
                                      transform=standard_transforms.Compose(val_input_transform),
                                      target_transform=standard_transforms.Compose(target_transform),
                                      joint_transform=joint_transforms.Compose(val_joint_transform_list_global),
                                      eval_mode=True,
                                      cv_split=0,
                                      image_in=args.image_in)

            val_sets.append(val_set)
            val_dataset_names.append('synthia')

        if 'mapillary' == dataset_name:
            dataset = mapillary

            eval_size = 1536
            mapillary_val_jointtransform_list = [
                joint_transforms.ResizeHeight(eval_size),
                joint_transforms.CenterCropPad(eval_size)]

            mapillary_val_jointtransform_list += val_joint_transform_list_global

            val_set = dataset.Mapillary(
                'semantic', 'val',
                joint_transform_list=mapillary_val_jointtransform_list,
                transform=standard_transforms.Compose(val_input_transform),
                target_transform=standard_transforms.Compose(target_transform),
                eval_mode=True,
                image_in=args.image_in,
                test=False)

            val_sets.append(val_set)
            val_dataset_names.append('mapillary')

    batch_size = 1

    extra_val_loader = {}
    for val_set,val_dataset_name in zip(val_sets,val_dataset_names):

        if args.syncbn:
            from datasets.sampler import DistributedSampler
            val_sampler = DistributedSampler(val_set, pad=False, permutation=False, consecutive_sample=False)
        else:
            val_sampler = None

        val_loader = DataLoader(val_set, batch_size=batch_size,
                                num_workers=args.num_workers // 2, shuffle=False, drop_last=False,
                                sampler=val_sampler)

        extra_val_loader[val_dataset_name] = val_loader

    return extra_val_loader


def get_net():
    """
    Get Network for evaluation
    """
    logging.info('Load model file: %s', args.snapshot)
    net = network.get_net(args, criterion=None)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = network.warp_network_in_dataparallel(net, args.local_rank)
    net, _, _, _, _ = restore_snapshot(net, optimizer=None, scheduler=None,
                              snapshot=args.snapshot, restore_optimizer_bool=False)

    net.eval()
    return net

class RunAbla():
    def __init__(self, output_dir, ablation_mode,selected_cls,imagenum_dom = args.imagenum_dom):
        self.output_dir = output_dir
        self.ablation_mode = ablation_mode
        self.imagenum_dom = imagenum_dom
        self.img_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(*mean_std)])
        self.to_pil = transforms.ToPILImage()
        self.tsne_path = os.path.join(self.output_dir, 'tsne_'+''.join([x[0] for x in args.dataset]))
        os.makedirs(self.tsne_path, exist_ok=True)
        self.mem_actpath = os.path.join(self.output_dir, 'memory_activation')
        self.mapping = {}
        self.selected_cls = selected_cls

        self.tsne_runner = RunTsne( self.tsne_path , selected_cls, domId2name, trainId2name, trainId2color=trainId2color, tsnecuda=args.tsnecuda, extention='.png',duplication=args.duplication)
        self.tsne_runner_updated = RunTsne( self.tsne_path , selected_cls, domId2name, trainId2name, trainId2color=trainId2color, tsnecuda=args.tsnecuda, extention='.png',duplication=args.duplication)

    def channelwise_minmax(self,AA):
        AA = AA.clone()
        c, h, w = AA.size()
        AA = AA.view(c, -1)
        AA -= AA.min(0, keepdim=True)[0]
        AA /= AA.max(0, keepdim=True)[0]
        return AA.view(c, h, w)

    def tsne_memact(self, data_loaders, net):
        ######################################################################
        # Run inference
        ######################################################################
        name2trainId = {v:k for k,v in trainId2name.items()}
        name2domId = {domId2name[x]: x for x in domId2name.keys()}
        self.selected_clsid = [name2trainId[x] for x in self.selected_cls]
        with torch.no_grad():
            # Validation after epochs, put source dataset into --val_dataset argument

            for dataset, val_loader in data_loaders.items():
                count = 0
                # Run Inference!
                pbar = tqdm(val_loader, desc='memory_activation & extract feature', smoothing=1.0)
                for val_idx, data in enumerate(pbar):
                    if count >= self.imagenum_dom:
                        break
                    inputs, gt_image, img_names, _ = data
                    # if img_names[0] == 'a91b7555-00001190':
                    if True:
                        input_pil = self.to_pil(inputs[0])
                        input = self.img_transform(input_pil)

                        C, H, W = input.shape
                        gt_image = gt_image.view(-1, H, W)
                        input = input.unsqueeze(dim=0)

                        assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
                        assert inputs.size()[2:] == gt_image.size()[1:]

                        input, gt_cuda = input.cuda(), gt_image.cuda()

                        with torch.no_grad():
                            if args.use_wtloss:
                                outputs = net(input, visualize=True)
                                output, f_cor_arr = outputs[0], outputs[1]
                            else:
                                outputs = net(input)
                                if args.memory:
                                    output, mem_outputs, features = outputs[0], outputs[-2], outputs[-1]
                                    softmax_score_memory = mem_outputs[1].permute(0, 3, 1, 2)
                                    updated_features = mem_outputs[-1]  # get refined features.
                                    self.tsne_runner_updated.input2basket(updated_features, gt_cuda, dataset)
                                else:
                                    output, features = outputs[0], outputs[-1]

                            assert output.size()[2:] == gt_image.size()[1:]
                            assert output.size()[1] == num_classes
                            self.tsne_runner.input2basket(features, gt_cuda, dataset)

                            count += 1
                        ######################################################################
                        # Dump Images(memory activation map)
                        ######################################################################
                        if args.mem_actmap:
                            mem_actpath = os.path.join(self.mem_actpath, dataset)
                            os.makedirs(mem_actpath, exist_ok=True)
                            img_name = img_names[0]
                            softmax_score_memory = F.interpolate(softmax_score_memory, [H, W], mode='bilinear',
                                                                 align_corners=True)
                            softmax_score_memory = softmax_score_memory.squeeze()
                            softmax_score_memory_refined = self.channelwise_minmax(softmax_score_memory.clone())
                            softmax_score_memory = (softmax_score_memory - softmax_score_memory.min()) / (
                                        softmax_score_memory.max() - softmax_score_memory.min())
                            softmax_score_memory = softmax_score_memory.cpu()
                            softmax_score_memory_refined = softmax_score_memory_refined.cpu()

                            for slot in self.selected_clsid:
                                cls_mem_score = softmax_score_memory_refined[slot]
                                cls_mem_score = np.array(cls_mem_score)
                                cls_mem_score = np.clip(cls_mem_score, 0, 1) * 255

                                cls_mem_score_map = cv2.applyColorMap(np.uint8(cls_mem_score), cv2.COLORMAP_VIRIDIS)
                                cls_mem_score_map = self.to_pil(cv2.cvtColor(cls_mem_score_map, cv2.COLOR_BGR2RGB))
                                act_img_name = '{}/{}_{}_memact.png'.format(mem_actpath, img_name,
                                                                                          trainId2name[slot])
                                cls_mem_score_map.save(act_img_name)

                                cls_mem_score_map = cv2.applyColorMap(np.uint8(cls_mem_score), cv2.COLORMAP_VIRIDIS)
                                cls_mem_score_map = self.to_pil(cv2.cvtColor(cls_mem_score_map, cv2.COLOR_BGR2RGB))
                                blend = Image.blend(input_pil.convert("RGBA"), cls_mem_score_map.convert("RGBA"), 0.65)
                                act_img_name = '{}/{}_{}_memact_blend.png'.format(mem_actpath, img_name, trainId2name[slot])
                                blend.save(act_img_name)
                    else:
                        continue

        if args.tsne:
            if args.memory:
                m_items = net.module.memory.m_items.clone().detach()
                self.tsne_runner.input_memory_item(m_items)
                self.tsne_runner_updated.input_memory_item(m_items)
            del net
            torch.cuda.empty_cache()
            #################################################
            # tsne plot
            #################################################
            # seen domain
            # domains2draw = args.source_domain
            # self.tsne_runner.draw_tsne(domains2draw,plot_memory=args.memory,clscolor=False)
            #
            # # unseen domain
            # domains2draw = args.target_domain
            # self.tsne_runner.draw_tsne(domains2draw, plot_memory=args.memory,clscolor=False)

            # all
            domains2draw = args.dataset
            self.tsne_runner.draw_tsne(domains2draw, plot_memory=args.memory,clscolor=False)

def main():
    """
    Main Function
    """
    if args.test_mode:
        ckpt_path = os.path.expanduser('~/experiment_pinmem/test')
    else:
        ckpt_path = os.path.join(os.path.split(args.snapshot)[0], os.path.splitext(os.path.split(args.snapshot)[1])[0])

    if args.outdir is not None:
        output_dir = args.outdir
    else:
        output_dir = ckpt_path
    os.makedirs(output_dir, exist_ok=True)
    save_log('abla', output_dir, date_str)
    logging.info("Network Arch: %s", args.arch)
    logging.info("Exp_name: %s", args.exp)
    logging.info("Ckpt path: %s", ckpt_path)

    # Set up network, loader, inference mode
    test_loaders = setup_loader()
    net = get_net()

    if args.all_class:
        selected_cls = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
                        'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                        'bicycle']
    else:
        selected_cls = ['building', 'vegetation', 'sky', 'car','sidewalk',
                        'pole']  # good memory learning for tsne


    runner = RunAbla(output_dir,ablation_mode=args.ablation_mode,selected_cls = selected_cls)
    runner.tsne_memact(test_loaders, net)

if __name__ == '__main__':
    main()
