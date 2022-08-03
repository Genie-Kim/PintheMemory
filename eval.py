"""
Evaluation Script
Support Two Modes: Pooling based inference and sliding based inference
"""
import os
import logging
import sys
import argparse
import re
import queue
import threading
from math import ceil
from datetime import datetime
from tqdm import tqdm
import cv2
from PIL import Image
import PIL

from torch.backends import cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
import transforms.transforms as extended_transforms
import transforms.joint_transforms as joint_transforms

from config import assert_and_infer_cfg
from datasets import cityscapes
from datasets import mapillary
from datasets import synthia
from datasets import bdd100k
from datasets import gtav
from datasets import idd
from datasets import video_folder
from optimizer import restore_snapshot

from utils.my_data_parallel import MyDataParallel
from utils.misc import fast_hist, save_log, per_class_iu, evaluate_eval_for_inference
from train import parse_for_modelassign

import time
import network

sys.path.append(os.path.join(os.getcwd()))

# # GFLOPS code
# from ptflops import get_model_complexity_info
# dummy_size = (3, 2048, 1024)
# m = 0
# for i in range(500):
#     start = time.time()
#     macs, params = get_model_complexity_info(net, dummy_size, as_strings=True,print_per_layer_stat=True, verbose=True)
#     end = time.time()
#     m+=end-start
# print(m/500)
# print('computational complexity: ', macs)
# print('number of parameters: ', params)

sys.path.append(os.path.join(os.getcwd(), '../'))

parser = argparse.ArgumentParser(description='evaluation')
parser = parse_for_modelassign(parser)
parser.add_argument('--dump_images', action='store_true', default=False)
parser.add_argument('--scales', type=str, default='0.5,1.0,2.0')
parser.add_argument('--dist_bn', action='store_true', default=False)
parser.add_argument('--profile', action='store_true', default=False)
parser.add_argument('--fixed_aspp_pool', action='store_true', default=False,
                    help='fix the aspp image-level pooling size to 105')

parser.add_argument('--sliding_overlap', type=float, default=1 / 3)
parser.add_argument('--no_flip', action='store_true', default=False,
                    help='disable flipping')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='cityscapes, video_folder')
parser.add_argument('--dataset_cls', type=str, default='cityscapes', help='cityscapes')
parser.add_argument('--trunk', type=str, default='resnet101', help='cnn trunk')
parser.add_argument('--dataset_dir', type=str, default=None,
                    help='Dataset Location')
parser.add_argument('--split', type=str, default='val')
parser.add_argument('--crop_size', type=int, default=1024) # scailing after cropping!!
parser.add_argument('--exp', type=str, default=None)
parser.add_argument('--snapshot', required=True, type=str, default='')
parser.add_argument('-im', '--inference_mode', type=str, default='sliding',
                    help='sliding or pooling')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='minimum testing (4 items evaluated) to verify nothing failed')
parser.add_argument('--cv_split', type=int, default=0)
parser.add_argument('--mode', type=str, default='fine')
parser.add_argument('--videopath', type=str, default=None, help='you should define video folder absolute path when you use video path')
parser.add_argument('--split_index', type=int, default=0)
parser.add_argument('--split_count', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--resume', action='store_true', default=False,
                    help='Resume Inference')
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by apex library')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                    help='url used to set up distributed training')

parser.add_argument('--syncbn', action='store_true', default=False,
                    help='Use Synchronized BN')
parser.add_argument('--image_in', action='store_true', default=False,
                    help='Image instance normalization')
parser.add_argument('--faster', action='store_true', default=False,
                    help='sliding with batch')
parser.add_argument('--visualize_feature', action='store_true', default=False,
                    help='Save intermediate feature map into NPY Files')
parser.add_argument('--trials', type=int, default=0.0)


args = parser.parse_args()
assert_and_infer_cfg(args, train_mode=False)
cudnn.benchmark = True
args.world_size = 1

if 'WORLD_SIZE' in os.environ:
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))

torch.cuda.set_device(args.local_rank)
print('My Rank:', args.local_rank)
args.dist_url = args.dist_url + str(8000 + (int(time.time()%1000))//10)
print('disturl : ' + args.dist_url)
torch.distributed.init_process_group(backend='nccl',
                                        init_method=args.dist_url,
                                        world_size=args.world_size, rank=args.local_rank)


mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
date_str = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

class MeanFusion:
    def __init__(self,img_orisize):
        self.buffer = torch.tensor((), dtype=torch.float64).new_zeros(1, 19, img_orisize[0],img_orisize[1])
        self.counter = 0

    def update(self, sem_logits):
        probs = torch.nn.functional.softmax(sem_logits, dim=1)
        self.counter += 1
        self.buffer.add_((probs - self.buffer) / self.counter)

    def output(self):
        probs, cls = self.buffer.max(1)
        return probs, cls


def sliding_window_cropping(data, scale=1.0):
    """
    Sliding Window Cropping
    Take the image and create a mapping and multiple crops
    """
    sliding_window_cropping = None
    mapping = {}
    crop_ctr = 0
    if scale < 1.0:
        scale = 1.0
    tile_size = (int(args.crop_size * scale), int(args.crop_size * scale))

    overlap = args.sliding_overlap

    for img_ctr in range(len(data)):

        h, w = data[img_ctr].shape[1:]
        mapping[img_ctr] = [w, h, []]
        stride = ceil(tile_size[0] * (1 - overlap))

        tile_rows = int(
            ceil((w - tile_size[0]) / stride) + 1)
        tile_cols = int(ceil((h - tile_size[1]) / stride) + 1)
        for row in range(tile_rows):
            for col in range(tile_cols):
                y1 = int(col * stride)
                x1 = int(row * stride)
                x2 = min(x1 + tile_size[1], w)
                y2 = min(y1 + tile_size[0], h)
                x1 = int(x2 - tile_size[1])
                y1 = int(y2 - tile_size[0])
                if x1 < 0:  # for portrait the x1 underflows sometimes
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if crop_ctr == 0:
                    sliding_window_cropping = data[img_ctr][:, y1:y2, x1:x2].unsqueeze(0)
                else:
                    sliding_window_cropping = torch.cat(
                        (sliding_window_cropping,
                         data[img_ctr][:, y1:y2, x1:x2].unsqueeze(0)),
                        dim=0)

                mapping[img_ctr][2].append((x1, y1, x2, y2))
                crop_ctr += 1

    return (mapping, sliding_window_cropping)


def resize_thread(flip, index, array, resizequeue, origw, origh):
    """
    Thread to resize the image size
    """
    if flip:
        resizequeue.put((index, cv2.resize(np.fliplr(array),
                                           (origw, origh),
                                           interpolation=cv2.INTER_LINEAR)))
    else:
        resizequeue.put((index, cv2.resize(array, (origw, origh),
                                           interpolation=cv2.INTER_LINEAR)))


def reverse_mapping(i, ctr, input_img, mapping, que, flip, origw, origh):
    """
    Reverse Mapping for sliding window
    """
    w, h, coords = mapping[i]
    full_probs = np.zeros((args.dataset_cls.num_classes, h, w))
    count_predictions = np.zeros((args.dataset_cls.num_classes, h, w))
    for j in range(len(coords)):
        x1, y1, x2, y2 = coords[j]
        count_predictions[y1:y2, x1:x2] += 1
        average = input_img[ctr]
        if full_probs[:, y1: y2, x1: x2].shape != average.shape:
            average = average[:, :y2 - y1, :x2 - x1]

        full_probs[:, y1:y2, x1:x2] += average
        ctr = ctr + 1

    # Accumulate and average overerlapping areas
    full_probs = full_probs / count_predictions.astype(np.float)
    out_temp = []
    out_y = []
    t_list = []
    resizequeue = queue.Queue()
    classes = full_probs.shape[0]
    for y_ in range(classes):
        t = threading.Thread(target=resize_thread, args=(flip, y_, full_probs[y_],
                                                         resizequeue, origw, origh))
        t.daemon = True
        t.start()
        t_list.append(t)

    for thread in t_list:
        thread.join()
        out_temp.append(resizequeue.get())

    dictionary = dict(out_temp)
    for iterator in range(classes):
        out_y.append(dictionary[iterator])

    que.put(out_y)


def reverse_sliding_window(mapping, input_img, flip_list, origw, origh, final_queue):
    """
    Take mapping and crops and reconstruct original image
    """

    batch_return = []
    ctr = 0
    # Loop through the maps and merge them together
    que = queue.Queue()
    t_list = []
    for i in range(len(mapping)):
        t = threading.Thread(target=reverse_mapping, args=(i, ctr, input_img, mapping, que,
                                                           flip_list[i], origw, origh))
        ctr = ctr + len(mapping[i][2])
        t.daemon = True
        t.start()
        t_list.append(t)

    for item in t_list:
        item.join()
        batch_return.append(que.get())

    final_queue.put(np.mean(batch_return, axis=0))


def pooled_eval(model, image, img_orisize):
    """
    Perform Pooled Evaluation
    """
    with torch.no_grad():
        y = model(image)[0]
        y = torch.nn.functional.interpolate(y, size=img_orisize, mode='bilinear')
        del image

    return y


def flip_tensor(x, dim):
    """
    Flip Tensor along a dimension
    """
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
                   else torch.arange(x.size(i) - 1, -1, -1).long()
                   for i in range(x.dim()))]

def flip_tensor2(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def inference_pool(model, img, scales):
    """
    Post Inference Pool Operations
    """


    for i in range(len(scales)):
        if scales[i] == 1:
            img_orisize = img[0][i].shape[2:]
            break
    fusion_cls = MeanFusion(img_orisize)

    if args.no_flip:
        flip_range = 1
    else:
        flip_range = 2

    y_tmp_with_flip = 0
    for flip in range(flip_range):
        y_tmp = None
        for i in range(len(scales)):
            if type(y_tmp) == type(None):
                y_tmp = pooled_eval(model, img[flip][i], img_orisize)
            else:
                y_tmp = pooled_eval(model, img[flip][i], img_orisize)
            if flip == 0:
                fusion_cls.update(y_tmp.cpu())
            else:
                fusion_cls.update(flip_tensor2(y_tmp.cpu(), -1))

    probs, preds = fusion_cls.output()
    pred = preds.cpu().numpy()

    return pred


def inference_sliding(model, img, scales):
    """
    Sliding Window Inference Function
    """

    w, h = img.size
    origw, origh = img.size
    preds = []
    if args.no_flip:
        flip_range = 1
    else:
        flip_range = 2

    finalque = queue.Queue()
    t_list = []
    for scale in scales:

        target_w, target_h = int(w * scale), int(h * scale)
        scaled_img = img.resize((target_w, target_h), Image.BILINEAR)
        y = []
        image_list = []
        flip_list = []
        for flip in range(flip_range):
            if flip:
                scaled_img = scaled_img.transpose(Image.FLIP_LEFT_RIGHT)

            img_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(*mean_std)])
            image = img_transform(scaled_img)
            image_list.append(image)
            flip_list.append(flip)

        mapping, input_crops = sliding_window_cropping(image_list, scale=scale)

        torch.cuda.empty_cache()
        with torch.no_grad():
            batch_num = input_crops.shape[0]


            if args.faster:
                output_scattered = model(input_crops.cuda())[0]
                del input_crops

            else:
                output_crops = []
                for i in range(batch_num):
                    input_crop = input_crops[i].unsqueeze(0).cuda()
                    output_crop = model(input_crop)[0]
                    output_crops.append(output_crop)
                    del input_crop
                output_scattered = torch.cat(output_crops, dim=0)

        output_scattered = output_scattered.data.cpu().numpy()

        t = threading.Thread(target=reverse_sliding_window, args=(mapping, output_scattered,
                                                                  flip_list, origw,
                                                                  origh, finalque))
        t.daemon = True
        t.start()
        t_list.append(t)

    for threads in t_list:
        threads.join()
        preds.append(finalque.get())

    return preds


def setup_loader():
    """
    Setup Data Loaders
    """
    val_input_transform = transforms.ToTensor()
    target_transform = extended_transforms.MaskToTensor()

    if args.dataset == 'cityscapes':
        args.dataset_cls = cityscapes
        eval_scales = None
        if args.inference_mode == 'pooling':
            eval_mode = 'pooling'
            eval_scales = args.scales
        elif args.inference_mode == 'sliding':
            eval_mode = 'sliding'
        else:
            raise Exception(f"Not implemented inference mode: {args.inference_mode}")

        test_set = args.dataset_cls.CityScapes(args.mode, args.split, 0,
                                               transform=val_input_transform,
                                               target_transform=target_transform,
                                               cv_split=0,#args.cv_split,
                                               eval_mode=eval_mode,
                                               eval_scales=eval_scales,
                                               eval_flip=not args.no_flip,
                                               image_in=args.image_in
                                               )
    elif args.dataset == 'idd':
        args.dataset_cls = idd
        eval_scales = None
        if args.inference_mode == 'pooling':
            eval_mode = 'pooling'
            eval_scales = args.scales
        elif args.inference_mode == 'sliding':
            eval_mode = 'sliding'
        else:
            raise Exception(f"Not implemented inference mode: {args.inference_mode}")

        test_set = args.dataset_cls.Idd(args.split, 0,
                                               transform=val_input_transform,
                                               target_transform=target_transform,
                                               cv_split=0,#args.cv_split,
                                               eval_mode=eval_mode,
                                               eval_scales=eval_scales,
                                               eval_flip=not args.no_flip,
                                               image_in=args.image_in
                                               )

    elif args.dataset == 'bdd100k':
        args.dataset_cls = bdd100k
        eval_scales = None
        if args.inference_mode == 'pooling':
            eval_mode = 'pooling'
            eval_scales = args.scales
        elif args.inference_mode == 'sliding':
            eval_mode = 'sliding'
        else:
            raise Exception(f"Not implemented inference mode: {args.inference_mode}")

        test_set = args.dataset_cls.BDD100K(args.split, 0,
                                               transform=val_input_transform,
                                               target_transform=target_transform,
                                               cv_split=0,#args.cv_split,
                                               eval_mode=eval_mode,
                                               eval_scales=eval_scales,
                                               eval_flip=not args.no_flip,
                                               image_in=args.image_in
                                               )
    elif args.dataset == 'gtav':
        args.dataset_cls = gtav
        eval_scales = None
        if args.inference_mode == 'pooling':
            eval_mode = 'pooling'
            eval_scales = args.scales
        elif args.inference_mode == 'sliding':
            eval_mode = 'sliding'
        else:
            raise Exception(f"Not implemented inference mode: {args.inference_mode}")

        test_set = args.dataset_cls.GTAV(args.split, 0,
                                               transform=val_input_transform,
                                               target_transform=target_transform,
                                               cv_split=0,#args.cv_split,
                                               eval_mode=eval_mode,
                                               eval_scales=eval_scales,
                                               eval_flip=not args.no_flip,
                                               image_in=args.image_in
                                               )

    elif args.dataset == 'synthia':
        args.dataset_cls = synthia
        eval_scales = None
        if args.inference_mode == 'pooling':
            eval_mode = 'pooling'
            eval_scales = args.scales
        elif args.inference_mode == 'sliding':
            eval_mode = 'sliding'
        else:
            raise Exception(f"Not implemented inference mode: {args.inference_mode}")

        test_set = args.dataset_cls.Synthia(args.split, 0,
                                               transform=val_input_transform,
                                               target_transform=target_transform,
                                               cv_split=0,#args.cv_split,
                                               eval_mode=eval_mode,
                                               eval_scales=eval_scales,
                                               eval_flip=not args.no_flip,
                                               image_in=args.image_in
                                               )
        args.dataset_cls.id_to_trainid = args.dataset_cls.trainid_to_trainid

    elif args.dataset == 'mapillary':
        args.dataset_cls = mapillary
        eval_scales = None
        if args.inference_mode == 'pooling':
            eval_mode = 'pooling'
            eval_scales = args.scales
        elif args.inference_mode == 'sliding':
            eval_mode = 'sliding'
        else:
            raise Exception(f"Not implemented inference mode: {args.inference_mode}")
        eval_size = 1536
        val_joint_transform_list = [
            joint_transforms.ResizeHeight(eval_size),
            joint_transforms.CenterCropPad(eval_size)]

        test_set = args.dataset_cls.Mapillary('semantic', args.split,joint_transform_list = val_joint_transform_list,
                                               transform = val_input_transform,
                                               target_transform = target_transform,
                                               eval_mode=eval_mode,
                                               eval_scales=eval_scales,
                                               eval_flip=not args.no_flip,
                                               image_in=args.image_in,
                                               test=False
                                               )
        args.dataset_cls.id_to_trainid = args.dataset_cls.id_to_ignore_or_group
    elif args.dataset == "video_folder":
        args.dataset_cls = video_folder
        eval_scales = None
        if args.inference_mode == 'pooling':
            eval_mode = 'pooling'
            eval_scales = args.scales
        elif args.inference_mode == 'sliding':
            eval_mode = 'sliding'
        else:
            raise Exception(f"Not implemented inference mode: {args.inference_mode}")

        assert args.videopath is not None, "you should define video folder absolute path"
        test_set = args.dataset_cls.video_folder(video_path = args.videopath,transform = val_input_transform,
                                               eval_mode=eval_mode,
                                               eval_scales=eval_scales,
                                               eval_flip=not args.no_flip,
                                               image_in=args.image_in
                                               )

        # args.dataset_cls.id_to_trainid = args.dataset_cls.trainid_to_trainid


    else:
        raise NameError('-------------Not Supported Currently-------------')

    if args.split_count > 1:
        test_set.split_dataset(args.split_index, args.split_count)

    batch_size = 1

    if args.syncbn:
        from datasets.sampler import DistributedSampler
        test_sampler = DistributedSampler(test_set, pad=False, permutation=False, consecutive_sample=False)
    else:
        test_sampler = None

    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=args.num_workers,
                             shuffle=False, drop_last=False, sampler = test_sampler)

    return test_loader


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


class RunEval():
    def __init__(self, output_dir, metrics, write_image, dataset_cls, inference_mode):
        self.output_dir = output_dir
        self.rgb_path = os.path.join(output_dir, 'rgb')
        self.pred_path = os.path.join(output_dir, 'pred')
        self.diff_path = os.path.join(output_dir, 'diff')
        self.compose_path = os.path.join(output_dir, 'compose')
        self.metrics = metrics

        self.write_image = write_image
        self.dataset_cls = dataset_cls
        self.inference_mode = inference_mode
        self.mapping = {}
        os.makedirs(self.rgb_path, exist_ok=True)
        os.makedirs(self.pred_path, exist_ok=True)
        os.makedirs(self.diff_path, exist_ok=True)
        os.makedirs(self.compose_path, exist_ok=True)

        if self.metrics:
            self.hist = np.zeros((self.dataset_cls.num_classes,
                                  self.dataset_cls.num_classes))
        else:
            self.hist = None

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

    def inf(self, imgs, img_names, gt, inference, net, scales, pbar, base_img):

        ######################################################################
        # Run inference
        ######################################################################
        to_pil = transforms.ToPILImage()
        if self.inference_mode == 'pooling':
            img = imgs
            pool_base_img = to_pil(base_img[0])
        else:
            img = to_pil(imgs[0])
        prediction_pre_argmax_collection = inference(net, img, scales)

        if self.inference_mode == 'pooling':
            prediction = prediction_pre_argmax_collection
            prediction = np.concatenate(prediction, axis=0)
        else:
            prediction_pre_argmax = np.mean(prediction_pre_argmax_collection, axis=0)
            prediction = np.argmax(prediction_pre_argmax, axis=0)

        if self.metrics:
            this_hist = fast_hist(prediction.flatten(), gt.cpu().numpy().flatten(),
                      self.dataset_cls.num_classes)
            self.hist += this_hist

            iou_w = round(np.nanmean(per_class_iu(self.hist)) * 100, 2)
            # acc_w = np.diag(self.hist).sum() / self.hist.sum()

            pbar.set_description("Mean IOU: %s" % (str(iou_w)))

        ######################################################################
        # Dump Images
        ######################################################################

        self.img_name = img_names[0]
        col_img_name = '{}/{}_color.png'.format(self.rgb_path,self.img_name)
        pred_img_name = '{}/{}.png'.format(self.pred_path, self.img_name)
        diff_img_name = '{}/{}_diff.png'.format(self.diff_path, self.img_name)
        compose_img_name = '{}/{}_compose.png'.format(self.compose_path, self.img_name)

        if self.write_image:

            if self.inference_mode == 'pooling':
                img = pool_base_img
            colorized = self.dataset_cls.colorize_mask(prediction)
            colorized.save(col_img_name)
            blend = Image.blend(img.convert("RGBA"), colorized.convert("RGBA"), 0.5)
            blend.save(compose_img_name)

            if gt is not None:
                gt = gt[0].cpu().numpy()
                # only write diff image if gt is valid
                diff = (prediction != gt)
                diff[gt == 255] = 0
                diffimg = Image.fromarray(diff.astype('uint8') * 255)
                PIL.ImageChops.lighter(
                    blend,
                    PIL.ImageOps.invert(diffimg).convert("RGBA")
                ).save(diff_img_name)

            label_out = np.zeros_like(prediction)
            for label_id, train_id in self.dataset_cls.id_to_trainid.items():
                label_out[np.where(prediction == train_id)] = label_id
            cv2.imwrite(pred_img_name, label_out)

    def final_dump(self):
        """
        Dump Final metrics on completion of evaluation
        """
        if self.metrics:
            print("Entire image")
            acc, acc_cls, mean_iu, fwavacc = evaluate_eval_for_inference(self.hist, args.dataset_cls)
            print("values: ", acc, acc_cls, mean_iu, fwavacc)


def main():
    """
    Main Function
    """
    scales = [float(x) for x in args.scales.split(',')]
    ckpt_path = os.path.join(os.path.split(args.snapshot)[0],os.path.splitext(os.path.split(args.snapshot)[1])[0])
    foldername = '{}_{}'.format(args.dataset,args.inference_mode)
    if args.videopath is not None:
        videopaths = args.videopath.strip(os.sep)
        videopaths = args.videopath.split(os.sep)
        foldername = '{}_{}_{}_{}'.format(foldername,videopaths[-3],videopaths[-2],videopaths[-1])
    if args.test_mode:
        output_dir = os.path.expanduser('~/experiment_pinmem/test/')
    else:
        output_dir = os.path.join(ckpt_path,foldername)
    os.makedirs(output_dir, exist_ok=True)
    save_log('eval', output_dir, date_str)
    logging.info("Network Arch: %s", args.arch)
    logging.info("CV split: %d", args.cv_split)
    logging.info("Exp_name: %s", args.exp)
    logging.info("Ckpt path: %s", ckpt_path)
    logging.info("Scales : %s", ' '.join(str(e) for e in scales))
    logging.info("Inference mode: %s", args.inference_mode)

    # Set up network, loader, inference mode
    if args.dataset == "video_folder":
        metrics=False
    else:
        metrics = True
    test_loader = setup_loader()

    runner = RunEval(output_dir, metrics,
                     write_image=args.dump_images,
                     dataset_cls=args.dataset_cls,
                     inference_mode=args.inference_mode)
    net = get_net()

    # Fix the ASPP pool size to 105, which is the tensor size if you train with crop
    # size of 840x840
    if args.fixed_aspp_pool:
        net.module.aspp.img_pooling = torch.nn.AvgPool2d(105)

    if args.inference_mode == 'sliding':
        inference = inference_sliding
    elif args.inference_mode == 'pooling':
        inference = inference_pool
    else:
        raise 'Not a valid inference mode: {}'.format(args.inference_mode)

    # Run Inference!
    pbar = tqdm(test_loader, desc='eval {}'.format(args.split), smoothing=1.0)
    for iteration, data in enumerate(pbar):
        if args.dataset == 'video_folder':
            if args.inference_mode == 'pooling':
                base_img, img, img_names = data
                base_img = base_img[0]
                imgs = img[0]
                gt = None
            else:
                imgs, img_names = data
                gt = None
                mask_aux = None
                base_img = None

        else:
            if args.inference_mode == 'pooling':
                base_img, gt_with_imgs, img_names = data
                base_img = base_img[0]
                imgs = gt_with_imgs[0]
                gt = gt_with_imgs[1]
            else:
                base_img = None
                imgs, gt, img_names, mask_aux = data

        runner.inf(imgs, img_names, gt, inference, net, scales, pbar, base_img)

        if iteration > 5 and args.test_mode:
            break

    # Calculate final overall statistics
    runner.final_dump()


if __name__ == '__main__':
    main()
