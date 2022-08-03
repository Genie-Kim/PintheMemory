import logging
import json
import os
import numpy as np
from PIL import Image, ImageCms
from skimage import color

from torch.utils import data
import torch
import torchvision.transforms as transforms
import datasets.uniform as uniform
import datasets.cityscapes_labels as cityscapes_labels
import copy

from config import cfg

trainid_to_name = cityscapes_labels.trainId2name
id_to_trainid = cityscapes_labels.label2trainid
num_classes = 19
ignore_label = 255

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153,
           153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
           255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)


def colorize_mask(mask):
    """
    Colorize a segmentation mask.
    """
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


def make_dataset_video(img_path):
    """
    Create Filename list for the dataset
    """

    items = []
    images = os.listdir(img_path)
    for name in images:
        items.append(os.path.join(img_path,name))
    return items

class video_folder(data.Dataset):

    def __init__(self,video_path,transform=None, dump_images=False, eval_mode=False,
                 eval_scales=None, eval_flip=False, image_in=False, extract_feature=False):
        self.transform = transform
        self.dump_images = dump_images
        self.eval_mode = eval_mode
        self.eval_flip = eval_flip
        self.eval_scales = None
        self.image_in = image_in
        self.extract_feature = extract_feature

        if eval_scales != None:
            self.eval_scales = [float(scale) for scale in eval_scales.split(",")]

        self.imgs = make_dataset_video(video_path)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def _eval_get_item(self, img, scales, flip_bool):
        return_imgs = []
        for flip in range(int(flip_bool) + 1):
            imgs = []
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for scale in scales:
                w, h = img.size
                target_w, target_h = int(w * scale), int(h * scale)
                resize_img = img.resize((target_w, target_h))
                tensor_img = transforms.ToTensor()(resize_img)
                final_tensor = transforms.Normalize(*self.mean_std)(tensor_img)
                imgs.append(final_tensor)
            return_imgs.append(imgs)
        return return_imgs

    def __getitem__(self, index):

        img_path = self.imgs[index]

        img = Image.open(img_path).convert('RGB')
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        if self.eval_mode == 'pooling':
            return [transforms.ToTensor()(img)], self._eval_get_item(img, self.eval_scales,
                                                                     self.eval_flip), img_name
        if self.transform is not None:
            img = self.transform(img)

        if not self.eval_mode:
            rgb_mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            if self.image_in:
                eps = 1e-5
                rgb_mean_std = ([torch.mean(img[0]), torch.mean(img[1]), torch.mean(img[2])],
                        [torch.std(img[0])+eps, torch.std(img[1])+eps, torch.std(img[2])+eps])
            img = transforms.Normalize(*rgb_mean_std)(img)

        # Debug
        if self.dump_images:
            outdir = '../../dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            out_img_fn = os.path.join(outdir, img_name + '.png')
            img.save(out_img_fn)

        return img,img_name

    def __len__(self):
        return len(self.imgs)
