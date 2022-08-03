#!/usr/bin/env bash
cd ..
python train.py \
--dataset gtav synthia \
--val_dataset cityscapes bdd100k mapillary \
--exp baseline \
--arch network.deepv3plus.DeepR50V3PlusD \
--lr 0.01 \
--bs_mult 4 \
--gblur \
--color_aug 0.5
