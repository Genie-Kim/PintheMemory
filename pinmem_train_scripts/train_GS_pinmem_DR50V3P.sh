#!/usr/bin/env bash
cd ..
python train.py \
--dataset gtav synthia \
--val_dataset cityscapes bdd100k mapillary \
--exp pinmem \
--arch network.deepv3plus.DeepR50V3PlusD \
--lr 0.01 \
--mldg \
--memory \
--bs_mult 4 \
--gblur \
--color_aug 0.5 \
--mem_readloss 0.02 \
--mem_divloss 0.4 \
--mem_clsloss 0.2 \
--mem_momentum 0.8 \
--inner_lr_anneal
