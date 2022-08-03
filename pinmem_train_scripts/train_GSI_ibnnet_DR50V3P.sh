#!/usr/bin/env bash
cd ..
python train.py \
--dataset gtav synthia idd \
--covstat_val_dataset gtav synthia idd \
--val_dataset cityscapes bdd100k mapillary \
--arch network.deepv3plus.DeepR50V3PlusD \
--exp ibnnet_DR50V3P16_GSI \
--lr 0.01 \
--max_cu_epoch 10000 \
--crop_size 768 \
--rrotate 0 \
--bs_mult 4 \
--gblur \
--color_aug 0.5 \
--wt_reg_weight 0.0 \
--relax_denom 0.0 \
--cov_stat_epoch 0 \
--wt_layer 0 0 4 4 4 0 0
