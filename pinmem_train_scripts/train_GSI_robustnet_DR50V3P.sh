#!/usr/bin/env bash
cd ..
python train.py \
--dataset gtav synthia idd \
--covstat_val_dataset gtav synthia idd \
--val_dataset cityscapes bdd100k mapillary \
--exp robustnet_DR50V3P16_GSI \
--arch network.deepv3plus.DeepR50V3PlusD \
--lr 0.01 \
--max_cu_epoch 10000 \
--rrotate 0 \
--bs_mult 4 \
--gblur \
--color_aug 0.5 \
--wt_reg_weight 0.6 \
--relax_denom 0.0 \
--clusters 3 \
--cov_stat_epoch 5 \
--trials 10 \
--wt_layer 0 0 2 2 2 0 0