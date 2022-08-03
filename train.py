"""
training code
"""
from __future__ import absolute_import
from __future__ import division
import argparse
import logging
import os
import torch
import datetime
import math
import pprint
import cv2

from config import cfg, assert_and_infer_cfg
from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
import datasets
import loss
import network
import optimizer
import time
import torchvision.utils as vutils
import torch.nn.functional as F
from network.mynn import freeze_weights, unfreeze_weights
import numpy as np
import random
from tqdm import tqdm
import PIL

# Enable CUDNN Benchmarking optimization
# torch.backends.cudnn.benchmark = True
random_seed = cfg.RANDOM_SEED  # 304
print("seed ", random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MemoryMetaFrameWork(object):

    def __init__(self, args):
        super(MemoryMetaFrameWork, self).__init__()

        self.args = args
        self.args.world_size = 1
        self.args.date = ''.join([x[0] for x in self.args.dataset])+'_source'
        exp_root_path = os.path.expanduser('~/experiment_pinmem/')
        self.args.ckpt = os.path.join(exp_root_path,self.args.ckpt)
        self.args.tb_path = os.path.join(exp_root_path,self.args.tb_path)

        if self.args.test_mode:
            self.args.exp = 'test'
            self.args.date = 'test'
            self.args.crop_size = 240
            self.args.bs_mult = 2
            self.args.trials = 1


        if 'WORLD_SIZE' in os.environ:
            # self.args.apex = int(os.environ['WORLD_SIZE']) > 1
            self.args.world_size = int(os.environ['WORLD_SIZE'])
            print("Total world size: ", int(os.environ['WORLD_SIZE']))

        torch.cuda.set_device(args.local_rank)
        print('My Rank:', self.args.local_rank)
        # Initialize distributed communication
        self.args.dist_url = self.args.dist_url + str(8000 + (int(time.time() % 1000)) // 10)

        torch.distributed.init_process_group(backend='nccl',
                                             init_method=args.dist_url,
                                             world_size=args.world_size,
                                             rank=args.local_rank)

        for i in range(len(args.wt_layer)):
            if self.args.wt_layer[i] == 1:
                self.args.use_wtloss = True
            if self.args.wt_layer[i] == 2:
                self.args.use_wtloss = True
                self.args.use_isw = True

        # Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
        assert_and_infer_cfg(self.args)
        self.writer = prep_experiment(self.args, parser)
        logging.info(pprint.pformat(vars(self.args)))
        self.train_loader, self.val_loaders, self.train_obj, self.extra_val_loaders, self.covstat_val_loaders = datasets.setup_loaders(self.args)

        self.criterion, self.criterion_val = loss.get_loss(self.args)
        self.criterion_aux = loss.get_loss_aux(self.args)
        self.net = network.get_net(self.args, self.criterion, self.criterion_aux)
        self.optim, self.scheduler = optimizer.get_optimizer(self.args, self.net)
        self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
        self.net = network.warp_network_in_dataparallel(self.net, self.args.local_rank)
        if self.args.mldg:
            self.updated_net = network.get_net(self.args, self.criterion, self.criterion_aux)
            self.updated_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.updated_net)
            self.updated_net = network.warp_network_in_dataparallel(self.updated_net, self.args.local_rank)
            self.updated_net2 = network.get_net(self.args, self.criterion, self.criterion_aux)
            self.updated_net2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.updated_net2)
            self.updated_net2 = network.warp_network_in_dataparallel(self.updated_net2, self.args.local_rank)

        self.epoch = 0
        self.i = 0
        if self.args.test_mode and not self.args.snapshot:
            self.args.max_iter = 60
            self.epoch = self.args.cov_stat_epoch

        if self.args.snapshot:
            self.epoch, mean_iu = optimizer.load_weights(self.net, self.optim, self.scheduler,
                                                    self.args.snapshot, self.args.restore_optimizer)
            if self.args.restore_optimizer is True:
                self.iter_per_epoch = len(self.train_loader)
                self.i = self.iter_per_epoch * self.epoch
            else:
                self.epoch = 0

        if self.args.memory and not self.args.snapshot:
            self.memory_initalize() # it should be before network init.

        self.tonum = lambda x: x.item() if type(x) == torch.Tensor else x


    # def target_norm_test(self): # For TSMLDG setting.(https://arxiv.org/abs/2003.12296)
    #     from optimizer import forgiving_state_restore
    #     checkpoint = torch.load(self.args.snapshot, map_location=torch.device('cpu'))
    #     if 'state_dict' in checkpoint:
    #         # target norm test
    #         print('target normalization!!!!!!validation batch size:', self.args.bs_mult_val)
    #         for dataset, val_loader in self.extra_val_loaders.items():
    #             self.net = forgiving_state_restore(self.net, checkpoint['state_dict'])
    #             logging.info("Checkpoint Load Compelete")
    #             self.validate(val_loader, dataset, self.criterion_val, save_pth=False)
    #     return 0

    def do_epoch(self):
        """
        Main Function
        """

        torch.cuda.empty_cache()

        while self.i < self.args.max_iter:
            # Update EPOCH CTR
            cfg.immutable(False)
            cfg.ITER = self.i
            cfg.immutable(True)
            # iteration start

            if self.args.memory:
                if self.args.mldg:
                    self.train_memory_mldg()
                else:
                    self.train_memory_agg()
            else:
                if self.args.mldg:
                    self.train_mldg()
                else:
                    self.train_agg()

            if self.args.snapshot and self.args.test_mode:
                print('lr : ', self.optim.param_groups[-1]['lr'])

            self.train_loader.sampler.set_epoch(self.epoch + 1)

            # for isw robustnet option
            if (self.args.dynamic and self.args.use_isw and self.epoch % (self.args.cov_stat_epoch + 1) == self.args.cov_stat_epoch) \
               or (self.args.dynamic is False and self.args.use_isw and self.epoch == self.args.cov_stat_epoch):
                self.net.module.reset_mask_matrix()
                for trial in range(self.args.trials):
                    for dataset, val_loader in self.covstat_val_loaders.items():  # For get the statistics of covariance
                        self.validate_for_cov_stat(val_loader)
                        self.net.module.set_mask_matrix()

            # class uniform sampler.
            if self.args.class_uniform_pct:
                if self.epoch >= self.args.max_cu_epoch:
                    self.train_obj.build_epoch(cut=True)
                    self.train_loader.sampler.set_num_samples()
                else:
                    self.train_obj.build_epoch()

            self.epoch += 1
            if self.epoch > self.args.max_epoch:
                # saving present model parameter
                if self.args.local_rank == 0:
                    print("Saving pth file...")
                    evaluate_eval(self.args, self.net, self.optim, self.scheduler, None, None, [],
                                  self.writer, self.epoch, "None", None, self.i, save_pth=True)

                for dataset, val_loader in self.extra_val_loaders.items():
                    print("Extra validating... This won't save pth file")
                    self.validate(val_loader, dataset, self.criterion_val, save_pth=False)

                break

    def meta_transform(self,justidx = False):
        # this must called before dataloader enumerate.
        # meteidx is meta test dataset idx, and it will be hard augmentated
        D = len(self.train_loader.dataset.datasets)
        split_idx = np.random.permutation(D)
        i = np.random.randint(1, D)
        metridx = split_idx[:i]
        meteidx = split_idx[i:]
        if ~justidx:
            for i in range(D):
                if i in meteidx:
                    self.train_loader.dataset.datasets[0].running_metatest = True
        return metridx, meteidx

    def calculate_loss(self,outputs,train_total_loss,batch_pixel_size):
        outputs_index = 0
        main_loss = outputs[outputs_index]
        outputs_index += 1
        aux_loss = outputs[outputs_index]
        outputs_index += 1
        if self.args.no_aux_loss:
            total_loss = main_loss + (0.0 * aux_loss)
        else:
            total_loss = main_loss + (0.4 * aux_loss)

        if self.args.use_wtloss and (not self.args.use_isw or (self.args.use_isw and self.epoch > self.args.cov_stat_epoch)):
            wt_loss = outputs[outputs_index]
            outputs_index += 1
            total_loss = total_loss + (self.args.wt_reg_weight * wt_loss)
        else:
            wt_loss = 0

        if self.args.memory:
            readloss = outputs[-2]
            writeloss = outputs[-3]
            total_loss = total_loss + self.args.mem_readloss * readloss + self.args.mem_divloss * writeloss[0] + self.args.mem_clsloss * writeloss[1]
        else:
            readloss = 0
            writeloss = [0,0]

        log_total_loss = total_loss.clone().detach_()
        torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
        log_total_loss = log_total_loss / self.args.world_size
        train_total_loss.update(log_total_loss.item(), batch_pixel_size)

        return total_loss, log_total_loss, outputs_index, wt_loss,readloss,writeloss,main_loss

    def get_updated_network(self,old, new, lr, load=False):
        updated_theta = {}
        state_dicts = old.state_dict()
        param_dicts = dict(old.named_parameters())

        for i, (k, v) in enumerate(state_dicts.items()):
            if k in param_dicts.keys() and param_dicts[k].grad is not None:
                updated_theta[k] = param_dicts[k] - lr * param_dicts[k].grad
            else:
                updated_theta[k] = state_dicts[k]
        if load:
            new.load_state_dict(updated_theta)
        else:
            new = self.put_theta(new, updated_theta)
        return new

    def put_theta(self,model, theta):
        def k_param_fn(tmp_model, name=None):
            if len(tmp_model._modules) != 0:
                for (k, v) in tmp_model._modules.items():
                    if name is None:
                        k_param_fn(v, name=str(k))
                    else:
                        k_param_fn(v, name=str(name + '.' + k))
            else:
                for (k, v) in tmp_model._parameters.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    tmp_model._parameters[k] = theta[str(name + '.' + k)]

        k_param_fn(model)
        return model

    def tfb_log(self,losses):
        for k, v in losses.items():
            self.writer.add_scalar('loss/' + k, v, self.i)

    # training framework
    def train_memory_agg(self):

        self.net.train()

        train_total_loss = AverageMeter()
        time_meter = AverageMeter()

        self.i = self.epoch * len(self.train_loader)

        for i, data in enumerate(self.train_loader):
            if self.i >= self.args.max_iter:
                break

            inputs, gts, _, aux_gts = data

            C, H, W = inputs.shape[-3:]

            # if train_loader is multi domains, merge multi domain to 1 batch.
            input = inputs.view(-1,C,H,W)
            gt = gts.view(-1,H,W)
            aux_gt = aux_gts.view(-1,H,W)

            img_gt = None
            input, gt, aux_gt = input.cuda(), gt.cuda(), aux_gt.cuda()

            batch_pixel_size = C * H * W
            start_ts = time.time()

            mem_t = self.net.module.memory.m_items.clone().detach()  # save initial memory

            self.optim.zero_grad()
            if self.args.use_isw:
                outputs = self.net(input, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=self.args.visualize_feature,
                              apply_wtloss=False if self.epoch <= self.args.cov_stat_epoch else True,memory_writing=True,writing_detach = False)
            else:
                outputs = self.net(input, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=self.args.visualize_feature,memory_writing=True,writing_detach = False)

            total_loss, _, outputs_index, wt_loss, reading_loss, writing_loss, seg_loss = self.calculate_loss(outputs, train_total_loss, batch_pixel_size)

            total_loss.backward()
            self.optim.step() # self.net update.

            if self.args.visualize_feature:
                f_cor_arr = outputs[outputs_index]
                outputs_index += 1

            with torch.no_grad(): # final memory update.
                self.net.eval()
                self.net.module.memory.m_items = mem_t # reset to initial memory
                self.net(input, gts=gt, aux_gts=aux_gt, img_gt=img_gt,
                                   visualize=self.args.visualize_feature, memory_writing=True)
                self.net.train()

            time_meter.update(time.time() - start_ts)
            # iteration done
            if self.args.local_rank == 0:
                if i % 50 == 49:
                    if self.args.visualize_feature:
                        self.visualize_matrix(f_cor_arr,'/Covariance/Feature-')

                    msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                        self.epoch, i + 1, len(self.train_loader), self.i, train_total_loss.avg,
                        self.optim.param_groups[-1]['lr'], time_meter.avg / self.args.train_batch_size)

                    logging.info(msg)
                    if self.args.use_wtloss:
                        print("Whitening Loss", wt_loss)

                    # Log tensorboard metrics for each iteration of the training phase
                    losses = {
                        'train_loss': train_total_loss.avg,
                        'seg_loss' : self.tonum(seg_loss),

                        'total_loss': self.tonum(total_loss),
                        'whitening_Loss': self.tonum(wt_loss),
                        'reading_Loss': self.tonum(reading_loss),
                        'writing_Loss_div': self.tonum(writing_loss[0]),
                        'writing_Loss_cls': self.tonum(writing_loss[1]),

                        'lr': self.optim.param_groups[-1]['lr'],
                    }
                    self.tfb_log(losses)
                    train_total_loss.reset()
                    time_meter.reset()

            self.i += 1
            self.scheduler.step()

            if i > 5 and self.args.test_mode:
                return 0
        return 0


    def train_mldg(self):

        self.net.train()

        train_total_loss = AverageMeter()
        time_meter = AverageMeter()

        self.i = self.epoch * len(self.train_loader)

        train_idx, test_idx = self.meta_transform(justidx=False)

        for i, data in enumerate(self.train_loader):
            if self.i >= self.args.max_iter:
                break

            inputs, gts, _, aux_gts = data

            # Multi source and AGG case
            assert len(inputs.shape) == 5, "mldg memory should dimension 5"
            B, D, C, H, W = inputs.shape
            num_domains = D
            inputs = inputs.transpose(0, 1).cuda()
            gts = gts.transpose(0, 1).squeeze(2).cuda()
            aux_gts = aux_gts.transpose(0, 1).squeeze(2).cuda()

            img_gt = None

            meta_train_input = inputs[train_idx,:].reshape(-1, C, H, W)
            meta_train_gt = gts[train_idx, :].reshape(-1, 1, H, W).squeeze(1)
            meta_train_auggt = aux_gts[train_idx, :].reshape(-1, 1, H, W).squeeze(1)

            meta_test_input = inputs[test_idx, :].reshape(-1, C, H, W)
            meta_test_gt = gts[test_idx, :].reshape(-1, 1, H, W).squeeze(1)
            meta_test_auggt = aux_gts[test_idx, :].reshape(-1, 1, H, W).squeeze(1)

            batch_pixel_size = C * H * W
            start_ts = time.time()
            self.optim.zero_grad()
            if self.args.use_isw:
                outputs = self.net(meta_train_input, gts=meta_train_gt, aux_gts=meta_train_auggt, img_gt=img_gt, visualize=self.args.visualize_feature,
                              apply_wtloss=False if self.epoch <= self.args.cov_stat_epoch else True)
            else:
                outputs = self.net(meta_train_input, gts=meta_train_gt, aux_gts=meta_train_auggt, img_gt=img_gt, visualize=self.args.visualize_feature)

            total_inner_loss, _, outputs_index, wt_inner_loss, _,_, seg_loss = self.calculate_loss(outputs, train_total_loss, batch_pixel_size)
            total_inner_loss.backward(retain_graph = True)

            self.updated_net = self.get_updated_network(self.net, self.updated_net, self.args.inner_lr).train().cuda()

            if self.args.visualize_feature:
                f_cor_arr = outputs[outputs_index]
                outputs_index += 1


            # meta test
            if self.args.use_isw:
                outputs = self.updated_net(meta_test_input, gts=meta_test_gt, aux_gts=meta_test_auggt, img_gt=img_gt,
                              visualize=self.args.visualize_feature,
                              apply_wtloss=False if self.epoch <= self.args.cov_stat_epoch else True)
            else:
                outputs = self.updated_net(meta_test_input, gts=meta_test_gt, aux_gts=meta_test_auggt, img_gt=img_gt,
                              visualize=self.args.visualize_feature)

            total_outer_loss, _, outputs_index, wt_outer_loss,  _,_, seg_loss = self.calculate_loss(outputs, train_total_loss, batch_pixel_size)

            total_outer_loss.backward()
            self.optim.step()

            time_meter.update(time.time() - start_ts)

            # iteration done

            if self.args.local_rank == 0:
                if i % 50 == 49:
                    if self.args.visualize_feature:
                        self.visualize_matrix(f_cor_arr, '/Covariance/Feature-')

                    msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                        self.epoch, i + 1, len(self.train_loader), self.i, train_total_loss.avg,
                        self.optim.param_groups[-1]['lr'], time_meter.avg / self.args.train_batch_size)

                    logging.info(msg)
                    if self.args.use_wtloss:
                        print("Whitening Loss", wt_outer_loss)

                    # Log tensorboard metrics for each iteration of the training phasage

                    losses = {
                        'train_loss' : train_total_loss.avg,
                        'seg_loss': self.tonum(seg_loss),

                        'total_outer_loss': self.tonum(total_outer_loss),
                        'outer_whitening_Loss' : self.tonum(wt_outer_loss),

                        'total_inner_loss': self.tonum(total_inner_loss),
                        'inner_whitening_Loss' : self.tonum(wt_inner_loss),

                        'lr': self.optim.param_groups[-1]['lr'],
                        'inner_lr':self.args.inner_lr
                    }
                    self.tfb_log(losses)

                    train_total_loss.reset()
                    time_meter.reset()

            self.i += 1
            self.scheduler.step()
            if self.args.inner_lr_anneal:
                self.args.inner_lr = self.optim.param_groups[-1]['lr']/4

            if i > 5 and self.args.test_mode:
                return 0
            # meta learning runtime augmentation.
            train_idx, test_idx = self.meta_transform(justidx=False)
        return 0

    def train_memory_mldg(self):
        self.net.train()

        train_total_loss = AverageMeter()
        time_meter = AverageMeter()

        self.i = self.epoch * len(self.train_loader)

        train_idx, test_idx = self.meta_transform(justidx=False)

        for i, data in enumerate(self.train_loader):
            if self.i >= self.args.max_iter:
                break

            inputs, gts, _, aux_gts = data

            # Multi source and AGG case
            assert len(inputs.shape) == 5, "mldg memory should dimension 5"
            B, D, C, H, W = inputs.shape
            num_domains = D
            inputs = inputs.transpose(0, 1).cuda()
            gts = gts.transpose(0, 1).squeeze(2).cuda()
            aux_gts = aux_gts.transpose(0, 1).squeeze(2).cuda()

            img_gt = None

            meta_train_input = inputs[train_idx,:].reshape(-1, C, H, W)
            meta_train_gt = gts[train_idx, :].reshape(-1, 1, H, W).squeeze(1)
            meta_train_auggt = aux_gts[train_idx, :].reshape(-1, 1, H, W).squeeze(1)

            meta_test_input = inputs[test_idx, :].reshape(-1, C, H, W)
            meta_test_gt = gts[test_idx, :].reshape(-1, 1, H, W).squeeze(1)
            meta_test_auggt = aux_gts[test_idx, :].reshape(-1, 1, H, W).squeeze(1)

            batch_pixel_size = C * H * W
            start_ts = time.time()

            mem_t = self.net.module.memory.m_items.clone().detach() # save initial memory
            with torch.autograd.set_detect_anomaly(True):
                self.optim.zero_grad()
                if self.args.use_isw:
                    outputs = self.net(meta_train_input, gts=meta_train_gt, aux_gts=meta_train_auggt, img_gt=img_gt, visualize=self.args.visualize_feature,
                                  apply_wtloss=False if self.epoch <= self.args.cov_stat_epoch else True,memory_writing=True,writing_detach = False)
                else:
                    outputs = self.net(meta_train_input, gts=meta_train_gt, aux_gts=meta_train_auggt, img_gt=img_gt, visualize=self.args.visualize_feature,memory_writing=True,writing_detach = False)

                total_inner_loss, _, outputs_index, wt_inner_loss, reading_inner_loss, writing_inner_loss, seg_loss = self.calculate_loss(outputs, train_total_loss, batch_pixel_size)

                total_inner_loss.backward(retain_graph = True)

                self.updated_net = self.get_updated_network(self.net, self.updated_net, self.args.inner_lr).train().cuda()


                self.updated_net2 = self.get_updated_network(self.net, self.updated_net2, self.args.inner_lr).train().cuda()
                self.updated_net2.module.memory.m_items = mem_t  # memory sync
                # freeze encoder
                for k, v in self.updated_net2.named_parameters():
                    if k.split(".")[1] is not 'memory':
                        v.detach_()
                        v.requires_grad_(False)

                # meta test sub step, seen domain memory write
                self.updated_net2(meta_train_input, gts=meta_train_gt, aux_gts=meta_train_auggt, img_gt=img_gt,
                                 visualize=self.args.visualize_feature, memory_writing=True,writing_detach = False)

                self.updated_net.module.memory.m_items = self.updated_net2.module.memory.m_items.clone()  # memory sync

                if self.args.visualize_feature:
                    f_cor_arr = outputs[outputs_index]
                    outputs_index += 1

                # meta test(memory_writing False -->  do not writing)
                if self.args.use_isw:
                    outputs = self.updated_net(meta_test_input, gts=meta_test_gt, aux_gts=meta_test_auggt, img_gt=img_gt,
                                  visualize=self.args.visualize_feature,
                                  apply_wtloss=False if self.epoch <= self.args.cov_stat_epoch else True,memory_writing=False)
                else:
                    outputs = self.updated_net(meta_test_input, gts=meta_test_gt, aux_gts=meta_test_auggt, img_gt=img_gt,
                                  visualize=self.args.visualize_feature,memory_writing=False)

                total_outer_loss, _, outputs_index, wt_outer_loss,reading_outer_loss, _, seg_loss = self.calculate_loss(outputs, train_total_loss, batch_pixel_size)

                total_outer_loss.backward()
                self.optim.step() # self.net update.

                with torch.no_grad(): # final memory update.
                    self.net.eval()
                    self.net.module.memory.m_items = mem_t # reset to initial memory
                    self.net(meta_train_input, gts=meta_train_gt, aux_gts=meta_train_auggt, img_gt=img_gt,
                                       visualize=self.args.visualize_feature, memory_writing=True)
                    self.net.train()

                time_meter.update(time.time() - start_ts)
                # iteration done

            if self.args.local_rank == 0:
                if i % 50 == 49:
                    if self.args.visualize_feature:
                        self.visualize_matrix(f_cor_arr, '/Covariance/Feature-')

                    msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                        self.epoch, i + 1, len(self.train_loader), self.i, train_total_loss.avg,
                        self.optim.param_groups[-1]['lr'], time_meter.avg / self.args.train_batch_size)

                    logging.info(msg)
                    if self.args.use_wtloss:
                        print("Whitening Loss", wt_outer_loss)

                    # Log tensorboard metrics for each iteration of the training phasage
                    losses = {
                        'train_loss' : train_total_loss.avg,
                        'seg_loss': self.tonum(seg_loss),

                        'total_outer_loss': self.tonum(total_outer_loss),
                        'outer_whitening_Loss' : self.tonum(wt_outer_loss),
                        'outer_reading_Loss': self.tonum(reading_outer_loss),

                        'total_inner_loss': self.tonum(total_inner_loss),
                        'inner_whitening_Loss' : self.tonum(wt_inner_loss),
                        'inner_reading_Loss': self.tonum(reading_inner_loss),
                        'inner_writing_Loss_div': self.tonum(writing_inner_loss[0]),
                        'inner_writing_Loss_cls': self.tonum(writing_inner_loss[1]),

                        'lr': self.optim.param_groups[-1]['lr'],
                        'inner_lr': self.args.inner_lr,
                    }
                    self.tfb_log(losses)
                    train_total_loss.reset()
                    time_meter.reset()

            self.i += 1
            self.scheduler.step()
            if self.args.inner_lr_anneal:
                self.args.inner_lr = self.optim.param_groups[-1]['lr']/4

            if i > 5 and self.args.test_mode:
                return 0
            # meta learning runtime augmentation.
            train_idx, test_idx = self.meta_transform(justidx=False)
        return 0

    def train_agg(self):
        """
        Runs the training loop per epoch
        train_loader: Data loader for train
        net: thet network
        optimizer: optimizer
        curr_epoch: current epoch
        writer: tensorboard writer
        return:
        """
        self.net.train()

        train_total_loss = AverageMeter()
        time_meter = AverageMeter()

        self.i = self.epoch * len(self.train_loader)

        for i, data in enumerate(self.train_loader):
            if self.i >= self.args.max_iter:
                break

            inputs, gts, _, aux_gts = data

            C, H, W = inputs.shape[-3:]

            # if train_loader is multi domains, merge multi domain to 1 batch.
            input = inputs.view(-1,C,H,W)
            gt = gts.view(-1,H,W)
            aux_gt = aux_gts.view(-1,H,W)

            batch_pixel_size = C * H * W
            start_ts = time.time()

            img_gt = None
            input, gt = input.cuda(), gt.cuda()

            self.optim.zero_grad()
            if self.args.use_isw:
                outputs = self.net(input, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=self.args.visualize_feature,
                            apply_wtloss=False if self.epoch<=self.args.cov_stat_epoch else True)
            else:
                outputs = self.net(input, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=self.args.visualize_feature)

            total_loss, _, outputs_index, wt_loss, _, _, seg_loss = self.calculate_loss(outputs,
                                                                                train_total_loss,
                                                                                batch_pixel_size)
            if self.args.visualize_feature:
                f_cor_arr = outputs[outputs_index]
                outputs_index += 1

            log_total_loss = total_loss.clone().detach_()
            torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
            log_total_loss = log_total_loss / self.args.world_size
            train_total_loss.update(log_total_loss.item(), batch_pixel_size)

            total_loss.backward()
            self.optim.step()

            time_meter.update(time.time() - start_ts)

            if self.args.local_rank == 0:
                if i % 50 == 49:
                    if self.args.visualize_feature:
                        self.visualize_matrix(f_cor_arr,'/Covariance/Feature-')

                    msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                        self.epoch, i + 1, len(self.train_loader), self.i, train_total_loss.avg,
                        self.optim.param_groups[-1]['lr'], time_meter.avg / self.args.train_batch_size)

                    logging.info(msg)
                    if self.args.use_wtloss:
                        print("Whitening Loss", wt_loss)

                    # Log tensorboard metrics for each iteration of the training phase
                    losses = {
                        'train_loss': train_total_loss.avg,
                        'seg_loss': self.tonum(seg_loss),

                        'total_loss': self.tonum(total_loss),
                        'whitening_Loss': self.tonum(wt_loss),
                        'lr': self.optim.param_groups[-1]['lr'],
                    }
                    self.tfb_log(losses)
                    train_total_loss.reset()
                    time_meter.reset()

            self.i += 1
            self.scheduler.step()

            if i > 5 and self.args.test_mode:
                return 0

        return 0


    def train_robustnetver(self):
        """
        Runs the training loop per epoch
        train_loader: Data loader for train
        net: thet network
        optimizer: optimizer
        curr_epoch: current epoch
        writer: tensorboard writer
        return:
        """
        self.net.train()

        train_total_loss = AverageMeter()
        time_meter = AverageMeter()

        self.i = self.epoch * len(self.train_loader)

        for i, data in enumerate(self.train_loader):
            if self.i >= self.args.max_iter:
                break

            inputs, gts, _, aux_gts = data

            # Multi source and AGG case
            if len(inputs.shape) == 5:
                B, D, C, H, W = inputs.shape
                num_domains = D
                inputs = inputs.transpose(0, 1)
                gts = gts.transpose(0, 1).squeeze(2)
                aux_gts = aux_gts.transpose(0, 1).squeeze(2)

                # put data per domain at each list index
                inputs = [input.squeeze(0) for input in torch.chunk(inputs, num_domains, 0)]
                gts = [gt.squeeze(0) for gt in torch.chunk(gts, num_domains, 0)]
                aux_gts = [aux_gt.squeeze(0) for aux_gt in torch.chunk(aux_gts, num_domains, 0)]
            else:
                B, C, H, W = inputs.shape
                num_domains = 1
                inputs = [inputs]
                gts = [gts]
                aux_gts = [aux_gts]

            batch_pixel_size = C * H * W
            # iterate per each domain
            for di, ingredients in enumerate(zip(inputs, gts, aux_gts)):
                input, gt, aux_gt = ingredients

                start_ts = time.time()

                img_gt = None
                input, gt = input.cuda(), gt.cuda()

                self.optim.zero_grad()
                if self.args.use_isw:
                    outputs = self.net(input, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=self.args.visualize_feature,
                                apply_wtloss=False if self.epoch<=self.args.cov_stat_epoch else True)
                else:
                    outputs = self.net(input, gts=gt, aux_gts=aux_gt, img_gt=img_gt, visualize=self.args.visualize_feature)
                outputs_index = 0
                main_loss = outputs[outputs_index]
                outputs_index += 1
                aux_loss = outputs[outputs_index]
                outputs_index += 1
                total_loss = main_loss + (0.4 * aux_loss)

                if self.args.use_wtloss and (not self.args.use_isw or (self.args.use_isw and self.epoch > self.args.cov_stat_epoch)):
                    wt_loss = outputs[outputs_index]
                    outputs_index += 1
                    total_loss = total_loss + (self.args.wt_reg_weight * wt_loss)
                else:
                    wt_loss = 0

                if self.args.visualize_feature:
                    f_cor_arr = outputs[outputs_index]
                    outputs_index += 1

                log_total_loss = total_loss.clone().detach_()
                torch.distributed.all_reduce(log_total_loss, torch.distributed.ReduceOp.SUM)
                log_total_loss = log_total_loss / self.args.world_size
                train_total_loss.update(log_total_loss.item(), batch_pixel_size)

                total_loss.backward()
                self.optim.step()

                time_meter.update(time.time() - start_ts)

                del total_loss, log_total_loss

                if self.args.local_rank == 0:
                    if i % 50 == 49:
                        if self.args.visualize_feature:
                            self.visualize_matrix(f_cor_arr,'/Covariance/Feature-')

                        msg = '[epoch {}], [iter {} / {} : {}], [loss {:0.6f}], [lr {:0.6f}], [time {:0.4f}]'.format(
                            self.epoch, i + 1, len(self.train_loader), self.i, train_total_loss.avg,
                            self.optim.param_groups[-1]['lr'], time_meter.avg / self.args.train_batch_size)

                        logging.info(msg)
                        if self.args.use_wtloss:
                            print("Whitening Loss", wt_loss)

                        # Log tensorboard metrics for each iteration of the training phase
                        losses = {
                            'train_loss': train_total_loss.avg,
                            'lr': self.optim.param_groups[-1]['lr'],
                        }
                        self.tfb_log(losses)
                        train_total_loss.reset()
                        time_meter.reset()

            self.i += 1
            self.scheduler.step()

            if i > 5 and self.args.test_mode:
                return 0

        return 0

    def validate(self, val_loader, dataset, criterion, save_pth=True):
        """
        Runs the validation loop after each training epoch
        val_loader: Data loader for validation
        dataset: dataset name (str)
        net: thet network
        criterion: loss fn
        optimizer: optimizer
        curr_epoch: current epoch
        writer: tensorboard writer
        return: val_avg for step function if required
        """
        self.net.eval()
        val_loss = AverageMeter()
        if self.args.memory:
            read_loss = AverageMeter()
        iou_acc = 0
        error_acc = 0
        dump_images = []

        for val_idx, data in enumerate(val_loader):
            # input        = torch.Size([1, 3, 713, 713])
            # gt_image           = torch.Size([1, 713, 713])
            inputs, gt_image, img_names, _ = data

            if len(inputs.shape) == 5:
                B, D, C, H, W = inputs.shape
                inputs = inputs.view(-1, C, H, W)
                gt_image = gt_image.view(-1, 1, H, W)

            assert len(inputs.size()) == 4 and len(gt_image.size()) == 3
            assert inputs.size()[2:] == gt_image.size()[1:]

            batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)
            inputs, gt_cuda = inputs.cuda(), gt_image.cuda()

            with torch.no_grad():
                if self.args.use_wtloss:
                    outputs = self.net(inputs, visualize=True)
                    output, f_cor_arr = outputs[0], outputs[1]
                else:
                    outputs = self.net(inputs)
                    output = outputs[0]

                if self.args.memory:
                    query = F.normalize(outputs[-1].clone(), dim=1)
                    query = query.permute(0, 2, 3, 1).contiguous()  # b X h X w X d
                    reading_loss = self.net.module.memory.get_score(query,gt_cuda,self.net.module.memory.m_items)[-1]
                    # read_loss.update(reading_loss.item(), 1)
                    read_loss.update(reading_loss.item(), batch_pixel_size)

            del inputs

            assert output.size()[2:] == gt_image.size()[1:]
            assert output.size()[1] == datasets.num_classes

            val_loss.update(criterion(output, gt_cuda).item(), batch_pixel_size)

            del gt_cuda

            # Collect data from different GPU to a single GPU since
            # encoding.parallel.criterionparallel function calculates distributed loss
            # functions
            predictions = output.data.max(1)[1].cpu()

            # Logging
            if val_idx % 20 == 0:
                if self.args.local_rank == 0:
                    logging.info("validating: %d / %d", val_idx + 1, len(val_loader))
            if val_idx > 10 and self.args.test_mode:
                break

            # Image Dumps
            if val_idx < 10:
                dump_images.append([gt_image, predictions, img_names])

            iou_acc += fast_hist(predictions.numpy().flatten(), gt_image.numpy().flatten(),
                                 datasets.num_classes)
            del output, val_idx, data

        iou_acc_tensor = torch.cuda.FloatTensor(iou_acc)
        torch.distributed.all_reduce(iou_acc_tensor, op=torch.distributed.ReduceOp.SUM)
        iou_acc = iou_acc_tensor.cpu().numpy()

        if self.args.local_rank == 0:
            evaluate_eval(self.args, self.net, self.optim, self.scheduler, val_loss, iou_acc, dump_images,
                        self.writer, self.epoch, dataset, None, self.i, save_pth=save_pth)
            if self.args.memory:
                self.writer.add_scalar('{}/read_loss'.format(dataset), read_loss.avg, self.i)
            if self.args.use_wtloss:
                self.visualize_matrix(f_cor_arr,'/Covariance/Feature-')

        return val_loss.avg

    def validate_for_cov_stat(self,val_loader):
        """
        Runs the validation loop after each training epoch
        val_loader: Data loader for validation
        dataset: dataset name (str)
        net: thet network
        criterion: loss fn
        optimizer: optimizer
        curr_epoch: current epoch
        writer: tensorboard writer
        return: val_avg for step function if required
        """
        # net.train()#eval()
        self.net.eval()

        for val_idx, data in enumerate(val_loader):
            img_or, img_photometric, img_geometric, img_name = data   # img_geometric is not used.
            img_or, img_photometric = img_or.cuda(), img_photometric.cuda()

            with torch.no_grad():
                self.net([img_photometric, img_or], cal_covstat=True)

            del img_or, img_photometric, img_geometric

            # Logging
            if val_idx % 20 == 0:
                if self.args.local_rank == 0:
                    logging.info("validating: %d / 100", val_idx + 1)
            del data

            if self.args.test_mode and val_idx >= 10:
                return
            elif val_idx >= 499:
                return


    def visualize_matrix(self, matrix_arr, title_str):
        stage = 'valid'

        for i in range(len(matrix_arr)):
            C = matrix_arr[i].shape[1]
            matrix = matrix_arr[i][0].unsqueeze(0)    # 1 X C X C
            matrix = torch.clamp(torch.abs(matrix), max=1)
            matrix = torch.cat((torch.ones(1, C, C).cuda(), torch.abs(matrix - 1.0),
                            torch.abs(matrix - 1.0)), 0)
            matrix = vutils.make_grid(matrix, padding=5, normalize=False, range=(0,1))
            self.writer.add_image(stage + title_str + str(i), matrix, self.i)


    def save_feature_numpy(self, feature_maps):
        file_fullpath = '/home/userA/projects/visualization/feature_map/'
        file_name = str(self.args.date) + '_' + str(self.args.exp)
        B, C, H, W = feature_maps.shape
        for i in range(B):
            feature_map = feature_maps[i]
            feature_map = feature_map.data.cpu().numpy()   # H X D
            file_name_post = '_' + str(self.i * B + i)
            np.save(file_fullpath + file_name + file_name_post, feature_map)

    def memory_initalize(self):
        self.net.eval()
        with torch.no_grad():
            basket = torch.zeros(size = self.net.module.memory.m_items.size()).cuda()
            count = torch.zeros(size = (self.args.mem_slot,1)).cuda()

            for epoch in range(2):
                for it, (inputs, gts, _, aux_gts) in enumerate(tqdm(self.train_loader,desc="memory initializing...epoch " + str(epoch))):

                    C, H, W = inputs.size()[-3:]
                    input = inputs.reshape(-1, C, H, W).cuda()
                    gt = gts.reshape(-1, H, W).cuda()
                    aux_gt = aux_gts.reshape(-1, H, W).cuda()
                    outputs = self.net(input, gts=gt, aux_gts=aux_gt)
                    query = outputs[-1]

                    query = F.normalize(query, dim=1) # normalize the input feature
                    batch_size, dims, h, w = query.size()

                    ### update supervised memory
                    query = query.view(batch_size, dims, -1)
                    gt[gt == 255] = self.args.mem_slot  # when supervised memory, memory size = class number
                    gt = F.one_hot(gt, num_classes=self.args.mem_slot + 1)

                    gt = F.interpolate(gt.permute(0, 3, 1, 2).contiguous().type(torch.float32), [h, w],
                                             mode='bilinear', align_corners=True).permute(0, 2, 3, 1).contiguous()

                    gt = gt.view(batch_size, -1, self.args.mem_slot + 1)
                    denominator = gt.sum(1).unsqueeze(dim=1)
                    nominator = torch.matmul(query, gt)

                    count += torch.t(denominator[:,:,:self.args.mem_slot].sum(0))
                    basket += torch.t(nominator[:,:,:self.args.mem_slot].sum(0))

                    if self.args.test_mode:
                        if it > 10:
                            break

            count[count == 0] = 1 # for nan
            init_prototypes = torch.div(basket, count)
            self.net.module.memory.m_items = F.normalize(init_prototypes, dim=1)

        self.net.train()


def parse_for_modelassign(parser):
    parser.add_argument('--arch', type=str, default='network.deepv3plus.DeepWV3Plus',
                        help='Network architecture. We have DeepSRNX50V3PlusD (backbone: ResNeXt50) \
                        and deepWV3Plus (backbone: WideResNet38).')
    # for loss
    parser.add_argument('--img_wt_loss', action='store_true', default=False,
                        help='per-image class-weighted loss')
    parser.add_argument('--cls_wt_loss', action='store_true', default=False,
                        help='class-weighted loss')
    parser.add_argument('--jointwtborder', action='store_true', default=False,
                        help='Enable boundary label relaxation')
    parser.add_argument('-wb', '--wt_bound', type=float, default=1.0,
                        help='Weight Scaling for the losses')
    # for whitning layers
    parser.add_argument('--wt_layer', nargs='*', type=int, default=[0, 0, 0, 0, 0, 0, 0],
                        help='0: None, 1: IW/IRW, 2: ISW, 3: IS, 4: IN (IBNNet: 0 0 4 4 4 0 0)')
    parser.add_argument('--clusters', type=int, default=50)
    parser.add_argument('--use_wtloss', action='store_true', default=False,
                        help='Automatic setting from wt_layer')
    parser.add_argument('--relax_denom', type=float, default=0.0) # robustnet : 2.0

    # for memory modules
    parser.add_argument('--memory', action='store_true', default=False,
                        help='Using memory network')
    parser.add_argument('--mem_slot', type=int, default=19,
                        help='number of memory slot')
    parser.add_argument('--mem_dim',  type=int, default=256,
                        help='memory feature dimension')
    parser.add_argument('--mem_momentum',  type=float, default=0.8,
                        help='memory update momentum')
    parser.add_argument('--mem_temp',  type=float, default=1,
                        help='memory reading loss temperature')
    parser.add_argument('--gumbel_off', action='store_true', default=False,
                        help='Do not Use gumbel softmax for read')
    return parser



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Semantic Segmentation')
    parser = parse_for_modelassign(parser)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dataset', nargs='*', type=str, default=['gtav','synthia'],
                        help='a list of datasets; cityscapes, mapillary, camvid, kitti, gtav, mapillary, synthia')
    parser.add_argument('--image_uniform_sampling', action='store_true', default=False,
                        help='uniformly sample images across the multiple source domains')
    parser.add_argument('--val_dataset', nargs='*', type=str, default=['cityscapes','bdd100k','mapillary'],
                        help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
    parser.add_argument('--covstat_val_dataset', nargs='*', type=str, default=['gtav'],
                        help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
    parser.add_argument('--cv', type=int, default=0,
                        help='cross-validation split id to use. Default # of splits set to 3 in config')
    parser.add_argument('--class_uniform_pct', type=float, default=0,
                        help='What fraction of images is uniformly sampled')
    parser.add_argument('--class_uniform_tile', type=int, default=1024,
                        help='tile size for class uniform sampling')
    parser.add_argument('--coarse_boost_classes', type=str, default=None,
                        help='use coarse annotations to boost fine data with specific classes')

    parser.add_argument('--batch_weighting', action='store_true', default=False,
                        help='Batch weighting for class (use nll class weighting using batch stats')

    parser.add_argument('--strict_bdr_cls', type=str, default='',
                        help='Enable boundary label relaxation for specific classes')
    parser.add_argument('--rlx_off_iter', type=int, default=-1,
                        help='Turn off border relaxation after specific epoch count')
    parser.add_argument('--rescale', type=float, default=1.0,
                        help='Warm Restarts new learning rate ratio compared to original lr')
    parser.add_argument('--repoly', type=float, default=1.5,
                        help='Warm Restart new poly exp')

    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Use Nvidia Apex AMP')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='parameter used by apex library')

    parser.add_argument('--sgd', action='store_true', default=True)
    parser.add_argument('--adam', action='store_true', default=False)
    parser.add_argument('--amsgrad', action='store_true', default=False)

    parser.add_argument('--freeze_trunk', action='store_true', default=False)
    parser.add_argument('--hardnm', default=0, type=int,
                        help='0 means no aug, 1 means hard negative mining iter 1,' +
                             '2 means hard negative mining iter 2')

    parser.add_argument('--trunk', type=str, default='resnet101',
                        help='trunk model, can be: resnet101 (default), resnet50')
    parser.add_argument('--max_iter', type=int, default=120000) # 40000 for original
    parser.add_argument('--epoch_per_val', type=int, default=2)

    parser.add_argument('--max_cu_epoch', type=int, default=10000,
                        help='Class Uniform Max Epochs')
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--crop_nopad', action='store_true', default=False)
    parser.add_argument('--rrotate', type=int,
                        default=0, help='degree of random roate')
    parser.add_argument('--color_aug', type=float,
                        default=0.0, help='level of color augmentation')
    parser.add_argument('--gblur', action='store_true', default=False,
                        help='Use Guassian Blur Augmentation')
    parser.add_argument('--bblur', action='store_true', default=False,
                        help='Use Bilateral Blur Augmentation')
    parser.add_argument('--lr_schedule', type=str, default='exp',
                        help='name of lr schedule: poly')
    parser.add_argument('--poly_exp', type=float, default=9,
                        help='polynomial LR exponent or gamma of exponential lr scheduler')
    parser.add_argument('--bs_mult', type=int, default=4,
                        help='Batch size for training per gpu')
    parser.add_argument('--bs_mult_val', type=int, default=1,
                        help='Batch size for Validation per gpu')
    parser.add_argument('--crop_size', type=int, default=768,
                        help='training crop size')
    parser.add_argument('--pre_size', type=int, default=None,
                        help='resize image shorter edge to this before augmentation')
    parser.add_argument('--scale_min', type=float, default=0.5,
                        help='dynamically scale training images down to this size')
    parser.add_argument('--scale_max', type=float, default=2.0,
                        help='dynamically scale training images up to this size')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--snapshot', type=str, default=None) # should increase maxiter option
    parser.add_argument('--restore_optimizer', action='store_true', default=False, help='restore optimizer and scheduler')

    parser.add_argument('--city_mode', type=str, default='train',
                        help='experiment directory date name')
    parser.add_argument('--date', type=str, default='1111',
                        help='experiment directory date name')
    parser.add_argument('--exp', type=str, default='default',
                        help='experiment directory name')
    parser.add_argument('--tb_tag', type=str, default='',
                        help='add tag to tb dir')
    parser.add_argument('--ckpt', type=str, default='./',
                        help='Save Checkpoint Point')
    parser.add_argument('--tb_path', type=str, default='./',
                        help='Save Tensorboard Path')
    parser.add_argument('--syncbn', action='store_true', default=True,
                        help='Use Synchronized BN')
    parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                        help='Dump Augmentated Images for sanity check')
    parser.add_argument('--test_mode', action='store_true', default=False,
                        help='Minimum testing to verify nothing failed, ' +
                             'Runs code for 1 epoch of train and val')
    parser.add_argument('--maxSkip', type=int, default=0,
                        help='Skip x number of  frames of video augmented dataset')
    parser.add_argument('--scf', action='store_true', default=False,
                        help='scale correction factor')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:', type=str,
                        help='url used to set up distributed training')

    parser.add_argument('--wt_reg_weight', type=float, default=0.0)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--dynamic', action='store_true', default=False)

    parser.add_argument('--image_in', action='store_true', default=False,
                        help='Input Image Instance Norm')
    parser.add_argument('--cov_stat_epoch', type=int, default=0,
                        help='cov_stat_epoch')
    parser.add_argument('--visualize_feature', action='store_true', default=False,
                        help='Visualize intermediate feature')
    parser.add_argument('--use_isw', action='store_true', default=False,
                        help='Automatic setting from wt_layer')

    parser.add_argument('--inner_lr', type=float, default=0.001)
    parser.add_argument('--mldg', action='store_true', default=False,
                        help='Do meta learning')
    parser.add_argument('--no_aux_loss', action='store_true', default=False,
                        help='Do meta learning')
    parser.add_argument('--mem_readloss', type=float, default=0.02)
    parser.add_argument('--mem_divloss', type=float, default=0.4)
    parser.add_argument('--mem_clsloss', type=float, default=0.2)
    parser.add_argument('--inner_lr_anneal', action='store_true', default=False,
                        help='Do meta learning inner learning rate annealing')



    args = parser.parse_args()
    framework = MemoryMetaFrameWork(args)
    # for training.
    framework.do_epoch()