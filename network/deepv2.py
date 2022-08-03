"""
# Code Adapted from:
# https://github.com/sthalles/deeplab_v3
#
# MIT License
#
# Copyright (c) 2018 Thalles Santos Silva
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
"""
import logging
import torch
from torch import nn
from network import Resnet
from network import Mobilenet
from network import Shufflenet
from network import memory
from network.cov_settings import CovMatrix_ISW, CovMatrix_IRW
from network.instance_whitening import instance_whitening_loss, get_covariance_matrix
from network.mynn import initialize_weights, Norm2d, Upsample, freeze_weights, unfreeze_weights

import torchvision.models as models


class _ASPPofDeeplabv2(nn.Module):
    def __init__(self, inplanes, dilation_series=[6, 12, 18, 24], padding_series=[6, 12, 18, 24], outdim=256):
        super(_ASPPofDeeplabv2, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Sequential(nn.Conv2d(inplanes, outdim, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                                         bias=False),
                               Norm2d(outdim),
                               nn.ReLU(inplace=True),
                               )
            )

    def forward(self, x):
        out0 = self.conv2d_list[0](x)
        out1 = out0 + self.conv2d_list[1](x)
        out2 = out1 + self.conv2d_list[2](x)
        out3 = out2 + self.conv2d_list[3](x)
        return out3


class DeepV2(nn.Module):
    """
    Implement DeepLab-V2 model
    No output stride option!!!! (os8)
    """

    def __init__(self, num_classes, trunk='resnet-101', criterion=None, criterion_aux=None,
                variant='D', skip='m1', skip_num=48, args=None):
        super(DeepV2, self).__init__()
        self.criterion = criterion
        self.criterion_aux = criterion_aux
        self.variant = variant
        self.args = args
        self.trunk = trunk

        channel_1st = 3
        channel_2nd = 64
        channel_3rd = 256
        channel_4th = 512
        prev_final_channel = 1024
        final_channel = 2048

        if trunk == 'resnet-18':
            channel_1st = 3
            channel_2nd = 64
            channel_3rd = 64
            channel_4th = 128
            prev_final_channel = 256
            final_channel = 512
            resnet = Resnet.resnet18(wt_layer=self.args.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-50':
            resnet = Resnet.resnet50(pretrained=True,wt_layer=self.args.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101': # three 3 X 3
            resnet = Resnet.resnet101(pretrained=True, wt_layer=self.args.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-152':
            resnet = Resnet.resnet152()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnext-50':
            resnet = models.resnext50_32x4d(pretrained=True)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnext-101':
            resnet = models.resnext101_32x8d(pretrained=True)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'wide_resnet-50':
            resnet = models.wide_resnet50_2(pretrained=True)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'wide_resnet-101':
            resnet = models.wide_resnet101_2(pretrained=True)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D':
            os = 8
            self.layer2[0].conv1.stride=(2,2)
            self.layer2[0].conv2.stride=(1,1)
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            raise 'unknown deepv2 variant: {}'.format(self.variant)
            # print("Not using Dilation ")

        self.output_stride = os
        self.aspp = _ASPPofDeeplabv2(final_channel)
        self.final1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))
        self.final2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))
        self.dsn = nn.Sequential(
            nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
            Norm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        initialize_weights(self.dsn)
        initialize_weights(self.aspp)
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        # if trunk == 'resnet-101': # robustnet's resnet101
        #     self.three_input_layer = True
        #     in_channel_list = [64, 64, 128, 256, 512, 1024, 2048]   # 8128, 32640, 130816
        #     out_channel_list = [32, 32, 64, 128, 256,  512, 1024]
        if trunk == 'resnet-18':
            self.three_input_layer = False
            in_channel_list = [0, 0, 64, 64, 128, 256, 512]   # 8128, 32640, 130816
            out_channel_list = [0, 0, 32, 32, 64,  128, 256]
        elif trunk == 'shufflenetv2':
            self.three_input_layer = False
            in_channel_list = [0, 0, 24, 116, 232, 464, 1024]
        elif trunk == 'mobilenetv2':
            self.three_input_layer = False
            in_channel_list = [0, 0, 16, 32, 64, 320, 1280]
        else: # ResNet-50
            self.three_input_layer = False
            in_channel_list = [0, 0, 64, 256, 512, 1024, 2048]   # 8128, 32640, 130816
            out_channel_list = [0, 0, 32, 128, 256,  512, 1024]

        self.cov_matrix_layer = []
        self.cov_type = []
        assert self.args.wt_layer==[0,0,0,0,0,0,0], "deeplabv2 did not fit with robustnet"
        for i in range(len(self.args.wt_layer)):
            if self.args.wt_layer[i] > 0:
                self.whitening = True
                if self.args.wt_layer[i] == 1:
                    self.cov_matrix_layer.append(CovMatrix_IRW(dim=in_channel_list[i], relax_denom=self.args.relax_denom))
                    self.cov_type.append(self.args.wt_layer[i])
                elif self.args.wt_layer[i] == 2:
                    self.cov_matrix_layer.append(CovMatrix_ISW(dim=in_channel_list[i], relax_denom=self.args.relax_denom, clusters=self.args.clusters))
                    self.cov_type.append(self.args.wt_layer[i])

        if self.args.memory:
            self.memory = memory.Memory_sup(memory_size=self.args.mem_slot, input_feature_dim=self.args.mem_dim,
                                            feature_dim=self.args.mem_dim, momentum=self.args.mem_momentum,
                                            temperature=self.args.mem_temp, gumbel_read=(not self.args.gumbel_off))


    def set_mask_matrix(self):
        for index in range(len(self.cov_matrix_layer)):
            self.cov_matrix_layer[index].set_mask_matrix()


    def reset_mask_matrix(self):
        for index in range(len(self.cov_matrix_layer)):
            self.cov_matrix_layer[index].reset_mask_matrix()


    def forward(self, x, gts=None, aux_gts=None, img_gt=None, visualize=False, cal_covstat=False, apply_wtloss=True,memory_writing = False,writing_detach = True):
        # reading detach false is also validation condition.
        w_arr = []

        if cal_covstat:
            x = torch.cat(x, dim=0)

        x_size = x.size()  # 800


        if self.three_input_layer: # resnet 101 setting
            x = self.layer0[0](x)
            if self.args.wt_layer[0] == 1 or self.args.wt_layer[0] == 2:
                x, w = self.layer0[1](x)
                w_arr.append(w)
            else:
                x = self.layer0[1](x)
            x = self.layer0[2](x)
            x = self.layer0[3](x)
            if self.args.wt_layer[1] == 1 or self.args.wt_layer[1] == 2:
                x, w = self.layer0[4](x)
                w_arr.append(w)
            else:
                x = self.layer0[4](x)
            x = self.layer0[5](x)
            x = self.layer0[6](x)
            if self.args.wt_layer[2] == 1 or self.args.wt_layer[2] == 2:
                x, w = self.layer0[7](x)
                w_arr.append(w)
            else:
                x = self.layer0[7](x)
            x = self.layer0[8](x)
            x = self.layer0[9](x)
        else:   # Single Input Layer
            x = self.layer0[0](x)
            if self.args.wt_layer[2] == 1 or self.args.wt_layer[2] == 2:
                x, w = self.layer0[1](x)
                w_arr.append(w)
            else:
                x = self.layer0[1](x)
            x = self.layer0[2](x)
            x = self.layer0[3](x)

        x_tuple = self.layer1([x, w_arr])  # 400
        low_level = x_tuple[0]

        x_tuple = self.layer2(x_tuple)  # 100
        x_tuple = self.layer3(x_tuple)  # 100
        aux_out = x_tuple[0]
        x_tuple = self.layer4(x_tuple)  # 100
        x = x_tuple[0]
        w_arr = x_tuple[1]

        dec0_up = self.aspp(x) # concat result of diverse kernel size of aspp
        inter_feature = dec0_up.clone()
        # memory operation
        if self.args.memory:
            # reading and writing
            dec0_up, softmax_score_query, softmax_score_memory, readloss, writeloss = self.memory(dec0_up, gts, memory_writing,writing_detach)
            # if memory_writing: # reading detach가 false 일때는 현재 framework에선 meta test 밖에 없음.
            #     # writing(permutation important)
            #     writeloss = self.memory.write(mem_input_feat, gts, writing_detach)

            mem_output = [softmax_score_query, softmax_score_memory, dec0_up.clone().detach()]

        ###################

        dec1 = self.final1(dec0_up)
        dec2 = self.final2(dec1)
        main_out = Upsample(dec2, x_size[2:])

        if self.training:
            loss1 = self.criterion(main_out, gts)

            if self.args.use_wtloss:
                wt_loss = torch.FloatTensor([0]).cuda()
                if apply_wtloss:
                    for index, f_map in enumerate(w_arr):
                        eye, mask_matrix, margin, num_remove_cov = self.cov_matrix_layer[index].get_mask_matrix()
                        loss = instance_whitening_loss(f_map, eye, mask_matrix, margin, num_remove_cov)
                        wt_loss = wt_loss + loss
                wt_loss = wt_loss / len(w_arr)

            aux_out = self.dsn(aux_out)
            if aux_gts.dim() == 1:
                aux_gts = gts
            aux_gts = aux_gts.unsqueeze(1).float()
            aux_gts = nn.functional.interpolate(aux_gts, size=aux_out.shape[2:], mode='nearest')
            aux_gts = aux_gts.squeeze(1).long()
            loss2 = self.criterion_aux(aux_out, aux_gts)

            return_loss = [loss1, loss2]

            if self.args.use_wtloss:
                return_loss.append(wt_loss)

            if self.args.use_wtloss and visualize:
                f_cor_arr = []
                for f_map in w_arr:
                    f_cor, _ = get_covariance_matrix(f_map)
                    f_cor_arr.append(f_cor)
                return_loss.append(f_cor_arr)
            # adding memory outputs last.
            if self.args.memory:
                return_loss.append(mem_output)
                return_loss.append(writeloss)
                return_loss.append(readloss)
            return_loss.append(inter_feature)

            return return_loss
        else:
            outputs = [main_out]
            if visualize:
                f_cor_arr = []
                for f_map in w_arr:
                    f_cor, _ = get_covariance_matrix(f_map)
                    f_cor_arr.append(f_cor)

                outputs.append(f_cor_arr)

            if self.args.memory:
                outputs.append(mem_output)
            outputs.append(inter_feature)

            return outputs


def get_final_layer(model):
    unfreeze_weights(model.final)
    return model.final


def DeepR50V2D(args, num_classes, criterion, criterion_aux):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv2, Backbone : ResNet-50")
    return DeepV2(num_classes, trunk='resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)


def DeepR101V2D(args, num_classes, criterion, criterion_aux):
    """
    Resnet 101 Based Network
    """
    print("Model : DeepLabv2, Backbone : ResNet-101")
    return DeepV2(num_classes, trunk='resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)