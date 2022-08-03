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
import torch
from torch import nn
from network.deepv3plus import DeepV3Plus
from network.instance_whitening import instance_whitening_loss, get_covariance_matrix
from network.mynn import initialize_weights, Norm2d, Upsample, freeze_weights, unfreeze_weights



class DeepV3(DeepV3Plus):
    """
    Implement DeepLab-V3 model
    No skip connections
    """

    def __init__(self, num_classes, trunk='resnet-101', criterion=None, criterion_aux=None,
                variant='D', skip='m1', skip_num=48, args=None):
        super(DeepV3, self).__init__(num_classes, trunk=trunk, criterion=criterion, criterion_aux=criterion_aux,
                variant=variant, skip=skip, skip_num=skip_num, args=args)

        self.final1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))
        self.final2 = nn.Sequential(
            nn.Conv2d(256, num_classes, kernel_size=1, bias=True))
        
        initialize_weights(self.final1)
        initialize_weights(self.final2)

        del self.bot_fine

    def forward(self, x, gts=None, aux_gts=None, img_gt=None, visualize=False, cal_covstat=False, apply_wtloss=True,memory_writing = False,writing_detach = True): # reading detach false is also validation condition.
        w_arr = []

        if cal_covstat:
            x = torch.cat(x, dim=0)

        x_size = x.size()  # 800

        if self.trunk == 'mobilenetv2' or self.trunk == 'shufflenetv2':
            x_tuple = self.layer0([x, w_arr])
            x = x_tuple[0]
            w_arr = x_tuple[1]
        else:   # ResNet
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

        if cal_covstat:
            for index, f_map in enumerate(w_arr):
                # Instance Whitening
                B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
                HW = H * W
                f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
                eye, reverse_eye = self.cov_matrix_layer[index].get_eye_matrix()
                f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW - 1) + (self.eps * eye)  # B X C X C / HW
                off_diag_elements = f_cor * reverse_eye
                #print("here", off_diag_elements.shape)
                self.cov_matrix_layer[index].set_variance_of_covariance(torch.var(off_diag_elements, dim=0))
            return 0

        x = self.aspp(x) # concat result of diverse kernel size of aspp
        dec0_up = self.bot_aspp(x) # concat to 256 dimension
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

        dec0 = self.final1(dec0_up)
        dec1 = self.final2(dec0)
        main_out = Upsample(dec1, x_size[2:])

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


def DeepR18V3D(args, num_classes, criterion, criterion_aux):
    """
    Resnet 18 Based Network
    """
    print("Model : DeepLabv3, Backbone : ResNet-18")
    return DeepV3(num_classes, trunk='resnet-18', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D32', skip='m1', args=args)


def DeepR50V3D_OS8(args, num_classes, criterion, criterion_aux):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3, Backbone : ResNet-50")
    return DeepV3(num_classes, trunk='resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepR50V3D(args, num_classes, criterion, criterion_aux):
    """
    Resnet 50 Based Network
    """
    print("Model : DeepLabv3, Backbone : ResNet-50")
    return DeepV3(num_classes, trunk='resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepR101V3D(args, num_classes, criterion, criterion_aux):
    """
    Resnet 101 Based Network
    """
    print("Model : DeepLabv3, Backbone : ResNet-101")
    return DeepV3(num_classes, trunk='resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepR101V3D_OS8(args, num_classes, criterion, criterion_aux):
    """
    Resnet 101 Based Network
    """
    print("Model : DeepLabv3, Backbone : ResNet-101")
    return DeepV3(num_classes, trunk='resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)


def DeepR152V3D_OS8(args, num_classes, criterion, criterion_aux):
    """
    Resnet 152 Based Network
    """
    print("Model : DeepLabv3, Backbone : ResNet-152")
    return DeepV3(num_classes, trunk='resnet-152', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)



def DeepResNext50V3D(args, num_classes, criterion, criterion_aux):
    """
    Resnext 50 Based Network
    """
    print("Model : DeepLabv3, Backbone : ResNext-50 32x4d")
    return DeepV3(num_classes, trunk='resnext-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepResNext101V3D(args, num_classes, criterion, criterion_aux):
    """
    Resnext 101 Based Network
    """
    print("Model : DeepLabv3, Backbone : ResNext-101 32x8d")
    return DeepV3(num_classes, trunk='resnext-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepWideResNet50V3D(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 50 Based Network
    """
    print("Model : DeepLabv3, Backbone : wide_resnet-50")
    return DeepV3(num_classes, trunk='wide_resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepWideResNet50V3D_OS8(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 50 Based Network
    """
    print("Model : DeepLabv3, Backbone : wide_resnet-50")
    return DeepV3(num_classes, trunk='wide_resnet-50', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepWideResNet101V3D(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 101 Based Network
    """
    print("Model : DeepLabv3, Backbone : wide_resnet-101")
    return DeepV3(num_classes, trunk='wide_resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepWideResNet101V3D_OS8(args, num_classes, criterion, criterion_aux):
    """
    Wide ResNet 101 Based Network
    """
    print("Model : DeepLabv3, Backbone : wide_resnet-101")
    return DeepV3(num_classes, trunk='wide_resnet-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)


def DeepResNext101V3D_OS8(args, num_classes, criterion, criterion_aux):
    """
    ResNext 101 Based Network
    """
    print("Model : DeepLabv3, Backbone : resnext-101")
    return DeepV3(num_classes, trunk='resnext-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepResNext101V3D_OS4(args, num_classes, criterion, criterion_aux):
    """
    ResNext 101 Based Network
    """
    print("Model : DeepLabv3, Backbone : resnext-101")
    return DeepV3(num_classes, trunk='resnext-101', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D4', skip='m1', args=args)

def DeepShuffleNetV3D_OS32(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3, Backbone : shufflenetv2")
    return DeepV3(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D32', skip='m1', args=args)


def DeepMNASNet05V3D(args, num_classes, criterion, criterion_aux):
    """
    MNASNET Based Network
    """
    print("Model : DeepLabv3, Backbone : mnas_0_5")
    return DeepV3(num_classes, trunk='mnasnet_05', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepMNASNet10V3D(args, num_classes, criterion, criterion_aux):
    """
    MNASNET Based Network
    """
    print("Model : DeepLabv3, Backbone : mnas_1_0")
    return DeepV3(num_classes, trunk='mnasnet_10', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)


def DeepShuffleNetV3D(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3, Backbone : shufflenetv2")
    return DeepV3(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepMobileNetV3D(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3, Backbone : mobilenetv2")
    return DeepV3(num_classes, trunk='mobilenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D16', skip='m1', args=args)

def DeepMobileNetV3D_OS8(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3, Backbone : mobilenetv2")
    return DeepV3(num_classes, trunk='mobilenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)

def DeepShuffleNetV3D_OS8(args, num_classes, criterion, criterion_aux):
    """
    ShuffleNet Based Network
    """
    print("Model : DeepLabv3, Backbone : shufflenetv2")
    return DeepV3(num_classes, trunk='shufflenetv2', criterion=criterion, criterion_aux=criterion_aux,
                    variant='D', skip='m1', args=args)
