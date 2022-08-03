import torch
import torch.nn as nn
import numpy as np
import math
import functools
from torch.nn import functional as F
from transforms.transforms import HideAndSeek

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()

def random_uniform(shape, low, high, cuda):
    x = torch.rand(*shape)
    result_cpu = (high - low) * x + low
    if cuda:
        return result_cpu.cuda()
    else:
        return result_cpu

def distance(a, b):
    return torch.sqrt(((a - b) ** 2).sum()).unsqueeze(0)


def distance_batch(a, b):
    bs, _ = a.shape
    result = distance(a[0], b)
    for i in range(bs - 1):
        result = torch.cat((result, distance(a[i], b)), 0)

    return result


def multiply(x):  # to flatten matrix into a vector
    return functools.reduce(lambda x, y: x * y, x, 1)


def flatten(x):
    """ Flatten matrix into a vector """
    count = multiply(x.size())
    return x.resize_(count)


def index(batch_size, x):
    idx = torch.arange(0, batch_size).long()
    idx = torch.unsqueeze(idx, -1)
    return torch.cat((idx, x), dim=1)


def MemoryLoss(memory):
    m, d = memory.size()
    memory_t = torch.t(memory)
    similarity = (torch.matmul(memory, memory_t)) / 2 + 1 / 2  # 30X30
    identity_mask = torch.eye(m).cuda()
    sim = torch.abs(similarity - identity_mask)

    return torch.sum(sim) / (m * (m - 1))

class Writingnet(nn.Module):
    def __init__(self,input_feature_dim, feature_dim):
        super(Writingnet, self).__init__()
        # refer object contextual represenation network fusion layer...
        assert input_feature_dim == feature_dim,"Should match when residual mode is on ({} != {})".format(input_feature_dim,feature_dim)


        self.writefeat = nn.Sequential(  # refer object contextual represenation network fusion layer...
            nn.Conv2d(input_feature_dim, feature_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(feature_dim)
        )
        self.relu = nn.ReLU(inplace=True)


        initialize_weights(self)

    def forward(self,x):
        output = x + self.writefeat(x)
        output = self.relu(output)

        return output






class Memory_sup(nn.Module):
    def __init__(self, memory_size, input_feature_dim, feature_dim, momentum, temperature, gumbel_read):
        super(Memory_sup, self).__init__()
        # Constants
        self.memory_size = memory_size  # when supervised memory, set same number with class num(19)
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.initial_momentum = momentum
        self.temperature = temperature
        self.output = nn.Sequential(  # refer object contextual represenation network fusion layer...
            nn.Conv2d(feature_dim * 2, input_feature_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(input_feature_dim),
            nn.ReLU(inplace=True),
        )
        self.writenet = Writingnet(input_feature_dim, feature_dim)


        self.mem_cls = torch.tensor([x for x in range(self.memory_size)]).cuda()
        self.clsfier = nn.Linear(in_features=self.feature_dim, out_features=self.memory_size, bias=True)


        self.celoss = nn.CrossEntropyLoss(ignore_index=255)
        self.gumbel_read = gumbel_read

        self.writeTF = lambda x: x.clone()

        self.m_items = F.normalize(torch.rand((memory_size, feature_dim), dtype=torch.float),
                                   dim=1).cuda()  # Initialize the memory items
        initialize_weights(self)

    def hard_neg_mem(self, mem, i):
        similarity = torch.matmul(mem, torch.t(self.keys_var))
        similarity[:, i] = -1
        _, max_idx = torch.topk(similarity, 1, dim=1)

        return self.keys_var[max_idx]

    def random_pick_memory(self, mem, max_indices):

        m, d = mem.size()
        output = []
        for i in range(m):
            flattened_indices = (max_indices == i).nonzero()
            a, _ = flattened_indices.size()
            if a != 0:
                number = np.random.choice(a, 1)
                output.append(flattened_indices[number, 0])
            else:
                output.append(-1)

        return torch.tensor(output)

    def get_update_query(self, mem, max_indices, update_indices, score, query):

        m, d = mem.size()
        query_update = torch.zeros((m, d)).cuda()
        random_update = torch.zeros((m, d)).cuda()
        for i in range(m):
            idx = torch.nonzero(max_indices.squeeze(1) == i)
            a, _ = idx.size()
            # ex = update_indices[0][i]
            if a != 0:
                # random_idx = torch.randperm(a)[0]
                # idx = idx[idx != ex]
                #                     query_update[i] = torch.sum(query[idx].squeeze(1), dim=0)
                query_update[i] = torch.sum(((score[idx, i] / torch.max(score[:, i])) * query[idx].squeeze(1)), dim=0)
                # random_update[i] = query[random_idx] * (score[random_idx,i] / torch.max(score[:,i]))
            else:
                query_update[i] = 0
                # random_update[i] = 0

            return query_update

    def get_score(self, query,mask,mem):
        bs, h, w, d = query.size()
        m, d = mem.size()

        score = torch.matmul(query, torch.t(mem))  # b X h X w X m
        if mask is not None:
            read_loss_score = score / self.temperature
            read_loss_score = read_loss_score.permute(0, 3, 1, 2).contiguous()  # b X m X h X w
            read_loss_score = F.interpolate(read_loss_score, size=mask.shape[1:], mode='bilinear', align_corners=True)
            readloss = self.celoss(read_loss_score, mask)
        else:
            readloss = 0
        score = score.view(bs * h * w, m)  # (b X h X w) X m

        if self.gumbel_read:
            # using Gumbel softmax for categorical memory sampling.
            score_query = F.gumbel_softmax(score, dim=0)  # 특정 메모리 슬롯이 들어오는 query들과 어떤 관계인지. 확률을 sampling.
            score_memory = F.gumbel_softmax(score, dim=1)  # 특정 query vector과 메모리들과 어떤 관계인지.
        else:
            score_query = F.softmax(score, dim=0)  # 특정 메모리 슬롯이 들어오는 query들과 어떤 관계인지.
            score_memory = F.softmax(score, dim=1)  # 특정 query vector과 메모리들과 어떤 관계인지.

        return score_query, score_memory, readloss

    def forward(self, query,mask=None, memory_writing=True,writing_detach = True):

        batch_size, dims, h, w = query.size()  # b X d X h X w

        # read
        updated_query, softmax_score_query, softmax_score_memory, readloss = self.read(query,mask,memory_writing)

        # write(permutation important)
        if memory_writing:
            writeloss = self.write(query, mask, writing_detach)
        else:
            writeloss = [0,0]

        return updated_query, softmax_score_query, softmax_score_memory, readloss, writeloss

    def write(self, input, mask, writing_detach = True):

        tempmask = mask.clone().detach()
        query = input.clone()
        if not writing_detach:
            query = self.writeTF(query) # only hide and seek for learning writing
        momentum = self.momentum
        ### get writing feature.
        query = self.writenet(query)
        query = F.normalize(query, dim=1)
        batch_size, dims, h, w = query.size()

        ### update supervised memory
        query = query.view(batch_size, dims, -1)
        tempmask[tempmask == 255] = self.memory_size  # when supervised memory, memory size = class number
        tempmask = F.one_hot(tempmask, num_classes=self.memory_size + 1)

        tempmask = F.interpolate(tempmask.permute(0, 3, 1, 2).contiguous().type(torch.float32), [h,w], mode='bilinear', align_corners=True).permute(0,2,3,1).contiguous()

        tempmask = tempmask.view(batch_size, -1, self.memory_size + 1)
        denominator = tempmask.sum(1).unsqueeze(dim=1)
        nominator = torch.matmul(query, tempmask)

        nominator = torch.t(nominator.sum(0))  # batchwise sum
        denominator = denominator.sum(0)  # batchwise sum
        denominator = denominator.squeeze()

        updated_memory = self.m_items.clone().detach() # memory write operation should detach from the previous memory
        for slot in range(self.memory_size):
            if denominator[slot] != 0: # If there is no corresponding class sample in the current batch, the memory is not updated.
                updated_memory[slot] = momentum * self.m_items[slot] + (
                            (1 - momentum) * nominator[slot] / denominator[slot])  # memory momentum update

        updated_memory = F.normalize(updated_memory, dim=1)  # normalize.

        ### get diversity loss about updated memory.
        div_loss = self.diversityloss(updated_memory)

        ### get classification loss about updated memory
        if type(self.clsfier) == type(None):
            cls_loss = torch.tensor(0).cuda()
        else:
            cls_loss = self.classification_loss(updated_memory)

        writing_loss = [div_loss, cls_loss]

        if writing_detach: # detaching memory from the present batch images.
            self.m_items = updated_memory.detach()
            return writing_loss
        else:
            self.m_items = updated_memory
            return writing_loss

    def classification_loss(self, mem):

        score = self.clsfier(mem)
        return self.celoss(score, self.mem_cls)

    def diversityloss(self, mem):
        # it is same with orthonomal constraints.
        cos_sim = torch.matmul(mem,torch.t(mem))
        margin = 0 # margin정도 까지는 similar해도 봐준다.(diversity loss는 orthonormal loss랑 다름!! cosin loss가 음수인 경우 orthonormal loss는 음수를 제곱해버려서 similar하지 않은데도 로스를 만들어냄.) large margin cosine loss 와 가까움.w
        # LMCL에 따르면 margin의 최적은 0.35
        cos_sim_pos = cos_sim-margin
        cos_sim_pos[cos_sim_pos<0]=0
        loss = (torch.sum(cos_sim_pos)-torch.trace(cos_sim_pos))/(self.memory_size*(self.memory_size-1))
        return loss

    def pointwise_gather_loss(self, query_reshape, keys, gathering_indices, train):
        n, dims = query_reshape.size()  # (b X h X w) X d
        loss_mse = torch.nn.MSELoss(reduction='none')

        pointwise_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        return pointwise_loss

    def spread_loss(self, query, keys):
        batch_size, h, w, dims = query.size()  # b X h X w X d

        loss = torch.nn.TripletMarginLoss(margin=1.0)

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size * h * w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 2, dim=1)

        # 1st, 2nd closest memories
        pos = keys[gathering_indices[:, 0]]
        neg = keys[gathering_indices[:, 1]]

        spreading_loss = loss(query_reshape, pos.detach(), neg.detach())

        return spreading_loss

    def gather_loss(self, query, keys):

        batch_size, h, w, dims = query.size()  # b X h X w X d

        loss_mse = torch.nn.MSELoss()

        softmax_score_query, softmax_score_memory = self.get_score(keys, query)

        query_reshape = query.contiguous().view(batch_size * h * w, dims)

        _, gathering_indices = torch.topk(softmax_score_memory, 1, dim=1)

        gathering_loss = loss_mse(query_reshape, keys[gathering_indices].squeeze(1).detach())

        return gathering_loss

    def read(self, query,mask, memory_writing):

        query = F.normalize(query.clone(), dim=1)
        query = query.permute(0, 2, 3, 1).contiguous()  # b X h X w X d
        batch_size, h, w, dims = query.size()  # b X h X w X d

        if memory_writing: # 메모리쪽으로 가는 gradient를 다 끊음.
            self.m_items = self.m_items.detach()

        softmax_score_query, softmax_score_memory, readloss = self.get_score(query, mask, self.m_items)
        query_reshape = query.view(batch_size * h * w, dims)
        concat_memory = torch.matmul(softmax_score_memory, self.m_items)  # (b X h X w) X d

        updated_query = torch.cat((query_reshape, concat_memory), dim=1)  # (b X h X w) X 2d
        updated_query = updated_query.view(batch_size, h, w, 2 * dims)
        updated_query = updated_query.permute(0, 3, 1, 2).contiguous()

        updated_query = self.output(updated_query)

        return updated_query, softmax_score_query.view(batch_size, h, w,19), softmax_score_memory.view(batch_size, h, w,19),readloss


    def unsupervised_memupdate(self, query):
        ### get writing feature.
        query_temp = F.normalize(query, dim=1).clone()
        m, d = self.m_items.size()

        query_temp = query_temp.permute(0, 2, 3, 1).contiguous()  # b X h X w X d
        batch_size, h, w, dims = query_temp.size()  # b X h X w X d
        score = torch.matmul(query_temp, torch.t(self.m_items))  # b X h X w X m

        score = score.view(batch_size * h * w, m)  # (b X h X w) X m

        softmax_score_memory = F.softmax(score, dim=1)
        # softmax_score_memory = (softmax_score_memory - softmax_score_memory.min()) / (softmax_score_memory.max() - softmax_score_memory.min())
        softmax_score_memory_deno = softmax_score_memory.sum(dim=0)
        for i in range(m):
            softmax_score_memory[:,i] = softmax_score_memory[:,i]/softmax_score_memory_deno[i]

        query_reshape = query_temp.contiguous().view(batch_size * h * w, dims)

        query_update = torch.matmul(torch.t(softmax_score_memory),query_reshape)

        query_update = F.normalize(query_update, dim=1)
        self.m_items = F.normalize((1-self.momentum)*query_update + self.momentum*self.m_items, dim=1).detach()
