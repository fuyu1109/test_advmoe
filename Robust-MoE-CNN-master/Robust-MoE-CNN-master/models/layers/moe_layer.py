import torch
from torch import autograd, nn as nn


class GetMask(autograd.Function):#实现二值化选择（即选择一个特定的专家进行计算）
    @staticmethod
    def forward(ctx, scores):  # scores为专家得分，[bs, n_expert]

        expert_pred = torch.argmax(scores, dim=1)  # 找到得分最高的专家索引
        expert_pred_one_hot = torch.zeros_like(scores).scatter_(1, expert_pred.unsqueeze(-1), 1)#将专家索引转换为独热向量，表示选择的专家

        return expert_pred, expert_pred_one_hot

    @staticmethod
    def backward(ctx, g1, g2):
        return g2


def get_device(x):
    gpu_idx = x.get_device()
    return f"cuda:{gpu_idx}" if gpu_idx >= 0 else "cpu"


class MoEBase(nn.Module):#管理分配给不同专家的分数
    def __init__(self):
        super(MoEBase, self).__init__()
        self.scores = None
        self.router = None

    def set_score(self, scores):
        self.scores = scores
        for module in self.modules():
            if hasattr(module, 'scores'):
                module.scores = self.scores


class MoEConv(nn.Conv2d, MoEBase):#多专家卷积操作
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, dilation=1, bias=False,
                 n_expert=5):
        super(MoEConv, self).__init__(in_channels, out_channels * n_expert, kernel_size, stride, padding, dilation,
            groups, bias, )
        self.in_channels = in_channels
        self.out_channels = out_channels * n_expert
        self.expert_width = out_channels

        self.n_expert = n_expert
        assert self.n_expert >= 1
        self.layer_selection = torch.zeros([n_expert, self.out_channels])#用于选择不同专家的矩阵。每个专家在layer_selection矩阵中对应一个子矩阵。
        for cluster_id in range(n_expert):
            start = cluster_id * self.expert_width
            end = (cluster_id + 1) * self.expert_width
            idx = torch.arange(start, end)
            self.layer_selection[cluster_id][idx] = 1
        self.scores = None

    def forward(self, x):#根据选择的专家在每个样本的输入上进行选择性卷积操作
        if self.n_expert > 1:
            if self.scores is None:
                self.scores = self.router(x)#计算专家得分
            expert_selection, expert_selection_one_hot = GetMask.apply(self.scores)#选择专家
            mask = torch.matmul(expert_selection_one_hot, self.layer_selection.to(x))  # 通过矩阵乘法生成选择掩码，每个样本只激活一个专家。[batch_size, self.out_channels]
            out = super(MoEConv, self).forward(x)#执行普通的卷积操作
            out = out * mask.unsqueeze(-1).unsqueeze(-1)#只保留激活专家的输出
            index = torch.where(mask.view(-1) > 0)[0]#在掩码中找到所有非零元素的位置索引，即激活的专家的输出通道位置。
            shape = out.shape
            out_selected = out.view(shape[0] * shape[1], shape[2], shape[3])[index].view(shape[0], -1, shape[2],
                                                                                         shape[3])
        else:
            out_selected = super(MoEConv, self).forward(x)
        self.scores = None
        return out_selected
