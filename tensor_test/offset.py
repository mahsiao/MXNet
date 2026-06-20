import numpy as np
import torch
import torchvision
from torch import nn


class dcn(nn.Module):
    def __init__(self):
        super(dcn, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # 原卷积

        self.conv_offset = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        init_offset = torch.Tensor(np.zeros([18, 3, 3, 3]))
        self.conv_offset.weight = torch.nn.Parameter(init_offset)  # 初始化为0

        self.conv_mask = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1)
        init_mask = torch.Tensor(np.zeros([9, 3, 3, 3]) + np.array([0.5]))
        self.conv_mask.weight = torch.nn.Parameter(init_mask)  # 初始化为0.5

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x))  # 保证在0到1之间
        out = torchvision.ops.deform_conv2d(input=x, offset=offset,
                                            weight=self.conv.weight,
                                            mask=mask, padding=(1, 1))
        return out