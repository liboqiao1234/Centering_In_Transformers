from .base_class import SingleStepQuantity
from ...extensions import ForwardOutputExtension
import torch


class OutputAngleMean(SingleStepQuantity):

    def _compute(self, global_step):
        data = self._module.output
        # 处理 data 可能是元组的情况
        if isinstance(data, tuple):
            data = data[0]  # 假设第一个元素是主要输出
        
        data = data.contiguous().view(data.shape[0], -1)
        ones = data.new_ones(data.shape)
        sum = data.sum(dim=1)
        norm = data.norm(p=2, dim=1) * ones.norm(p=2, dim=1)
        theta = torch.acos(sum/norm) * (180.0 / torch.pi)
        return theta.mean()

    def forward_extensions(self):
        extensions = [ForwardOutputExtension()]
        return extensions
    

class OutputAngleStd(SingleStepQuantity):
    def _compute(self, global_step):
        data = self._module.output
        # 处理 data 可能是元组的情况
        if isinstance(data, tuple):
            data = data[0]  # 假设第一个元素是主要输出
            
        data = data.contiguous().view(data.shape[0], -1)
        ones = data.new_ones(data.shape)
        sum = data.sum(dim=1)
        norm = data.norm(p=2, dim=1) * ones.norm(p=2, dim=1)
        theta = torch.acos(sum/norm) * (180.0 / torch.pi)
        return theta.std()

    def forward_extensions(self):
        extensions = [ForwardOutputExtension()]
        return extensions


if __name__ == '__main__':
    import torch
    from torch import nn as nn

    l = nn.Linear(2, 3)
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x = torch.randn((4, 2))
    x_c = torch.randn((4, 1, 3, 3))
    quantity_l = OutputAngleMean(l)
    quantity_c = OutputAngleMean(cov)
    for hook in quantity_l.forward_extensions():
        l.register_forward_hook(hook)
    for hook in quantity_c.forward_extensions():
        cov.register_forward_hook(hook)

    for i in range(3):
        y = l(x)
        y_c = cov(x_c)
        quantity_l.track(i)
        quantity_c.track(i)

    print(quantity_l.get_output()[0])
    # print(y.shape)
    # print(y.mean())
    print(quantity_c.get_output()[0])
    # print(y_c.mean())
