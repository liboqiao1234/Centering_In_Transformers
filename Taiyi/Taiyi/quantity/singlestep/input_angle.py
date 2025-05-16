from .base_class import SingleStepQuantity
from ...extensions import ForwardInputExtension
import torch


class InputAngleMean(SingleStepQuantity):

    def _compute(self, global_step):
        data = self._module.input
        data = data.contiguous().view(data.shape[0], -1)
        ones = data.new_ones(data.shape)
        sum = data.sum(dim=1)
        norm = data.norm(p=2, dim=1) * ones.norm(p=2, dim=1)
        theta = torch.acos(sum/norm) * (180.0 / torch.pi)
        return theta.mean()

    def forward_extensions(self):
        extensions = [ForwardInputExtension()]
        return extensions
    

class InputAngleStd(SingleStepQuantity):
    def _compute(self, global_step):
        data = self._module.input
        data = data.contiguous().view(data.shape[0], -1)
        ones = data.new_ones(data.shape)
        sum = data.sum(dim=1)
        norm = data.norm(p=2, dim=1) * ones.norm(p=2, dim=1)
        theta = torch.acos(sum/norm) * (180.0 / torch.pi)
        return theta.std()

    def forward_extensions(self):
        extensions = [ForwardInputExtension()]
        return extensions


if __name__ == '__main__':
    import torch
    from torch import nn as nn

    l = nn.Linear(2, 3)
    cov = nn.Conv2d(1, 2, 3, 1, 1)
    x = torch.randn((4, 2))
    x_c = torch.randn((4, 1, 3, 3))
    quantity_l = InputAngleMean(l)
    quantity_c = InputAngleMean(cov)
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
    # print(x.shape)
    # print(x.mean())
    print(quantity_c.get_output()[0])
    # print(x_c.mean())
