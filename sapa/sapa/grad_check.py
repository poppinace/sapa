import os
import os.path as osp
import sys

import torch
from torch.autograd import gradcheck
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
sys.path.append(osp.abspath(osp.join(__file__, '../../../')))
from sapa_func import sim, atn  # noqa: E402, isort:skip

q = torch.randn(2, 6, 6, 64, requires_grad=True, device='cuda:0').double()
k = torch.randn(2, 3, 3, 64, requires_grad=True, device='cuda:0').double()
attn = torch.randn(2, 6, 6, 25, requires_grad=True, device='cuda:0').sigmoid().double()
v = torch.randn(2, 3, 3, 20, requires_grad=True, device='cuda:0').double()

print('Gradcheck for sim...')
test = gradcheck(sim, (q, k, 5, 2))
print(test)

print('Gradcheck for atn...')
test = gradcheck(atn, (attn, v, 5, 2))
print(test)
