# SAPA used to conduct experiments

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from sapa import sim, atn


def nearest_interpolate(x, scale_factor=2):
    b, h, w, c = x.shape  # channels last
    return x.repeat(1, 1, 1, scale_factor ** 2).reshape(b, h, w, scale_factor, scale_factor, c).permute(
        0, 1, 3, 2, 4, 5).reshape(b, scale_factor * h, scale_factor * w, c)


class SAPAExp(nn.Module):
    def __init__(self, dim_y, dim_x=None, out_dim=None,
                 q_mode='encoder_only', v_embed=False,
                 up_factor=2, up_kernel_size=5, embedding_dim=64,
                 qkv_bias=True, norm=nn.LayerNorm):
        super().__init__()
        dim_x = dim_x if dim_x is not None else dim_y
        out_dim = out_dim if out_dim is not None else dim_x

        self.up_factor = up_factor
        self.up_kernel_size = up_kernel_size
        self.embedding_dim = embedding_dim

        self.norm_y = norm(dim_y)
        self.norm_x = norm(dim_x)

        self.q_mode = q_mode
        if q_mode == 'encoder_only':
            self.q = nn.Linear(dim_y, embedding_dim, bias=qkv_bias)
        elif q_mode == 'cat':
            self.q = nn.Linear(dim_x + dim_y, embedding_dim, bias=qkv_bias)
        elif q_mode == 'gate':
            self.qy = nn.Linear(dim_y, embedding_dim, bias=qkv_bias)
            self.qx = nn.Linear(dim_x, embedding_dim, bias=qkv_bias)
            self.gate = nn.Linear(dim_x, 1, bias=qkv_bias)
        else:
            raise NotImplementedError

        self.k = nn.Linear(dim_x, embedding_dim, bias=qkv_bias)

        if v_embed or out_dim != dim_x:
            self.v = nn.Linear(dim_x, out_dim, bias=qkv_bias)

        self.apply(self._init_weights)

    def forward(self, y, x):
        y = y.permute(0, 2, 3, 1).contiguous()
        x = x.permute(0, 2, 3, 1).contiguous()
        y = self.norm_y(y)
        x_ = self.norm_x(x)

        if self.q_mode == 'encoder_only':
            q = self.q(y)
        elif self.q_mode == 'cat':
            q = self.q(torch.cat([y, nearest_interpolate(x, self.up_factor)], dim=-1))
        elif self.q_mode == 'gate':
            gate = nearest_interpolate(torch.sigmoid(self.gate(x_)), self.up_factor)
            q = gate * self.qy(y) + (1 - gate) * self.qx(nearest_interpolate(x, self.up_factor))
        else:
            raise NotImplementedError

        k = self.k(x_)

        if hasattr(self, 'v'):
            x = self.v(x_)

        return self.attention(q, k, x).permute(0, 3, 1, 2).contiguous()

    def attention(self, q, k, v):
        attn = F.softmax(sim(q, k, self.up_kernel_size, self.up_factor), dim=-1)
        return atn(attn, v, self.up_kernel_size, self.up_factor)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


if __name__ == '__main__':
    x = torch.randn(2, 2, 2, 2).to('cuda')
    y = torch.randn(2, 2, 4, 4).to('cuda')
    sapa = SAPAExp(2).to('cuda')
    print(sapa(y, x).shape)
