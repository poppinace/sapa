
import torch
import torch.nn.functional as F
from sapa_func import sim, atn


def sim_(q, k, kernel_size=5, scale=2):
    B, H, W, C = k.shape
    q = q.view(B, H, scale, W, scale, C)
    k = F.unfold(k.permute(0, 3, 1, 2), kernel_size=kernel_size, padding=kernel_size // 2).reshape(
            B, C, kernel_size ** 2, H, W)
    return torch.einsum('ijklmn,inojl->ijklmo', q, k).reshape(
            B, scale * H, scale * W, kernel_size ** 2).contiguous()


def atn_(attn, x, kernel_size=5, scale=2):
    B, H, W, C = x.shape
    attn = attn.view(B, H, scale, W, scale, kernel_size ** 2)
    x = F.unfold(x.permute(0, 3, 1, 2), kernel_size=kernel_size, stride=1, padding=kernel_size // 2).view(
        B, C, kernel_size ** 2, H, W)
    return torch.einsum('ijklmn,ionjl->ijklmo', attn, x).contiguous().view(B, H * scale, W * scale, C)


def forward_check():
    print("forward check...")
    b = 4
    h = 10
    w = 12
    c = 8
    up_kernel_size=5
    up_factor=2
    q = torch.randn(b, up_factor * h, up_factor * w, c).to('cuda')
    k = torch.randn(b, h, w, c).to('cuda')
    sim_check = torch.allclose(sim_(q, k, up_kernel_size, up_factor),
                              sim(q, k, up_kernel_size, up_factor), atol=1e-5)
    print("sim check:", sim_check)

    attn = torch.randn(b, up_factor * h, up_factor * w, up_kernel_size ** 2).to('cuda')
    v = torch.randn(b, h, w, c).to('cuda')
    atn_check = torch.allclose(atn_(attn, v, up_kernel_size, up_factor),
                               atn(attn, v, up_kernel_size, up_factor), atol=1e-5)
    print("atn_check:", atn_check)


if __name__ == '__main__':
    forward_check()
