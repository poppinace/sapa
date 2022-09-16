# SAPA by https://github.com/Poppuppy
# Modified from CARAFE https://github.com/myownskyW7/CARAFE

import torch
from torch.autograd import Function

import sim_ext, atn_ext


class QKFunction(Function):

    @staticmethod
    def forward(ctx, query, key, kernel_size, scale_factor):
        assert scale_factor >= 1
        assert query.size(0) == key.size(0)
        assert query.size(-1) == key.size(-1)

        assert query.size(-2) == key.size(-2) * scale_factor
        assert query.size(-3) == key.size(-3) * scale_factor
        assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1
        ctx.kernel_size = kernel_size
        ctx.scale_factor = scale_factor
        ctx.query_size = query.size()
        ctx.key_size = key.size()

        b, h, w, c = query.size()
        output = query.new_zeros((b, h, w, kernel_size ** 2))
        if query.is_cuda:
            sim_ext.forward(query, key, kernel_size, scale_factor, output)
        else:
            raise NotImplementedError

        if query.requires_grad or key.requires_grad:
            ctx.save_for_backward(query, key)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda

        query, key = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        scale_factor = ctx.scale_factor

        grad_query = torch.zeros_like(query)
        grad_key = torch.zeros_like(key)
        sim_ext.backward(grad_output.contiguous(), query, key,
                        kernel_size, scale_factor,
                        grad_query, grad_key)

        return grad_query, grad_key, None, None, None


sim = QKFunction.apply


class ATNFunction(Function):

    @staticmethod
    def forward(ctx, attn, value, kernel_size, scale_factor):
        assert scale_factor >= 1
        assert attn.size(-1) == kernel_size * kernel_size
        assert attn.size(-2) == value.size(-2) * scale_factor
        assert attn.size(-3) == value.size(-3) * scale_factor
        assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1
        ctx.kernel_size = kernel_size
        ctx.scale_factor = scale_factor
        ctx.attn_size = attn.size()
        ctx.value_size = value.size()

        b, h, w, c = value.size()
        output = attn.new_zeros((b, h * scale_factor, w * scale_factor, c))
        if attn.is_cuda:
            atn_ext.forward(attn, value, kernel_size, scale_factor, output)
        else:
            raise NotImplementedError

        if attn.requires_grad or value.requires_grad:
            ctx.save_for_backward(attn, value)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda

        attn, value = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        scale_factor = ctx.scale_factor

        grad_attn = torch.zeros_like(attn)
        grad_value = torch.zeros_like(value)
        atn_ext.backward(grad_output.contiguous(), attn, value,
                        kernel_size, scale_factor,
                        grad_attn, grad_value)

        return grad_attn, grad_value, None, None, None


atn = ATNFunction.apply

