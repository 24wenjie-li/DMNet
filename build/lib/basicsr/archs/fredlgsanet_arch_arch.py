import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from basicsr.utils.registry import ARCH_REGISTRY

from .arch_util import to_2tuple, trunc_normal_

from collections import OrderedDict

# for restormer
import numbers
from pdb import set_trace as stx

from einops import rearrange

# for idynamic
from .dlgsanet_idynamicdwconv_util import *
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------------------------------
# Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):    # for better performance and less params we set bias=False
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# FFN
class FeedForward(nn.Module):
    """
        GDFN in Restormer: [github] https://github.com/swz30/Restormer
    """
    def __init__(self, dim, ffn_expansion_factor, bias, input_resolution=None):
        super(FeedForward, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

    def flops(self):
        h, w = self.input_resolution
        N = h*w
        flops = 0

        flops += N * self.dim * self.dim * self.ffn_expansion_factor * 2
        flops += self.dim * self.ffn_expansion_factor * 2 * 9
        flops += N * self.dim * self.ffn_expansion_factor * self.dim
        return flops


# FFN
class BaseFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        # base feed forward network in SwinIR
        super(BaseFeedForward, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.body = nn.Sequential(
            nn.Conv2d(dim, hidden_features, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(hidden_features, dim, 1, bias=bias),
        )

    def forward(self, x):
        # shortcut outside
        return self.body(x)
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# IDynamicDWConvBlock
class IDynamicDWConvBlock(nn.Module):
    """
        code based on: [github] https://github.com/Atten4Vis/DemystifyLocalViT/blob/master/models/dwnet.py
        but we remove reductive Norm Layers and Activation Layers for better performance in SR-task
    """
    def __init__(self, dim, window_size, dynamic=True, heads=None, bias=True, input_resolution=None):
        super().__init__()

        # for flops counting
        self.input_resolution = input_resolution

        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.dynamic = dynamic
        self.heads = heads

        # pw-linear
        # in pw-linear layer we inherit settings from DWBlock. Set bias=False
        self.conv0 = nn.Conv2d(dim, dim, 1, bias=False)
        self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)

        if dynamic:
            self.conv = IDynamicDWConv(dim, kernel_size=window_size, group_channels=heads, bias=bias)
        else:
            self.conv = nn.Conv2d(dim, dim, kernel_size=window_size, stride=1, padding=window_size // 2, groups=dim, bias=bias)

    def forward(self, x):
        # shortcut outside the block
        x = self.conv0(x)
        x = self.conv(x)
        x = self.conv1(x)
        return x

    def flops(self):
        # calculate flops for windows with token length of N
        h, w = self.input_resolution
        N = h * w

        flops = 0
        # x = self.conv0(x)
        flops += N * self.dim * self.dim

        # x = self.conv(x)
        if self.dynamic:
            flops += (N * self.dim * self.dim / 4 + N * self.dim * self.window_size * self.window_size + N * self.dim / 4 * self.dim / self.heads * self.window_size * self.window_size)

        flops += N * self.dim * self.window_size * self.window_size
        #  x = self.conv2(x)
        flops += N * self.dim * self.dim
        return flops


# ---------------------------------------------------------------------------------------------------------------------
class SparseAttention(nn.Module):
    """
        SparseGSA is based on MDTA
        MDTA in Restormer: [github] https://github.com/swz30/Restormer
        TLC: [github] https://github.com/megvii-research/TLC
        We use TLC-Restormer in forward function and only use it in test mode
    """
    def __init__(self, dim, num_heads, bias, tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None):
        super(SparseAttention, self).__init__()
        self.tlc_flag = tlc_flag    # TLC flag for validation and test

        self.dim = dim
        self.input_resolution = input_resolution

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.act = nn.Identity()

        # ['gelu', 'sigmoid'] is for ablation study
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()

        # [x2, x3, x4] -> [96, 72, 48]
        self.kernel_size = [tlc_kernel, tlc_kernel]

    def _forward(self, qkv):
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # attn = attn.softmax(dim=-1)
        attn = self.act(attn)     # Sparse Attention due to ReLU's property

        out = (attn @ v)

        return out

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        if self.training or not self.tlc_flag:
            out = self._forward(qkv)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

            out = self.project_out(out)
            return out

        # Then we use the TLC methods in test mode
        qkv = self.grids(qkv)  # convert to local windows
        out = self._forward(qkv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=qkv.shape[-2], w=qkv.shape[-1])
        # print("out", out.size()) [1, 48, 72, 72]
        out = self.grids_inverse(out)  # reverse
        # print("out", out.size()) [1, 48, 72, 72]

        out = self.project_out(out)
        return out

    # Code from [megvii-research/TLC] https://github.com/megvii-research/TLC
    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c // 3, h, w) # [1, 48, 72, 72]
        # print("self.original_size", self.original_size)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size # [1, 48, 72, 72]
        # print("self.original_size", self.original_size)

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def flops(self):
        # calculate flops for window with token length of N
        h, w = self.input_resolution
        N = h * w

        flops = 0
        # x = self.qkv(x)
        flops += N * self.dim * self.dim * 3
        # x = self.qkv_dwconv(x)
        flops += N * self.dim * 3 * 9

        # qkv
        # CxC
        N_k = self.kernel_size[0] * self.kernel_size[1]
        N_num = ((h - 1)//self.kernel_size[0] + 1) * ((w - 1) // self.kernel_size[1] + 1)

        flops += N_num * self.num_heads * self.dim // self.num_heads * N_k * self.dim // self.num_heads
        # CxN CxC
        flops += N_num * self.num_heads * self.dim // self.num_heads * self.dim // self.num_heads * N_k

        # x = self.project_out(x)
        flops += N * self.dim * self.dim
        return flops
# ---------------------------------------------------------------------------------------------------------------------
# IDynamicDWBlock with GDFN
class IDynamicLayerBlock(nn.Module):
    def __init__(self, dim, window_size=7, idynamic_num_heads=6, idynamic_ffn_type='GDFN', idynamic_ffn_expansion_factor=2., idynamic=True, input_resolution=None):
        super(IDynamicLayerBlock, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')

        # IDynamic Local Feature Calculate
        self.IDynamicDWConv = IDynamicDWConvBlock(dim, window_size=window_size, dynamic=idynamic, heads=idynamic_num_heads, input_resolution=input_resolution)

        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')

        # FeedForward Network
        if idynamic_ffn_type == 'GDFN':
            self.IDynamic_ffn = FeedForward(dim, ffn_expansion_factor=idynamic_ffn_expansion_factor, bias=False, input_resolution=input_resolution)
        elif idynamic_ffn_type == 'BaseFFN':
            self.IDynamic_ffn = BaseFeedForward(dim, ffn_expansion_factor=idynamic_ffn_expansion_factor, bias=True)
        else:
            raise NotImplementedError(f'Not supported FeedForward Net type{idynamic_ffn_type}')

    def forward(self, x):
        x = self.IDynamicDWConv(self.norm1(x)) + x
        x = self.IDynamic_ffn(self.norm2(x)) + x
        return x

    def flops(self):
        flops = 0
        h, w = self.input_resolution
        flops += self.dim * h * w
        flops += self.dim * h * w

        flops += self.IDynamicDWConv.flops()
        flops += self.IDynamic_ffn.flops()
        return flops


class SparseAttentionLayerBlock(nn.Module):
    def __init__(self, dim, restormer_num_heads=6, restormer_ffn_type='GDFN', restormer_ffn_expansion_factor=2., tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None):
        super(SparseAttentionLayerBlock, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.norm3 = LayerNorm(dim, LayerNorm_type='WithBias')

        # We use SparseGSA inplace MDTA
        self.restormer_attn = SparseAttention(dim, num_heads=restormer_num_heads, bias=False, tlc_flag=tlc_flag, tlc_kernel=tlc_kernel, activation=activation, input_resolution=input_resolution)

        self.norm4 = LayerNorm(dim, LayerNorm_type='WithBias')

        # Restormer FeedForward
        if restormer_ffn_type == 'GDFN':
            # FIXME: new experiment, test bias
            self.restormer_ffn = FeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=False, input_resolution=input_resolution)
        elif restormer_ffn_type == 'BaseFFN':
            self.restormer_ffn = BaseFeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=True)
        else:
            raise NotImplementedError(f'Not supported FeedForward Net type{restormer_ffn_type}')

    def forward(self, x):
        x = self.restormer_attn(self.norm3(x)) + x
        x = self.restormer_ffn(self.norm4(x)) + x
        return x

    def flops(self):
        flops = 0
        h, w = self.input_resolution
        flops += self.dim * h * w
        flops += self.dim * h * w

        flops += self.restormer_attn.flops()
        flops += self.restormer_ffn.flops()
        return flops
# ---------------------------------------------------------------------------------------------------------------------
import pywt
import numpy as np
from torch.autograd import Function

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H//2, W//2)

            dx = dx.transpose(1,2).reshape(B, -1, H//2, W//2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None

class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        # self.filters = self.filters.to(dtype=torch.float32)

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)

class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        # self.w_ll = self.w_ll.to(dtype=torch.float32)
        # self.w_lh = self.w_lh.to(dtype=torch.float32)
        # self.w_hl = self.w_hl.to(dtype=torch.float32)
        # self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

# class WaveAttention(nn.Module):
#     def __init__(self, dim, num_heads, bias=False, tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None):
#         super(WaveAttention, self).__init__()
#         self.tlc_flag = tlc_flag    # TLC flag for validation and test

#         self.dim = dim
#         self.input_resolution = input_resolution

#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.dwt = DWT_2D(wave='haar')
#         self.idwt = IDWT_2D(wave='haar')
#         self.reduce = nn.Sequential(
#             nn.Conv2d(dim, dim//4, kernel_size=1, padding=0, stride=1),
#             nn.ReLU(inplace=True),
#         )
#         self.filter = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim),
#             nn.ReLU(inplace=True),
#         )

#         self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
#         self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
#         self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
#         self.project_out = nn.Conv2d(dim * 5 // 4, dim, kernel_size=1, bias=bias)

#         self.act = nn.Identity()

#         # ['gelu', 'sigmoid'] is for ablation study
#         if activation == 'relu':
#             self.act = nn.ReLU()
#         elif activation == 'gelu':
#             self.act = nn.GELU()
#         elif activation == 'sigmoid':
#             self.act = nn.Sigmoid()

#         # [x2, x3, x4] -> [96, 72, 48]
#         self.kernel_size = [tlc_kernel, tlc_kernel]

#     def _forward(self, x):
#         q = self.q_dwconv(self.q(x))

#         x_dwt = self.dwt(self.reduce(x))
#         x_dwt = self.filter(x_dwt)
#         x_idwt = self.idwt(x_dwt)

#         kv = self.kv_dwconv(self.kv(x_dwt))
#         k, v = kv.chunk(2, dim=1)

#         q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
#         k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
#         v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1)) * self.temperature

#         attn = self.act(attn)     # Sparse Attention due to ReLU's property

#         out = (attn @ v)

#         return out, x_idwt

#     def forward(self, x):
#         b, c, h, w = x.shape

#         qkv = x

#         if self.training or not self.tlc_flag:
#             out, x_idwt = self._forward(qkv)
#             out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)

#             out = self.project_out(torch.cat([out, x_idwt], dim=1))
#             return out

#         # Then we use the TLC methods in test mode
#         qkv = self.grids(qkv)  # convert to local windows
#         out, x_idwt = self._forward(qkv)
#         out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=qkv.shape[-2], w=qkv.shape[-1])

#         out = self.project_out(torch.cat([out, x_idwt], dim=1))
#         out = self.grids_inverse(out)  # reverse

#         return out

#     # Code from [megvii-research/TLC] https://github.com/megvii-research/TLC
#     def grids(self, x):
#         b, c, h, w = x.shape
#         print("self.original_size", self.original_size)
#         assert b == 1
#         k1, k2 = self.kernel_size
#         k1 = min(h, k1)
#         k2 = min(w, k2)
#         num_row = (h - 1) // k1 + 1
#         num_col = (w - 1) // k2 + 1
#         self.nr = num_row
#         self.nc = num_col

#         import math
#         step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
#         step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

#         parts = []
#         idxes = []
#         i = 0  # 0~h-1
#         last_i = False
#         while i < h and not last_i:
#             j = 0
#             if i + k1 >= h:
#                 i = h - k1
#                 last_i = True
#             last_j = False
#             while j < w and not last_j:
#                 if j + k2 >= w:
#                     j = w - k2
#                     last_j = True
#                 parts.append(x[:, :, i:i + k1, j:j + k2])
#                 idxes.append({'i': i, 'j': j})
#                 j = j + step_j
#             i = i + step_i

#         parts = torch.cat(parts, dim=0)
#         self.idxes = idxes
#         return parts

#     def grids_inverse(self, outs):
#         preds = torch.zeros(self.original_size).to(outs.device)
#         b, c, h, w = self.original_size

#         count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
#         k1, k2 = self.kernel_size
#         k1 = min(h, k1)
#         k2 = min(w, k2)

#         for cnt, each_idx in enumerate(self.idxes):
#             i = each_idx['i']
#             j = each_idx['j']
#             preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
#             count_mt[0, 0, i:i + k1, j:j + k2] += 1.

#         del outs
#         torch.cuda.empty_cache()
#         return preds / count_mt


# class WaveAttention(nn.Module):
#     def __init__(self, dim, num_heads, bias=False, tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None):
#         super(WaveAttention, self).__init__()
#         self.tlc_flag = tlc_flag    # TLC flag for validation and test

#         self.dim = dim
#         self.input_resolution = input_resolution

#         self.num_heads = num_heads
#         self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

#         self.dwt = DWT_2D(wave='haar')
#         self.idwt = IDWT_2D(wave='haar')
#         self.reduce = nn.Sequential(
#             nn.Conv2d(dim, dim//4, kernel_size=1, padding=0, stride=1),
#             nn.ReLU(inplace=True),
#         )
#         self.filter = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim),
#             nn.ReLU(inplace=True),
#         )

#         self.wsize = window_sizes
#         self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
#         self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
#         self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
#         self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
#         self.project_out = nn.Conv2d(dim * 5 // 4, dim, kernel_size=1, bias=bias)

#         self.act = nn.Identity()

#         # ['gelu', 'sigmoid'] is for ablation study
#         if activation == 'relu':
#             self.act = nn.ReLU()
#         elif activation == 'gelu':
#             self.act = nn.GELU()
#         elif activation == 'sigmoid':
#             self.act = nn.Sigmoid()

#         # [x2, x3, x4] -> [96, 72, 48]
#         self.kernel_size = [tlc_kernel, tlc_kernel]

#     def _forward(self, x):
#         q = self.q_dwconv(self.q(x))

#         x_dwt = self.dwt(self.reduce(x))
#         x_dwt = self.filter(x_dwt)
#         x_idwt = self.idwt(x_dwt)

#         kv = self.kv_dwconv(self.kv(x_dwt))
#         k, v = kv.chunk(2, dim=1)

#         q = rearrange(q, 'b (c head) (h dh) (w dw) -> b head (h w) (dh dw) c', dh=wsize, dw=wsize, head=self.num_heads)
#         k = rearrange(q, 'b (c head) (h dh) (w dw) -> b head (h w) (dh dw) c', dh=wsize, dw=wsize, head=self.num_heads)
#         v = rearrange(q, 'b (c head) (h dh) (w dw) -> b head (h w) (dh dw) c', dh=wsize, dw=wsize, head=self.num_heads)

#         q = torch.nn.functional.normalize(q, dim=-1)
#         k = torch.nn.functional.normalize(k, dim=-1)

#         attn = (q @ k.transpose(-2, -1)) * self.temperature

#         attn = self.act(attn)     # Sparse Attention due to ReLU's property

#         out = (attn @ v)

#         return out, x_idwt

#     def forward(self, x):
#         wsize = self.wsize
#         b, c, h, w = x.shape

#         qkv = x

#         if self.training or not self.tlc_flag:
#             out, x_idwt = self._forward(qkv)
#             out = rearrange(out, 'b head (h w) (dh dw) c -> b (c head) (h dh) (w dw)', head=self.num_heads, h=w=w//wsize, dh=dw=wsize)

#             out = self.project_out(torch.cat([out, x_idwt], dim=1))
#             return out

#         # Then we use the TLC methods in test mode
#         qkv = self.grids(qkv)  # convert to local windows
#         out, x_idwt = self._forward(qkv)
#         out = rearrange(out, 'b head (h w) (dh dw) c -> b (c head) (h dh) (w dw)', head=self.num_heads, h=w=qkv.shape[-2]//wsize, dh=dw=wsize)

#         out = self.project_out(torch.cat([out, x_idwt], dim=1))
#         out = self.grids_inverse(out)  # reverse

#         return out

#     # Code from [megvii-research/TLC] https://github.com/megvii-research/TLC
#     def grids(self, x):
#         b, c, h, w = x.shape
#         print("self.original_size", self.original_size)
#         assert b == 1
#         k1, k2 = self.kernel_size
#         k1 = min(h, k1)
#         k2 = min(w, k2)
#         num_row = (h - 1) // k1 + 1
#         num_col = (w - 1) // k2 + 1
#         self.nr = num_row
#         self.nc = num_col

#         import math
#         step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
#         step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

#         parts = []
#         idxes = []
#         i = 0  # 0~h-1
#         last_i = False
#         while i < h and not last_i:
#             j = 0
#             if i + k1 >= h:
#                 i = h - k1
#                 last_i = True
#             last_j = False
#             while j < w and not last_j:
#                 if j + k2 >= w:
#                     j = w - k2
#                     last_j = True
#                 parts.append(x[:, :, i:i + k1, j:j + k2])
#                 idxes.append({'i': i, 'j': j})
#                 j = j + step_j
#             i = i + step_i

#         parts = torch.cat(parts, dim=0)
#         self.idxes = idxes
#         return parts

#     def grids_inverse(self, outs):
#         preds = torch.zeros(self.original_size).to(outs.device)
#         b, c, h, w = self.original_size

#         count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
#         k1, k2 = self.kernel_size
#         k1 = min(h, k1)
#         k2 = min(w, k2)

#         for cnt, each_idx in enumerate(self.idxes):
#             i = each_idx['i']
#             j = each_idx['j']
#             preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
#             count_mt[0, 0, i:i + k1, j:j + k2] += 1.

#         del outs
#         torch.cuda.empty_cache()
#         return preds / count_mt


class WaveAttention(nn.Module):
    def __init__(self, dim, num_heads, bias=False, tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None):
        super(WaveAttention, self).__init__()
        self.tlc_flag = tlc_flag    # TLC flag for validation and test

        self.dim = dim
        self.input_resolution = input_resolution

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.reduce = nn.Sequential(
            nn.Conv2d(dim, dim//4, kernel_size=1, padding=0, stride=1),
            nn.ReLU(inplace=True),
        )
        self.filter = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim),
            nn.ReLU(inplace=True),
        )

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim * 2 // 4, dim, kernel_size=1, bias=bias)
        self.idwt_sa = IDWT_2D(wave='haar')

        self.act = nn.Identity()

        # ['gelu', 'sigmoid'] is for ablation study
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()

        # [x2, x3, x4] -> [96, 72, 48]
        self.kernel_size = [tlc_kernel, tlc_kernel]

    def _forward(self, x):
        x_dwt = self.dwt(self.reduce(x))
        x_dwt = self.filter(x_dwt)
        x_idwt = self.idwt(x_dwt)

        qkv = self.qkv_dwconv(self.qkv(x_dwt))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = self.act(attn)     # Sparse Attention due to ReLU's property

        out = (attn @ v)

        return out, x_idwt

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = x

        if self.training or not self.tlc_flag:
            out, x_idwt = self._forward(qkv)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h//2, w=w//2)
            out = self.idwt_sa(out)

            out = self.project_out(torch.cat([out, x_idwt], dim=1))
            return out

        # Then we use the TLC methods in test mode
        qkv = self.grids(qkv)  # convert to local windows
        out, x_idwt = self._forward(qkv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=qkv.shape[-2]//2, w=qkv.shape[-1]//2)
        out = self.idwt_sa(out)

        out = self.project_out(torch.cat([out, x_idwt], dim=1))
        out = self.grids_inverse(out)  # reverse

        return out

    # Code from [megvii-research/TLC] https://github.com/megvii-research/TLC
    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt


class WaveAttentionLayerBlock(nn.Module):
    def __init__(self, dim, restormer_num_heads=6, restormer_ffn_type='GDFN', restormer_ffn_expansion_factor=2., tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None):
        super(WaveAttentionLayerBlock, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.norm3 = LayerNorm(dim, LayerNorm_type='WithBias')

        # We use SparseGSA inplace MDTA
        self.restormer_attn = WaveAttention(dim=dim, num_heads=restormer_num_heads, bias=False, tlc_flag=tlc_flag, tlc_kernel=tlc_kernel, activation=activation, input_resolution=input_resolution)

        self.norm4 = LayerNorm(dim, LayerNorm_type='WithBias')

        # Restormer FeedForward
        if restormer_ffn_type == 'GDFN':
            # FIXME: new experiment, test bias
            self.restormer_ffn = FeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=False, input_resolution=input_resolution)
        elif restormer_ffn_type == 'BaseFFN':
            self.restormer_ffn = BaseFeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=True)
        else:
            raise NotImplementedError(f'Not supported FeedForward Net type{restormer_ffn_type}')

    def forward(self, x):
        x = self.restormer_attn(self.norm3(x)) + x
        x = self.restormer_ffn(self.norm4(x)) + x
        return x

    def flops(self):
        flops = 0
        h, w = self.input_resolution
        flops += self.dim * h * w
        flops += self.dim * h * w

        flops += self.restormer_attn.flops()
        flops += self.restormer_ffn.flops()
        return flops
# ---------------------------------------------------------------------------------------------------------------------
def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

class PhaseProcess(nn.Module):
    def __init__(self, in_nc=48, out_nc=48):
        super(PhaseProcess,self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.contrast1 = stdv_channels

    def forward(self,x_amp,x):  # x_amp=trans, x=cnn
        x_amp_freq = torch.fft.rfft2(x_amp, norm='backward')
        x_freq = torch.fft.rfft2(x, norm='backward')
        x_amp_freq_amp = torch.abs(x_amp_freq)
        x_amp_freq_pha = torch.angle(x_amp_freq)

        x_freq_amp = torch.abs(x_freq)
        x_freq_pha = torch.angle(x_freq)

        real = x_freq_amp * torch.cos(x_amp_freq_pha)
        imag = x_freq_amp * torch.sin(x_amp_freq_pha)
        x_recom = torch.complex(real, imag)
        x_recom_trans = torch.fft.irfft2(x_recom)

        real = x_amp_freq_amp * torch.cos(x_freq_pha)
        imag = x_amp_freq_amp * torch.sin(x_freq_pha)
        x_recom = torch.complex(real, imag)
        x_recom_cnn = torch.fft.irfft2(x_recom)

        x_recom_trans = self.contrast(x_recom_trans) + self.avgpool(x_recom_trans) * x_recom_trans
        x_recom_cnn = self.contrast1(x_recom_cnn) + self.avgpool1(x_recom_cnn) * x_recom_cnn

        return x_recom_cnn, x_recom_trans
# ---------------------------------------------------------------------------------------------------------------------
# BuildBlocks
class BuildBlock(nn.Module):
    # Sorry for the redundant parameter setting
    # it is easier for ablation study while during experiment
    # if necessary it can be changed to **args
    def __init__(self, dim, blocks=3, buildblock_type='edge',
                 window_size=7, idynamic_num_heads=6, idynamic_ffn_type='GDFN', idynamic_ffn_expansion_factor=2., idynamic=True,
                 restormer_num_heads=6, restormer_ffn_type='GDFN', restormer_ffn_expansion_factor=2., tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None
                 ):
        super(BuildBlock, self).__init__()

        self.input_resolution = input_resolution

        # those all for extra_repr
        # --------
        self.dim = dim
        self.blocks = blocks
        self.buildblock_type = buildblock_type
        self.window_size = window_size
        self.num_heads = (idynamic_num_heads, restormer_num_heads)
        self.ffn_type = (idynamic_ffn_type, restormer_ffn_type)
        self.ffn_expansion = (idynamic_ffn_expansion_factor, restormer_ffn_expansion_factor)
        self.idynamic = idynamic
        self.tlc = tlc_flag
        # ---------

        # buildblock body
        # ---------
        body = []
        if buildblock_type == 'sparseedge':  #this
            for _ in range(blocks):
                body.append(IDynamicLayerBlock(dim, window_size, idynamic_num_heads, idynamic_ffn_type, idynamic_ffn_expansion_factor, idynamic, input_resolution=input_resolution))
                body.append(SparseAttentionLayerBlock(dim, restormer_num_heads, restormer_ffn_type, restormer_ffn_expansion_factor, tlc_flag, tlc_kernel, activation, input_resolution=input_resolution))

        elif buildblock_type == 'idynamic':
            for _ in range(blocks):
                body.append(IDynamicLayerBlock(dim, window_size, idynamic_num_heads, idynamic_ffn_type, idynamic_ffn_expansion_factor, idynamic))

        elif buildblock_type == 'FFT':
            for _ in range(blocks):
                body.append(FFTAttentionLayerBlock(dim, restormer_num_heads, restormer_ffn_type, restormer_ffn_expansion_factor, tlc_flag, tlc_kernel, activation, input_resolution=input_resolution))

        elif buildblock_type == 'Wave':
            for _ in range(blocks):
                body.append(SparseAttentionLayerBlock(dim, restormer_num_heads, restormer_ffn_type, restormer_ffn_expansion_factor, tlc_flag, tlc_kernel, activation, input_resolution=input_resolution))
                body.append(SparseAttentionLayerBlock(dim, restormer_num_heads, restormer_ffn_type, restormer_ffn_expansion_factor, tlc_flag, tlc_kernel, activation, input_resolution=input_resolution))
        # --------HybridAttentionBlock

        body.append(nn.Conv2d(dim, dim, 3, 1, 1))   # as like SwinIR, we use one Conv3x3 layer after buildblock
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x) + x     # shortcut in buildblock

        # for i in range(self.blocks):
        #     if i == 0: ## only calculate attention for the 1-st module
        #         x, x_recom_cnn, x_recom_trans = self.modules_lfe['lfe_{}'.format(i)](x, None, None)
        #     else:
        #         x, x_recom_cnn, x_recom_trans = self.modules_lfe['lfe_{}'.format(i)](x, x_recom_cnn, x_recom_trans)
        # return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, blocks={self.blocks}, buildblock_type={self.buildblock_type}, ' \
               f'window_size={self.window_size}, num_heads={self.num_heads}, ffn_type={self.ffn_type}, ' \
               f'ffn_expansion={self.ffn_expansion}, idynamic={self.idynamic}, tlc={self.tlc}'

    def flops(self):
        flops = 0
        h, w = self.input_resolution

        for i in range(len(self.body) - 1):
            flops += self.body[i].flops()

        flops += h*w * self.dim * self.dim * 9

        return flops
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

       but for our model, we give up Traditional Upsample and use UpsampleOneStep for better performance not only in
       lightweight SR model, Small/XSmall SR model, but also for our base model.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


# Traditional Upsample from SwinIR EDSR RCAN
class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------
# Network
@ARCH_REGISTRY.register()
class FreDLGSANet(nn.Module):
    r""" DLGSANet
        A PyTorch impl of : DLGSANet: Lightweight Dynamic Local and Global Self-Attention Network for Image Super-Resolution
        'IDynamic' using the idynamic transformer block
        'Restormer' using the Restormer transformer block
        'Edge' a new way inspired by EdgeViTs and EdgeNeXt
        'SparseEdge' a new way of using ReLU's properties for Sparse Attention

    Args:
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 90
        depths (tuple(int)): Depth of each BuildBlock
        num_heads (tuple(int)): Number of attention heads in different layers
        window_size (int): Window size. Default: 7
        ffn_expansion_factor (float): Ratio of feedforward network hidden dim to embedding dim. Default: 2
        ffn_type (str): feedforward network type, such as GDFN and BaseFFN
        bias (bool): If True, add a learnable bias to layers. Default: True
        body_norm (bool): Normalization layer. Default: False
        idynamic (bool): using idynamic for local attention. Default: True
        tlc_flag (bool): using TLC during validation and test. Default: True
        tlc_kernel (int): TLC kernel_size [x2, x3, x4] -> [96, 72, 48]
        upscale: Upscale factor. 2/3/4 for image SR
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction module. 'pixelshuffle'/'pixelshuffledirect'
    """

    def __init__(self,
                 in_chans=3,
                 dim=48,
                 groups=4,
                 blocks=3,
                 buildblock_type='sparseedge',
                 window_size=7, idynamic_num_heads=6, idynamic_ffn_type='GDFN', idynamic_ffn_expansion_factor=2.,
                 idynamic=True,
                 restormer_num_heads=6, restormer_ffn_type='GDFN', restormer_ffn_expansion_factor=2., tlc_flag=True, tlc_kernel=48, activation='relu',
                 upscale=4,
                 img_range=1.,
                 upsampler='',
                 body_norm=False,
                 input_resolution=None,     # input_resolution = (height, width)
                 **kwargs):
        super(FreDLGSANet, self).__init__()

        # for flops counting
        self.dim = dim
        self.input_resolution = input_resolution

        # MeanShift for Image Input
        # ---------
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        # -----------

        # Upsample setting
        # -----------
        self.upscale = upscale
        self.upsampler = upsampler
        # -----------

        # ------------------------- 1, shallow feature extraction ------------------------- #
        # the overlap_embed: remember to set it into bias=False
        self.overlap_embed = nn.Sequential(OverlapPatchEmbed(in_chans, dim, bias=False))

        # ------------------------- 2, deep feature extraction ------------------------- #
        m_body = []

        # Base on the Transformer, When we use pre-norm we need to build a norm after the body block
        if body_norm:       # Base on the SwinIR model, there are LayerNorm Layers in PatchEmbed Layer between body
            m_body.append(LayerNorm(dim, LayerNorm_type='WithBias'))

        for i in range(groups):
            m_body.append(BuildBlock(dim, blocks, buildblock_type,
                 window_size, idynamic_num_heads, idynamic_ffn_type, idynamic_ffn_expansion_factor, idynamic,
                 restormer_num_heads, restormer_ffn_type, restormer_ffn_expansion_factor, tlc_flag, tlc_kernel, activation, input_resolution=input_resolution))

        if body_norm:
            m_body.append(LayerNorm(dim, LayerNorm_type='WithBias'))

        m_body.append(nn.Conv2d(dim, dim, kernel_size=(3, 3), padding=(1, 1)))

        self.deep_feature_extraction = nn.Sequential(*m_body)

        # ------------------------- 3, high quality image reconstruction ------------------------- #

        # setting for pixelshuffle for big model, but we only use pixelshuffledirect for all our model
        # -------
        num_feat = 64
        embed_dim = dim
        num_out_ch = in_chans
        # -------

        if self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch, input_resolution=self.input_resolution)

        elif self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        pass    # all are in forward function including deep feature extraction

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.overlap_embed(x)
            x = self.deep_feature_extraction(x) + x
            x = self.upsample(x)

        elif self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.overlap_embed(x)
            x = self.deep_feature_extraction(x) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        else:
            # for image denoising and JPEG compression artifact reduction
            x = self.overlap_embed(x)
            x = self.deep_feature_extraction(x) + x
            x = self.conv_last(x)

        x = x / self.img_range + self.mean

        return x

    def flops(self):
        flops = 0
        h, w = self.input_resolution

        # overlap_embed layer
        flops += h * w * 3 * self.dim * 9

        # BuildBlock:
        for i in range(len(self.deep_feature_extraction) - 1):
            flops += self.deep_feature_extraction[i].flops()

        # conv after body
        flops += h * w * 3 * self.dim * self.dim
        flops += self.upsample.flops()

        return flops


if __name__ == '__main__':
    # use fvcore for flops accounting
    # from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count_table

    upscale = 4
    # window_size = 8
    height = (1280 // upscale)
    width = (720 // upscale)
    window_size = 7
    idynamic_num_heads = 6
    restormer_num_heads = 6
    print(f'information of input: [upscale: {upscale}] [height: {height}] [weight: {width} \n')

    dim = 48 #90
    groups = 3 #6
    blocks = 3 #4
    print(f'loading model TIPEIRNet with [dim: {dim}] [groups: {groups}] [blocks: {blocks} \n')
    model = FreDLGSANet(dim=dim, upscale=upscale, groups=groups, blocks=blocks, window_size=window_size, idynamic_num_heads=idynamic_num_heads, restormer_num_heads=restormer_num_heads, upsampler='pixelshuffledirect', input_resolution=(height, width))

    print('======'*50)
    # print(model)

    print('======'*50)
    print('fvcore for model flops counting...' + '-'*50)
    x = torch.randn((1, 3, height, width))
    # x = model(x)

    net_params = sum(map(lambda x: x.numel(), model.parameters()))
    print(f"network params: {net_params}")
    print(f"network flops: {model.flops() / 1e9}")

    import numpy as np
    from torchvision.models import resnet50
    import torch
    from torch.backends import cudnn
    import tqdm

    cudnn.benchmark = True

    device = 'cuda:0'

    repetitions = 10

    model = model.to(device)
    dummy_input = torch.rand(1, 3, height, width).to(device)

    # warm up
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)

    # synchronize / wait for all the GPU process then back to cpu
    torch.cuda.synchronize()

    # testing CUDA Event
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # initialize
    timings = np.zeros((repetitions, 1))

    print('testing ...\n')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()  # wait for ending
            curr_time = starter.elapsed_time(ender)  # from starter to ender (/ms)
            timings[rep] = curr_time

    avg = timings.sum() / repetitions
    print('\navg={}\n'.format(avg))

    # with torch.no_grad():
    #     flop = FlopCountAnalysis(model, x)
    #     print(flop_count_table(flop, max_depth=4, show_param_shapes=False))
    #     # print(flop_count_str(flop))
    #     print("Total", flop.total() / 1e9)
    #
    # print('======'*50)
    # print('check output shape: ')
    # print(x.shape)





# ---------------------------------------------------------------------------------------------------------------------
##  Top-K Sparse Attention (TKSA)
class TopkAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, tlc_kernel):
        super(TopkAttention, self).__init__()
        self.num_heads = num_heads

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        # [x2, x3, x4] -> [96, 72, 48]
        self.kernel_size = [tlc_kernel, tlc_kernel]

    def forward_(self, x):
        b, _, _, _ = x.shape
        q, k, v = x.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        _, _, C, _ = q.shape

        mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)

        attn = (q @ k.transpose(-2, -1)) * self.temperature # b, head, C/head, C/head

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        return out

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        if self.training:
            out = self.forward_(qkv)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

            out = self.project_out(out)
            return out

        # Then we use the TLC methods in test mode
        qkv = self.grids(qkv)  # convert to local windows
        out = self.forward_(qkv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=qkv.shape[-2], w=qkv.shape[-1])
        out = self.grids_inverse(out)  # reverse

        out = self.project_out(out)
        return out

    # Code from [megvii-research/TLC] https://github.com/megvii-research/TLC
    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c // 3, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

##  Sparse Transformer Block (STB)
class TopKTransformer(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, tlc_kernel, bias=False, LayerNorm_type='WithBias'):
        super(TopKTransformer, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = TopkAttention(dim, num_heads, bias, tlc_kernel)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
# ---------------------------------------------------------------------------------------------------------------------
class NonLocalAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, tlc_kernel, depth, input_resolution=None):
        super(NonLocalAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.act = nn.ReLU()

        self.test = nn.Identity()
        self.window_size = [8,8]
        self.depth = depth
        self.shift_size = self.window_size[0]//2
        self.kernel_size = [tlc_kernel, tlc_kernel]


    def forward(self, x):
        if self.depth % 2:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        b,c,h,w = x.shape
        w_size = self.window_size
        qkv = self.qkv_dwconv(self.qkv(x)) # train [16, 144, 48, 48]
        if not self.training:
            qkv = self.grids(qkv) # test [4, 144, 48, 48]
        qkv_ = rearrange(qkv, 'b c (h b0) (w b1) -> (b h w) c b0 b1', b0=w_size[0], b1=w_size[1])

        q,k,v = qkv_.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # attn = attn.softmax(dim=-1)
        attn = self.act(attn)     # Sparse Attention due to ReLU's property

        attn = self.test(attn)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=w_size[1], w=w_size[1]) #train [576, 48, 8, 8]

        if not self.training:
            out = rearrange(out, '(b h w) c b0 b1 -> b c (h b0)  (w b1)', h=qkv.shape[-2] // w_size[0], w=qkv.shape[-1] // w_size[1],
                            b0=w_size[0])
        else:
            out = rearrange(out, '(b h w) c b0 b1 -> b c (h b0)  (w b1)', h=h // w_size[0], w=w // w_size[1],
                            b0=w_size[0])

        # if not self.training:
        if not self.training:
            out = self.grids_inverse(out)

        out = self.project_out(out)
        if self.depth % 2:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        return out

    # Code from [megvii-research/TLC] https://github.com/megvii-research/TLC
    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c//3, h, w)
        # assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

class NonLocalAttentionBlock(nn.Module):
    def __init__(self, dim, restormer_num_heads=6, restormer_ffn_type='GDFN', restormer_ffn_expansion_factor=2., tlc_kernel=48, depth=0, input_resolution=None):
        super(NonLocalAttentionBlock, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.norm3 = LayerNorm(dim, LayerNorm_type='WithBias')

        # We use SparseGSA inplace MDTA
        self.restormer_attn = NonLocalAttention(dim, num_heads=restormer_num_heads, bias=False, tlc_kernel=tlc_kernel, depth=depth, input_resolution=input_resolution)

        self.norm4 = LayerNorm(dim, LayerNorm_type='WithBias')

        # Restormer FeedForward
        if restormer_ffn_type == 'GDFN':
            # FIXME: new experiment, test bias
            self.restormer_ffn = FeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=False, input_resolution=input_resolution)
        elif restormer_ffn_type == 'BaseFFN':
            self.restormer_ffn = BaseFeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=True)
        else:
            raise NotImplementedError(f'Not supported FeedForward Net type{restormer_ffn_type}')

    def forward(self, x):
        x = self.restormer_attn(self.norm3(x)) + x
        x = self.restormer_ffn(self.norm4(x)) + x
        return x

# ---------------------------------------------------------------------------------------------------------------------
class LowPassModule(nn.Module):
    def __init__(self, in_channel, sizes=(1, 3, 5, 7)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        self.relu = nn.ReLU()
        ch =  in_channel // 4
        self.channel_splits = [ch, ch, ch, ch]

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        feats = torch.split(feats, self.channel_splits, dim=1)
        priors = [F.upsample(input=self.stages[i](feats[i]), size=(h, w), mode='bilinear') for i in range(4)]
        bottle = torch.cat(priors, 1)

        return self.relu(bottle)


class FilterModule(nn.Module):
    def __init__(self, Ch, h, window={
                3: 3,
                5: 3,
                7: 2
            }):
        super().__init__()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]
        self.LP = LowPassModule(Ch * h)

    def forward(self, q, v, size):
        B, h, Ch, N = q.shape
        H, W = size[0], size[1]

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img = rearrange(v, "B h Ch (H W) -> B (h Ch) H W", H=H, W=W)
        LP = self.LP(v_img)
        # Split according to channels.
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        HP_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        HP = torch.cat(HP_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        HP = rearrange(HP, "B (h Ch) H W -> B h Ch (H W)", h=h)
        LP = rearrange(LP, "B (h Ch) H W -> B h Ch (H W)", h=h)

        dynamic_filters = q * HP + LP
        return dynamic_filters


class FilterAttention(nn.Module):
    def __init__(self, dim, num_heads, bias=False, tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None):
        super(FilterAttention, self).__init__()
        self.tlc_flag = tlc_flag    # TLC flag for validation and test

        self.dim = dim
        self.input_resolution = input_resolution

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.act = nn.Identity()

        # ['gelu', 'sigmoid'] is for ablation study
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'sigmoid':
            self.act = nn.Sigmoid()

        # [x2, x3, x4] -> [96, 72, 48]
        self.kernel_size = [tlc_kernel, tlc_kernel]

        self.fft = nn.Parameter(torch.ones((1, num_heads, dim // num_heads, tlc_kernel * tlc_kernel // 2 + 1)))

        self.crpe = FilterModule(Ch=dim // num_heads, h=num_heads)
        self.scale = (dim // num_heads) ** -0.5
        self.norm = LayerNorm(dim, LayerNorm_type='WithBias')

    def _forward(self, qkv):
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        # attn = attn.softmax(dim=-1)
        attn = self.act(attn)     # Sparse Attention due to ReLU's property

        crpe = self.crpe(q, v, size=self.kernel_size)

        out = (attn @ v) # b head c (h w)

        out = self.scale * out + crpe

        return out

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(self.norm(x)))

        if self.training or not self.tlc_flag:
            out = self._forward(qkv)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

            out = self.project_out(out)
            return out

        # Then we use the TLC methods in test mode
        qkv = self.grids(qkv)  # convert to local windows
        out = self._forward(qkv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=qkv.shape[-2], w=qkv.shape[-1])
        out = self.grids_inverse(out)  # reverse

        out = self.project_out(out)
        return out + x

    # Code from [megvii-research/TLC] https://github.com/megvii-research/TLC
    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c // 3, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

class FreFeedForward(nn.Module):
    """
        GDFN in Restormer: [github] https://github.com/swz30/Restormer
    """
    def __init__(self, dim, ffn_expansion_factor, bias, kernel_size=7, reduction_ratio=4, input_resolution=None):
        super(FreFeedForward, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.conv_abs = nn.Conv2d(hidden_features, hidden_features, 5, 2, 1, groups=hidden_features, bias=bias)
        self.conv_angle = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bias=bias)

    def forward(self, x):
        _, _, h, w = x.shape
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x_enc = torch.fft.rfft2(x1, norm='backward')
        x_dec = torch.fft.rfft2(x2, norm='backward')
        x_freq_amp = torch.abs(x_enc)
        x_freq_pha = torch.angle(x_dec)
        x_freq_pha = self.conv_angle(x_freq_pha)
        x_freq_amp = self.conv_abs(x_freq_amp)
        real = x_freq_amp * torch.cos(x_freq_pha)
        imag = x_freq_amp * torch.sin(x_freq_pha)
        x_recom = torch.complex(real, imag)
        x = torch.fft.irfft2(x_recom)

        x = self.project_out(x)
        return x


# class WaveAttention(nn.Module):
#     def __init__(self, dim, num_heads, sr_ratio):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim**-0.5
#         self.sr_ratio = sr_ratio

#         self.dwt = DWT_2D(wave='haar')
#         self.idwt = IDWT_2D(wave='haar')
#         self.reduce = nn.Sequential(
#             nn.Conv2d(dim, dim//4, kernel_size=1, padding=0, stride=1),
#             nn.BatchNorm2d(dim//4),
#             nn.ReLU(inplace=True),
#         )
#         self.filter = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(inplace=True),
#         )
#         self.kv_embed = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
#         self.q = nn.Linear(dim, dim)
#         self.kv = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, dim * 2)
#         )
#         self.proj = nn.Linear(dim+dim//4, dim)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()

#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

#         x = x.view(B, H, W, C).permute(0, 3, 1, 2)
#         x_dwt = self.dwt(self.reduce(x))
#         x_dwt = self.filter(x_dwt)

#         x_idwt = self.idwt(x_dwt)
#         x_idwt = x_idwt.view(B, -1, x_idwt.size(-2)*x_idwt.size(-1)).transpose(1, 2)

#         kv = self.kv_embed(x_dwt).reshape(B, C, -1).permute(0, 2, 1)
#         kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv[0], kv[1]

#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(torch.cat([x, x_idwt], dim=-1))
#         return x


class FFT_Block(nn.Module):
    def __init__(self, ch_in, reduction=6):
        super(FFT_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // reduction, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in // reduction, ch_in, 3, 1, 1),
            nn.Sigmoid()
        )

        self.seq1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // 2 + 1, 1),
            nn.ReLU(inplace=True),
        )
        self.seq2 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in // 2 + 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        y_ = y.view(b, 1, 1, c)

        y_seq1, y_seq2 = self.seq1(y).view(b, 1, 1, -1), self.seq2(y).view(b, 1, 1, -1)

        y_freq = torch.fft.rfft2(y_, norm='backward')
        y_freq_amp = torch.abs(y_freq) * y_seq1
        y_freq_pha = torch.angle(y_freq) * y_seq2

        real = y_freq_amp * torch.cos(y_freq_pha)
        imag = y_freq_amp * torch.sin(y_freq_pha)
        x_recom = torch.complex(real, imag)
        x = torch.fft.irfft2(x_recom).view(b, c, 1, 1)

        return x * y.expand_as(x)

class FFTAttentionLayerBlock(nn.Module):
    def __init__(self, dim, restormer_num_heads=6, restormer_ffn_type='GDFN', restormer_ffn_expansion_factor=2., tlc_flag=True, tlc_kernel=48, activation='relu', input_resolution=None):
        super(FFTAttentionLayerBlock, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')

        self.norm3 = LayerNorm(dim, LayerNorm_type='WithBias')

        # We use SparseGSA inplace MDTA
        self.restormer_attn = SparseAttention(dim, num_heads=restormer_num_heads, bias=False, tlc_flag=tlc_flag, tlc_kernel=tlc_kernel, activation=activation, input_resolution=input_resolution)

        self.FFT = FFT_Block(dim)

        self.norm4 = LayerNorm(dim, LayerNorm_type='WithBias')

        # Restormer FeedForward
        if restormer_ffn_type == 'GDFN':
            # FIXME: new experiment, test bias
            self.restormer_ffn = FeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=False, input_resolution=input_resolution)
            self.restormer_ffn_fft = FeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=False, input_resolution=input_resolution)
        elif restormer_ffn_type == 'BaseFFN':
            self.restormer_ffn = BaseFeedForward(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=True)
        else:
            raise NotImplementedError(f'Not supported FeedForward Net type{restormer_ffn_type}')

    def forward(self, x):
        x = self.restormer_attn(self.norm3(x)) + x
        x = self.restormer_ffn(self.norm4(x)) + x
        x = self.FFT(self.norm1(x)) + x
        x = self.restormer_ffn_fft(self.norm2(x)) + x
        return x