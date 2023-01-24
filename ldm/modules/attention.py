import importlib
import math
import sys
from inspect import isfunction
from typing import Any, Optional

import psutil
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import einsum, nn

from ldm.modules.diffusionmodules.util import checkpoint
from nataili import disable_xformers
from nataili.util import logger

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

from numba import cuda as ncuda

device = ncuda.get_current_device()

xformers_available = False
logger.init("xformers optimizations", status="Checking")
if disable_xformers.active:
    xformers = None
    logger.init_err("xformers optimizations", status="Disabled")
elif (6, 1) <= torch.cuda.get_device_capability(device) <= (9, 0):
    xformers_available = importlib.util.find_spec("xformers") is not None
    try:
        importlib_metadata.version("xformers")

    except importlib_metadata.PackageNotFoundError:
        xformers_available = False
    if xformers_available:
        import xformers
        import xformers.ops
        logger.init_ok("xformers optimizations", status="Loaded")
    else:
        xformers = None
        logger.init_err("xformers optimizations", status="Missing")
else:
    xformers = None
    logger.init_err("xformers optimizations", status="Not Possible")


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        if torch.cuda.is_available():
            self.einsum_op = self.einsum_op_cuda
        else:
            self.mem_total = psutil.virtual_memory().total / (1024**3)
            self.einsum_op = self.einsum_op_mps_v1 if self.mem_total >= 32 else self.einsum_op_mps_v2
    
    def einsum_op_compvis(self, q, k, v, r1):
        s1 = einsum('b i d, b j d -> b i j', q, k) * self.scale # faster
        s2 = s1.softmax(dim=-1, dtype=q.dtype)
        del s1
        r1 = einsum('b i j, b j d -> b i d', s2, v)
        del s2
        return r1

    def einsum_op_mps_v1(self, q, k, v, r1):
        if q.shape[1] <= 4096: # (512x512) max q.shape[1]: 4096
            r1 = self.einsum_op_compvis(q, k, v, r1)
        else:
            slice_size = math.floor(2**30 / (q.shape[0] * q.shape[1]))
            for i in range(0, q.shape[1], slice_size):
                end = i + slice_size
                s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k) * self.scale
                s2 = s1.softmax(dim=-1, dtype=r1.dtype)
                del s1  
                r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
                del s2
        return r1

    def einsum_op_mps_v2(self, q, k, v, r1):
        if self.mem_total >= 8 and q.shape[1] <= 4096:
                r1 = self.einsum_op_compvis(q, k, v, r1)
        else:
            slice_size = 1
            for i in range(0, q.shape[0], slice_size):
                end = min(q.shape[0], i + slice_size)
                s1 = einsum('b i d, b j d -> b i j', q[i:end], k[i:end])
                s1 *= self.scale
                s2 = s1.softmax(dim=-1, dtype=r1.dtype)
                del s1
                r1[i:end] = einsum('b i j, b j d -> b i d', s2, v[i:end])
                del s2
        return r1

    def einsum_op_cuda(self, q, k, v, r1):
        stats = torch.cuda.memory_stats(q.device)
        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch

        gb = 1024 ** 3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * 4
        mem_required = tensor_size * 2.5
        steps = 1

        if mem_required > mem_free_total:
            steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))

        if steps > 64:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                            f'Need: {mem_required/64/gb:0.1f}GB free, Have:{mem_free_total/gb:0.1f}GB free')

        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]  
        for i in range(0, q.shape[1], slice_size):
            end = min(q.shape[1], i + slice_size)
            s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k)# * self.scale
            s2 = s1.softmax(dim=-1, dtype=r1.dtype)
            del s1
            r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
            del s2 
        return r1

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        del x
        k = self.to_k(context) * self.scale
        v = self.to_v(context)
        del context

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
        r1 = self.einsum_op(q, k, v, r1)
        del q, k, v
        r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
        del r1
        return self.to_out(r2)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        if xformers_available:
            self.attn1 = MemoryEfficientCrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
            self.attn2 = MemoryEfficientCrossAttention(query_dim=dim, context_dim=context_dim,
                                        heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        else:
            self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
            self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                        heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = x.contiguous() if x.device.type == 'mps' else x
        x += self.attn1(self.norm1(x))
        x += self.attn2(self.norm2(x), context=context)
        x += self.ff(self.norm3(x))
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


class MemoryEfficientCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None):
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_k(context), self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.heads).contiguous(), (q, k, v))
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)
        out = rearrange(out, 'b n h d -> b n (h d)', h=self.heads)

        return self.to_out(out)
