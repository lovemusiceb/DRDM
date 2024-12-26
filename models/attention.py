from diffusion_util import default,Normalize,zero_module,checkpoint
from diffusion_util import conv_nd,normalization,count_flops_attn
from abc import abstractmethod
from torch import einsum
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import math
from einops import repeat,rearrange





# feed-forward 增强注意力后的特征表达能力
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




class CrossAttention(nn.Module): #交叉注意力机制
    def __init__(self, query_dim, oimage_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        oimage_dim = default(oimage_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(oimage_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(oimage_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, oimage=None):
        h = self.heads
        q = self.to_q(x)
        oimage = default(oimage, x) # 如果没有context就会变成自注意力机制
        k = self.to_k(oimage)
        v = self.to_v(oimage)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)




class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., oimage_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, oimage_dim=oimage_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, oimage=None):
        return checkpoint(self._forward, (x, oimage), self.parameters(), self.checkpoint)

    def _forward(self, x, oimage=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), oimage=oimage) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer块用于图像类数据。首先，将输入(又称嵌入)投影到b, t, d并重塑。
    然后应用标准的transformer动作。最后，重塑图像
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., oimage_dim=None):
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
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, oimage_dim=oimage_dim) for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, oimage=None):
        # note: if no context is given, 交叉注意力默认为自注意力
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, oimage=oimage)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in



class TimestepBlock(nn.Module):
    """
    任何forward()将timestep嵌入作为第二个参数的模块.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        给定`emb`时间步嵌入，将模块应用于`x`.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    一个顺序模块，将timestep嵌入作为额外输入传递给支持它的子模块.
    """

    def forward(self, x, emb=None, oimage=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
                # 将时间编码嵌入
            elif isinstance(layer, AttentionBlock):
                x = layer(x, oimage)
                # 将条件编码嵌入
            else:
                x = layer(x)
        return x


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    创建正弦时间步嵌入。
    :param timesteps: N个指标的1-D张量，每个批次元素一个。
    这些可能是小部分。
    :param dim:输出的尺寸。
    :参数max_period:控制嵌入的最小频率。
    :返回:一个位置嵌入的[N x dim]张量。
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding



class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        print(weight.shape)
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        print(weight.shape)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)



class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)





class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.to_kv = conv_nd(1, channels, channels * 2, 1)
        self.to_q = conv_nd(1, channels, channels * 1, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.selfattention = QKVAttention(self.num_heads)
            self.crossattention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.selfattention = QKVAttentionLegacy(self.num_heads)
            self.crossattention = QKVAttentionLegacy(self.num_heads)

        self.proj_out1 = zero_module(conv_nd(1, channels, channels, 1))
        self.proj_out2 = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, oimage):
        return checkpoint(self._forward, (x, oimage), self.parameters(), self.use_checkpoint)

    def _forward(self, x, oimage):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        oimage = oimage.reshape(b, c, -1)
        kv = self.to_kv(oimage)
        q = self.to_q(self.norm(x))
        qkv = th.cat([q, kv], 1)
        h = self.crossattention(qkv)
        h = self.proj_out2(h)

        return (x + h).reshape(b, c, *spatial)



