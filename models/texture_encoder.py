
from nn import (conv_nd, normalization, zero_module)
from diffusion_util import avg_pool_nd
from diffusion_util import zero_module
from diffusion_util import conv_nd,normalization,count_flops_attn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import math



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
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
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
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.to_kv = conv_nd(1, channels, channels * 2, 1)
        self.to_q = conv_nd(1, channels, channels * 1, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.selfattention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.selfattention = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))


    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.selfattention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)






class Upsample(nn.Module): #上采样
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x




class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)









class ResBlock(nn.Module):
    """
    一种残差块，可以选择性地改变通道的数量。

    :param channels:输入通道数。

    :param emb_channels:时间步嵌入通道数。

    :param dropout:辍学率。

    :param out_channels:如果指定，out通道的数量。

    :param use_conv:如果是True并且指定了out_channels，则使用spatial卷积而不是较小的1x1卷积来改变skip连接中的通道。

    :参数调光:表示信号是1D、2D还是3D。

    :param use_checkpoint:如果为真，在此模块上使用梯度检查点。

    :param up:如果为真，使用此块进行上采样。

    :param down:如果为真，使用此块进行下采样。
    """

    def __init__(self,channels,dropout,out_channels=None,use_conv=False,use_scale_shift_norm=False,dims=2,use_checkpoint=False,up=False,down=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()



        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            h = out_rest(h)
        else:
            h = h
            h = self.out_layers(h)
        return self.skip_connection(x) + h




class Tex_Encoder(nn.Module):

    def __init__(
        self,
        in_channels=3,
        model_channels=32,
        num_res_blocks=2,
        dropout=0,
        channel_mult=(1,1,2,2,4,4),
        dims=2,
        use_fp16=False,
    ):
        super().__init__()
        model_channels_1 = 32
        model_channels_2 = 64
        model_channels_3 = 16
        model_channels_4 = 16
        ch_1 = model_channels_1
        ch_2 = model_channels_2
        ch_3 = model_channels_3
        ch_4 = model_channels_4

        self.dtype = th.float16 if use_fp16 else th.float32

        self.pant_encoder = []
        self.pant_encoder += [conv_nd(dims, in_channels, model_channels_1, 3, padding=1)]
        self.upper_clothes_encoder = []
        self.upper_clothes_encoder += [conv_nd(dims, in_channels, model_channels_2, 3, padding=1)]
        self.head_encoder = []
        self.head_encoder += [conv_nd(dims, in_channels, model_channels_3, 3, padding=1)]
        self.background_endcoder = []
        self.background_endcoder += [conv_nd(dims, in_channels, model_channels_4, 3, padding=1)]

        for level, mult in enumerate(channel_mult):  # 1 1 2 2 4 4
            for _ in range(num_res_blocks):
                self.pant_encoder += [ResBlock(ch_1, dropout, out_channels=mult * model_channels_1, dims=dims)]
                self.upper_clothes_encoder += [ResBlock(ch_2, dropout, out_channels=mult * model_channels_2, dims=dims)]
                self.head_encoder += [ResBlock(ch_3, dropout, out_channels=mult * model_channels_3, dims=dims)]
                self.background_endcoder += [ResBlock(ch_4, dropout, out_channels=mult * model_channels_4, dims=dims)]
                ch_1 = mult * model_channels_1
                ch_2 = mult * model_channels_2
                ch_3 = mult * model_channels_3
                ch_4 = mult * model_channels_4

                if (level == 3 and _ == 0) or (level == 4 and _ == 0):
                    self.pant_encoder += [AttentionBlock(ch_1, num_heads=1, num_head_channels=-1)]
                    self.upper_clothes_encoder += [AttentionBlock(ch_2, num_heads=1, num_head_channels=-1)]
                    self.head_encoder += [AttentionBlock(ch_3, num_heads=1, num_head_channels=-1)]
                    self.background_endcoder += [AttentionBlock(ch_4, num_heads=1, num_head_channels=-1)]
            if level != len(channel_mult) - 1:
                out_ch_1 = ch_1
                out_ch_2 = ch_2
                out_ch_3 = ch_3
                out_ch_4 = ch_4
                self.pant_encoder += [ResBlock(ch_1, dropout, out_channels=out_ch_1, dims=dims, down=True)]
                self.upper_clothes_encoder += [ResBlock(ch_2, dropout, out_channels=out_ch_2, dims=dims, down=True)]
                self.head_encoder += [ResBlock(ch_3, dropout, out_channels=out_ch_3, dims=dims, down=True)]
                self.background_endcoder += [ResBlock(ch_4, dropout, out_channels=out_ch_4, dims=dims, down=True)]
                ch_1 = out_ch_1
                ch_2 = out_ch_2
                ch_3 = out_ch_3
                ch_4 = out_ch_4

        # self.pant_encoder += [AttentionBlock(ch_1, num_heads=1, num_head_channels=-1)]
        # self.upper_clothes_encoder += [AttentionBlock(ch_2, num_heads=1, num_head_channels=-1)]
        # self.head_encoder += [AttentionBlock(ch_3, num_heads=1, num_head_channels=-1)]
        # self.background_endcoder += [AttentionBlock(ch_4, num_heads=1, num_head_channels=-1)]

        self.pant_encoder = nn.Sequential(*self.pant_encoder)
        self.upper_clothes_encoder = nn.Sequential(*self.upper_clothes_encoder)
        self.head_encoder = nn.Sequential(*self.head_encoder)
        self.background_endcoder = nn.Sequential(*self.background_endcoder)

    def forward(self, x,sem):
        xi = []
        for i in range(sem.size(1)):
            semi = sem[:, i, :, :]
            semi = torch.unsqueeze(semi, 1)
            semi = semi.repeat(1, x.size(1), 1, 1)
            xs = x.mul(semi)
            xi.append(xs)  # 把这个部位特征提取出来
            # sem's dims: 0 is background, 1 and 3 are pants and skirts, 2 is hair, 4 is face, 5 is upper clothes, 6 is arms, 7 is legs.
            # For clothes style transfer, you can replace some part in the "sem" with that in another person image's appearance latent code.
            # 把这几个xi拼接起来作为8*3维的特征，送入特征提取器
        xa = []
        xa.append(xi[1]+ xi[7])  # 1下半身
        xa.append(xi[2] + xi[4])  # 2
        xa.append(xi[5])  # 3
        xa.append(xi[6]  + xi[0]+ xi[3])  # 4
        hs = []

        h0 = xa[0].type(self.dtype)
        h1 = xa[1].type(self.dtype)
        h2 = xa[2].type(self.dtype)
        h3 = xa[3].type(self.dtype)
        for i in range(len(self.head_encoder)):
            # print(self.pant_encoder)
            h0 = self.pant_encoder[i](h0)
            h1 = self.head_encoder[i](h1)
            h2 = self.upper_clothes_encoder[i](h2)
            h3 = self.background_endcoder[i](h3)
            # print(h0.shape,h1.shape,h2.shape,h3.shape)
            feature=torch.cat([h0,h1,h2,h3],dim=1)
            hs.append(feature)

        return hs

















# enc_appearance = Tex_Encoder()
#
# x=torch.zeros(4,3,256,192)
# s=torch.zeros(4,8,256,192)
#
# x=enc_appearance(x=x,sem=s)
#
# print(x.shape)
#
# xx=nn.Linear(6016,512)
# x=xx(x)
# print(x.shape)