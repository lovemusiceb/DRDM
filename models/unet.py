import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from diffusion_util import checkpoint,linear,normalization,zero_module,avg_pool_nd,conv_nd
from attention import TimestepBlock,TimestepEmbedSequential,AttentionBlock,SpatialTransformer,timestep_embedding
from texture_encoder import Tex_Encoder
from pose_encoder import Pose_encoder



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









class ResBlock(TimestepBlock):
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

    def __init__(self,channels,emb_channels,dropout,out_channels=None,use_conv=False,use_scale_shift_norm=False,dims=2,use_checkpoint=False,up=False,down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
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

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

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

    def forward(self, x, emb):
        """
        将块应用于张量，以时间步嵌入为条件。
        :param x: an [N x C x…特征张量。
        :param emb:时间步嵌入的一个[N x emb_channels]张量。
        :return: an [N x C x…输出张量。
        """
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)


    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h













class UNetModel(nn.Module):
    """
    具有注意力和时间步长嵌入的完整 UNet 模型
    :param in_channels: 输入张量中的通道
    :param model_channels: 模型的基本通道计数
    :param out_channels: 输出张量中的通道。
    :param num_res_blocks: 每个下采样的剩余块数
    :param attention_resolutions: 注意力发生的下行抽样率的集合。可以是集合、列表或元组。例如，如果这包含4，那么在4倍下采样时，将使用注意力。
    :param dropout: 失活概率
    :param channel_mult: Unet每个级别的通道乘数。
    :param conv_resample: 如果为 True，则使用学习卷积进行上抽样和下抽样。
    :param dims: 确定信号是1D、2D 还是3D。
    :param use_checkpoint: 使用渐变检查点来减少内存使用。
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: 如果指定了，忽略 num _ head，而是使用每个注意力头的固定通道宽度
    :param num_heads_upsample: 使用 num _ head 设置不同数量的头以进行上抽样
    :param use_scale_shift_norm: 使用类似于 FiLM 的调节机制。
    :param resblock_updown: 使用剩余块进行上/下采样。
    :param use_new_attention_order: 使用不同的注意模式可以提高效率
    """

    def __init__(
        self,
        image_size=256,
        in_channels=7,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(32,16,8),#32,16,8
        dropout=0.,
        channel_mult=(1,1,2,2,4,4),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=True,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'
            #必须设置这两个参数
        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.style_encoder = Tex_Encoder()
        self.pose_encoder = Pose_encoder()


        # 时间编码
        time_embed_dim = model_channels * 4

        # 使用两层线性层将步长编码
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # 输入块 这里conv_nd为 conv_2d 这个块的作用是创建一个可以根据要求，TimestepEmbedSequential的作用是选择将时间编码嵌入x 还是将条件编码嵌入x的二维卷积
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1))])

        # 特征大小
        self._feature_size = model_channels
        # 输入块通道数
        input_block_chans = [model_channels]

        ch = model_channels
        ds = 1

        # 下采样
        for level, mult in enumerate(channel_mult): # 1 1 2 2 4 4
            for _ in range(num_res_blocks):
                # ch 越大效果越好，训练越慢，内存占用越大 时间编码维度为ch*4  dims 2维度为图片  use_checkpoint 如果为ture可以减少内存损耗但可能降低效果 use_scale_shift_norm是否使用fim优化
                # 这层是上下采样
                layers = [ResBlock(ch,time_embed_dim,dropout,out_channels=mult * model_channels,dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,)]
                ch = mult * model_channels

                if ds in attention_resolutions and _==num_res_blocks-1:
                    layers.append(
                        AttentionBlock(ch,use_checkpoint=use_checkpoint,num_heads=num_heads,num_head_channels=num_head_channels,
                        use_new_attention_order=use_new_attention_order,)
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch,time_embed_dim,dropout,out_channels=out_ch,dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,down=True,)if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # TimestepEmbedSequential的作用是选择将时间编码嵌入x 还是将条件编码嵌入x
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch,time_embed_dim,dropout,dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,),
            AttentionBlock(ch,use_checkpoint=use_checkpoint,num_heads=num_heads,num_head_channels=num_head_channels,use_new_attention_order=use_new_attention_order,),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, ),)
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks+1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich,time_embed_dim,dropout,out_channels=model_channels * mult,dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,)]
                ch = model_channels * mult
                # print('c',mult,'ds',ds)
                if ds in attention_resolutions and i==num_res_blocks-2:
                    layers.append(AttentionBlock(ch,use_checkpoint=use_checkpoint,num_heads=num_heads_upsample,num_head_channels=num_head_channels,use_new_attention_order=use_new_attention_order,))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResBlock(ch,time_embed_dim,dropout,out_channels=out_ch,dims=dims,use_checkpoint=use_checkpoint,use_scale_shift_norm=use_scale_shift_norm,up=True,)if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        # print(self.output_blocks[0])

    def forward(self, x, timesteps, oimage=None,pose=None,sem_oimage=None,y=None, **kwargs):
        """
        将模型应用于输入批处理。
        :param x: an [N x C x…输入张量。
        :param timesteps:一维的时间步。
        :param上下文:条件插入通过crossattn
        :参数y:一个标签的张量，如果是类条件的。
        :return: an [N x C x…输出张量。
        """
        oimage = self.style_encoder(oimage,sem_oimage)
        pose = self.pose_encoder(pose)
        x = torch.cat([x,pose],dim=1)
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb) # 时间编码器
        h = x.type(self.dtype)
        i=0
        for module in self.input_blocks:
            if i ==11:
                h = module(h, emb, oimage=oimage[12])
            elif i ==14:
                h = module(h, emb, oimage=oimage[16])
            elif i ==17:
                h = module(h, emb, oimage=oimage[19])
            else:
                h = module(h, emb,oimage=None)
            hs.append(h)
            i+=1
        h = self.middle_block(h, emb, oimage=oimage[19])
        i=17
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            if i == 17:
                h = module(h, emb, oimage=oimage[19])
            elif i == 14:
                h = module(h, emb, oimage=oimage[16])
            elif i == 11:
                h = module(h, emb, oimage=oimage[12])
            else:
                h = module(h, emb,oimage=None)
            i-=1
        h = h.type(x.dtype)

        return self.out(h)


# g=UNetModel()
# print(g)