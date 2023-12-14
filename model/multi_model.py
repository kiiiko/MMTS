import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

__all__ = [
    'pvt_tiny', 'pvt_small', 'pvt_medium', 'pvt_large'
]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    # 定义Mlp模块：Mlp类是一个多层感知机（MLP），它由两个线性层和激活函数构成。MLP常用于对特征进行非线性变换和映射。
    # 这里的Mlp用于PVT模型中的每个Transformer块内部定义Mlp模块：Mlp类是一个多层感知机（MLP），它由两个线性层和激活函数构成。
    # MLP常用于对特征进行非线性变换和映射。这里的Mlp用于PVT模型中的每个Transformer块内部

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    # 定义Attention模块：Attention类是PVT模型中的注意力机制。它包含一个多头注意力层，用于计算输入特征之间的关联程度，并生成加权后的输出特征。


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
    # 定义Block模块：Block类是PVT模型中的基本构建块，由一个注意力模块和一个MLP模块组成。
    # 每个Block将输入特征先进行注意力处理，然后再通过MLP进行非线性变换。


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)
    # 定义PatchEmbed模块：PatchEmbed类用于将输入图像分割成小的图像块（或称为patches），然后对这些patches进行线性变换。这是PVT模型的输入处理模块。

class PyramidVisionTransformer(nn.Module):
    # 定义PyramidVisionTransformer模块：PyramidVisionTransformer类是整个PVT模型的主体部分，它由多个Block和PatchEmbed组成。
    # 该模型将图像分割成不同尺度的patches，并利用Transformer结构进行特征提取和分类。
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=5, embed_dims=[32, 64, 128, 256],
                 num_heads=[1, 1, 2, 4], mlp_ratios=[2, 2, 2, 2], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 3, 4, 2], sr_ratios=[8, 4, 2, 1], num_stages=4):
        super().__init__()
        self.stage_features=[]
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                     patch_size=patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

        self.norm = norm_layer(embed_dims[3])

        # cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims[3]))

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            trunc_normal_(pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
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
        # return {'pos_embed', 'cls_token'} # has pos_embed may be better
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x)

            if i == self.num_stages - 1:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                pos_embed_ = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
                pos_embed = torch.cat((pos_embed[:, 0:1], pos_embed_), dim=1)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            for blk in block:
                x = blk(x, H, W)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            self.stage_features.append(x)  # 保存每个阶段的特征
        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x, self.stage_features


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
def pvt_tiny(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[32, 64, 128,256], num_heads=[1, 2, 2, 4], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 1 ,1, 1], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_small(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_medium(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_large(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_huge_v2(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[128, 256, 512, 768], num_heads=[2, 4, 8, 12], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 10, 60, 3], sr_ratios=[8, 4, 2, 1],
        # drop_rate=0.0, drop_path_rate=0.02)
        **kwargs)
    model.default_cfg = _cfg()

    return model






def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register_model
def pvt_tiny(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[32, 64, 128,256], num_heads=[1, 2, 2, 4], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[1, 1 ,1, 1], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_small(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_medium(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_large(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    model.default_cfg = _cfg()

    return model


@register_model
def pvt_huge_v2(pretrained=False, **kwargs):
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[128, 256, 512, 768], num_heads=[2, 4, 8, 12], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 10, 60, 3], sr_ratios=[8, 4, 2, 1],
        # drop_rate=0.0, drop_path_rate=0.02)
        **kwargs)
    model.default_cfg = _cfg()

    return model



#transformer

import torch
import numpy as np
import torch.nn as nn
import math

d_model = 512   # 字 Embedding 的维度512
d_ff = 128# 前向传播隐藏层维度 2048
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 1    # 有多少个encoder和decoder
n_heads = 2     # Multi-Head Attention设置为8

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device=None):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([[pos / np.power(10000, 2 * (i // 2) / d_model) for i in range(d_model)]
                              for pos in range(max_len)])
        pos_table[:, 0::2] = np.sin(pos_table[:, 0::2])
        pos_table[:, 1::2] = np.cos(pos_table[:, 1::2])
        self.register_buffer('pos_table', torch.FloatTensor(pos_table))

    def forward(self, x):
        assert x.size(1) <= self.pos_table.size(0), "Sequence length exceeds positional table size!"
        pos_table = self.pos_table.to(self.device) if self.device else self.pos_table
        return x + pos_table[:x.size(1), :].unsqueeze(0)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, device=None):
        super(ScaledDotProductAttention, self).__init__()
        self.device = device

    def forward(self, Q, K, V, attn_mask):
        batch_size, n_heads, len_q, _ = Q.size()
        d_k = Q.size(-1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        return context.to(self.device), attn.to(self.device)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, device=None):
        super(MultiHeadAttention, self).__init__()
        self.device = device
        self.n_heads = n_heads
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False,device=device)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False,device=device)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False,device=device)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False,device=device)
        self.scaled_dot_product_attention = ScaledDotProductAttention(device=device)  ##
        self.lay_norm = nn.LayerNorm(self.d_model,device=device)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).expand(batch_size, self.n_heads, attn_mask.size(-2), attn_mask.size(-1))

        context, attn = self.scaled_dot_product_attention(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * d_v)
        output = self.fc(context)
        return self.lay_norm(output + residual), attn



class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, device):
        super(PoswiseFeedForwardNet, self).__init__()
        self.device = device
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False,device=device),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False,device=device))
        self.lay_norm=nn.LayerNorm(d_model,device=device)
    def forward(self, inputs):  # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return self.lay_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, device):
        super(EncoderLayer, self).__init__()
        self.device = device
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, device=device)  # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet(device=device)   # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, device):
        super(Encoder, self).__init__()
        self.device = device
        self.src_emb = nn.Linear(2, d_model,device=device)
        self.pos_emb = PositionalEncoding(d_model,device=device)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, device=device) for _ in range(n_layers)])

        self.src_emb.to(device)
        self.pos_emb.to(device)

    def forward(self, enc_inputs, enc_mask):
        enc_outputs = self.src_emb(enc_inputs)
        # enc_outputs = self.pos_emb(enc_outputs)位置编码
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

def create_padding_mask(seq):
    return (torch.sum(seq, dim=-1) == 0).unsqueeze(1)

def get_attn_pad_mask(seq_q, seq_k):
    mask_q = create_padding_mask(seq_q)
    mask_k = create_padding_mask(seq_k)

    expanded_mask_q = mask_q.unsqueeze(1).unsqueeze(1)
    expanded_mask_k = mask_k.unsqueeze(1).unsqueeze(2)

    attn_pad_mask = expanded_mask_k.expand(-1, seq_q.size(1), seq_k.size(1), -1)
    return attn_pad_mask



class Transformer(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, device):
        super(Transformer, self).__init__()
        self.device = device
        self.Encoder = Encoder(d_model, n_layers, n_heads, device=device)
        self.projection = nn.Linear(d_model, 5, bias=False, device=device).to(device)

    def forward(self, enc_inputs):
        # print(f"Input Shape: {enc_inputs.shape}")
        enc_mask = create_padding_mask(enc_inputs)
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs, enc_mask.to(self.device))
        masked_outputs = enc_outputs * enc_mask.unsqueeze(-1)

        dec_logits = self.projection(masked_outputs[:, -1, :])

        # print("dec_logits shape:", dec_logits.shape)
        output_avg = torch.mean(dec_logits, dim=1)

        # print("output_avg shape:", output_avg.shape)
        return output_avg, enc_self_attns



class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img_feats, ts_feats):
        q = self.query(img_feats)
        k = self.key(ts_feats)
        v = self.value(ts_feats)

        attn_score = torch.matmul(q, k.transpose(-2, -1))
        attn_prob = self.softmax(attn_score)
        out = torch.matmul(attn_prob, v)
        return out


class MultiModalFusion(nn.Module):
    def __init__(self, pvt_model, transformer_model, num_classes=5):
        super(MultiModalFusion, self).__init__()
        self.pvt_model = pvt_model
        self.transformer_model = transformer_model

        self.cross_modal_attention = nn.ModuleList([CrossModalAttention(512) for _ in range(4)])
        self.classifier = nn.Linear(512 * 4, num_classes)

    def forward(self, images, time_series):
        pvt_features = self.pvt_model(images)
        transformer_feature, _ = self.transformer_model(time_series)

        # Adjust feature sizes
        resized_pvt_features = [F.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1) for feat in pvt_features]
        transformer_feature = torch.mean(transformer_feature, dim=1)

        # Perform cross-modal attention
        cross_modal_features = []
        for i, pvt_feature in enumerate(resized_pvt_features):
            cross_modal_feature = self.cross_modal_attention[i](pvt_feature, transformer_feature)
            cross_modal_features.append(cross_modal_feature)

        # Concatenate and classify
        fusion_feature = torch.cat(cross_modal_features, dim=-1)
        class_output = self.classifier(fusion_feature)

        return class_output


import torch
from torchvision import transforms
from PIL import Image
from model import pvt_reduce  # 确保这个导入语句与你的模型路径匹配
from model import multi_model
# 加载模型
def predict_image(image_path):
    # 打开图像并应用变换
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # 增加一个批处理维度

    # 进行预测
    with torch.no_grad():
        outputs, feature_maps = model(image)
        _, predicted = torch.max(outputs.data, 1)

    return predicted.item(), feature_maps  # 返回预测的类别


if __name__ == '__main__':
    pvt_model = pvt_tiny(num_classes=5, pretrained=False)
    transformer_model = Transformer(d_model=512, n_layers=1, n_heads=8,device="cuda").to('cuda')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pvt_model.load_state_dict(torch.load('E:/zlx2/pythonProject1/save_model/pvt_tiny_5_classes.pth'))  # 加载训练好的权重
    model =pvt_model.to(device)


    # 定义图像预处理步骤
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])




    # 使用模型进行预测
    path = 'E:/zlx2/pythonProject1/data/pic/hangji1/trajectory_map1.png'
    predicted_class, feature_maps = predict_image(path)
    print(f'The predicted class is: {predicted_class}')
    print("feature_maps:\n")
    print(feature_maps)












