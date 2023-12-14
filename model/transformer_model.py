
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
        return output_avg, enc_outputs

