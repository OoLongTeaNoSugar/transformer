#-*- ecoding:utf-8 -*-

import torch
import torch.nn as nn
from transformer.utils import MultiHeadAttention
from transformer.utils import PositionalEncoding
from transformer.beam import PositionalWiseFeedForward

class Encoderlayer(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        # self attention
        context, attention = self.attention(inputs, inputs, inputs, padding_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Decoderlayer(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(Decoderlayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self,
                dec_inputs,
                enc_outputs,
                self_atten_mask = None,
                con_atten_mask=None):

        dec_output, self_attention = self.attention(dec_inputs, dec_inputs, dec_inputs, self_atten_mask)

        context, con_attentin = self.attention(dec_output, enc_outputs, enc_outputs, con_atten_mask)

        output = self.feed_forward(context)

        return output, self_attention, con_attentin


