#!/usr/bin/env python
__author__ = 'arenduchintala'
import torch
import torch.nn as nn
import pdb
from PIL import Image
import numpy as np


class FLCEncoder(nn.Module):
    def __init__(self, embed_tokens, dropout_in=0.1):
        super(FLCEncoder, self).__init__()
        self.embed_tokens = embed_tokens
        embed_dim = self.embed_tokens.embedding_dim
        self.robust_ff = nn.Sequential(nn.Linear(3 * embed_dim, 3 * embed_dim),
                                       nn.ReLU(),
                                       nn.Dropout(dropout_in),
                                       nn.Linear(3 * embed_dim, embed_dim),
                                       nn.ReLU(),
                                       nn.Dropout(dropout_in))

    def forward(self, src_tokens):
        assert src_tokens.dim() == 3
        src_tokens_f = src_tokens[:, :, 0]
        src_tokens_l = src_tokens[:, :, 1]
        src_tokens_boc = src_tokens[:, :, 2:]
        emb_f = self.embed_tokens(src_tokens_f)
        emb_l = self.embed_tokens(src_tokens_l)
        emb_boc = self.embed_tokens(src_tokens_boc).mean(dim=2)
        emb = torch.cat([emb_f, emb_boc, emb_l], dim=2)
        return self.robust_ff(emb)


class VisualEncoder(nn.Module):
    def __init__(self, embed_dim, img_r, img_c, img_emb, dropout_in=0.1):
        super(VisualEncoder, self).__init__()
        self.img_emb = img_emb
        self.img_emb.weight.requires_grad = False
        self.img_r = img_r
        self.img_c = img_c
        self.conv2d = torch.nn.Conv2d(1, 10, kernel_size=(26, 20), stride=(1, 5))
        self.flatten = torch.nn.Linear(450, embed_dim)

    def save_img_debug(self, img_t, name):
        img = img_t.cpu().numpy() * 255.0
        Image.fromarray(img).convert("L").save(name)
        print('save_img_debug' + name)

    def forward(self, src_tokens):
        assert src_tokens.dim() == 3
        bsz, seqlen, num_chars = src_tokens.shape
        emb = self.img_emb(src_tokens)  # (bsz, seqlen, num_chars, 312)
        emb = emb.view(bsz, seqlen, num_chars, self.img_r, self.img_c)  # (bsz, seqlen, num_chars, 26, 12)
        emb = emb.view(-1, num_chars, self.img_r, self.img_c)  # (bsz * seqlen, num_chars, 26, 12)
        tiled_emb = torch.cat([emb[:, i, :, :] for i in range(num_chars)], dim=2) # (bsz * seqlen, 26, 12 * num_chars) TODO:replace with view op
        emb = tiled_emb.unsqueeze(1)
        emb = self.conv2d(emb).squeeze(2).view(-1, 10 * 45)
        emb = self.flatten(emb)
        emb = emb.view(bsz, seqlen, -1) # (bsz, seqlen, embed_dim)
        return emb
