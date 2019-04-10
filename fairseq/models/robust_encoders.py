#!/usr/bin/env python
__author__ = 'arenduchintala'
import torch
import torch.nn as nn
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
import numpy as np

import pdb


class SobelLayer(nn.Module):
    def __init__(self):
        super(SobelLayer, self).__init__()
        self.sx = nn.Conv2d(1, 1, kernel_size=(2, 2), stride=1, padding=(1, 1), bias=None)
        self.sx.weight.data[0, 0, :, :] = torch.Tensor([[-1, 1], [-1, 1]])
        self.sy = nn.Conv2d(1, 1, kernel_size=(2, 2), stride=1, padding=(1, 1), bias=None)
        self.sy.weight.data[0, 0, :, :] = torch.Tensor([[1, 1], [-1, -1]])

        self.sx.weight.requires_grad = False
        self.sy.weight.requires_grad = False

    def forward(self, x):
        g_x = self.sx(x)
        g_y = self.sy(x)
        g = torch.sqrt(g_x ** 2 + g_y ** 2)
        return g


class GaussianLayer(nn.Module):
    def __init__(self):
        super(GaussianLayer, self).__init__()
        self.blur = nn.Conv2d(1, 1, kernel_size=(9, 9), stride=1, padding=(4, 4), bias=None)
        n = np.zeros((9, 9))
        n[4, 4] = 1
        k = gaussian_filter(n, sigma=1.6)
        self.blur.weight.data[0, 0, :, :] = torch.from_numpy(k)

    def forward(self, x):
        return self.blur(x)


class HighwayNetwork(nn.Module):

    def __init__(self, input_size):
        super(HighwayNetwork, self).__init__()
        # transform gate
        self.trans_gate = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid())
        # highway
        self.activation = nn.ReLU()

        self.h_layer = nn.Sequential(nn.Linear(input_size, input_size),
                                     self.activation)
        self.trans_gate[0].weight.data.uniform_(-0.05, 0.05)
        self.h_layer[0].weight.data.uniform_(-0.05, 0.05)
        self.trans_gate[0].bias.data.fill_(0)
        self.h_layer[0].bias.data.fill_(0)

    def forward(self, x):
        t = self.trans_gate(x)
        h = self.h_layer(x)
        z = torch.mul(t, h) + torch.mul(1.0 - t, x)
        return z


class CharCNNEncoder(nn.Module):
    def __init__(self, embed_tokens, num_chars, dropout_in=0.1):
        super(CharCNNEncoder, self).__init__()
        self.num_chars = num_chars
        self.embed_tokens = embed_tokens
        self.cnn = nn.Conv1d(self.embed_tokens.embedding_dim, 512, kernel_size=6, padding=0)
        self.mp = nn.MaxPool1d(num_chars - 5)
        self.cnn.weight.data.uniform_(-0.05, 0.05)
        out_channel, in_channel, kernel_size = self.cnn.weight.shape
        # approximating many smaller kernel sizes with one large kernel by zeroing out kernel elements
        for i in [1, 2, 3, 4, 5]:
            x = torch.Tensor(85, 512, 6).uniform_(-0.05, 0.05)
            x[:, :, torch.arange(i, 6).long()] = 0
            self.cnn.weight.data[torch.arange((i - 1) * 85, i * 85).long(), :, :] = x
        self.cnn.bias.data.fill_(0.0)
        self.highway = HighwayNetwork(self.embed_tokens.embedding_dim)

    def forward(self, src_tokens):
        assert src_tokens.dim() == 3
        bsz, seqlen, num_chars = src_tokens.shape
        src_tokens_reshape = src_tokens.view(-1, num_chars)
        emb = self.embed_tokens(src_tokens_reshape) # (bsz * seqlen, num_chars, 512)
        emb = emb.transpose(1, 2)  #(bsz * seqlen, 512, num_chars)
        emb = self.mp(self.cnn(emb)).squeeze(2)
        emb = self.highway(emb)
        return emb.view(bsz, seqlen, -1) # (bsz, seqlen, embed_dim)


class MultiFeatEncoder(nn.Module):
    def __init__(self, embed_tokens):
        self.embed_tokens = embed_tokens

    def forward(self, src_tokens):
        assert src_tokens.dim() == 3
        bsz, seqlen, num_feat = src_tokens.size()
        feat_tokens = [src_tokens[:, :, i] for i in range(1, num_feat)]
        feat_x = [self.embed_tokens(fx) for fx in feat_tokens]
        x = self.embed_tokens(src_tokens[:, :, 0])
        for fx in feat_x:
            x = x + fx
        return x


class OldFLCEncoder(nn.Module):
    def __init__(self, embed_tokens_boc, embed_tokens_f, embed_tokens_l, dropout_in=0.1):
        super(OldFLCEncoder, self).__init__()
        self.embed_tokens_boc = embed_tokens_boc
        self.embed_tokens_f = embed_tokens_f
        self.embed_tokens_l = embed_tokens_l
        embed_dim = self.embed_tokens_boc.embedding_dim
        self.robust_ff = nn.Sequential(nn.Linear(3 * embed_dim, 3 * embed_dim),
                                       nn.ReLU(),
                                       nn.Linear(3 * embed_dim, embed_dim),
                                       nn.ReLU())

    def forward(self, src_tokens):
        assert src_tokens.dim() == 3
        src_tokens_f = src_tokens[:, :, 0]
        src_tokens_l = src_tokens[:, :, 1]
        src_tokens_boc = src_tokens[:, :, 2:]
        emb_f = self.embed_tokens_f(src_tokens_f)
        emb_l = self.embed_tokens_l(src_tokens_l)
        emb_boc = self.embed_tokens_boc(src_tokens_boc).mean(dim=2)
        emb = torch.cat([emb_f, emb_boc, emb_l], dim=2)
        return self.robust_ff(emb)


class FLCEncoder(nn.Module):
    def __init__(self, embed_tokens_boc, embed_tokens_f, embed_tokens_l, dropout_in=0.1):
        super(FLCEncoder, self).__init__()
        self.embed_tokens_boc = embed_tokens_boc
        self.embed_tokens_f = embed_tokens_f
        self.embed_tokens_l = embed_tokens_l
        embed_dim = self.embed_tokens_boc.embedding_dim
        self.robust_ff = nn.Sequential(nn.Linear(3 * embed_dim, 3 * embed_dim),
                                       nn.ReLU(),
                                       nn.Dropout(dropout_in),
                                       nn.Linear(3 * embed_dim, 3 * embed_dim),
                                       nn.ReLU(),
                                       nn.Dropout(dropout_in),
                                       nn.Linear(3 * embed_dim, embed_dim))

    def forward(self, src_tokens):
        assert src_tokens.dim() == 3
        src_tokens_f = src_tokens[:, :, 0]
        src_tokens_l = src_tokens[:, :, 1]
        src_tokens_boc = src_tokens[:, :, 2:]
        emb_f = self.embed_tokens_f(src_tokens_f)
        emb_l = self.embed_tokens_l(src_tokens_l)
        emb_boc = self.embed_tokens_boc(src_tokens_boc).mean(dim=2)
        emb = torch.cat([emb_f, emb_boc, emb_l], dim=2)
        return self.robust_ff(emb)


class VisualEncoder(nn.Module):
    def __init__(self, embed_dim, img_r, img_c, img_emb, num_chars, dropout_in=0.1):
        super(VisualEncoder, self).__init__()
        self.img_emb = img_emb
        self.num_chars = num_chars
        self.embed_dim = embed_dim
        self.img_emb.weight.requires_grad = False
        self.img_r = img_r
        self.img_c = img_c
        self.g_layer = GaussianLayer()
        self.g_layer.blur.weight.requires_grad = False
        self.cnn = nn.Conv1d(self.img_r, 512, kernel_size=6 * self.img_c, padding=0)
        self.mp = nn.MaxPool1d(num_chars * self.img_c - (6 * img_c - 1))
        self.cnn.weight.data.uniform_(-0.05, 0.05)
        out_channel, in_channel, kernel_size = self.cnn.weight.shape
        # approximating many smaller kernel sizes with one large kernel by zeroing out kernel elements
        for i in [1, 2, 3, 4, 5]:
            x = torch.Tensor(85, self.img_r, 6 * self.img_c).uniform_(-0.05, 0.05)
            x[:, :, torch.arange(i * self.img_c, 6 * self.img_c).long()] = 0
            self.cnn.weight.data[torch.arange((i - 1) * 85, i * 85).long(), :, :] = x
        self.cnn.bias.data.fill_(0.0)
        self.highway = HighwayNetwork(self.embed_dim)

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
        emb = emb.squeeze(1)
        emb = self.mp(self.cnn(emb)).squeeze(2)
        emb = self.highway(emb)
        return emb.view(bsz, seqlen, -1) # (bsz, seqlen, embed_dim)


class VisualEdgeEncoder(nn.Module):
    def __init__(self, embed_dim, img_r, img_c, img_emb, num_chars, edge_threshold, dropout_in=0.1):
        super(VisualEdgeEncoder, self).__init__()
        self.img_emb = img_emb
        self.num_chars = num_chars
        self.embed_dim = embed_dim
        self.img_emb.weight.requires_grad = False
        self.img_r = img_r
        self.img_c = img_c
        self.g_layer = GaussianLayer()
        self.g_layer.blur.weight.requires_grad = False
        self.threshold = edge_threshold
        self.s_layer = SobelLayer()
        self.cnn = nn.Conv1d(self.img_r + 1, 512, kernel_size=6 * self.img_c, padding=0)
        self.mp = nn.MaxPool1d(num_chars * self.img_c - (6 * img_c - 1) + 1)
        self.cnn.weight.data.uniform_(-0.05, 0.05)
        out_channel, in_channel, kernel_size = self.cnn.weight.shape
        # approximating many smaller kernel sizes with one large kernel by zeroing out kernel elements
        for i in [1, 2, 3, 4, 5]:
            x = torch.Tensor(85, self.img_r + 1, 6 * self.img_c).uniform_(-0.05, 0.05)
            x[:, :, torch.arange(i * self.img_c, 6 * self.img_c).long()] = 0
            self.cnn.weight.data[torch.arange((i - 1) * 85, i * 85).long(), :, :] = x
        self.cnn.bias.data.fill_(0.0)
        self.highway = HighwayNetwork(self.embed_dim)

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
        #pdb.set_trace()
        #for j in [10, 20, 30, 40, 50, 80, 12, 23, 34, 456, 57, 23, 35, 8, 23, 450,1213, 135, 1353, 250, 94]:
        #    foo = emb[j, 0, :, :]
        #    foo = foo.unsqueeze(0).unsqueeze(0)
        #    self.save_img_debug(foo[0, 0, :, :], str(j) + '.' + "normal.png")
        #    foo = self.g_layer(foo)
        #    self.save_img_debug(foo[0, 0, :, :], str(j) + '.' + "blur.png")
        #    for t in [0.025, 0.05, 0.1, 0.15]:
        #        _f = foo.clone()
        #        _f[foo > t] = 1
        #        _f[foo <= t] = 0
        #        self.save_img_debug(_f[0, 0, :, :], str(j) + '.' + str(t) + 'thresh.png')
        #        s_f = self.s_layer(_f)
        #        self.save_img_debug(s_f[0, 0, :, :], str(j) + '.' + str(t) + 'sobel.thresh.png')
        #pdb.set_trace()
        emb = self.g_layer(emb)
        emb[emb > self.threshold] = 1
        emb[emb <= self.threshold] = 0
        emb = self.s_layer(emb)
        emb = emb.squeeze(1)
        emb = self.mp(self.cnn(emb)).squeeze(2)
        emb = self.highway(emb)
        return emb.view(bsz, seqlen, -1) # (bsz, seqlen, embed_dim)
