#!/usr/bin/env python
__author__ = 'arenduchintala'
import torch
import torch.nn as nn
from PIL import Image


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
                                       nn.Linear(3 * embed_dim, 3 * embed_dim),
                                       nn.ReLU(),
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


class VisualEncoder_old(nn.Module):
    def __init__(self, embed_dim, img_r, img_c, img_emb, dropout_in=0.1):
        super(VisualEncoder, self).__init__()
        self.img_emb = img_emb
        self.img_emb.weight.requires_grad = False
        self.img_r = img_r
        self.img_c = img_c
        self.flatten = torch.nn.Linear(450, embed_dim)
        self.conv2d = torch.nn.Conv2d(1, 10, kernel_size=(26, 20), stride=(1, 5))

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


class VisualEncoder(nn.Module):
    def __init__(self, embed_dim, img_r, img_c, img_emb, dropout_in=0.1):
        super(VisualEncoder, self).__init__()
        self.img_emb = img_emb
        self.embed_dim = embed_dim
        self.img_emb.weight.requires_grad = False
        self.img_r = img_r
        self.img_c = img_c
        #self.conv_seq = nn.Sequential(torch.nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(3, 3)),
        #                              torch.nn.Dropout(dropout_in),
        #                              torch.nn.Conv2d(32, 32, kernel_size=(7, 7), stride=(3, 3)),
        #                              torch.nn.Dropout(dropout_in),
        #                              torch.nn.Conv2d(32, 512, kernel_size=(1, 24), stride=(1, 1)))

        self.conv_seq = nn.Sequential(torch.nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 3), padding=(1, 2)),
                                      torch.nn.Dropout(dropout_in),
                                      torch.nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 3), padding=(1, 2)),
                                      torch.nn.MaxPool2d(kernel_size=(1, 28))
                                      )
        self.flatten = torch.nn.Linear(16 * 26, self.embed_dim)

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
        emb = self.conv_seq(emb)
        emb = emb.squeeze(3).view(-1, 16 * 26)
        emb = self.flatten(emb)
        emb = emb.view(bsz, seqlen, self.embed_dim) # (bsz, seqlen, embed_dim)
        return emb
