#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse
import torch
import numpy as np
from PIL import Image, ImageFont, ImageDraw

import pdb

np.set_printoptions(precision=1)
chars = "abcdefghijklmnopqrstuvwzyxABCDEFGHIJKLMNOPQRSTUVWXY123"
if __name__ == '__main__':
    opt = argparse.ArgumentParser(description="write program description here")

    # insert options here
    opt.add_argument('-f', action='store', dest='preprocessed_dir', required=True)
    opt.add_argument('--font_path', action='store', dest='font_path',
                     default='/usr/share/fonts/truetype/inconsolata/Inconsolata.otf')
    options = opt.parse_args()
    src_dict = open(options.preprocessed_dir + '/dict.en.clean.txt', 'r', encoding='utf-8').readlines()
    src_dict = [i.split()[0] for i in src_dict]
    src_dict = ['<lua>', '<pad>', '</s>', '<unk>'] + src_dict
    font = ImageFont.truetype(options.font_path, 24)
    _W, H = font.getsize(chars)
    W = _W // len(chars)  # this size was set manually for the default font and font size
    mat = np.zeros((len(src_dict), H, W))
    for idx, c in enumerate(src_dict):
        if idx >= 4:
            print(c, idx)
            im = Image.new('L', (W + _W, H))
            draw = ImageDraw.Draw(im)
            #w, h = font.getsize(c + chars)
            draw.text((0, 0), c + chars, font=font, fill="white")
            del draw
            m = np.asarray(im) / 255.0
            m = m[:, :12]
            Image.fromarray(m * 255.0).convert("L").save(options.preprocessed_dir + "/img_" + c + ".png")
            if idx % 25 == 0:
                pass
        else:
            if idx == 1:
                m = np.zeros((H, W))
            else:
                m = np.random.uniform(0.0, 1.0, (H, W))
        mat[idx, :, :] = m
    t = torch.Tensor(mat)
    torch.save(t, options.preprocessed_dir + '/tmp.dict.img')
