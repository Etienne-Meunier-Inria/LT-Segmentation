import flowiz
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
from glob import glob
from natsort import natsorted
import numpy as np

def r(x, w=240, h=240) :
    x = torch.tensor(x)
    x = x.permute(2, 0, 1)[None]
    x = nn.functional.interpolate(x, size=(w, h), mode='bilinear', align_corners=False)
    return x[0].permute(1, 2, 0).numpy()

sf = lambda x : flowiz.convert_from_flow(x)

def load_flo(fp) :
    flow = flowiz.read_flow(fp)
    return torch.tensor(flow).permute(2, 0, 1)

def load_mask(mp, sh=(400, 400)) :
    m = (np.array(Image.open(mp).resize((sh[0], sh[1])))/255).astype('bool')
    cfd = 0.95 # Confidence of the model
    w_mask = np.stack([m*(cfd - (1 - cfd)) + ( 1-cfd),
                       (1 - m)*(cfd - (1 - cfd)) + ( 1-cfd)], axis=-1)
    return torch.tensor(w_mask, dtype=torch.float32).permute(2,0,1)


def load_image(ip, sh=(400, 400)) :
    im = (np.array(Image.open(ip).resize((sh[0], sh[1])))/255)
    return torch.tensor(im, dtype=torch.float32).permute(2,0,1)

def load_sequence(sps, loader, n_elements=-1) :
    elements = []
    for i, sp in enumerate(natsorted(glob(sps))) :
        elements.append(loader(sp))
        if i == n_elements :
            break
    elements = torch.stack(elements)
    return elements

def write_file(name_prefix, type_split, df_train, df_val, flp='../DataSplit/') :
    split_file_val = f'{flp}{name_prefix}_{type_split}_val.csv'
    split_file_train = f'{flp}{name_prefix}_{type_split}_train.csv'
    df_train.to_csv(split_file_train, header=True, index=False)
    df_val.to_csv(split_file_val, header=True, index=False)