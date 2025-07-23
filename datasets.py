"""
Datasets file. Code adapted from LOST: https://github.com/valeoai/LOST
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms

root = "/home/huifang/workspace/code/registration/data"

class Sennet_image_pair(Dataset):
    def __init__(self, datalist="train_data_list.txt",image_transformer=None):
        super(Sennet_image_pair, self).__init__()
        self.root = "/media/huifang/data/sennet/hf_aligned_data/"
        self.image_transformer = image_transformer
        f = open(datalist, 'r')
        self.data = f.readlines()

    def __getitem__(self, index):
        # make consistent with all other datasets
        # return a PIL Image
        data_pair = self.data[index % len(self.data)].strip()
        data_pair = data_pair.split(' ')
        xenium_sampleid, xenium_regionid, codex_sampleid, codex_regionid = data_pair

        xenium_img_name = f"xenium_{xenium_sampleid}_{xenium_regionid}_enhanced.png"
        codex_img_name = f"codex_{codex_sampleid}_{codex_regionid}_enhanced.png"

        fixed_image = Image.open(self.root + xenium_img_name)
        moving_image = Image.open(self.root + codex_img_name)


        if self.image_transformer is not None:
            fixed_image = self.image_transformer(fixed_image)
            moving_image = self.image_transformer(moving_image)
        out = {'fixed_image': fixed_image,'moving_image': moving_image}
        return out
    def __len__(self):
        return len(self.data)