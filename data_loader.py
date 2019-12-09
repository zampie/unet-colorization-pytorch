import torch.utils.data as data
import torch
from PIL import Image

import os
import os.path
import sys
import numpy as np
import glob


class Img_tran_AB(data.Dataset):

    def __init__(self, root, transform_A=None, transform_B=None, test=False):
        self.root = root
        self.transform_A = transform_A
        self.transform_B = transform_B
        self.test = test

        self.dataset_A = []
        self.dataset_B = []

        self.walk_data(root)

    def __len__(self):
        return len(self.dataset_A)

    def __getitem__(self, idx):
        img_A = self.dataset_A[idx]
        img_B = self.dataset_B[idx]

        img_A = Image.open(img_A)
        img_B = Image.open(img_B)

        img_A = img_A.convert('L')
        # img_A = img_A.convert('RGB')

        img_B = img_B.convert('RGB')

        if self.transform_A:
            img_A = self.transform_A(img_A)

        if self.transform_B:
            img_B = self.transform_B(img_B)

        return img_A, img_B

    def walk_data(self, root):
        if self.test:
            root_A = os.path.join(root, 'testA')
            root_B = os.path.join(root, 'testB')
        else:
            root_A = os.path.join(root, 'trainA')
            root_B = os.path.join(root, 'trainB')

        for dirpath, dirnames, filenames in os.walk(root_A):
            for filename in filenames:
                # print(filename)
                img_A = os.path.join(dirpath, filename)
                self.dataset_A.append(img_A)

        for dirpath, dirnames, filenames in os.walk(root_B):
            for filename in filenames:
                img_B = os.path.join(dirpath, filename)
                self.dataset_B.append(img_B)

        if len(self.dataset_A) != len(self.dataset_B):
            raise TypeError('Different size of A and B')



class SimpleImageFolder(data.Dataset):

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.filename_list = glob.glob(os.path.join(self.root, '*'))

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        img = Image.open(self.filename_list[idx])

        img = img.convert('L')

        if self.transform:
            img = self.transform(img)

        return img