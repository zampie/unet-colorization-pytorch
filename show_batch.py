import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from IPython.display import HTML
import torch.functional as F
from data_loader import Img_tran_AB, SimpleImageFolder

if __name__ == '__main__':
    data_path = './datasets/konaedge'

    batch_size = 16
    workers = 0
    # image_size = 128
    transform_A = transforms.Compose([
        # transforms.Resize(image_size),
        # transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    transform_B = transforms.Compose([
        # transforms.Resize(image_size),
        # transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = Img_tran_AB(root=data_path, transform_A=transform_A, transform_B=transform_B)


    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)

    print('Data length: ',len(dataset))
    # Plot some training images
    real_batch = next(iter(dataloader))

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Training Images A")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[0], nrow=4, padding=4, normalize=True).cpu(), (1, 2, 0)))

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Training Images B")
    plt.imshow(
        np.transpose(vutils.make_grid(real_batch[1], nrow=4, padding=4, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
