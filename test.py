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
import torch.functional as F
from unet_model import UNet
# from model_PaintsChainer import UNet
from data_loader import Img_tran_AB, SimpleImageFolder

if __name__ == '__main__':

    batch_size = 1  # Batch size during training
    # image_size = 128  # All images will be resized to this size using a transformer.

    # Root directory for dataset
    data_path = './datasets/konaedge/'
    samples_path = './samples_0/'
    ckpt_path = './samples_0/ckpt_200000.tar'
    os.makedirs(os.path.join(samples_path, 'result'), exist_ok=True)

    workers = 0  # Number of workers for dataloader
    # device = torch.device("cpu")
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

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

    # Create the dataset
    dataset = Img_tran_AB(root=data_path,
                          transform_A=transform_A,
                          transform_B=transform_B,
                          test=True
                          )

    # dataset = SimpleImageFolder(root=data_path, transform=transform_A)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=workers)


    def get_layer_param(model):
        return sum([torch.numel(param) for param in model.parameters()])


    net = UNet(1, 3).to(device)

    print(net)
    print('parameters:', get_layer_param(net))

    print("Loading checkpoint...")

    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['net_state_dict'])
    net.eval()



    print("Starting Test...")
    for i, data in enumerate(dataloader, 0):
        # -----------------------------------------------------------
        # Initial batch
        data_A = data[0].to(device)
        data_B = data[1].cpu()

        real_batch_size = data_A.size(0)

        # -----------------------------------------------------------
        # Generate fake img:
        fake_B = net(data_A)
        # -----------------------------------------------------------
        # Output training stats
        print('[%2d/%d]' % (i, len(dataloader)))
        vutils.save_image(data_A, os.path.join(samples_path, 'result', '%s_data_A.jpg' % str(i).zfill(6)),
                          padding=0, nrow=2, normalize=True)
        vutils.save_image(data_B, os.path.join(samples_path, 'result', '%s_data_B.jpg' % str(i).zfill(6)),
                          padding=0, nrow=2, normalize=True)
        vutils.save_image(fake_B, os.path.join(samples_path, 'result', '%s_fake_B.jpg' % str(i).zfill(6)),
                          padding=0, nrow=1, normalize=True)
