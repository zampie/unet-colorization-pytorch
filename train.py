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
from data_loader import Img_tran_AB

if __name__ == '__main__':

    max_epochs = 5  # Number of training epochs
    lr = 0.0002  # Learning rate for optimizers
    batch_size = 10  # Batch size during training
    # image_size = 128  # All images will be resized to this size using a transformer.

    # Root directory for dataset
    data_path = './datasets/konaedge'
    samples_path = './samples_0'
    os.makedirs(samples_path, exist_ok=True)

    workers = 0  # Number of workers for dataloader
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
                          )
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=workers)


    def get_layer_param(model):
        return sum([torch.numel(param) for param in model.parameters()])


    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    net = UNet(1, 3).to(device)

    # Apply the weights_init function to randomly initialize all weights
    net.apply(weights_init)

    print(net)
    print('parameters:', get_layer_param(net))

    # Initialize BCELoss function
    criterion = nn.MSELoss()

    # Setup Adam optimizers for both G and D
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))


    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    # define a method to save loss image
    def save_loss_image(list_loss, path):
        plt.figure(figsize=(10, 5))
        plt.title("Loss During Training")
        plt.plot(list_loss, label="loss")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(path, 'loss.jpg'))
        plt.close()


    list_loss = []
    iteration = 0
    print("Starting Training Loop...")
    for epoch in range(0, max_epochs):
        for i, data in enumerate(dataloader, 0):
            # -----------------------------------------------------------
            # Initial batch
            data_A = data[0].to(device)
            data_B = data[1].to(device)
            real_batch_size = data_A.size(0)

            # -----------------------------------------------------------
            # Update network:
            net.zero_grad()

            fake_B = net(data_A)
            # Calculate loss
            loss = criterion(fake_B, data_B)

            # Update G
            loss.backward()
            optimizer.step()

            # -----------------------------------------------------------
            # Output training stats
            with torch.no_grad():

                if i % 50 == 0:
                    print(
                        '[%d/%d][%2d/%d]\tLoss: %.4f'
                        % (epoch, max_epochs, i, len(dataloader), loss.item()))
                    list_loss.append(loss.item())

                # Check how the generator is doing by saving G's output on sample_noise
                if (iteration % 100 == 0) or ((epoch == max_epochs - 1) and (i == len(dataloader) - 1)):
                    vutils.save_image(data_A, os.path.join(samples_path, '%s_data_A.jpg' % str(iteration).zfill(6)),
                                      padding=2, nrow=2, normalize=True)
                    vutils.save_image(data_B, os.path.join(samples_path, '%s_data_B.jpg' % str(iteration).zfill(6)),
                                      padding=2, nrow=2, normalize=True)
                    vutils.save_image(fake_B, os.path.join(samples_path, '%s_fake_B.jpg' % str(iteration).zfill(6)),
                                      padding=2, nrow=2, normalize=True)
                    save_loss_image(list_loss, samples_path)

                iteration += 1

        # -----------------------------------------------------------
        # Save model
        if ((epoch + 1) % 5 == 0) or (epoch == max_epochs - 1):
            print('save model')
            save_path = os.path.join(samples_path, 'ckpt_%d.tar' % iteration)
            torch.save({
                'epoch': epoch,
                'iteration': iteration,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'list_loss': list_loss,
            }, save_path)
