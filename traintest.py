import numpy as np
import gc
import random
from PIL import Image
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image, make_grid
from dataloader import *
from model import *
from evaluator import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# custom weights initialization called on generator and discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def training(generator, discriminator, image_size, latent_size, lr_g, lr_d, batch_size, num_epochs):
    real_label = 1
    fake_label = 0
    
    g_loss = []
    d_loss = []
    accuracy_list = []
    
    # load data
    traindata = DataLoader('train', image_size=64)
    train_data = data.DataLoader(traindata, batch_size, num_workers=2, shuffle=True)

    # init Loss & optimizer
    Loss = nn.BCELoss()
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(2e-4, 0.999))
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(2e-4, 0.999))

    # init fixed noise
    fixed_noise = torch.randn(32, latent_size, 1, 1, device=device)
    
    # start training
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for datas in train_data:
            b_size = datas[0].size(0)
            img = datas[0].to(device)
            condition = datas[1].to(device)
            
            # train discriminator
            discriminator.zero_grad()
            train_label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(img, condition).view(-1)
            
            errD_real = Loss(output, train_label)
            errD_real.backward()

            noise = torch.randn(b_size, latent_size, 1, 1, device=device)
            fake = generator(noise, condition)
            train_label.fill_(fake_label)
            
            output = discriminator(fake.detach(), condition).view(-1)
            
            errD_fake = Loss(output, train_label)
            errD_fake.backward()

            errD = errD_fake + errD_real
            
            optimizer_d.step()
            
            # train generator
            generator.zero_grad()
            train_label.fill_(real_label)
            output = discriminator(fake, condition).view(-1)
            
            errG = Loss(output, train_label)
            errG.backward()
            
            optimizer_g.step()

        # store model and print information
        accuracy,_ = testing(generator, fixed_noise, latent_size, batch_size)
        accuracy_list.append(accuracy)
        if accuracy>=max(accuracy_list) :
            print ("Model save...")
            torch.save(generator, "generator.pkl")
            torch.save(discriminator, "discriminator.pkl")
            
        if epoch % 1 == 0:
            print('(%d %d%%) Accuracy: %.4f Loss_D: %.4f Loss_G: %.4f'
                            % (epoch, epoch/num_epochs * 100, accuracy, errD.item(), errG.item()))
        
        g_loss.append(errG.item())
        d_loss.append(errD.item())
        gc.collect()
        torch.cuda.empty_cache()

    np.save("GLoss.npy", g_loss)
    np.save("DLoss.npy", d_loss)
    np.save("Accuracy", accuracy_list)
    return accuracy_list

def testing(generator, noise=None, latent_size=100, batch_size=32):
    E = evaluation_model()
    
    img_list = []
    accuracy_list = []

    if noise is None:
        noise = torch.randn(batch_size, latent_size, 1, 1, device=device)
    
    # load data 
    testdata = DataLoader('test')
    test_data = data.DataLoader(testdata, batch_size, num_workers=2)
    
    with torch.no_grad():
        for condition in test_data:
            condition = condition.to(device)
            fake = generator(noise, condition).detach()
            
            accuracy_list.append(E.eval(fake, condition))
            img_list.append(make_grid(fake, padding=2, normalize=True).cpu())

    return sum(accuracy_list)/len(accuracy_list), img_list
