import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from dataloader import *
from traintest import *
from model import *
import time
from Time import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def show1(a1, Epochs):
    plt.title('Accuracy Curve', fontsize = 18)
    plt.xlabel('Epochs')
    
    x = []
    for i in range(Epochs):
        x.append(i)
        plt.plot(i, a1[i], 'bo')
    
    plt.plot(x, a1)
    plt.show()
    
def show2(a, b, Epochs):
    plt.title('generator and discriminator loss', fontsize = 18)
    x = []
    for i in range(Epochs):
        x.append(i)
        plt.plot(i, a[i], 'bo')
        plt.plot(i, b[i], 'ro')
    
    plt.plot(x, a, b)
    plt.show()
    


image_size = 64
lr_G = 0.0001
lr_D = 0.0001
batch_size = 64
num_epochs = 1000
input_size_g = 124
hidden_size_g = 64
output_size_g = 3
input_size_d = 4
hidden_size_d = 64
output_size_d = 1
latent_size = 100


manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

generator = Generator(input_size_g, hidden_size_g, output_size_g).to(device)
generator.apply(weights_init)
discriminator = Discriminator(input_size_d, hidden_size_d, output_size_d).to(device)
discriminator.apply(weights_init)

print("model complete ...")

start = time.time()

accuracy = training(generator, discriminator, image_size, latent_size, lr_G, lr_D, batch_size, num_epochs)

print("Time: ", timeSince(start, 1 / 100))

print("Max Accuracy: ", max(accuracy))



g_loss = np.load("Gloss.npy")
d_loss = np.load("Dloss.npy")
accuracy = np.load("Accuracy.npy")


show1(accuracy, 1200)

show2(g_loss, d_loss, 1200)


"""
generator = torch.load('generator.pkl', map_location = torch.device('cpu'))
acc, imgs = testing(generator)
print("Accuracy: ", acc)

fig = plt.figure(figsize=(15,15))
plt.imshow(np.transpose(imgs[0],(1,2,0)))
"""


