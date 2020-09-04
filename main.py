import numpy as np
import random
import matplotlib.pyplot as plt
from dataloader import *
from traintest import *
from model import *
import time
from Time import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_size = 64
lr_G = 0.0002
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
