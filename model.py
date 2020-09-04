import numpy as np
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        ##layer 1
        self.convt2d_1 = nn.ConvTranspose2d(input_size, hidden_size*8, kernel_size=4, stride=1, padding=0, bias=False)
        self.batchnorm2d_1 = nn.BatchNorm2d(hidden_size*8)
        self.activ_1 = nn.LeakyReLU()

        ##layer 2
        self.convt2d_2 = nn.ConvTranspose2d(hidden_size*8, hidden_size*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2d_2 = nn.BatchNorm2d(hidden_size*4)
        self.activ_2 = nn.LeakyReLU()

        ##layer 3
        self.convt2d_3 = nn.ConvTranspose2d(hidden_size*4, hidden_size*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2d_3 = nn.BatchNorm2d(hidden_size*2)
        self.activ_3 = nn.LeakyReLU()

        ##layer 4
        self.convt2d_4 = nn.ConvTranspose2d(hidden_size*2, hidden_size, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2d_4 = nn.BatchNorm2d(hidden_size)
        self.activ_4 = nn.LeakyReLU()

        ##layer 5
        self.convt2d_5 = nn.ConvTranspose2d(hidden_size, output_size, kernel_size=4, stride=2, padding=1, bias=False)
        self.activ_5 = nn.Tanh()

    def forward(self, input, condition):
        input = torch.cat((input, condition.view(input.size(0), -1, 1, 1)), 1)
        y = self.convt2d_1(input)
        y = self.batchnorm2d_1(y)
        y = self.activ_1(y)
        y = self.convt2d_2(y)
        y = self.batchnorm2d_2(y)
        y = self.activ_2(y)
        y = self.convt2d_3(y)
        y = self.batchnorm2d_3(y)
        y = self.activ_3(y)
        y = self.convt2d_4(y)
        y = self.batchnorm2d_4(y)
        y = self.activ_4(y)
        y = self.convt2d_5(y)
        y = self.activ_5(y)
        return y

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.linear = nn.Linear(24, hidden_size*hidden_size)
        
        ##layer 1
        self.conv2d_1 = nn.Conv2d(input_size, hidden_size, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2d_1 = nn.BatchNorm2d(hidden_size)
        self.activ_1 = nn.LeakyReLU()

        ##layer 2
        self.conv2d_2 = nn.Conv2d(hidden_size, hidden_size*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2d_2 = nn.BatchNorm2d(hidden_size*2)
        self.activ_2 = nn.LeakyReLU()

        ##layer 3
        self.conv2d_3 = nn.Conv2d(hidden_size*2, hidden_size*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2d_3 = nn.BatchNorm2d(hidden_size*4)
        self.activ_3 = nn.LeakyReLU()

        ##layer 4
        self.conv2d_4 = nn.Conv2d(hidden_size*4, hidden_size*8, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2d_4 = nn.BatchNorm2d(hidden_size*8)
        self.activ_4 = nn.LeakyReLU()

        ##layer 5
        self.conv2d_5 = nn.Conv2d(hidden_size * 8, output_size, kernel_size=4, stride=1, padding=0, bias=False)
        self.activ_5 = nn.Sigmoid()

    def forward(self, input, condition):
        condition = self.linear(condition).view(input.size(0), 1, self.hidden_size, self.hidden_size)
        input = torch.cat((input, condition), 1)
        y = self.conv2d_1(input)
        y = self.batchnorm2d_1(y)
        y = self.activ_1(y)
        y = self.conv2d_2(y)
        y = self.batchnorm2d_2(y)
        y = self.activ_2(y)
        y = self.conv2d_3(y)
        y = self.batchnorm2d_3(y)
        y = self.activ_3(y)
        y = self.conv2d_4(y)
        y = self.batchnorm2d_4(y)
        y = self.activ_4(y)
        y = self.conv2d_5(y)
        y = self.activ_5(y)
        return y