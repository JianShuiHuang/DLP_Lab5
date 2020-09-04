import numpy as np
import json
from PIL import Image
import torch
from torch.utils import data
import torchvision.transforms as transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data(mode):
    data = json.load(open('./lab5_dataset/'+mode+'.json', 'r'))
    if mode == 'train':
        data = [i for i in data.items()]
    return np.squeeze(data)

def dictionary():
    return json.load(open('./lab5_dataset/objects.json', 'r'))

class DataLoader(data.Dataset):
    def __init__(self, mode, image_size=64):
        self.mode = mode   
        self.data = get_data(mode)
        self.obj_dict = dictionary()
        self.transformation = transforms.Compose([
                                  transforms.Resize(image_size),
                                  transforms.CenterCrop(image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ])
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'train': 
            img_name = self.data[index][0]
            objects = []
            for item in self.data[index][1]:
                objects.append(self.obj_dict[item])

            img = np.array(Image.open('./iclevr/'+img_name))[...,:-1]
            img = self.transformation(Image.fromarray(img))
            
            condition = torch.zeros(24)
            condition = torch.tensor([v+1 if i in objects else v for i,v in enumerate(condition)])
            
            data = (img, condition)
        else:
            objects = []
            for item in self.data[index][1]:
                objects.append(self.obj_dict[item])
            condition = torch.zeros(24)
            data = torch.tensor([v+1 if i in objects else v for i,v in enumerate(condition)])
        
        return data     