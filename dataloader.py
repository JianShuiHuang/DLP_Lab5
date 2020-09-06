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
        Data = []
        for item in data.items():
            Data.append(item)

        return np.squeeze(Data)
    else:
        return data

def dictionary():
    return json.load(open('./lab5_dataset/objects.json', 'r'))

class DataLoader(data.Dataset):
    def __init__(self, mode, image_size=64):
        self.mode = mode   
        self.data = get_data(mode)
        self.dict = dictionary()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode == 'train': 
            img_name = self.data[index][0]
            objects = []
            for item in self.data[index][1]:
                objects.append(self.dict[item])

            img = Image.open('./lab5_dataset/iclevr/'+img_name)
            img = img.resize((64, 64),Image.ANTIALIAS)
            img = np.asarray(img)/255
            img = np.transpose(img, (2,0,1))
            img = torch.from_numpy(img)
            
            
            
            condition = torch.zeros(24)
            condition = torch.tensor([v+1 if i in objects else v for i,v in enumerate(condition)])
            
            data = (img, condition)
        else:
            objects = []
            for item in self.data[index]:
                objects.append(self.dict[item])
            condition = torch.zeros(24)
            data = torch.tensor([v+1 if i in objects else v for i,v in enumerate(condition)])
        
        return data     
