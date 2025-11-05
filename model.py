import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.images_size = (32,32)
        self.w = 32
        self.h = 32
        self.input_channels = 3
        self.kernal_size = 3
        
        self.conv_layer1 = nn.Conv2d(self.input_channels, 32 , self.kernal_size); self.pool = nn.MaxPool2d(2,2)
        self.conv_layer2 = nn.Conv2d(32, 64 , self.kernal_size)
        self.conv_layer3 = nn.Conv2d(64, 64 , self.kernal_size) 

        self.fc1 = nn.Linear(64*4*4, 64)
        self.fc2 = nn.Linear(64, 10)
    

    
    def forward(self,x):
        x = self.pool(F.relu(self.conv_layer1(x)))
        x = self.pool(F.relu(self.conv_layer2(x)))
        x = (F.relu(self.conv_layer3(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


        