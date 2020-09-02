'''
Loads checkpoint from specified location and returns model
'''
import torch
from torch import nn
from torch import optim
from torchvision import models

def load_checkpoint(path):
    
    #Code to load checkpoint    
    checkpoint = torch.load(path)
    model = checkpoint['model_arch']
    optimizer = optim.Adam(model.classifier.parameters(), lr = checkpoint['lr'])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    epoch = checkpoint['epochs']
    class_to_idx = checkpoint['class_to_idx']
    
    return model